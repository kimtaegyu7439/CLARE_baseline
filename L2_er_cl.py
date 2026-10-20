#!/usr/bin/env python
"""LOTUS 셋팅 **ER(experience replay)** 기준선 — 과거 에피소드를 배치에 섞는다.

    python L2_er_cl.py --suite object --seed 42 --out results/lotus_er_object_seed42

L2_seq_cl.py 와 **완전히 같은 조건**에서 재생 버퍼만 더한 것이다. 비교 대상이므로
다른 것을 바꾸면 안 된다. 같은 것:

    · base 체크포인트(Lotus_clare_libero_<suite>_pretrain)에서 출발
    · 스테이지당 --steps(기본 10000), 옵티마이저/스케줄러를 스테이지마다 재생성
    · 같은 FM 경로: prep_batch -> rgb_cls -> cond_tail -> sample_fm -> velocity_net -> fm_loss
    · 같은 grad clip, 같은 평가(칸당 --eval_episodes 롤아웃, --eval_seed 고정)
    · 같은 report/metrics 형식

다른 것은 하나뿐이다.

    seq   배치 32 = 현재 32                     과거 보존 장치 없음
    ER    배치 32 = 현재 24 + 버퍼 8            과거 에피소드를 섞어 넣는다

★ 총 배치를 32 로 맞춘 이유: seq-FT 기준선과 스텝당 표본 수가 같아야 "버퍼를 넣은
  효과"만 남는다. 배치를 40(32+8)으로 키우면 표본이 늘어난 효과와 섞인다.
  bash/er/ER_suite.sh 의 CLARE 셋팅 ER 도 같은 24+8 분할을 쓴다.

★ 버퍼 범위: base 스플릿 + 지난 스테이지 전부. base 태스크도 평가 대상(R 의 열)이라
  보존 대상에 들어간다. 과거 **태스크당** --buffer_ep 에피소드를 무작위로 고르고,
  그 선택은 --seed 로 고정한다.

  goal/object  base 6 task x 5 = 30 ep + 지난 stage 마다 1 task x 5
  50           base 25 task x 5 = 125 ep + 지난 stage 마다 5 task x 5

    python L2_er_cl.py --suite 50 --seed 7 --out results/lotus_er_50_seed7
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lerobot_lsy" / "src"))

import L2_V4 as V4                                                     # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata    # noqa: E402
from lerobot.datasets.utils import cycle                               # noqa: E402
from lerobot.optim.factory import make_optimizer_and_scheduler         # noqa: E402
from lerobot.policies.factory import make_policy                       # noqa: E402
from lerobot.scripts.E0 import to_device                               # noqa: E402
from lerobot.utils.random_utils import set_seed                        # noqa: E402
from lerobot.utils.train_utils import (                                # noqa: E402
    get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint,
)
from lerobot.utils.utils import get_safe_torch_device, init_logging    # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
#  재생 버퍼
# ═════════════════════════════════════════════════════════════════════════════
def _episode_task(ds):
    """(프레임별 episode_index, 프레임별 task_index) 를 numpy 로."""
    hf = ds.hf_dataset
    ep = np.asarray(hf.data.column("episode_index").to_pylist(), dtype=np.int64)
    ti = np.asarray(hf.data.column("task_index").to_pylist(), dtype=np.int64)
    return ep, ti


def build_replay(splits: list[Path], policy_cfg, frac: float, seed: int):
    """과거 스플릿들에서 **태스크당 에피소드의 frac 비율**만 골라 ConcatDataset 으로.

    비율로 두는 이유: base 는 태스크당 50 demo, stage 는 10 demo 라 고정 개수를 쓰면
    스플릿마다 보관 비율이 달라진다. 10% 면 base 5 개 / stage 1 개로 일정하다.

    ★ LeRobotDataset(episodes=[...]) 를 쓰지 않는다. episode_data_index 가 선택
      에피소드 기준으로 다시 매겨지는데 __getitem__ 은 원본 episode_index 로
      인덱싱해서 어긋난다(E0.py:197 의 주석과 같은 이유). 통째로 열고 Subset 으로 자른다.
    """
    subs, n_ep, n_frame = [], 0, 0
    for sroot in splits:
        ds = V4.load_split(sroot, policy_cfg)
        ep, ti = _episode_task(ds)
        rng = np.random.default_rng(seed + hash(sroot.name) % 10_000)
        keep = []
        for t in np.unique(ti):
            eps_t = np.unique(ep[ti == t])
            k = max(1, int(round(frac * len(eps_t))))      # 50 -> 5, 10 -> 1
            keep += list(rng.choice(eps_t, min(k, len(eps_t)), replace=False))
        idx = np.nonzero(np.isin(ep, keep))[0]
        subs.append(torch.utils.data.Subset(ds, idx.tolist()))
        n_ep += len(keep)
        n_frame += len(idx)
    return torch.utils.data.ConcatDataset(subs), n_ep, n_frame


def merge_batch(b1: dict, b2: dict) -> dict:
    """현재 배치와 버퍼 배치를 배치 축으로 잇는다. 정규화 전에 합쳐야 한다."""
    out = {}
    for k, v1 in b1.items():
        v2 = b2.get(k)
        if v2 is None:
            out[k] = v1
        elif torch.is_tensor(v1):
            out[k] = torch.cat([v1, v2], dim=0)
        elif isinstance(v1, list):
            out[k] = list(v1) + list(v2)
        else:
            out[k] = v1
    return out


def write_er_report(path: Path, R, learned_at, instr, args, n_task, spec) -> dict:
    """ER 전용 리포트. 지표 계산은 V4.metrics 로 공유하고 본문만 따로 쓴다."""
    m = V4.metrics(R, learned_at, n_task)
    n_stage = len(spec["stage_tasks"])
    L = ["=" * 92,
         f"LOTUS 셋팅 lifelong — libero_{args.suite}   **ER (경험 재생 기준선)**",
         "=" * 92, "",
         f"손실     L = L_FM(현재 {args.batch_size - args.replay_batch_size} + "
         f"버퍼 {args.replay_batch_size})",
         f"버퍼     base + 지난 스테이지, 과거 태스크당 에피소드의 "
         f"{args.buffer_frac:.0%} (base 5 / stage 1)",
         f"프로토콜 base 체크포인트에서 출발, stage 당 {args.steps:,} step, "
         f"총 배치 {args.batch_size} (seq-FT 와 동일)",
         f"평가     칸당 {args.eval_episodes} rollout, train seed {args.seed} / "
         f"eval seed {args.eval_seed}", "",
         "행 = 스테이지 k 까지 배운 뒤, 열 = 평가 태스크. 단위 %.", "",
         f"{'stage':<8}" + "".join(f"{'t'+str(t):>6}" for t in range(n_task))]
    for k in range(n_stage + 1):
        L.append(f"{k:<8}" + "".join(
            (f"{R[k][t]:6.0f}" if R[k][t] is not None else f"{'':>6}")
            for t in range(n_task)))
    f = lambda v: "n/a" if v is None else f"{v:.1f}"                    # noqa: E731
    L += ["", f"  지표  AvgSR {f(m['AvgSR'])}   FWT↑ {f(m['FWT'])}   "
              f"NBT↓ {f(m['NBT'])}   AUC↑ {f(m['AUC'])}", "",
          "태스크별 명령어"]
    L += [f"  t{t:<3} (배운 스테이지 {learned_at[t]})  {instr[t]}" for t in range(n_task)]
    L += ["", "주의",
          f"  · 칸당 {args.eval_episodes} rollout 이라 이항 표준오차가 개별 칸 최대 "
          f"±{50 / (args.eval_episodes ** 0.5):.0f}%p 다. 행/열 평균으로 읽어야 한다.",
          "  · seq-FT 와 총 배치가 같다. 차이는 그중 8 개가 과거에서 온다는 것뿐이다.",
          ""]
    path.write_text("\n".join(L) + "\n")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["goal", "object", "50"], required=True)
    ap.add_argument("--steps", type=int, default=10000, help="lifelong 스테이지당 스텝")
    ap.add_argument("--batch_size", type=int, default=32, help="현재+버퍼 합")
    ap.add_argument("--replay_batch_size", type=int, default=8, help="그중 버퍼 몫")
    ap.add_argument("--buffer_frac", type=float, default=0.10,
                    help="과거 태스크당 보관 에피소드 **비율**. base 50 demo -> 5 개, "
                         "stage 10 demo -> 1 개. 최소 1 개는 남긴다.")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--replay_num_workers", type=int, default=3)
    ap.add_argument("--n_stages", type=int, default=0, help="0 = 전부. 스모크용 축소")
    ap.add_argument("--seed", type=int, default=42, help="학습 시드")
    ap.add_argument("--eval_seed", type=int, default=7, help="롤아웃 시작 시드")
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=20)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    # report/eval 쪽이 getattr 로 읽는 v4 전용 인자들 — ER 에서는 쓰이지 않는다.
    ap.add_argument("--sampler_mode", default="er")
    ap.add_argument("--anchor_weight", default="none")
    ap.add_argument("--lambda_level", type=float, default=0.0)
    ap.add_argument("--k_per_demo", type=int, default=0)
    ap.add_argument("--k_min", type=int, default=0)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--w_clip", type=float, default=0.0)
    args = ap.parse_args()

    cur_bs = args.batch_size - args.replay_batch_size
    if cur_bs <= 0:
        raise SystemExit("--replay_batch_size 가 --batch_size 보다 작아야 한다")

    init_logging()
    spec = V4.suite_spec(args.suite)
    if args.n_stages and args.n_stages < len(spec["stage_tasks"]):
        spec["stage_tasks"] = spec["stage_tasks"][:args.n_stages]
        keep = spec["n_base"] + sum(len(x) for x in spec["stage_tasks"])
        spec["n_task"] = keep
        spec["env_task"] = {t: v for t, v in spec["env_task"].items() if t < keep}
        spec["instr"] = {t: v for t, v in spec["instr"].items() if t < keep}
        spec["learned_at"] = spec["learned_at"][:keep]
        logging.info(f"[er] ★ 스모크 축소: stage {args.n_stages}개, task {keep}개")

    root = spec["root"]
    n_base, n_task = spec["n_base"], spec["n_task"]
    stage_tasks = spec["stage_tasks"]
    n_stages = len(stage_tasks)
    env_task, instr = spec["env_task"], spec["instr"]
    learned_at = spec["learned_at"]

    base_ckpt = V4.MODEL_ROOT / f"Lotus_clare_libero_{args.suite}_pretrain"
    if not (base_ckpt / "model.safetensors").exists():
        raise SystemExit(f"base 체크포인트가 없다: {base_ckpt}")
    out_dir = Path(args.out) if args.out else REPO / "results" / f"lotus_er_{args.suite}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), (out_dir / "config.json").open("w"), indent=2, ensure_ascii=False)

    device = get_safe_torch_device(args.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    R: list[list[float | None]] = [[None] * n_task for _ in range(n_stages + 1)]

    # 명령어 대조 — seq/v4 와 같은 검사.
    for split, ids in [("base", list(range(n_base)))] + [
            (f"stage{i+1}", ids) for i, ids in enumerate(stage_tasks)]:
        have = set(V4.instructions_of(root / split).values())
        miss = [instr[t] for t in ids if instr[t] not in have]
        if miss:
            raise SystemExit(f"{split}: lotus_setting.json 의 명령어가 데이터셋에 없다 "
                             f"({len(miss)}개, 예: {miss[0]!r})")
    logging.info(f"[er] suite={args.suite}  base {n_base} task  "
                 f"{n_stages} stage x {len(stage_tasks[0])} task  총 {n_task} task  "
                 f"배치 {cur_bs}+{args.replay_batch_size}  버퍼 {args.buffer_frac:.0%}/task  "
                 f"seed={args.seed} eval_seed={args.eval_seed}")

    # ── stage 0: base 체크포인트 ─────────────────────────────────────────────
    cfg = V4.build_eval_cfg(args, str(base_ckpt), env_task[0])
    base_meta = LeRobotDatasetMetadata("local/lotus", root=str(root / "base"))
    policy = make_policy(cfg=cfg.policy, ds_meta=base_meta)
    policy.to(device)
    logging.info(f"[er] base 체크포인트 로드 -> {base_ckpt}")

    for t in range(n_base):
        R[0][t] = V4.rollout_sr(policy, cfg, env_task[t], args)
        logging.info(f"[er] SR stage0 task{t} ({env_task[t]}) = {R[0][t]}")
        write_er_report(out_dir / "report.txt", R, learned_at, instr, args, n_task, spec)

    # ── stage 1..N: 현재 + 버퍼 ──────────────────────────────────────────────
    t_start = time.perf_counter()
    for stage in range(1, n_stages + 1):
        new_tasks = stage_tasks[stage - 1]
        n_seen = new_tasks[-1] + 1
        sroot = root / f"stage{stage}"
        ds = V4.load_split(sroot, policy.config)
        loader = torch.utils.data.DataLoader(
            ds, num_workers=args.num_workers, batch_size=cur_bs, shuffle=True,
            pin_memory=device.type == "cuda", drop_last=False,
            multiprocessing_context="spawn" if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0)
        dl = cycle(loader)

        # 버퍼 = base + 지난 스테이지 전부
        past = [root / "base"] + [root / f"stage{s}" for s in range(1, stage)]
        rep_ds, n_ep, n_fr = build_replay(past, policy.config, args.buffer_frac, args.seed)
        rep_loader = torch.utils.data.DataLoader(
            rep_ds, num_workers=args.replay_num_workers,
            batch_size=args.replay_batch_size, shuffle=True,
            pin_memory=device.type == "cuda", drop_last=True,
            multiprocessing_context="spawn" if args.replay_num_workers > 0 else None,
            persistent_workers=args.replay_num_workers > 0)
        rep_dl = cycle(rep_loader)

        cfg.steps = args.steps
        optimizer, sched = make_optimizer_and_scheduler(cfg, policy)
        policy.train()
        logging.info(f"══ [er] stage {stage}  새 태스크 t{new_tasks[0]}..t{new_tasks[-1]} "
                     f"({len(new_tasks)}개)  {ds.num_frames} frame  "
                     f"버퍼 {len(past)} 스플릿 / {n_ep} ep / {n_fr} frame")

        for it in range(args.steps):
            batch = merge_batch(next(dl), next(rep_dl))
            batch = V4.prep_batch(policy, to_device(batch, device))
            cls = V4.rgb_cls(policy, batch)
            tail = V4.cond_tail(policy, batch, cls)
            x_t, t_, target = V4.sample_fm(policy, batch)          # target = a − ε
            cond = V4.make_cond(V4.encode_lang(policy, list(batch["task"])), tail)
            pred = policy.dit_flow.velocity_net(noisy_actions=x_t, time=t_,
                                                global_cond=cond)
            loss_fm = V4.fm_loss(pred, target)
            loss_fm.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           cfg.optimizer.grad_clip_norm,
                                           error_if_nonfinite=False)
            optimizer.step()
            optimizer.zero_grad()
            if sched is not None:
                sched.step()

            if it % args.log_every == 0 or it == args.steps - 1:
                logging.info(f"[er] stage{stage} step={it:6d} "
                             f"fm={float(loss_fm.detach()):.4f} "
                             f"lr={optimizer.param_groups[0]['lr']:.2e}")
                with (out_dir / "diag.csv").open("a") as f:
                    f.write(f"{stage},{it},{float(loss_fm.detach()):.6f},0.0\n")

        sdir = out_dir / "checkpoints" / f"stage{stage}"
        ck = get_step_checkpoint_dir(sdir, args.steps, args.steps)
        save_checkpoint(ck, args.steps, cfg, policy, optimizer, sched)
        update_last_checkpoint(ck)
        del loader, dl, ds, rep_loader, rep_dl, rep_ds
        torch.cuda.empty_cache()

        for t in range(n_seen):
            R[stage][t] = V4.rollout_sr(policy, cfg, env_task[t], args)
            logging.info(f"[er] SR stage{stage} task{t} ({env_task[t]}) = {R[stage][t]}")
            write_er_report(out_dir / "report.txt", R, learned_at, instr, args, n_task, spec)

    m = write_er_report(out_dir / "report.txt", R, learned_at, instr, args, n_task, spec)
    json.dump({"suite": args.suite, "method": "er", "R": R, "learned_at": learned_at,
               "stage_tasks": stage_tasks,
               "env_task": {str(k): v for k, v in env_task.items()},
               "metrics": m, "instructions": {str(k): v for k, v in instr.items()},
               "wall_hours": (time.perf_counter() - t_start) / 3600},
              (out_dir / "metrics.json").open("w"), indent=2, ensure_ascii=False)
    print((out_dir / "report.txt").read_text())
    logging.info(f"[er] 완료 -> {out_dir}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
