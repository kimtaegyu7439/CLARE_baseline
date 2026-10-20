#!/usr/bin/env python
"""LOTUS 셋팅 **seq-FT** 기준선 — 아무 대책 없는 순차 미세조정.

    python L2_seq_cl.py --suite object --seed 42 --out results/lotus_seq_object_seed42

L2_V4.py 와 **완전히 같은 조건**에서 앵커만 뺀 것이다. 비교 대상이므로 다른 것을
바꾸면 안 된다. 구체적으로 L2_V4.main() 과 같은 것:

    · base 체크포인트(Lotus_clare_libero_<suite>_pretrain)에서 출발
    · 스테이지당 --steps(기본 10000), batch 32, 옵티마이저/스케줄러를 스테이지마다 재생성
    · 같은 DataLoader 설정(shuffle, drop_last=False, spawn, persistent_workers)
    · 같은 FM 경로: prep_batch -> rgb_cls -> cond_tail -> sample_fm -> velocity_net -> fm_loss
    · 같은 grad clip, 같은 평가(칸당 --eval_episodes 롤아웃, --eval_seed 고정)
    · 같은 report/metrics 형식

다른 것은 하나뿐이다.

    v4    L = L_FM(현재) + λ_a·λ_lvl·L_anchor      코드북·teacher 유지
    seq   L = L_FM(현재)                           코드북·teacher 없음

★ `L2_V4.py --lambda_level 0` 으로 대신하지 않는 이유: 앵커 경로가 randn 을 소비해
  FM 쪽 RNG 스트림이 밀린다. 같은 seed 42 에서의 대조가 목적이므로 앵커 계산 자체를
  하지 않는 전용 드라이버를 쓴다. (덤으로 스테이지당 1~1.5 시간이 빠르다)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

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


def write_seq_report(path: Path, R, learned_at, instr, args, n_task, spec) -> dict:
    """seq-FT 전용 리포트. V4.write_report 는 헤더에 코드북·λ_lvl 을 박으므로
    (seq 에는 없는 것들) 지표 계산만 V4.metrics 로 공유하고 본문은 따로 쓴다."""
    m = V4.metrics(R, learned_at, n_task)
    n_stage = len(spec["stage_tasks"])
    L = ["=" * 92,
         f"LOTUS 셋팅 lifelong — libero_{args.suite}   **seq-FT (기준선, 앵커·리플레이 없음)**",
         "=" * 92, "",
         f"손실     L = L_FM(현재)  그것뿐이다. 과거 보존 장치가 없다.",
         f"프로토콜 base 체크포인트에서 출발, stage 당 {args.steps:,} step",
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
          "  · seq-FT 는 망각의 상한(최악)을 보는 기준선이다. 방법들이 이보다 나쁘면 문제다.",
          ""]
    path.write_text("\n".join(L) + "\n")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["goal", "object", "50"], required=True)
    ap.add_argument("--steps", type=int, default=10000, help="lifelong 스테이지당 스텝")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--n_stages", type=int, default=0, help="0 = 전부. 스모크용 축소")
    ap.add_argument("--seed", type=int, default=42, help="학습 시드")
    ap.add_argument("--eval_seed", type=int, default=7, help="롤아웃 시작 시드")
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=20)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    # report/eval 쪽이 getattr 로 읽는 v4 전용 인자들 — seq 에서는 쓰이지 않지만
    # write_report 가 args.<이름> 을 참조하므로 같은 이름으로 채워 둔다.
    ap.add_argument("--sampler_mode", default="seq_ft")
    ap.add_argument("--anchor_weight", default="none")
    ap.add_argument("--lambda_level", type=float, default=0.0)
    ap.add_argument("--k_per_demo", type=int, default=0)
    ap.add_argument("--k_min", type=int, default=0)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--w_clip", type=float, default=0.0)
    args = ap.parse_args()

    init_logging()
    spec = V4.suite_spec(args.suite)
    if args.n_stages and args.n_stages < len(spec["stage_tasks"]):
        spec["stage_tasks"] = spec["stage_tasks"][:args.n_stages]
        keep = spec["n_base"] + sum(len(x) for x in spec["stage_tasks"])
        spec["n_task"] = keep
        spec["env_task"] = {t: v for t, v in spec["env_task"].items() if t < keep}
        spec["instr"] = {t: v for t, v in spec["instr"].items() if t < keep}
        spec["learned_at"] = spec["learned_at"][:keep]
        logging.info(f"[seq] ★ 스모크 축소: stage {args.n_stages}개, task {keep}개")

    root = spec["root"]
    n_base, n_task = spec["n_base"], spec["n_task"]
    stage_tasks = spec["stage_tasks"]
    n_stages = len(stage_tasks)
    env_task, instr = spec["env_task"], spec["instr"]
    learned_at = spec["learned_at"]

    base_ckpt = V4.MODEL_ROOT / f"Lotus_clare_libero_{args.suite}_pretrain"
    if not (base_ckpt / "model.safetensors").exists():
        raise SystemExit(f"base 체크포인트가 없다: {base_ckpt}")
    out_dir = Path(args.out) if args.out else REPO / "results" / f"lotus_seq_{args.suite}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), (out_dir / "config.json").open("w"), indent=2, ensure_ascii=False)

    device = get_safe_torch_device(args.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    R: list[list[float | None]] = [[None] * n_task for _ in range(n_stages + 1)]

    # 명령어 대조 — v4 와 같은 검사를 한다. seq 는 앵커가 없어 이 문자열을 학습에
    # 쓰지는 않지만, 평가 핸들과 스펙이 어긋났는지 여기서 먼저 걸러야 한다.
    for split, ids in [("base", list(range(n_base)))] + [
            (f"stage{i+1}", ids) for i, ids in enumerate(stage_tasks)]:
        have = set(V4.instructions_of(root / split).values())
        miss = [instr[t] for t in ids if instr[t] not in have]
        if miss:
            raise SystemExit(f"{split}: lotus_setting.json 의 명령어가 데이터셋에 없다 "
                             f"({len(miss)}개, 예: {miss[0]!r})")
    logging.info(f"[seq] suite={args.suite}  base {n_base} task  "
                 f"{n_stages} stage x {len(stage_tasks[0])} task  총 {n_task} task  "
                 f"seed={args.seed} eval_seed={args.eval_seed}")

    # ── stage 0: base 체크포인트 ─────────────────────────────────────────────
    cfg = V4.build_eval_cfg(args, str(base_ckpt), env_task[0])
    base_meta = LeRobotDatasetMetadata("local/lotus", root=str(root / "base"))
    policy = make_policy(cfg=cfg.policy, ds_meta=base_meta)
    policy.to(device)
    logging.info(f"[seq] base 체크포인트 로드 -> {base_ckpt}")

    for t in range(n_base):
        R[0][t] = V4.rollout_sr(policy, cfg, env_task[t], args)
        logging.info(f"[seq] SR stage0 task{t} ({env_task[t]}) = {R[0][t]}")
        write_seq_report(out_dir / "report.txt", R, learned_at, instr, args, n_task, spec)

    # ── stage 1..N: FM 손실만 ────────────────────────────────────────────────
    t_start = time.perf_counter()
    for stage in range(1, n_stages + 1):
        new_tasks = stage_tasks[stage - 1]
        n_seen = new_tasks[-1] + 1
        sroot = root / f"stage{stage}"
        ds = V4.load_split(sroot, policy.config)
        loader = torch.utils.data.DataLoader(
            ds, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True,
            pin_memory=device.type == "cuda", drop_last=False,
            multiprocessing_context="spawn" if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0)
        dl = cycle(loader)

        cfg.steps = args.steps
        optimizer, sched = make_optimizer_and_scheduler(cfg, policy)
        policy.train()
        logging.info(f"══ [seq] stage {stage}  새 태스크 t{new_tasks[0]}..t{new_tasks[-1]} "
                     f"({len(new_tasks)}개)  {ds.num_frames} frame  앵커 없음")

        for it in range(args.steps):
            batch = V4.prep_batch(policy, to_device(next(dl), device))
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
                logging.info(f"[seq] stage{stage} step={it:6d} "
                             f"fm={float(loss_fm.detach()):.4f} "
                             f"lr={optimizer.param_groups[0]['lr']:.2e}")
                with (out_dir / "diag.csv").open("a") as f:
                    f.write(f"{stage},{it},{float(loss_fm.detach()):.6f},0.0\n")

        sdir = out_dir / "checkpoints" / f"stage{stage}"
        ck = get_step_checkpoint_dir(sdir, args.steps, args.steps)
        save_checkpoint(ck, args.steps, cfg, policy, optimizer, sched)
        update_last_checkpoint(ck)
        del loader, dl, ds
        torch.cuda.empty_cache()

        for t in range(n_seen):
            R[stage][t] = V4.rollout_sr(policy, cfg, env_task[t], args)
            logging.info(f"[seq] SR stage{stage} task{t} ({env_task[t]}) = {R[stage][t]}")
            write_seq_report(out_dir / "report.txt", R, learned_at, instr, args, n_task, spec)

    m = write_seq_report(out_dir / "report.txt", R, learned_at, instr, args, n_task, spec)
    json.dump({"suite": args.suite, "method": "seq_ft", "R": R, "learned_at": learned_at,
               "stage_tasks": stage_tasks,
               "env_task": {str(k): v for k, v in env_task.items()},
               "metrics": m, "instructions": {str(k): v for k, v in instr.items()},
               "wall_hours": (time.perf_counter() - t_start) / 3600},
              (out_dir / "metrics.json").open("w"), indent=2, ensure_ascii=False)
    print((out_dir / "report.txt").read_text())
    logging.info(f"[seq] 완료 -> {out_dir}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
