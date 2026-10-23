#!/usr/bin/env python
"""Lotus_rollout_only — 학습 끝난 LOTUS 런의 체크포인트로 SR 행렬만 채운다.

    python Lotus_rollout_only.py --run results/lotus_v7_dea_b3.0_50_seed7
    python Lotus_rollout_only.py --run <dir> --stages "1 2" --eval_episodes 20
    python Lotus_rollout_only.py --run <dir> --watch        # 새 체크포인트를 기다린다

왜 따로 있는가
──────────────────────────────────────────────────────────────────────────────
Lotus_l2cb_cl.py 는 학습 루프 안에서 rollout_sr 을 직접 부른다. LIBERO-50 은
스테이지마다 **최대 50 칸**을 돌아서 롤아웃이 학습을 아주 오래 막는다.
그래서 학습은 --skip_eval 로 체크포인트만 남기고(GPU 하나), 롤아웃은 이
스크립트로 **다른 GPU** 에서 따로 돈다. 두 과정이 서로를 안 막는다.

CLARE 쪽 CLARE_rollout_only.py 와 같은 역할이고, 다른 점은 LOTUS 의
스테이지/태스크 구조(base n개 joint + stage별 추가)를 따른다는 것이다.

무엇을 읽는가
──────────────────────────────────────────────────────────────────────────────
    <run>/config.json                        suite, eval_episodes, eval_seed 등
    <run>/checkpoints/stage{k}/checkpoints/last/pretrained_model
                                             스테이지 k 학습 직후의 정책

무엇을 쓰는가
──────────────────────────────────────────────────────────────────────────────
    <run>/sr_rollout.json    R 행렬. 칸을 채울 때마다 갱신한다(중단 복구용)
    <run>/report.txt         Lotus_l2cb_cl.write_report 와 같은 형식
    <run>/metrics.json       학습 쪽과 같은 형식

★ 한 칸씩 저장하므로 중간에 죽어도 이미 잰 칸은 다시 안 돈다. 다시 실행하면
  빈 칸만 이어서 채운다 (--redo 로 전부 다시).

★ sr_rollout.json 이 **durable 한 기록**이다. metrics.json 은 거기서 매번 다시
  만든다. --skip_eval 로 돈 학습 런도 끝에 metrics.json 을 쓰는데(R 이 전부
  null 이다), 그게 이 스크립트의 결과를 덮을 수 있다. 정상 순서라면 학습이
  먼저 끝나고 롤아웃이 그 뒤에 계속 쓰므로 최종 상태는 맞다. 다만 이 스크립트가
  먼저 죽은 채로 학습이 끝나면 metrics.json 만 null 로 남는다 — 그때는 이걸
  (--watch 없이) 다시 한 번 돌리면 sr_rollout.json 에서 복원된다.

★ base 열에 대하여: 학습을 --skip_base_eval 로 돌렸으면 base 태스크(stage 0
  에서 joint 로 배운 것들)는 애초에 안 잰다. 여기서도 같은 기본값을 쓴다 —
  LIBERO-50 은 base 가 25 태스크라 스테이지마다 25 칸이 더 붙고 그게 전체
  시간의 대부분이다. --base_eval 로 켤 수 있다.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import Lotus_l2cb_cl as LC

REPO = Path(__file__).resolve().parent


def stage_policy_path(run: Path, k: int) -> Path | None:
    """스테이지 k 의 pretrained_model 경로. 없으면 None."""
    base = run / "checkpoints" / f"stage{k}" / "checkpoints"
    for cand in ("last", None):
        d = base / cand if cand else None
        if d is not None and (d / "pretrained_model" / "model.safetensors").exists():
            return d / "pretrained_model"
    # last 심링크가 없으면 숫자 폴더 중 가장 큰 것
    if base.is_dir():
        nums = sorted((p for p in base.iterdir() if p.name.isdigit()),
                      key=lambda p: int(p.name))
        for p in reversed(nums):
            if (p / "pretrained_model" / "model.safetensors").exists():
                return p / "pretrained_model"
    return None


def load_R(run: Path, n_stage: int, n_task: int):
    """저장된 R 을 읽어 온다. 모양이 안 맞으면 새로 만든다."""
    p = run / "sr_rollout.json"
    if p.exists():
        try:
            R = json.loads(p.read_text())["R"]
            if len(R) == n_stage + 1 and all(len(r) == n_task for r in R):
                return R
            logging.warning(f"[rollout] sr_rollout.json 모양 불일치 — 새로 만든다")
        except Exception as e:
            logging.warning(f"[rollout] sr_rollout.json 읽기 실패({e}) — 새로 만든다")
    return [[None] * n_task for _ in range(n_stage + 1)]


def save_R(run: Path, R, spec, learned_at, instr, args, n_task):
    (run / "sr_rollout.json").write_text(
        json.dumps({"R": R, "updated": time.strftime("%F %T")},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    m = LC.write_report(run / "report.txt", R, learned_at, instr, args, n_task, spec)
    json.dump({"suite": args.suite, "R": R, "learned_at": learned_at,
               "stage_tasks": spec["stage_tasks"],
               "env_task": {str(k): v for k, v in spec["env_task"].items()},
               "metrics": m,
               "instructions": {str(k): v for k, v in instr.items()}},
              (run / "metrics.json").open("w"), indent=2, ensure_ascii=False)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="학습 런의 출력 폴더")
    ap.add_argument("--stages", default="all", help='"all" 또는 "1 2 3"')
    ap.add_argument("--eval_episodes", type=int, default=0, help="0 = config.json 값")
    ap.add_argument("--eval_seed", type=int, default=-1, help="-1 = config.json 값")
    ap.add_argument("--base_eval", action="store_true",
                    help="base 태스크 열도 잰다 (LIBERO-50 은 스테이지마다 25 칸 추가)")
    ap.add_argument("--redo", action="store_true", help="이미 채운 칸도 다시 잰다")
    ap.add_argument("--watch", action="store_true",
                    help="체크포인트가 더 나올 때까지 기다리며 반복한다")
    ap.add_argument("--poll", type=int, default=300, help="--watch 의 확인 간격(초)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    LC.init_logging()
    run = Path(a.run)
    if not run.is_dir():
        raise SystemExit(f"런 폴더가 없다: {run}")
    cfgf = run / "config.json"
    if not cfgf.exists():
        raise SystemExit(f"config.json 이 없다: {cfgf}  (학습이 시작되긴 했나?)")
    cj = json.loads(cfgf.read_text())

    # ★ config.json 을 **통째로** args 로 되살린다. write_report 가 ridge_rho,
    #   h_scale, grad_min_frames 등 학습 인자를 두루 읽어서, 필요한 것만 골라
    #   담으면 AttributeError 가 하나씩 튀어나온다. 학습 때 쓴 값을 그대로
    #   쓰는 게 맞기도 하다 — eval_episodes/seed 가 달라지면 같은 표 안에서
    #   칸끼리 비교가 안 된다.
    args = SimpleNamespace(**cj)
    args.device = a.device
    args.skip_base_eval = not a.base_eval
    if a.eval_episodes:
        args.eval_episodes = a.eval_episodes
        args.eval_batch_size = a.eval_episodes
    if a.eval_seed >= 0:
        args.eval_seed = a.eval_seed
    for k, v in (("eval_episodes", 20), ("eval_batch_size", 20), ("eval_seed", 7),
                 ("steps", 10000), ("num_workers", 6), ("batch_size", 32),
                 ("n_stages", 0), ("seed", 42)):
        if not hasattr(args, k):
            setattr(args, k, v)

    spec = LC.suite_spec(args.suite)
    if args.n_stages and args.n_stages < len(spec["stage_tasks"]):
        spec["stage_tasks"] = spec["stage_tasks"][:args.n_stages]
        keep = spec["n_base"] + sum(len(x) for x in spec["stage_tasks"])
        spec["n_task"] = keep
        spec["env_task"] = {t: v for t, v in spec["env_task"].items() if t < keep}
        spec["instr"] = {t: v for t, v in spec["instr"].items() if t < keep}
        spec["learned_at"] = spec["learned_at"][:keep]

    n_task = spec["n_task"]
    n_stage = len(spec["stage_tasks"])
    env_task, instr = spec["env_task"], spec["instr"]
    learned_at = spec["learned_at"]
    want = (list(range(1, n_stage + 1)) if a.stages == "all"
            else [int(x) for x in a.stages.split()])

    logging.info(f"[rollout] {run.name}  suite={args.suite}  "
                 f"{n_stage} stage, {n_task} task, 칸당 {args.eval_episodes} rollout, "
                 f"eval seed {args.eval_seed}, base 열 {'측정' if a.base_eval else '건너뜀'}")

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy
    root = spec["root"]
    LC.get_safe_torch_device(args.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    while True:
        R = load_R(run, n_stage, n_task)
        did = 0
        missing_ckpt = 0        # 아직 학습이 안 끝난 스테이지 수
        for k in want:
            pol_path = stage_policy_path(run, k)
            if pol_path is None:
                logging.info(f"[rollout] stage{k} 체크포인트 아직 없음 — 건너뜀")
                missing_ckpt += 1
                continue
            # 이 스테이지에서 평가할 태스크: 여기까지 본 것 전부
            n_seen = spec["stage_tasks"][k - 1][-1] + 1
            todo = [t for t in range(n_seen)
                    if (a.redo or R[k][t] is None)
                    and (a.base_eval or learned_at[t] != 0)]
            if not todo:
                logging.info(f"[rollout] stage{k} 이미 다 채움")
                continue

            logging.info(f"[rollout] stage{k}: {len(todo)}칸  ({pol_path})")
            cfg = LC.build_eval_cfg(args, str(pol_path), env_task[todo[0]])
            meta = LeRobotDatasetMetadata("local/lotus", root=str(root / f"stage{k}"))
            policy = make_policy(cfg=cfg.policy, ds_meta=meta)
            policy.to(args.device)
            try:
                for t in todo:
                    t0 = time.perf_counter()
                    R[k][t] = LC.rollout_sr(policy, cfg, env_task[t], args)
                    did += 1
                    logging.info(f"[rollout] SR stage{k} task{t} ({env_task[t]}) = "
                                 f"{R[k][t]}  ({time.perf_counter()-t0:.0f}s)")
                    save_R(run, R, spec, learned_at, instr, args, n_task)
            finally:
                del policy
                torch.cuda.empty_cache()

        filled = sum(1 for row in R for c in row if c is not None)
        logging.info(f"[rollout] 이번 바퀴 {did}칸, 누적 {filled}칸 -> {run}/report.txt")
        if not a.watch:
            break
        # ★ --watch 의 종료 조건: 요청한 스테이지가 **전부 체크포인트를 갖고**
        #   더 채울 칸이 없을 때. 이게 없으면 학습이 다 끝나도 영원히 돌면서
        #   GPU 를 붙잡고 있어 다음 시드로 못 넘어간다.
        if missing_ckpt == 0 and did == 0:
            logging.info("[rollout] 모든 스테이지 완료 — 종료")
            break
        if did == 0:
            logging.info(f"[rollout] 남은 스테이지 {missing_ckpt}개 — 대기 {a.poll}s")
            time.sleep(a.poll)

    m = save_R(run, R, spec, learned_at, instr, args, n_task)
    print((run / "report.txt").read_text())
    logging.info(f"[rollout] 완료 — {m}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
