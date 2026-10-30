#!/usr/bin/env python3
"""학습이 끝난 LOTUS 런을 **평가 시드만 바꿔** 여러 번 다시 굴린다 — 평가 노이즈 측정 / 시드 고정용.

기존 --eval_seed 는 롤아웃에서 사실상 아무것도 고정하지 않는다: gym_libero 는 reset(seed) 를
init state 선택에 쓰지 않고(프로세스 전역 카운터), 정책의 flow 초기 노이즈는 전역 RNG 에서 나온다.
여기서는 평가 시드 s 가 두 가지를 **결정**하게 한다.
  1. init state 묶음: 칸마다 카운터를 (s*20) % 50 으로 되돌린다 → 그 칸은 init state
     offset..offset+19 (mod 50) 를 쓴다. s 가 같으면 어떤 순서로 돌려도 같은 배치다.
  2. 정책 노이즈: task t 마다 torch.Generator(s*1000 + t) 를 policy(.dit_flow).eval_generator 로
     꽂는다 — 같은 task 는 stage 가 달라도 같은 노이즈 (modeling_dit_flow_mt.select_action 이 읽는다).
  + cudnn.benchmark 끄고 deterministic 알고리즘(warn_only) → 같은 GPU 에서는 재실행이 일치해야 한다.

    python Lotus_eval_seeds.py --run results/<run> --seeds "0 1 2 3" [--repeat 0]
      --stages "1 2 3 4" (기본 전부)   --device cuda
결과: <run>/eval_seeds/seed<s>[_r<i>].json (R + 지표) 와 results/LOTUS_eval_seed_sweep.txt 요약.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from Lotus_L3_V7dea import (LeRobotDatasetMetadata, _stage_policy_path, build_eval_cfg,  # noqa: E402
                            make_policy, metrics, suite_spec)

OUT_TXT = Path("results/LOTUS_eval_seed_sweep.txt")


def eval_cell(policy, cfg, handle, args, seed, k, t):
    import copy
    from gym_libero.env import LiberoEnv as GLEnv
    from lerobot.envs.factory import make_env
    from lerobot.scripts.eval import eval_policy
    GLEnv.init_state_couter = (seed * 20) % 50
    # task 기준으로 시드를 고정한다 — 같은 task 는 어느 stage 에서 재도 같은 노이즈 (stage 간 차이=NBT 의 분산을 줄인다)
    g = torch.Generator(device=args.device).manual_seed(seed * 1000 + t)
    policy.eval_generator = g
    if hasattr(policy, "dit_flow"):
        policy.dit_flow.eval_generator = g       # select_action/conditional_sample 은 dit_flow 안에 있다
    env_cfg = copy.deepcopy(cfg.env); env_cfg.task = handle
    env = make_env(env_cfg, n_envs=args.eval_batch_size, use_async_envs=False)
    try:
        policy.eval()
        with torch.no_grad():
            info = eval_policy(env, policy, args.eval_episodes, start_seed=seed)
        return float(info["aggregated"]["pc_success"])
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", default="0 1 2 3")
    ap.add_argument("--repeat", type=int, default=None, help="이 시드는 한 번 더 돌려 재현 여부를 본다")
    ap.add_argument("--stages", default="all")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    run = Path(a.run)
    cj = json.loads((run / "config.json").read_text())
    args = argparse.Namespace(**cj); args.device = a.device
    for k, v in (("eval_episodes", 20), ("eval_batch_size", 20), ("steps", 10000),
                 ("num_workers", 4), ("batch_size", 32)):
        if not hasattr(args, k):
            setattr(args, k, v)
    spec = suite_spec(args.suite)
    n_task, n_stage = spec["n_task"], len(spec["stage_tasks"])
    learned_at, env_task = spec["learned_at"], spec["env_task"]
    stages = list(range(1, n_stage + 1)) if a.stages == "all" else [int(x) for x in a.stages.split()]
    jobs = [(int(s), "") for s in a.seeds.split()]
    if a.repeat is not None:
        jobs.append((a.repeat, "_r1"))
    out = run / "eval_seeds"; out.mkdir(exist_ok=True)

    for seed, suffix in jobs:
        f = out / f"seed{seed}{suffix}.json"
        if f.exists():
            logging.info(f"⏭ {f.name} 있음"); continue
        R = [[None] * n_task for _ in range(n_stage + 1)]
        t0 = time.perf_counter()
        for k in stages:
            pol = _stage_policy_path(run, k)
            if pol is None:
                logging.warning(f"stage{k} 체크포인트 없음"); continue
            n_seen = spec["stage_tasks"][k - 1][-1] + 1
            todo = [t for t in range(n_seen) if learned_at[t] != 0]
            cfg = build_eval_cfg(args, str(pol), env_task[todo[0]])
            meta = LeRobotDatasetMetadata("local/lotus", root=str(spec["root"] / f"stage{k}"))
            policy = make_policy(cfg=cfg.policy, ds_meta=meta).to(a.device)
            try:
                for t in todo:
                    t1 = time.perf_counter()
                    R[k][t] = eval_cell(policy, cfg, env_task[t], args, seed, k, t)
                    logging.info(f"seed{seed}{suffix} stage{k} t{t} = {R[k][t]:.0f}  ({time.perf_counter() - t1:.0f}s)")
            finally:
                del policy; torch.cuda.empty_cache()
        m = metrics(R, learned_at, n_task)
        rec = {"run": run.name, "seed": seed, "suffix": suffix, "R": R, "metrics": m,
               "init_offset": (seed * 20) % 50, "sec": time.perf_counter() - t0,
               "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
        f.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        line = (f"{run.name:58s} seed{seed:<3d}{suffix:4s} init{rec['init_offset']:>2d}..  "
                f"AvgSR {m['AvgSR']:5.1f}  FWT {m['FWT']:5.1f}  NBT {m['NBT']:+5.1f}  AUC {m['AUC']:5.1f}  "
                f"t{n_task - 1} {R[-1][n_task - 1]:4.0f}   {time.strftime('%m-%d %H:%M')}")
        logging.info(line)
        with OUT_TXT.open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX); fh.write(line + "\n")


if __name__ == "__main__":
    main()
