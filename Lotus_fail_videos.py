#!/usr/bin/env python3
"""LOTUS 런의 스테이지 체크포인트로 한 태스크를 굴려 **실패한 에피소드**만 영상으로 남긴다.

학습 때의 평가와 같은 조건(칸당 eval_episodes 롤아웃, eval_seed, 배치 = eval_batch_size)으로
굴린 뒤 실패한 에피소드 가운데 앞에서 n_fail 개를 mp4 로 쓴다. 각 프레임은
[agentview | 손목 카메라] 를 나란히 붙인 것이고, 요약 PNG(시작/¼/½/¾/끝 프레임)와
에피소드별 표(JSON)도 같이 남긴다.

    python Lotus_fail_videos.py --run results/<run> --stage 4 --task 9 --n_fail 3

★ gym_libero 의 init_state 카운터는 프로세스 전역이라, 새 프로세스에서 태스크 하나만 돌리면
  init state 0..N-1 이 쓰인다. 학습 중 평가는 같은 프로세스에서 t6..t9 를 이어 돌리므로
  칸마다 다른 init state 를 썼다 — 그래서 여기 SR 이 report.txt 의 칸과 조금 다를 수 있다.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from Lotus_L3_V7dea import (LeRobotDatasetMetadata, _stage_policy_path, build_eval_cfg,
                            make_policy, suite_spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--task", type=int, default=-1, help="스위트 태스크 id (기본: 그 스테이지가 배운 마지막 태스크)")
    ap.add_argument("--n_fail", type=int, default=3)
    ap.add_argument("--n_success", type=int, default=0, help="비교용 성공 에피소드도 이만큼 저장")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--res", type=int, default=384, help="카메라 한 장의 해상도")
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run = Path(a.run)
    cj = json.loads((run / "config.json").read_text())
    args = argparse.Namespace(**cj)
    args.device = a.device
    for k, v in (("eval_episodes", 20), ("eval_batch_size", 20), ("eval_seed", 7),
                 ("steps", 10000), ("num_workers", 4), ("batch_size", 32)):
        if not hasattr(args, k):
            setattr(args, k, v)
    spec = suite_spec(args.suite)
    task = a.task if a.task >= 0 else spec["stage_tasks"][a.stage - 1][-1]
    handle, instr = spec["env_task"][task], spec["instr"][task]
    pol = _stage_policy_path(run, a.stage)
    if pol is None:
        raise SystemExit(f"stage{a.stage} 체크포인트가 없다: {run}")
    out = Path(a.out) if a.out else run / f"videos_stage{a.stage}_t{task}"
    out.mkdir(parents=True, exist_ok=True)
    logging.info(f"{run.name}  stage{a.stage} → task{task} {handle}  '{instr}'\n  ckpt {pol}\n  out {out}")

    from lerobot.envs.factory import make_env
    from lerobot.scripts.eval import rollout
    from lerobot.utils.io_utils import write_video

    cfg = build_eval_cfg(args, str(pol), handle)
    meta = LeRobotDatasetMetadata("local/lotus", root=str(spec["root"] / f"stage{a.stage}"))
    policy = make_policy(cfg=cfg.policy, ds_meta=meta).to(a.device)
    policy.eval()

    n = args.eval_episodes
    env = make_env(cfg.env, n_envs=args.eval_batch_size, use_async_envs=False)
    raws = [e.env.env for e in env.envs]            # gym_libero.LiberoEnv

    def frame(raw) -> np.ndarray:
        sim = raw._env.sim
        a_ = np.flip(sim.render(camera_name="agentview", height=a.res, width=a.res), axis=0)
        w_ = np.flip(sim.render(camera_name="robot0_eye_in_hand", height=a.res, width=a.res), axis=0)
        return np.concatenate([a_, w_], axis=1)

    frames: list[np.ndarray] = []                  # step -> (b, H, 2W, 3)

    def cb(_env):
        frames.append(np.stack([frame(r) for r in raws]))

    seeds = list(range(args.eval_seed, args.eval_seed + env.num_envs))
    t0 = time.perf_counter()
    with torch.no_grad():
        rd = rollout(env, policy, seeds=seeds, render_callback=cb)
    # rollout() 이 reset 뒤에 _init_state_id 를 num_envs 만큼 앞당겨 둔다(다음 reset 용) → 되돌린다
    init_ids = [(int(r._init_state_id) - env.num_envs) % 50 for r in raws]
    n_steps = rd["done"].shape[1]
    done_idx = torch.argmax(rd["done"].to(int), dim=1)
    mask = (torch.arange(n_steps)[None, :] <= (done_idx + 1)[:, None]).int()
    succ = ((rd["success"] * mask).sum(1) > 0).tolist()[:n]
    maxr = (rd["reward"] * mask).max(1).values.tolist()[:n]
    logging.info(f"롤아웃 {time.perf_counter() - t0:.0f}s  SR {100 * np.mean(succ):.1f}%  "
                 f"({sum(succ)}/{n})  최대 {n_steps} 스텝")
    stacked = np.stack(frames, axis=1)            # (b, T+1, H, 2W, 3)

    rows, saved = [], {"fail": 0, "success": 0}
    for i in range(n):
        T = int(done_idx[i]) + 1
        rows.append({"ep": i, "seed": seeds[i], "init_state": init_ids[i], "success": bool(succ[i]),
                     "steps": T, "max_reward": float(maxr[i])})
        kind = "success" if succ[i] else "fail"
        quota = a.n_success if succ[i] else a.n_fail
        if saved[kind] >= quota:
            continue
        clip = stacked[i, : T + 1]
        stem = f"{kind}{saved[kind]}_ep{i}_init{init_ids[i]}_{T}step"
        write_video(str(out / f"{stem}.mp4"), clip, a.fps)
        idx = [0, len(clip) // 4, len(clip) // 2, 3 * len(clip) // 4, len(clip) - 1]
        strip = np.concatenate([clip[j] for j in idx], axis=1)
        try:
            import imageio
            imageio.imwrite(str(out / f"{stem}.png"), strip)
        except Exception as e:    # noqa: BLE001
            logging.warning(f"PNG 저장 실패: {e}")
        saved[kind] += 1
        logging.info(f"  저장 {stem}.mp4  ({len(clip)} 프레임)")
    summ = {"run": run.name, "stage": a.stage, "task": task, "handle": handle, "instruction": instr,
            "ckpt": str(pol), "sr": float(100 * np.mean(succ)), "episodes": rows}
    (out / "episodes.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    env.close()
    print(f"\n{run.name} stage{a.stage} task{task}  SR {summ['sr']:.1f}%")
    print(f"{'ep':>3} {'seed':>5} {'init':>4} {'succ':>5} {'steps':>5} {'maxR':>5}")
    for r in rows:
        print(f"{r['ep']:>3} {r['seed']:>5} {r['init_state']:>4} {str(r['success']):>5} {r['steps']:>5} {r['max_reward']:>5.2f}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
