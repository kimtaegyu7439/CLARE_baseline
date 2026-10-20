#!/usr/bin/env python
"""학습된 체크포인트로 **롤아웃만** 돌려 SR 행렬을 채운다. 학습은 하지 않는다.

    # median 팔: 이미 학습된 체크포인트로 20 rollout 만
    python CLARE_rollout_only.py \
        --ckpt_dir outputs/CLARE_V4/libero_goal_seed7/libero_goal_seed7_ours \
        --out results/clare_mar_goal_median_seed7 \
        --suite libero_goal --rows 0-6 --eval_episodes 20

    # 같은 출력 폴더에 다른 GPU 가 나머지 행을 채운다
    CUDA_VISIBLE_DEVICES=1 python CLARE_rollout_only.py ... --rows 7-9

왜 필요한가
    L2_V4_seq.py 의 --resume 은 **마지막 행의 빈 칸만** 채운다("그 시점 정책이
    없다"). 하지만 ckpt_root 아래에 task_0..task_9 체크포인트가 전부 남아 있으면
    행 k 는 task_k 체크포인트로 되살려 롤아웃할 수 있다. 이 파일이 그걸 한다.

    덕분에 학습과 평가를 **다른 GPU 로 분리**할 수 있다. 학습이 도는 동안 끝난
    스테이지의 행을 다른 카드가 채우면 전체 시간이 크게 준다.

★ 같은 --out 을 여러 프로세스가 공유해도 되게, 행 단위로 나눠 맡고 **자기 행만**
  쓴다. sr_matrix.csv 는 매번 디스크에서 다시 읽어 병합하므로 서로의 결과를
  지우지 않는다. 다만 동시에 같은 행을 맡기면 안 된다.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

import L2_V4_seq as B1


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True,
                   help="task_0..task_{K-1} 이 들어 있는 폴더 "
                        "(예: outputs/CLARE_V4/libero_goal_seed7/libero_goal_seed7_ours)")
    p.add_argument("--out", required=True, help="sr_matrix.csv 를 쓸 결과 폴더")
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--num_tasks", type=int, default=10)
    p.add_argument("--rows", default="all",
                   help="맡을 행. 'all' 또는 '0-6' 또는 '3,5,7'")
    p.add_argument("--eval_episodes", type=int, default=20)
    p.add_argument("--eval_batch_size", type=int, default=20)
    p.add_argument("--eval_seed", type=int, default=7)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps_per_task", type=int, default=10000)
    # write_outputs 가 보고서 머리말에 적으려고 읽는다. 롤아웃 자체엔 영향이 없다.
    p.add_argument("--episodes_per_task", type=int, default=50)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--overwrite", action="store_true",
                   help="이미 값이 있는 칸도 다시 잰다(기본은 건너뛴다)")
    return p.parse_args()


def parse_rows(spec: str, K: int) -> list[int]:
    if spec == "all":
        return list(range(K))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [k for k in out if 0 <= k < K]


def read_matrix(out_dir: Path, K: int) -> list[list[float | None]]:
    """디스크의 sr_matrix.csv 를 읽어 R 로. 없으면 빈 행렬."""
    R: list[list[float | None]] = [[None] * K for _ in range(K)]
    p = out_dir / "sr_matrix.csv"
    if not p.exists():
        return R
    body = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
    for line in body[1:]:
        c = line.split(",")
        k = int(c[0])
        if k >= K:
            continue
        for i, v in enumerate(c[1:K + 1]):
            if v.strip():
                R[k][i] = float(v)
    return R


def main() -> int:
    a = parse()
    B1.init_logging()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(a.ckpt_dir)
    K = a.num_tasks
    rows = parse_rows(a.rows, K)
    _ds_prefix, env_prefix = B1.suite_prefixes(a.suite)

    logging.info(f"[rollout] ckpt={ckpt_dir}")
    logging.info(f"[rollout] out={out_dir}  행 {rows}  칸당 {a.eval_episodes} rollout")

    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy

    t_all = time.perf_counter()
    n_done = 0
    for k in rows:
        link = ckpt_dir / f"task_{k}" / "checkpoints" / "last" / "pretrained_model"
        if not (link / "model.safetensors").exists():
            logging.warning(f"[rollout] 행 {k} 건너뜀 — 체크포인트 없음: {link}")
            continue

        # ★ 디스크에서 매번 다시 읽는다. 다른 GPU 프로세스가 채운 칸을 덮어쓰지 않기 위해서다.
        R = read_matrix(out_dir, K)
        todo = [i for i in range(k + 1) if a.overwrite or R[k][i] is None]
        if not todo:
            logging.info(f"[rollout] 행 {k} 이미 다 참 — 건너뜀")
            continue

        cfg = B1.build_cfg(a, k, str(link), out_dir)
        dataset = make_dataset(cfg)
        policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
        policy.eval()
        del dataset

        for i in todo:
            t0 = time.perf_counter()
            sr = B1.rollout_sr(policy, cfg, f"{env_prefix}{i}", a)
            # 쓰기 직전에 다시 읽어 병합한다 — 그 사이 다른 프로세스가 쓴 칸을 보존한다.
            R = read_matrix(out_dir, K)
            R[k][i] = sr
            B1.write_outputs(out_dir, R, K, a)
            n_done += 1
            logging.info(f"[rollout] SR stage {k} task {i} = {sr}  ({time.perf_counter()-t0:.0f}s)")

        del policy
        torch.cuda.empty_cache()

    logging.info(f"[rollout] 끝 — {n_done} 칸, {(time.perf_counter()-t_all)/60:.1f}분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
