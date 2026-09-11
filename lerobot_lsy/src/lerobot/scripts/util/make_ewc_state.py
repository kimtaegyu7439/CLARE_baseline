#!/usr/bin/env python
"""저장된 체크포인트 하나에서 EWC state(Fisher + anchor)만 만들어 저장한다.

왜 필요한가
    E0.py는 lambda>0 인 팔에서만 ewc_state.pt 를 남긴다. 그래서 lam0(순차 파인튜닝)의
    task_0 에는 그 파일이 없다. EWC 팔을 lam0 의 task_0 위에 다시 세우려면
    (두 팔의 stage 0 이 같아야 비교가 성립한다) 그 체크포인트에 대한 Fisher/anchor 를
    따로 만들어 줘야 한다. 학습은 하지 않는다.

    Fisher 추정 자체는 E0.build_ewc_state 를 그대로 호출한다. 복사본을 두면
    "E0 가 만든 Fisher"와 "여기서 만든 Fisher"가 조용히 갈라진다.

사용 예
    python make_ewc_state.py \
        --ckpt=outputs/E0/libero_spatial/seed_42/lam0/task_0/checkpoints/last/pretrained_model \
        --repo_id=continuallearning/libero_spatial_image_task_0 \
        --out=outputs/E0/libero_spatial/seed_42/lam100_rebased/task_0/ewc_state.pt
"""

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.scripts.E0 import build_ewc_state, split_episodes
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="pretrained_model 디렉터리")
    p.add_argument("--repo_id", required=True, help="이 태스크의 데이터셋")
    p.add_argument("--out", required=True, help="저장할 ewc_state.pt 경로")
    p.add_argument("--holdout_episodes", type=int, default=5, help="E0와 같은 분할이어야 한다")
    p.add_argument("--fisher_batches", type=int, default=100)
    p.add_argument("--fisher_batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    init_logging()
    set_seed(args.seed)

    cfg_policy = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg_policy.pretrained_path = args.ckpt
    device = get_safe_torch_device(cfg_policy.device, log=True)

    ds_meta = LeRobotDatasetMetadata(args.repo_id)
    policy = make_policy(cfg=cfg_policy, ds_meta=ds_meta)
    policy.train()

    dataset = LeRobotDataset(
        args.repo_id, delta_timestamps=resolve_delta_timestamps(cfg_policy, ds_meta)
    )
    train_eps, _ = split_episodes(args.repo_id, None, args.holdout_episodes)
    logging.info(f"Fisher 추정: {len(train_eps)} 에피소드, {args.fisher_batches} 배치")

    # build_ewc_state 가 쓰는 필드만 채운 가짜 cfg. episode_sampler 는 cfg.policy 를 본다.
    cfg = SimpleNamespace(
        policy=cfg_policy,
        fisher_batches=args.fisher_batches,
        fisher_batch_size=args.fisher_batch_size,
    )
    state = build_ewc_state(cfg, policy, dataset, train_eps, device, None)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: {n: v.cpu() for n, v in d.items()} for k, d in state.items()}, out)
    logging.info(f"EWC state saved -> {out}")


if __name__ == "__main__":
    main()
