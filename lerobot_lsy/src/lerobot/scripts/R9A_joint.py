#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""R9A_joint — N개 태스크 joint 통제군 학습 (R7.train_joint의 N-태스크 판).

왜 따로 있는가
    R7.train_joint는 정확히 두 태스크(task_a, task_b)만 섞는다. R9_A는 CL이 0..3을
    순차로 배운 것과 견주므로 joint도 0..3을 함께 배워야 한다. R7 파일은 건드리지 않고
    같은 방식(로더를 스텝마다 번갈아 먹이기)을 N개로 일반화한다.

    ConcatDataset을 쓰지 않는 이유는 R7과 같다 — LeRobotDataset의 episode_data_index가
    데이터셋 로컬 인덱스라, 합치면 EpisodeAwareSampler가 어긋난다. 번갈아 먹이면 각
    태스크가 자기 샘플러를 그대로 쓴다.

CL과 무엇을 맞추는가
    seq CL이 "이미 배운 체크포인트에서 이어서" task2, task3를 배우므로 joint도 같게 한다:
    기존 joint(0+1) 체크포인트에서 출발해 0..3을 섞어 추가 학습한다. 시작점과 추가
    업데이트 예산이 양쪽에서 같아야 "데이터를 섞었는가"만 다른 비교가 된다.

        seq CL : pretrain -> t0(5k) -> t1(5k) -> t2(5k) -> t3(5k)      총 20k
        joint  : pretrain -> {t0,t1}(10k) -> {t0,t1,t2,t3}(10k)        총 20k

    --policy.path 로 출발 체크포인트를, --joint_steps 로 추가 스텝을 준다.
    나머지(옵티마이저·배치·holdout 분할)는 E0의 순차 학습과 같게 둔다.

사용 예
    python R9A_joint.py --policy.path=<기존 joint ckpt> --output_dir=<새 dir> \
        --tasks=0,1,2,3 --joint_steps=10000 --batch_size=32 --holdout_episodes=5
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import torch
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.scripts.E0 import episode_sampler, split_episodes, to_device, update_policy

try:                                    # torch 2.x 경로 차이 흡수
    from torch.amp import GradScaler
except ImportError:                     # pragma: no cover
    from torch.cuda.amp import GradScaler


@dataclass
class JointConfig(TrainPipelineConfig):
    tasks: str = "0,1,2,3"                  # 섞을 태스크 (쉼표)
    dataset_prefix: str = "continuallearning/libero_spatial_image_task_"
    joint_steps: int = 10000                # 이번 실행에서 돌 스텝 수
    holdout_episodes: int = 5               # E0와 같은 분할. 뒤 5 에피소드는 학습에서 제외

    def validate(self):
        out = self.output_dir
        if isinstance(out, Path) and out.is_dir() and (out / ".done").exists():
            # 이미 끝난 산출물이면 덮어쓰기 검사를 우회하고 바로 종료할 수 있게 둔다.
            self.output_dir = None
            super().validate()
            self.output_dir = out
        else:
            super().validate()


@parser.wrap()
def main(cfg: JointConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))
    out = Path(cfg.output_dir)
    if (out / ".done").exists():
        logging.info(f"[joint] 이미 완료됨: {out}")
        return

    tasks = [int(x) for x in cfg.tasks.split(",") if x.strip() != ""]
    logging.info(colored(f"[joint] {len(tasks)}개 태스크 동시 학습: {tasks}",
                         "green", attrs=["bold"]))
    if cfg.seed is not None:
        set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    datasets, loaders = [], []
    for k in tasks:
        repo = f"{cfg.dataset_prefix}{k}"
        meta = LeRobotDatasetMetadata(repo)
        ds = LeRobotDataset(repo, delta_timestamps=resolve_delta_timestamps(cfg.policy, meta),
                            video_backend=cfg.dataset.video_backend)
        train_eps, hold = split_episodes(repo, None, cfg.holdout_episodes)
        logging.info(f"[joint] task {k}: train {len(train_eps)} ep / held-out {len(hold)} ep")
        datasets.append(ds)
        loaders.append(torch.utils.data.DataLoader(
            ds, num_workers=cfg.num_workers, batch_size=cfg.batch_size,
            sampler=episode_sampler(cfg, ds, train_eps),
            pin_memory=device.type == "cuda", drop_last=False,
            multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
            persistent_workers=cfg.num_workers > 0))

    policy = make_policy(cfg=cfg.policy, ds_meta=datasets[0].meta)
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    iters = [cycle(dl) for dl in loaders]
    policy.train()
    tracker = MetricsTracker(
        cfg.batch_size, sum(d.num_frames for d in datasets),
        sum(d.num_episodes for d in datasets),
        {"loss": AverageMeter("loss", ":.3f"), "mse": AverageMeter("mse", ":.3f"),
         "penalty": AverageMeter("pen", ":.3e"), "grad_norm": AverageMeter("grdn", ":.3f"),
         "lr": AverageMeter("lr", ":0.1e"), "update_s": AverageMeter("updt_s", ":.3f"),
         "dataloading_s": AverageMeter("data_s", ":.3f")},
        initial_step=0)

    logging.info(f"[joint] {cfg.joint_steps} 스텝 (task당 {cfg.joint_steps // len(tasks)})")
    for step in range(cfg.joint_steps):
        t0 = time.perf_counter()
        batch = to_device(next(iters[step % len(tasks)]), device)     # ★ 번갈아
        tracker.dataloading_s = time.perf_counter() - t0
        tracker, _ = update_policy(tracker, policy, batch, optimizer,
                                   cfg.optimizer.grad_clip_norm, grad_scaler=grad_scaler,
                                   lr_scheduler=lr_scheduler, use_amp=cfg.policy.use_amp)
        tracker.step()
        if cfg.log_freq > 0 and (step + 1) % cfg.log_freq == 0:
            logging.info(tracker)
            tracker.reset_averages()

    ckpt = get_step_checkpoint_dir(cfg.output_dir, cfg.joint_steps, cfg.joint_steps)
    save_checkpoint(ckpt, cfg.joint_steps, cfg, policy, optimizer, lr_scheduler)
    update_last_checkpoint(ckpt)
    (out / ".done").write_text(f"joint_steps={cfg.joint_steps}\ntasks={tasks}\n"
                               f"init={cfg.policy.pretrained_path}\n")
    logging.info(colored(f"[joint] 체크포인트 -> {ckpt}", "green", attrs=["bold"]))


if __name__ == "__main__":
    init_logging()
    main()
