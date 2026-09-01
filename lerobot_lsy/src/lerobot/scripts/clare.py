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
"""CLARE: 어댑터를 늘려 가며 태스크를 순차 학습하는 continual learning 스크립트.

train.py를 복사해 개조한 파일이라 뼈대(설정 -> 데이터셋 -> 정책 -> 옵티마이저 -> 루프)는
그대로다. 다른 곳만 알면 되므로 train.py와의 차이를 먼저 정리한다.

═══════════════════════════════════════════════════════════════════════════════
 train.py와 달라지는 지점 (파일 순서대로)
═══════════════════════════════════════════════════════════════════════════════
  #  위치            train.py                     clare.py
 ─────────────────────────────────────────────────────────────────────────────
  1  설정 클래스     TrainPipelineConfig          PEFTTrainPipelineConfig (상속 + 15개 필드)
  2  정책 래핑       (없음)                       PeftWrapperPolicy -> PeftModel
  3  train/eval 전환 policy.train()               set_peft_module_train() (base_layer는 eval 고정)
  4  분포 이동 감지  (없음)                       detect_distribution_shift() -- 200스텝, 통째로 신규
  5  어댑터 확장     (없음)                       z-score로 층별 확장 여부 결정 (train()의 424-542행)
  6  옵티마이저      1개 (전체 파라미터)          2개 (adapter_params / discriminator_params)
  7  손실            policy_loss 고정             단계에 따라 policy_loss 또는 판별기 loss
  8  루프 길이       cfg.steps                    cfg.steps + train_discriminators_steps
  9  단계 전환       (없음)                       cfg.steps번째에 어댑터->판별기로 스위치
 10  평가            env 1개                      태스크마다 env 1개 (지금까지 배운 전부)
 11  저장            정책만                       정책 + adapter/ 디렉터리 (다음 스테이지 입력)
 12  __main__        mp.set_start_method("spawn") (없음)

한 스테이지(= 태스크 하나)의 전체 흐름:

    [A] 분포 이동 감지    200 스텝   detect_distribution_shift()  ※ 첫 태스크는 건너뜀
    [B] 확장 결정         즉시       층별 z-score > expand_threshold 이면 어댑터 추가
    [C] 어댑터 학습    20,000 스텝   adapter_optimizer
    [D] 판별기 학습     2,000 스텝   discriminator_optimizer
                     ─────────────
                       22,000 스텝   ← 로그의 step 수가 22000까지 가는 이유

[C]와 [D]는 별개 루프가 아니라 하나의 for 문 안에서 스위치로 갈린다(653행, 662행).

베이스 모델은 절대 학습되지 않는다. 옵티마이저에 adapter_params / discriminator_params만
등록되므로(548행, 554행), 베이스 weight는 gradient가 흘러도 갱신될 경로가 없다.
그래서 셸 스크립트가 스테이지마다 --policy.path를 같은 사전학습 체크포인트로 고정하고,
스테이지 간에 이어지는 것은 --peft_weight_path가 가리키는 adapter/ 뿐이다.
"""

import logging
import os          # EVAL_SEED (롤아웃 시드) 를 환경변수로 받기 위해
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any, Literal
from dataclasses import dataclass, field
from pathlib import Path
import copy
import re

import torch
from termcolor import colored
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.optimizers import OptimizerConfig, AdamWConfig
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.optim.schedulers import LRSchedulerConfig, LRScheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
# from lerobot.scripts.eval import eval_policy
from lerobot.scripts.eval_peft import eval_policy_with_env_init
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed, load_rng_state
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
    load_training_step,
    load_optimizer_state,
    load_scheduler_state
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger

from peft import get_peft_model, PeftConfig, PeftModel
from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING

# ─────────────────────────────────────────────────────────────────────────────
# [차이 2] 정책 래핑 -- train.py에는 없는 계층
# ─────────────────────────────────────────────────────────────────────────────
class PeftWrapperPolicy(torch.nn.Module):
    """정책을 nn.Module 하나로 한 번 더 감싸는 껍데기. 필드가 policy 하나뿐이다.

    왜 필요한가: PEFT는 target_modules를 "policy.model.encoder.xxx" 같은 이름 경로로
    찾는데, 정책을 바로 넘기면 최상위 이름이 비어 경로가 어긋난다. 이 래퍼가 있으면
    모든 대상 모듈 앞에 "policy." 접두사가 붙어 설정 파일의 경로와 일치하게 된다.

    학습 로직은 전혀 없다. forward도 정의하지 않으므로 이 객체를 직접 호출하면 안 되고,
    아래 코드도 항상 안쪽의 policy를 꺼내 policy.forward(batch)를 부른다.
    """
    policy: PreTrainedPolicy

    def __init__(self, policy: PreTrainedPolicy):
        super().__init__()
        self.policy = policy


# ─────────────────────────────────────────────────────────────────────────────
# [차이 1] 설정 클래스 -- TrainPipelineConfig를 상속해 CLARE 전용 인자를 추가
#
# 상속이므로 --batch_size, --steps, --dataset.repo_id 등 train.py의 인자는 전부 그대로
# 쓸 수 있고, 여기 추가된 필드만 CLI에 새로 생긴다. 본인 방법론을 만들 때도 같은 패턴이다.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PEFTTrainPipelineConfig(TrainPipelineConfig):
    # 둘 중 정확히 하나를 준다(아래 __post_init__의 assert).
    #   peft_cfg_path    : 어댑터 "설계도"만 읽어 새로 만든다  -> 첫 태스크(task_0)
    #   peft_weight_path : 이전 스테이지의 어댑터 가중치를 이어받는다 -> task_1 이후
    # 이 한 줄이 스테이지를 사슬로 잇는 유일한 연결고리다. 베이스 모델은 이어지지 않는다.
    peft_cfg_path: Path | None = None
    peft_weight_path: Path | None = None

    # [A단계] 새 태스크 데이터가 기존 어댑터에게 얼마나 낯선지 재는 구간의 길이.
    # 학습이 아니라 순수 추론이라 짧아도 된다.
    detect_distribution_shift_steps:int = 200
    detect_distribution_shift_batch_size: int = 32
    detect_distribution_shift_num_workers: int = 16
    detect_distribution_shift_log_freq:int = 10
    
    # [D단계] 판별기 학습 길이. cfg.steps(어댑터) 뒤에 이어 붙는다.
    # 20000 + 2000 = 22000이 한 스테이지의 총 루프 횟수가 된다(653행).
    train_discriminators_steps: int = 2000
    train_discriminators_batch_size: int = 32   # 주의: 루프는 cfg.batch_size를 쓰므로 미사용
    train_discriminators_num_workers: int = 16  # 주의: 위와 동일하게 미사용
    train_discriminators_log_freq: int = 50     # 판별기 구간의 로깅 주기(어댑터 구간과 별도)
    train_discriminators_save_freq: int = 2000
    train_discriminators_eval_freq: int = 2000
    # 판별기 전용 옵티마이저 설정. 어댑터와 학습 성격이 달라 lr을 따로 준다.
    train_discriminator_optimizer: OptimizerConfig = field(
        default_factory=lambda: AdamWConfig(
            lr = 0.0005,
            weight_decay=0.01,
            grad_clip_norm=10.0,
            betas=(0.9,0.999),
            eps=1e-08
        )
    )
    train_discriminator_lr_scheduler: LRSchedulerConfig | None = None

    # [B단계] 확장 정책을 정하는 세 인자.
    #   maximum_expand   한 스테이지에서 새로 만들 수 있는 어댑터 수 상한.
    #                    기본 10000이라 사실상 무제한이고 셸 스크립트도 안 건드린다.
    #   expand_threshold 층별 z-score 문턱. 모든 z-score가 이 값을 넘으면 "충분히 낯선
    #                    태스크"로 보고 그 층에 어댑터를 새로 단다. 셸에서는 1.0.
    #                    0.0(기본)이면 거의 항상 확장 -> 어댑터가 태스크 수만큼 늘어난다.
    #   at_least_expand  어느 층도 확장 신호를 못 냈을 때의 보험. 그래도 최소 한 층은
    #                    확장한다. 안 그러면 새 태스크가 배울 자리가 아예 없어진다.
    maximum_expand: int = 10000
    expand_threshold: float = 0.0
    at_least_expand: Literal["shallowest", "deepest"] = field(
        default="shallowest", metadata={"help": "At least expand which layer. Can be 'shallowest' or 'deepest'"}
    )

    max_episodes_rendered: int = 4

    def __post_init__(self):
        # 주의: 부모의 __post_init__(self.checkpoint_path = None)을 호출하지 않고 덮어쓴다.
        # cfg.validate()가 checkpoint_path를 따로 채우므로 지금은 문제가 없지만,
        # 부모에 초기화 로직이 추가되면 조용히 누락된다.
        assert self.peft_cfg_path or self.peft_weight_path, "One from (peft_cfg_path,peft_weight_path) must be specified"


# ─────────────────────────────────────────────────────────────────────────────
# [차이 3] policy.train() 대신 쓰는 함수
#
# train.py는 그냥 policy.train()을 부르지만, 여기서는 그러면 안 된다. 베이스 모델까지
# 학습 모드가 되어 dropout이 켜지고 BatchNorm running stats가 갱신되기 때문이다.
# 베이스는 얼려 둔 상태여야 하므로 "어댑터 쪽만 train, base_layer는 eval"로 나눈다.
# ─────────────────────────────────────────────────────────────────────────────
def set_peft_module_train(peft_modules:list, train: bool = True):
    # peft_type에 대응하는 이름 접두사(예: LoRA면 "lora_"). 이 접두사가 이름에 있으면
    # PEFT가 주입한 모듈이라는 뜻이다.
    prefix = PEFT_TYPE_TO_PREFIX_MAPPING[peft_modules[0].peft_config.peft_type]
    for peft_module in peft_modules:
        for name, module in peft_module.named_modules():
            # 어댑터 모듈 + 컨테이너 자신(name == '')만 학습 모드로.
            if prefix in name or name == '':
                module.train(train)
            # 원본 층은 무조건 eval. 위 조건과 겹칠 수 있어 뒤에 두어 덮어쓴다(순서 중요).
            if 'base_layer' in name:
                module.train(False)
    return peft_modules   # 제자리 변경이므로 반환값은 사실 같은 객체다


# ─────────────────────────────────────────────────────────────────────────────
# [차이 4] 분포 이동 감지 -- train.py에 대응물이 전혀 없는 신규 단계
# ─────────────────────────────────────────────────────────────────────────────
def detect_distribution_shift(cfg: PEFTTrainPipelineConfig,
                              wandb_logger: WandBLogger,
                              global_steps: int,
                              policy: PreTrainedPolicy,
                              peft_modules: list,
                              dataset,
                              device):
    """새 태스크 데이터가 기존 판별기들에게 얼마나 "낯선지"를 층별로 측정한다.

    학습이 아니다. gradient를 만들지 않고 200스텝 동안 추론만 하며 통계를 모은다.
    각 판별기는 자기가 담당한 태스크의 특징 분포를 running_mean/std로 기억하고 있고,
    새 데이터가 들어오면 그 분포에서 몇 표준편차만큼 벗어났는지를 z-score로 돌려준다.

        z-score 큼  -> 기존 어댑터가 모르는 데이터 -> 새 어댑터가 필요
        z-score 작음 -> 이미 비슷한 걸 배웠음      -> 기존 어댑터 재사용

    반환: (z_scores_mean, losses_mean, global_steps + step)
        z_scores_mean  {"layer.id": tensor([판별기 수])}  층별 판별기별 평균 z-score
                       -> 호출부에서 expand_threshold와 비교해 확장 여부를 결정
        losses_mean    같은 형태. argmin이 "가장 잘 맞는 판별기"이므로 재사용할
                       어댑터를 고르는 데 쓰인다
        세 번째        소비한 스텝 수를 더한 값. 이후 학습 step 번호가 여기서 이어진다
                       (그래서 로그의 step이 0이 아니라 200부터 시작한다)

    첫 태스크(new_task_id == 0)에서는 비교할 판별기가 아직 없으므로 호출되지 않는다.
    """
    infer_metrics = {
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # z-score 계산을 켠다. 평상시에는 꺼져 있어(오버헤드) 이 구간에서만 활성화하고,
    # 아래 220행대에서 다시 끈다.
    for peft_module in peft_modules:
        peft_module.track_z_score(True)
        for discriminator_id in range(peft_module.num_discriminators):
            key = f"{peft_module.layer_name}.{peft_module.layer_id}.{discriminator_id}"

            infer_metrics[f"loss_{key}"] = AverageMeter(f"loss_{key}", ":.3f")
            infer_metrics[f"z_score_{key}"] = AverageMeter(f"z_score_{key}", ":.3f")
    
    detect_tracker = MetricsTracker(
        cfg.detect_distribution_shift_batch_size, dataset.num_frames, dataset.num_episodes, infer_metrics, initial_step=0
    )

    # 학습용과 별개의 DataLoader. EpisodeAwareSampler를 쓰지 않고 단순 shuffle이다.
    # 여기서는 액션 예측 품질이 아니라 특징 분포만 보므로 경계 패딩이 문제가 안 된다.
    detect_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.detect_distribution_shift_num_workers,
        batch_size=cfg.detect_distribution_shift_batch_size,
        shuffle=True,
        sampler=None,
        pin_memory=device.type != "cpu",
        drop_last=True,   # 마지막 자투리 배치를 버려 통계가 배치 크기에 흔들리지 않게 한다
    )
    detect_iter = cycle(detect_dataloader)

    policy.eval()   # dropout 끄기. 이 구간은 순수 측정이다.

    z_scores_sum = {}
    losses_sum = {}

    step = 0

    # infer on new dataset only for 1 epoch
    for _ in range(cfg.detect_distribution_shift_steps):
        batch = next(detect_iter)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        # inference_mode는 no_grad보다 강하다(버전 카운터도 안 만든다). 반환값을 쓰지 않는
        # 이유는 forward의 부작용이 목적이기 때문이다 -- 통과하는 동안 각 peft_module이
        # 자기 info_dicts에 loss와 z_score를 채워 넣고, 아래에서 그걸 읽어 간다.
        with torch.inference_mode():
            _, _ = policy.forward(batch)


        # forward가 채워 놓은 통계를 층 x 판별기 단위로 누적한다.
        for peft_module in peft_modules:
            info_dicts = peft_module.info_dicts
            for discriminator_id in range(peft_module.num_discriminators):

                discriminator_info_dict = info_dicts[f"discriminator_{discriminator_id}"]

                # 배치 안 샘플별 값을 평균 내 스칼라 하나로 줄인다.
                loss = discriminator_info_dict["loss"].mean().item()
                z_score = discriminator_info_dict["z_score"].mean().item()

                key = f"{peft_module.layer_name}.{peft_module.layer_id}"

                # 첫 스텝에는 리스트를 만들고, 이후에는 같은 자리에 더한다.
                # (defaultdict를 안 쓴 탓에 분기가 길어졌을 뿐 동작은 단순 누적)
                if key in z_scores_sum:
                    if len(z_scores_sum[key]) <= discriminator_id:
                        z_scores_sum[key].append(z_score)
                        losses_sum[key].append(loss)
                    else:
                        z_scores_sum[key][discriminator_id] += z_score
                        losses_sum[key][discriminator_id] += loss
                else:
                    z_scores_sum[key] = [z_score]
                    losses_sum[key] = [loss]

                log_key = f"{key}.{discriminator_id}"
                detect_tracker.__setattr__(f"loss_{log_key}", loss)
                detect_tracker.__setattr__(f"z_score_{log_key}", z_score)

        step += 1
        detect_tracker.step()
        is_log_step = cfg.detect_distribution_shift_log_freq > 0 and step % cfg.detect_distribution_shift_log_freq == 0

        if is_log_step:
            logging.info(detect_tracker)
            if wandb_logger:
                wandb_log_dict = detect_tracker.to_dict()
                wandb_step = step
                if global_steps > 0:
                    wandb_step += global_steps
                wandb_logger.log_dict(wandb_log_dict, wandb_step, mode='continual_learning')
            detect_tracker.reset_averages()

    # ── 누적합 -> 평균으로 마무리 ─────────────────────────────────────────────
    z_scores_mean = {}
    losses_mean = {}

    for peft_module in peft_modules:
        peft_module.track_z_score(False)   # 켜 뒀던 z-score 추적을 원상 복구

        key = f"{peft_module.layer_name}.{peft_module.layer_id}"

        # step으로 나눠 200스텝 평균. 여기 나오는 벡터의 길이가 그 층의 판별기 개수다.
        z_scores_mean_current_layer = torch.tensor(z_scores_sum[key], device="cpu") / step
        losses_mean_current_layer = torch.tensor(losses_sum[key], device="cpu") / step

        logging.info(f"Distribution shift of {key}")
        logging.info(f"Average z_scores: {[f'{z_score:.4f}' for z_score in z_scores_mean_current_layer.tolist()]}")
        logging.info(f"Average losses: {[f'{loss:.4f}' for loss in losses_mean_current_layer.tolist()]}")

        z_scores_mean[key] = z_scores_mean_current_layer
        losses_mean[key] = losses_mean_current_layer

    return z_scores_mean, losses_mean, global_steps + step


def load_discriminator_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None
) -> tuple[int, Optimizer, LRScheduler | None]:
    """
    Loads the training step, optimizer state, scheduler state, and rng state.
    This is used to resume a training run.

    Args:
        checkpoint_dir (Path): The checkpoint directory. Should contain a 'training_state' dir.
        optimizer (Optimizer): The optimizer to load the info_dict to.
        scheduler (LRScheduler | None): The scheduler to load the info_dict to (can be None).

    Raises:
        NotADirectoryError: If 'checkpoint_dir' doesn't contain a 'training_state' dir

    Returns:
        tuple[int, Optimizer, LRScheduler | None]: training step, optimizer and scheduler with their
            info_dict loaded.
    """
    training_state_dir = checkpoint_dir / "discriminator_training_state"
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)

    return step, optimizer, scheduler


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    peft_modules: list,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """train.py의 update_policy와 뼈대는 같고 세 가지가 다르다.

      [차이 3] policy.train() -> set_peft_module_train() (베이스는 eval 유지)
      [차이 7] 손실이 고정이 아니다. peft_module의 플래그를 보고 매 스텝 결정한다:
                 어댑터 구간 -> policy_loss (flow matching MSE, train.py와 동일)
                 판별기 구간 -> 판별기 loss들의 합 (정책 손실은 계산만 하고 버려진다)
      [추가]    peft_modules 인자가 늘었다. 판별기 통계를 읽어 로깅하기 위해서다.

    grad_norm_after_clip은 train.py에만 있고 여기엔 없다.
    """
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    # policy.train()  <- train.py의 이 줄을 아래로 대체했다. 베이스를 학습 모드로 만들면 안 된다.
    peft_modules = set_peft_module_train(peft_modules)
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        # 두 구간 모두 forward는 반드시 돈다. 판별기 구간에서도 forward를 해야 판별기가
        # 자기 info_dicts를 채우기 때문이다(정책 손실 자체는 그때 쓰이지 않는다).
        policy_loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

        # 플래그 하나로 전체 학습 목표가 바뀐다. 이 플래그는 메인 루프 662-666행에서
        # cfg.steps번째 스텝에 True로 넘어간다. 모든 peft_module이 같이 바뀌므로
        # 대표로 [0]번만 확인한다.
        if peft_modules[0]._train_discriminator:
            discriminators_loss = []
            for peft_module in peft_modules:
                # discriminator_id = peft_module._forwarded_discriminator_id
                for discriminator_id in range(peft_module.num_discriminators):
                    discriminator_info_dict = peft_module.info_dicts[f"discriminator_{discriminator_id}"]

                    
                    discriminator_running_mean = discriminator_info_dict["running_mean"]
                    discriminator_running_std = discriminator_info_dict["running_std"]
                    discriminator_num_batches_tracked = discriminator_info_dict["num_batches_tracked"]

                    key = f"{peft_module.layer_name}.{peft_module.layer_id}.{discriminator_id}"
                    
                    train_metrics.__setattr__(f"running_mean_{key}", discriminator_running_mean.item())
                    train_metrics.__setattr__(f"running_std_{key}", discriminator_running_std.item())
                    train_metrics.__setattr__(f"num_batches_tracked_{key}", discriminator_num_batches_tracked.item())

                    # 통계는 전 판별기에 대해 로깅하지만, 손실에 넣는 건 이번 태스크용으로
                    # 새로 붙인 판별기 하나뿐이다. 과거 판별기까지 학습하면 이전 태스크의
                    # 라우팅 기준이 흔들려 망각이 생긴다.
                    if discriminator_id == peft_module._forwarded_discriminator_id:
                        discriminator_loss = discriminator_info_dict["loss"].mean()
                        discriminators_loss.append(discriminator_loss)
                        train_metrics.__setattr__(f"loss_{key}", discriminator_loss.item())
            # 층별 판별기 손실의 단순 합. 가중치 없이 더한다.
            loss = sum(discriminators_loss)
        else:
            loss = policy_loss

    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    # policy.parameters() 전체를 넘기지만, 얼린 파라미터는 grad가 None이라 무시된다.
    # 실제로 잘리는 건 어댑터/판별기뿐이다.
    # 주의: 호출부가 항상 cfg.optimizer.grad_clip_norm을 넘기므로(680행), 판별기 구간에서도
    # 어댑터용 임계값이 쓰인다. train_discriminator_optimizer.grad_clip_norm(10.0)은
    # 설정에만 있고 아무 데서도 읽히지 않는다.
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: PEFTTrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    # ── [차이 10] 평가 환경이 하나가 아니라 여러 개 ──────────────────────────
    # train.py: make_env(cfg.env) 하나.
    # 여기: --env.task="Task_0,Task_1,..." 를 쉼표로 쪼개 태스크마다 설정을 만든다.
    # 셸 스크립트가 스테이지마다 태스크를 누적해 넘기므로(task_k 학습 후 Task_0..Task_k),
    # 지금까지 배운 전부를 매번 재평가하게 된다 -- 망각을 측정하는 표준 CL 프로토콜.
    #
    # 여기서는 env를 실제로 만들지 않고 "설정만" 담아 둔다(398행). 10개 시뮬레이터를
    # 학습 내내 띄워 두면 메모리를 낭비하므로, 평가 시점에 하나씩 만들었다 버린다
    # (그래서 eval_policy가 아니라 eval_policy_with_env_init을 쓴다).
    eval_envs = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        task_list = [task.strip() for task in cfg.env.task.split(",") if task.strip()]
        if not task_list:
            raise ValueError("No valid tasks found in env.task")

        # env_cfg = copy.deepcopy(cfg.env)

        eval_envs = {}
        for task in task_list:
            env_cfg = copy.deepcopy(cfg.env)
            env_cfg.task = task
            # libero_40처럼 여러 suite가 섞인 시퀀스에서는 --env.benchmark 하나로
            # 전부 덮으면 안 된다. 핸들과 짝이 맞는 benchmark로 맞춰 둔다
            # (LiberoEnv.resolved_benchmark 주석 참고).
            if hasattr(env_cfg, "benchmark"):
                env_cfg.benchmark = env_cfg.resolved_benchmark
            # eval_env = make_env(
            #     env_cfg, 
            #     n_envs=cfg.eval.batch_size, 
            #     use_async_envs=cfg.eval.use_async_envs
            # )
            # eval_envs[task] = eval_env
            eval_envs[task] = env_cfg

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    # train.py는 여기서 policy.train()이지만 여기는 eval이다. 베이스는 끝까지 추론 모드로
    # 두고, 학습 모드 전환은 어댑터에 대해서만 set_peft_module_train이 담당한다.
    policy.eval()

    # ── [차이 2] PEFT 래핑 ──────────────────────────────────────────────────
    logging.info("Wrapping policy with peft module")

    peft_wrapper_policy = PeftWrapperPolicy(policy=policy)

    # 여기가 스테이지 사슬의 분기점이다.
    if cfg.peft_weight_path:
        # task_1 이후: 이전 스테이지가 저장한 adapter/를 통째로 읽는다. 지금까지 쌓인
        # 어댑터/판별기와 그 개수(peft_config.structure)가 전부 복원된다.
        # is_trainable=True가 없으면 PEFT가 추론 전용으로 얼려 버려 학습이 안 된다.
        peft_policy = PeftModel.from_pretrained(peft_wrapper_policy, cfg.peft_weight_path, is_trainable=True, autocast_adapter_dtype=False)
        peft_config = peft_policy.peft_config["default"]
    else:
        # task_0: 가중치 없이 "설계도"만 읽어 빈 상태로 만든다.
        peft_cfg = PeftConfig.from_pretrained(cfg.peft_cfg_path)
        peft_cfg.inference_mode = False
        peft_policy = get_peft_model(peft_wrapper_policy, peft_cfg)
        peft_config = peft_policy.peft_config["default"]

    # 어댑터가 주입된 층들의 리스트. 이후 모든 CLARE 로직이 이 리스트를 순회한다.
    peft_modules = peft_policy.base_model.adapter_layers

    step = 0  # number of policy updates (forward + backward + optim)

    # ── [차이 5] 어댑터 확장 결정 ───────────────────────────────────────────
    logging.info("Explore new task")

    # 이 두 리스트가 아래에서 옵티마이저에 등록될 파라미터의 전부다.
    # 여기 담기지 않은 것은 학습되지 않는다 = 베이스 모델이 안 변하는 이유.
    adapter_params = []
    discriminator_params = []

    # 지금까지 몇 개 태스크를 배웠는지가 곧 이번 태스크의 id다. 체크포인트에 함께
    # 저장되므로 스테이지를 넘어가며 자동으로 0,1,2...로 증가한다(542행에서 +1).
    # new_task_id start from 0
    new_task_id = peft_config.num_learned_task

    if new_task_id == 0:
        # 첫 태스크: 비교할 기존 어댑터가 없으니 감지 단계를 건너뛰고 무조건 전 층 확장.
        logging.info("Learning the first new task")
        logging.info("Expand all finetuned layers with new adapter and new discriminator")

        for peft_module in peft_modules:
            adapter_param, discriminator_param = \
                peft_module.add_adapter_and_discriminator(new_task_id)
            adapter_params += adapter_param
            discriminator_params += discriminator_param

            key = f"{peft_module.layer_name}.{peft_module.layer_id}"

            peft_module._forwarded_adapter_id = peft_module.num_adapters - 1
            peft_module._forwarded_discriminator_id = peft_module.num_discriminators - 1

            peft_config.structure[key] = \
                [peft_module.num_adapters, peft_module.num_discriminators]
            logging.info(f"Add both adapter and discriminator into layer {key}")
            logging.info(f"Only forward adapter id: {peft_module._forwarded_adapter_id}")
            logging.info(f"Only forward discriminator id: {peft_module._forwarded_discriminator_id}")
    else:
        # ── [A단계] 200스텝 추론으로 층별 z-score / loss 측정 ────────────────
        # step을 반환값으로 덮어쓰는 데 주의. 이후 학습 루프의 step이 0이 아니라
        # 200부터 시작한다(init_step = step, 653행).
        z_scores_mean, losses_mean, step = \
            detect_distribution_shift(
                cfg,
                wandb_logger,
                step,
                policy,
                peft_modules,
                dataset,
                device
            )

        # ── [B단계] 층마다 "확장할까 / 기존 걸 쓸까" 결정 ────────────────────
        only_forward_ids = []    # 재사용할 경우 어느 어댑터를 쓸지
        to_expand_or_not = []    # 층별 확장 여부 (bool)

        for peft_module in peft_modules:
            key = f"{peft_module.layer_name}.{peft_module.layer_id}"
            logging.info(f"For layer {key}")

            z_scores_mean_current_layer = z_scores_mean[key]
            losses_mean_current_layer = losses_mean[key]

            # 손실이 가장 낮은 판별기 = 새 데이터를 가장 잘 설명하는 판별기.
            # 그 판별기에 묶인 어댑터가 "가장 비슷한 과거 태스크"의 어댑터다.
            # 확장하지 않기로 하면 이 어댑터를 재사용한다.
            closest_discriminator_id = torch.argmin(losses_mean_current_layer).item()
            connected_adapter_id = peft_module.get_adapter_id_by_discriminator_id(closest_discriminator_id)
            only_forward_ids.append(connected_adapter_id)

            expand_the_module = False

            # all()에 주의: "모든" 판별기가 낯설다고 해야 확장한다. 하나라도 익숙하다고
            # 하면(z-score가 낮으면) 그 판별기에 붙은 어댑터를 재사용한다. 보수적인 조건이라
            # 어댑터가 불필요하게 늘어나는 걸 억제한다.
            if all(z_scores_mean_current_layer > cfg.expand_threshold):
                logging.info(f"All z-scores in layer {key} exceed threshold {cfg.expand_threshold}")
                logging.info(f"Will try to add new adapter and new discriminator in layer {key}")
                expand_the_module = True
            else:
                logging.info(f"At least one z_score in layer {key} is lower than threshold {cfg.expand_threshold}")
                logging.info(f"Will try to only add new discriminator in layer {key}")
            
            if expand_the_module:
                if sum(to_expand_or_not) < cfg.maximum_expand:
                    logging.info("The number of new adapter is within limit")
                else:
                    logging.info("The number of new adapter reaches the expansion limit")
                    expand_the_module = False

            to_expand_or_not.append(expand_the_module)
            
        # 보험: 어느 층도 확장 신호를 못 냈다면 새 태스크가 배울 자리가 없다.
        # 그래서 최소 한 층은 강제로 확장한다. only_forward_id를 -1로 두는 건
        # "재사용할 어댑터 없음"을 의미한다.
        if sum(to_expand_or_not) == 0:
            logging.info("No layer have expansion signal. But still expand")
            if cfg.at_least_expand == "shallowest":
                logging.info("Still expand the shallowest layer.")
                to_expand_or_not[0] = True
                only_forward_ids[0] = -1
            elif cfg.at_least_expand == "deepest":
                logging.info("Still expand the deepest layer.")
                to_expand_or_not[-1] = True
                only_forward_ids[-1] = -1
        
        
        # ── 결정을 실제로 반영 ──────────────────────────────────────────────
        # 판별기는 두 경우 모두 새로 추가된다는 점이 핵심이다. 어댑터를 재사용하더라도
        # "이번 태스크의 입력은 이렇게 생겼다"를 기억할 판별기는 있어야 추론 시 라우팅이 된다.
        #   to_expand=True  -> 어댑터 + 판별기 둘 다 추가
        #   to_expand=False -> 판별기만 추가하고 기존 어댑터에 연결
        for peft_module_id, (to_expand, only_forward_id) in enumerate(zip(to_expand_or_not, only_forward_ids)):
            peft_module = peft_modules[peft_module_id]

            peft_module.train_discriminator(False)
            key = f"{peft_module.layer_name}.{peft_module.layer_id}"
            logging.info(f"For layer {key}")

            if to_expand:
                adapter_param, discriminator_param = \
                    peft_module.add_adapter_and_discriminator(new_task_id)
                adapter_params += adapter_param
                discriminator_params += discriminator_param

                peft_module._forwarded_adapter_id = peft_module.num_adapters - 1
                peft_module._forwarded_discriminator_id = peft_module.num_discriminators - 1
                logging.info(f"Add both adapter and discriminator into layer {key}")
            else:
                discriminator_param = \
                    peft_module.add_discriminator(only_forward_id, new_task_id)
                discriminator_params += discriminator_param
        
                peft_module._forwarded_adapter_id = only_forward_id
                peft_module._forwarded_discriminator_id = peft_module.num_discriminators - 1

                attached_adapter_task_id = peft_module.clare_func_adapters[peft_module.adapter_name][only_forward_id].task_id.item()
                logging.info(f"Only add discriminator into layer {key}, attatch it with adapter of task_id {attached_adapter_task_id}")
            
            logging.info(f"Only forward adapter id: {peft_module._forwarded_adapter_id}")
            logging.info(f"Only forward discriminator id: {peft_module._forwarded_discriminator_id}")
            peft_module._active_task = new_task_id
            peft_config.structure[key] = \
                [peft_module.num_adapters, peft_module.num_discriminators]

    # 배운 태스크 수 +1. peft_config는 adapter/에 함께 저장되므로 다음 스테이지가
    # 이 값을 읽어 자기 new_task_id를 안다.
    peft_config.num_learned_task += 1

    # ── [차이 6] 옵티마이저가 하나가 아니라 둘 ──────────────────────────────
    logging.info("Creating optimizer and scheduler")
    # 아래 주석 처리된 두 줄이 train.py의 원래 코드다. make_optimizer_and_scheduler는
    # policy.parameters() 전체를 대상으로 삼기 때문에 여기서는 쓸 수 없다.
    # optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    # grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    # ★ 베이스 모델이 학습되지 않는 이유가 정확히 이 줄이다.
    #   옵티마이저는 자기 param group에 없는 텐서를 갱신할 수 없는데, 여기 들어가는 건
    #   방금 새로 만든 adapter_params뿐이다. 베이스 weight는 물론이고 과거 태스크의
    #   어댑터조차 등록되지 않으므로, 이전에 배운 것은 물리적으로 변경 불가능하다.
    #   -> catastrophic forgetting이 구조적으로 발생할 수 없는 근거.
    adapter_optimizer = cfg.optimizer.build(adapter_params)
    if cfg.scheduler:
        adapter_lr_scheduler = cfg.scheduler.build(adapter_optimizer, cfg.steps)
    else:
        adapter_lr_scheduler = None

    # 판별기용 별도 옵티마이저. 어댑터와 학습 목표가 다르므로 lr도 따로 준다.
    # 두 옵티마이저는 파라미터가 겹치지 않아 서로 간섭하지 않는다.
    discriminator_optimizer = cfg.train_discriminator_optimizer.build(discriminator_params)
    if cfg.train_discriminator_lr_scheduler:
        discriminator_lr_scheduler = cfg.train_discriminator_lr_scheduler.build(discriminator_optimizer, cfg.steps)
    else:
        discriminator_lr_scheduler = None

    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    if cfg.resume:
        step, adapter_optimizer, adapter_lr_scheduler = load_training_state(cfg.checkpoint_path, adapter_optimizer, adapter_lr_scheduler)
        step, discriminator_optimizer, discriminator_lr_scheduler = load_discriminator_training_state(cfg.checkpoint_path, discriminator_optimizer, discriminator_lr_scheduler)

    # 이 네 숫자를 비교하면 "정말 베이스가 안 학습되는가"를 로그만 보고 확인할 수 있다.
    #   num_adapter_params + num_discriminator_params  << num_total_params
    # 여야 정상이다. num_learnable_params(requires_grad 기준)가 크더라도, 옵티마이저에
    # 등록된 건 adapter/discriminator뿐이라 실제 갱신 대상은 아래 두 값의 합이다.
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_adapter_params = sum(p.numel() for p in adapter_params)
    num_discriminator_params = sum(p.numel() for p in discriminator_params)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_adapter_params=} ({format_big_number(num_adapter_params)})")
    logging.info(f"{num_discriminator_params=} ({format_big_number(num_discriminator_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    # train.py의 DataLoader와 세 군데 다르다:
    #   drop_last              False -> True   (자투리 배치를 버린다)
    #   multiprocessing_context "spawn" -> 없음 (기본 fork를 쓴다)
    #   persistent_workers     True -> 없음    (cycle()이 재순회할 때마다 worker 재생성)
    # 뒤의 둘은 의도된 변경이라기보다 개조 과정에서 빠진 것으로 보인다. 특히 spawn이
    # 빠진 탓에 __main__에도 mp.set_start_method가 없다(827행). num_workers=16이면
    # 매 재순회마다 16개 프로세스를 다시 띄우는 비용이 든다.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)

    # policy.train()  <- 여기서도 train.py의 이 줄을 어댑터 전용 버전으로 대체
    peft_modules = set_peft_module_train(peft_modules)

    # ── 지표 그릇도 두 벌 ────────────────────────────────────────────────────
    # 어댑터 구간과 판별기 구간이 서로 다른 값을 추적하므로 tracker를 따로 만들고,
    # 루프 안에서 train_tracker 변수를 갈아 끼운다(669행).
    train_adapter_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_adapter_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_adapter_metrics, initial_step=step
    )

    train_discriminator_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    for peft_module in peft_modules:
        for discriminator_id in range(peft_module.num_discriminators):
            key = f"{peft_module.layer_name}.{peft_module.layer_id}.{discriminator_id}"
            train_discriminator_metrics[f"loss_{key}"] = AverageMeter(f"loss_{key}", ":.3f")
            train_discriminator_metrics[f"running_mean_{key}"] = AverageMeter(f"running_mean_{key}", ":.3f")
            train_discriminator_metrics[f"running_std_{key}"] = AverageMeter(f"running_std_{key}", ":.3f")
            train_discriminator_metrics[f"num_batches_tracked_{key}"] = AverageMeter(f"num_batches_tracked_{key}", ":.0f")

    # 판별기 tracker는 initial_step이 step + cfg.steps다. 판별기 구간이 어댑터 구간
    # 뒤에서 시작하므로 wandb 그래프의 x축이 이어지게 맞춘 것이다.
    train_discriminator_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_discriminator_metrics, initial_step= step + cfg.steps
    )

    # ═════════════════════════════════════════════════════════════════════════
    # [차이 8/9] 학습 루프 -- 하나의 for 문 안에서 두 단계가 순차로 진행된다
    #
    #   step        init_step        init_step+cfg.steps      +train_discriminators_steps
    #               │                │                        │
    #               ├─── [C] 어댑터 학습 ───┼─── [D] 판별기 학습 ───┤
    #                   20,000 스텝            2,000 스텝
    #
    # 아래 여섯 변수(optimizer/lr_scheduler/train_tracker/wandb_mode/log_freq/save_freq)를
    # 전환 시점에 통째로 갈아 끼우는 방식이다. 루프 본문은 그대로 두고 "무엇을 쓸지"만
    # 바꾸므로 코드 중복이 없다.
    # ═════════════════════════════════════════════════════════════════════════
    logging.info("Start offline training on a fixed dataset")
    logging.info("Training func adapters")
    # [C] 어댑터 구간의 초기 설정.
    #   train_discriminator(False) -> update_policy의 손실이 policy_loss가 된다
    #   update_stats(False)        -> 판별기의 running_mean/std를 아직 갱신하지 않는다
    #                                 (어댑터가 안정되기 전 통계를 잡으면 기준이 흔들린다)
    for peft_module in peft_modules:
        peft_module.train_discriminator(False)
        peft_module.update_stats(False)
    optimizer = adapter_optimizer
    lr_scheduler = adapter_lr_scheduler
    train_tracker = train_adapter_tracker
    wandb_mode="train"
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq

    # detect_distribution_shift가 소비한 스텝(200)이 여기 들어 있다. 첫 태스크면 0.
    init_step = step
    for _ in range(init_step, init_step + cfg.steps + cfg.train_discriminators_steps):
        start_time = time.perf_counter()
        # 두 구간이 같은 dl_iter를 공유한다. 판별기도 같은 태스크 데이터로 학습한다.
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        # ── [C] -> [D] 전환. 딱 한 번 실행된다 ──────────────────────────────
        if step == init_step + cfg.steps:
            logging.info("Training discriminator")
            # train_discriminator(True): update_policy의 손실이 판별기 loss로 바뀐다
            # update_stats(True):        이제부터 running_mean/std를 갱신한다. 이 통계가
            #                            다음 태스크의 detect_distribution_shift에서
            #                            z-score를 재는 기준이 된다.
            for peft_module in peft_modules:
                peft_module.train_discriminator(True)
                peft_module.update_stats(True)
            optimizer = discriminator_optimizer
            lr_scheduler = discriminator_lr_scheduler
            train_tracker = train_discriminator_tracker
            wandb_mode="train_discriminator"   # wandb에서 별도 계열로 그려진다
            log_freq = cfg.train_discriminators_log_freq
            save_freq = cfg.train_discriminators_save_freq

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            peft_modules,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )


        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()

        # 구간마다 주기 판정 기준이 다르다. train.py는 이 분기가 없다.
        # (step - init_step)을 쓰는 이유: init_step이 200일 수 있어 절대 step으로 나누면
        # 주기가 어긋난다. 다만 어댑터 구간의 is_log_step만 (step - init_step)이 아니라
        # step을 그대로 쓴다 -- 두 구간의 로깅 시점이 200 만큼 어긋난다.
        if step <= init_step + cfg.steps:
            is_log_step = log_freq > 0 and step % log_freq == 0
            is_saving_step = (step - init_step) % save_freq == 0 or step == init_step + cfg.steps
            is_eval_step = cfg.eval_freq > 0 and (step - init_step) % cfg.eval_freq == 0
        else:
            is_log_step = log_freq > 0 and (step - init_step) % log_freq == 0
            is_saving_step = (step - init_step) % save_freq == 0 or step == init_step + cfg.steps + cfg.train_discriminators_steps
            is_eval_step = cfg.train_discriminators_eval_freq > 0 and (step - init_step) % cfg.train_discriminators_eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step, mode=wandb_mode)
            train_tracker.reset_averages()

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            
            # torch.no_grad()만으로는 부족해서 requires_grad를 직접 끈다. 어댑터 내부에
            # 통계 갱신 같은 부작용이 있어 평가 중 상태가 오염될 수 있기 때문이다.
            # 어느 파라미터가 원래 True였는지 이름으로 기억해 두었다가 763행에서 복원한다.
            logging.info("Stopping gradients during evaluation")
            to_train_module_list = []
            for peft_module in peft_modules:
                for name, parameter in peft_module.named_parameters():
                    if parameter.requires_grad:
                        to_train_module_list.append(name)
                        parameter.requires_grad = False

            eval_infos = {}

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }

            # 지금까지 배운 모든 태스크를 하나씩 평가한다. 스테이지가 진행될수록
            # 이 루프가 길어져 평가 비용이 선형으로 증가한다
            # (마지막 스테이지: 10개 태스크 x n_episodes=100 = 1000회 롤아웃).
            for task in eval_envs.keys():
                # eval_env = eval_envs[task]
                eval_env_cfg = eval_envs[task]
                logging.info(f"Eval task {task}")
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
                    
                    # eval_info = eval_policy(
                    #     eval_env,
                    #     policy,
                    #     cfg.eval.n_episodes,
                    #     videos_dir=cfg.output_dir / "eval" / task / f"videos_step_{step_id}",
                    #     max_episodes_rendered=cfg.max_episodes_rendered,
                    #     start_seed=cfg.seed,
                    # )
                    # train.py의 eval_policy와 달리 env 객체가 아니라 env "설정"을 받아
                    # 안에서 만들었다 닫는다. 태스크 10개의 시뮬레이터를 학습 내내 띄워
                    # 두지 않기 위한 변경이다(위 388-398행에서 설정만 저장한 이유).
                    eval_info = eval_policy_with_env_init(
                        eval_env_cfg,
                        cfg.eval.batch_size,
                        False,          # use_async_envs
                        policy,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / task / f"videos_step_{step_id}",
                        max_episodes_rendered=cfg.max_episodes_rendered,
                        start_seed=int(os.environ.get("EVAL_SEED", cfg.seed)),
                    )
                    eval_infos[task] = eval_info

                eval_metrics[f"avg_sum_reward_{task}"] = AverageMeter(f"∑rwrd_{task}", ":.3f")
                eval_metrics[f"pc_success_{task}"] = AverageMeter(f"success_{task}", ":.1f")
                eval_metrics[f"eval_s_{task}"] = AverageMeter(f"eval_s_{task}", ":.3f")

            logging.info("Restoring gradients during evaluation")
            for peft_module in peft_modules:
                for name, parameter in peft_module.named_parameters():
                    if name in to_train_module_list:
                        parameter.requires_grad = True

            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )

            sum_avg_sum_reward = 0.0
            sum_pc_success = 0.0
            sum_eval_s = 0.0

            for task in eval_infos.keys():
                eval_info = eval_infos[task]
                avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                pc_success = eval_info["aggregated"].pop("pc_success")
                eval_s = eval_info["aggregated"].pop("eval_s")
                
                sum_avg_sum_reward += avg_sum_reward
                sum_pc_success += pc_success
                sum_eval_s += eval_s

                eval_tracker.__setattr__(f"avg_sum_reward_{task}", avg_sum_reward)
                eval_tracker.__setattr__(f"pc_success_{task}", pc_success)
                eval_tracker.__setattr__(f"eval_s_{task}", eval_s)

            # 태스크별 성공률의 단순 평균. CL 논문에서 보고하는 average accuracy에 해당한다.
            # 태스크별 값은 위에서 pc_success_{task}로 따로 로깅되므로, 어느 태스크를
            # 잊었는지는 개별 값을 봐야 한다.
            mean_avg_sum_reward = sum_avg_sum_reward / len(eval_infos.keys())
            mean_pc_success = sum_pc_success / len(eval_infos.keys())

            eval_tracker.avg_sum_reward = mean_avg_sum_reward
            eval_tracker.pc_success = mean_pc_success
            eval_tracker.eval_s = sum_eval_s

            logging.info(eval_tracker)
            if wandb_logger:
                # 주의: eval_info는 위 for 루프의 마지막 태스크 것만 남아 있다.
                # 집계값(eval_tracker)은 전 태스크를 반영하지만, 여기 병합되는 원시 dict와
                # 아래 영상은 마지막 태스크 하나에 국한된다.
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][-1], step, mode="eval")

        # ── [차이 11] 저장: 정책 외에 adapter/를 따로 남긴다 ────────────────
        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            # 주의: 판별기 구간의 step은 cfg.steps를 넘어서므로 zero-pad 자릿수를 넘을 수 있다.
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)

            # ★ 이 줄이 continual learning의 사슬을 잇는다.
            #   다음 스테이지의 --peft_weight_path가 가리키는 게 바로 이 디렉터리이며,
            #   어댑터/판별기 가중치와 peft_config(structure, num_learned_task)가 들어간다.
            #   아래 save_checkpoint가 저장하는 정책 전체는 다음 스테이지가 읽지 않는다.
            peft_policy.save_pretrained(str(checkpoint_dir / "adapter"))

            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            # checkpoints/last 심볼릭 링크 갱신. 셸 스크립트가
            # .../checkpoints/last/adapter 로 이전 스테이지 결과를 참조하는 근거다.
            update_last_checkpoint(checkpoint_dir)
            # if wandb_logger:
            #     wandb_logger.log_policy(checkpoint_dir)

            

    # if eval_env:
    #     eval_env.close()
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    # [차이 12] train.py에 있는 mp.set_start_method("spawn", force=True)가 여기엔 없고,
    # 위 DataLoader에도 multiprocessing_context="spawn"이 없다. 기본 fork로 worker가
    # 뜨는데, worker가 CUDA를 직접 만지지 않아 지금은 동작한다. 다만 train.py가 굳이
    # spawn을 명시한 이유(CUDA 컨텍스트 fork 문제)를 생각하면 잠재적 위험 요소다.
    init_logging()
    train()
