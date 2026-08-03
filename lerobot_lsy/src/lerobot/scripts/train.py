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

# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger attach…")
# debugpy.wait_for_client()
# print("Hello, Debugging!")

"""표준 오프라인 학습 루프. 데이터셋 하나로 정책 하나를 학습한다.

CLARE와 무관하다 -- 이 파일에는 peft/discriminator 관련 코드가 한 줄도 없다.
따라서 본인 방법론을 만들 때는 scripts/clare.py(827줄)에서 CLARE를 걷어내는 것보다
이 파일(345줄)을 복사해 시작하는 편이 훨씬 낫다.

continual learning 베이스라인(naive sequential fine-tuning)은 코드 수정 없이도 된다.
make_policy가 --policy.path를 from_pretrained로 처리하므로(policies/factory.py),
셸에서 아래처럼 이으면 그대로 순차 파인튜닝이다.

    python train.py --dataset.repo_id=...task_0 --policy.type=ditflow_mt --output_dir=OUT0
    python train.py --dataset.repo_id=...task_1 --policy.path=OUT0/checkpoints/last/pretrained_model ...

전체 흐름:
    설정 파싱 -> 데이터셋 -> (평가환경) -> 정책 -> 옵티마이저 -> 루프
    루프 안에서: 배치 꺼내기 -> update_policy -> 로깅/체크포인트/평가

본인 로직을 끼워 넣을 자리는 보통 update_policy() 한 곳이다.
추가 CLI 인자가 필요하면 TrainPipelineConfig를 상속한 dataclass를 만들면 된다
(scripts/clare.py의 PEFTTrainPipelineConfig가 그 패턴의 예시다).

참고: --eval_freq=0으로 두면 make_env가 아예 호출되지 않아 gym_libero 의존성 없이
학습만 돌릴 수 있다(152행).
"""

import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
import torch.multiprocessing as mp
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval import eval_policy
from lerobot.scripts.eval_episode import create_episode_plot, eval_episode_no_loader
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger

# ═════════════════════════════════════════════════════════════════════════════
#  위 import들이 각각 무엇을 하고 무엇을 돌려주는지
#
#  이 파일이 외부 함수를 많이 부르는 이유는, train.py가 "학습 로직"이 아니라
#  "조립 순서"만 담당하기 때문이다. 실제 구현은 전부 factory 함수 뒤에 숨어 있고
#  이 파일은 그것들을 정해진 순서로 엮는다. 아래 표만 알면 이 파일은 자립적으로 읽힌다.
#
# ── PyTorch / 표준 라이브러리 ─────────────────────────────────────────────────
#  nullcontext        아무 일도 안 하는 with 블록. `with A() if flag else nullcontext()`
#                     관용구로 AMP를 켜고 끄는 데 쓴다(111행, 329행).
#  pformat            dict를 들여쓰기된 문자열로. 시작 시 설정 전체를 로그에 찍는 용도(158행).
#  mp                 맨 아래 set_start_method("spawn")용(397행). CUDA를 초기화한 프로세스를
#                     fork하면 죽으므로 DataLoader worker를 spawn으로 띄워야 한다.
#  colored            터미널 색상 문자열. 로그 가독성용이며 기능적 역할은 없다.
#  GradScaler         AMP(fp16) 전용. fp16은 표현 범위가 좁아 작은 gradient가 0으로
#                     사라지므로(underflow), loss를 크게 곱해 backward한 뒤 되돌린다.
#                     use_amp=False면 enabled=False라 전 과정이 no-op이 된다(203행).
#  Optimizer          타입 힌트 전용. 런타임 동작 없음.
#
# ── 설정 (configs/) ──────────────────────────────────────────────────────────
#  parser.wrap()      데코레이터. train(cfg)를 감싸 sys.argv를 파싱하고 cfg를 만들어 주입한다.
#                     내부적으로 draccus.parse를 부르되 세 가지를 더 한다:
#                       (1) --policy.path 류의 '.path' 인자를 미리 걷어냄
#                           -> cfg.validate()가 나중에 처리(157행)
#                       (2) --config_path가 있으면 from_pretrained로 설정 복원
#                       (3) --xxx.type 문자열로 어느 서브클래스를 만들지 결정
#                     반환: 없음(데코레이터). 결과는 train()의 인자 cfg로 들어온다.
#  TrainPipelineConfig
#                     실행 하나의 전체 설정을 담는 dataclass. 중첩 구조가 CLI 점 표기법과
#                     1:1 대응한다: cfg.dataset(DatasetConfig) / cfg.policy(PreTrainedConfig)
#                     / cfg.env / cfg.optimizer / cfg.eval / cfg.wandb + batch_size, steps 등.
#
# ── 데이터 (datasets/) ───────────────────────────────────────────────────────
#  make_dataset(cfg)  -> LeRobotDataset
#                     2단계 로딩. ① 메타데이터만 읽어 fps를 알아낸 뒤
#                     ② 정책의 delta_indices(스텝)를 fps로 나눠 delta_timestamps(초)로
#                     번역해 본 데이터셋을 만든다. 마지막에 meta.stats의 이미지 mean/std를
#                     ImageNet 상수로 교체한다(use_imagenet_stats). 로컬에 없으면 HF Hub에서
#                     자동 다운로드($HF_LEROBOT_HOME/<repo_id>).
#  LeRobotDataset     torch Dataset. dataset[i]가 dict 하나를 돌려준다.
#                     delta_timestamps를 받았으면 한 프레임이 시간 창으로 확장된다:
#                       observation.* -> (2, ...) / action -> (16, 7) / *_is_pad -> bool 마스크
#                     안 받았으면(180행의 평가용) 확장 없이 원본 단일 프레임 그대로.
#  EpisodeAwareSampler
#                     -> 인덱스를 yield하는 Sampler. episode_data_index의 {from, to}를 훑어
#                     각 에피소드의 앞뒤 n프레임을 후보에서 빼고 나머지 인덱스 리스트를 만든다.
#                     "에피소드 경계를 모르는 무작위 셔플"을 "경계를 아는 셔플"로 바꾸는 장치.
#  cycle(dataloader)  -> 무한 제너레이터. DataLoader가 소진되면 iter()를 다시 만들어 이어 간다.
#                     itertools.cycle은 첫 epoch 결과를 메모리에 캐싱해 버려 쓸 수 없다.
#                     이 덕분에 이 파일에는 epoch 루프가 없고 스텝 루프만 존재한다(270행).
#
# ── 환경 (envs/) ─────────────────────────────────────────────────────────────
#  make_env(cfg, ...) -> gym.vector.VectorEnv (병렬 환경 묶음)
#                     cfg.type="libero"로부터 "gym_libero" 패키지를 import한다. 이 import는
#                     부작용이 목적으로, 로드되면서 gym 레지스트리에 환경 핸들이 등록된다.
#                     패키지가 없으면 여기서 ModuleNotFoundError로 죽는다.
#
# ── 최적화 (optim/) ──────────────────────────────────────────────────────────
#  make_optimizer_and_scheduler(cfg, policy) -> (Optimizer, LRScheduler | None)
#                     use_policy_training_preset=True(기본)면 policy.get_optim_params()가
#                     주는 param group을 쓴다. 정책이 "백본은 lr 1/10" 같은 규칙을 스스로
#                     정할 수 있게 하는 통로다. 스케줄러는 설정에 없으면 None.
#
# ── 정책 (policies/) ─────────────────────────────────────────────────────────
#  make_policy(cfg, ds_meta) -> PreTrainedPolicy (nn.Module)
#                     이 파일에서 가장 중요한 함수. 두 가지를 한다:
#                       (1) input_features/output_features를 ds_meta에서 채운다. 즉 정책의
#                           입출력 차원은 config가 아니라 데이터셋이 결정한다.
#                       (2) cfg.pretrained_path(= --policy.path)가 있으면 from_pretrained로
#                           체크포인트를 불러오고, 없으면 새로 초기화한다.
#                     (2) 때문에 "이전 태스크 체크포인트 -> 다음 태스크 시작점" 순차
#                     파인튜닝이 코드 수정 없이 셸 인자만으로 가능하다.
#  PreTrainedPolicy   정책들의 추상 부모. 타입 힌트로만 쓰인다.
#  get_device_from_parameters(module) -> torch.device
#                     파라미터 하나를 꺼내 .device를 읽는다. update_policy가 device 인자를
#                     따로 안 받아도 되게 하는 편의 함수(109행).
#
# ── 평가 (scripts/) ──────────────────────────────────────────────────────────
#  eval_policy(env, policy, n_episodes, ...) -> dict
#                     시뮬레이터에서 실제로 롤아웃해 성공률을 잰다. 학습 손실과 달리
#                     이쪽이 논문에 보고되는 진짜 지표다.
#                     반환 dict: {"aggregated": {pc_success, avg_sum_reward, eval_s},
#                                 "per_episode": [...], "video_paths": [...]}
#                     비쌈: n_episodes번의 롤아웃을 끝까지 굴린다.
#  eval_episode_no_loader(dataset, policy, episode) -> (targets, preds, times)
#                     시뮬레이터 없는 저비용 대용품. 데이터셋에서 한 에피소드를 골라
#                     프레임마다 policy.select_action()을 부르고 시연 정답과 나란히 모은다.
#                     반환: targets [T,7](정답 액션), preds [T,7](예측), times [T](timestamp)
#                     select_action은 단일 프레임 입력을 기대하므로, 여기에 넘기는
#                     dataset_eval은 delta_timestamps 없이 만들어야 한다(180행).
#  create_episode_plot(targets, preds, times, save_path) -> None
#                     위 셋을 받아 액션 7차원을 시간축 그래프로 그려 png로 저장.
#
# ── 로깅/체크포인트 (utils/) ─────────────────────────────────────────────────
#  AverageMeter(name, fmt)
#                     값 하나의 누적 평균기. .update(v)로 넣고 .avg로 읽는다. 매 스텝의
#                     loss를 log_freq 구간 동안 평균 내 노이즈를 줄이는 용도.
#  MetricsTracker(batch_size, num_frames, num_episodes, metrics, initial_step)
#                     AverageMeter들을 묶고 스텝 수로부터 samples/episodes/epochs를 파생시킨다.
#                     __getattr__/__setattr__을 덮어써서 `tracker.loss = 0.3`이 내부적으로
#                     metrics["loss"].update(0.3)이 되게 한다 -- 그래서 dict 접근처럼 안 보인다.
#                     .to_dict()로 wandb에 넘기고, .reset_averages()로 구간 평균을 리셋한다.
#  set_seed(seed)     random / numpy / torch / cuda 시드를 한 번에 고정. 반환 없음.
#  get_step_identifier(step, total) -> str   "000500" 같은 zero-pad 문자열(정렬용).
#  get_step_checkpoint_dir(out, total, step) -> Path   output_dir/checkpoints/000500
#  save_checkpoint(dir, step, cfg, policy, optimizer, scheduler) -> None
#                     아래 구조를 만든다. pretrained_model/만 있으면 추론이 되고,
#                     training_state/까지 있어야 --resume으로 학습을 이어갈 수 있다.
#                       000500/pretrained_model/{config.json, model.safetensors, train_config.json}
#                       000500/training_state/{optimizer_state, scheduler_state, rng_state, training_step}
#  update_last_checkpoint(dir) -> Path
#                     checkpoints/last 심볼릭 링크를 방금 저장한 디렉터리로 다시 건다.
#                     --policy.path=OUT/checkpoints/last/pretrained_model 이 항상 최신을
#                     가리키게 되는 이유이며, 순차 파인튜닝 셸 스크립트가 이에 의존한다.
#  load_training_state(dir, optimizer, scheduler) -> (step, optimizer, scheduler)
#                     --resume 전용. optimizer/scheduler/rng 상태를 제자리에 복원하고
#                     중단됐던 step 번호를 돌려준다. 정책 가중치는 여기가 아니라
#                     make_policy가 이미 불러온 상태다.
#  format_big_number(n) -> str        200000 -> "200K". 로그 가독성용.
#  get_safe_torch_device(s) -> torch.device
#                     "cuda" 문자열을 검증 후 torch.device로. 사용 불가면 경고 후 대체.
#  has_method(obj, name) -> bool      hasattr + callable. 선택적 훅 존재 여부 검사(143행).
#  init_logging()                     logging 포맷터 설정. main에서 한 번만 부른다.
#  WandBLogger(cfg)   wandb run을 열고 아래 메서드를 제공한다. cfg.wandb.enable이 꺼져 있으면
#                     아예 생성하지 않고 None을 쓰므로, 이 파일 전체에 `if wandb_logger:` 가드가
#                     붙어 있다. .log_dict(d, step, mode) / .log_policy(dir) /
#                     .log_video(path, step) / .log_figure(path, step)
# ═════════════════════════════════════════════════════════════════════════════


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """한 스텝의 순전파 + 역전파 + 옵티마이저 갱신.

    본인 방법론의 추가 손실 항이나 정규화는 대개 여기에 들어간다.
    반환하는 output_dict는 그대로 wandb에 기록되므로 추가 지표는 policy.forward가
    dict로 돌려주게 하면 자동으로 로깅된다(DiT-Flow는 현재 None을 반환).

    반환: (train_metrics, output_dict)
        train_metrics  인자로 받은 MetricsTracker를 제자리에서 갱신해 그대로 돌려준다.
                       (새 객체가 아니다 -- 호출부가 같은 이름에 재대입하는 건 관례일 뿐)
        output_dict    policy.forward의 두 번째 반환값. 로깅 외에는 쓰이지 않는다.

    AMP가 켜졌을 때의 순서가 미묘하다. scale -> backward -> unscale_ -> clip -> step:
    clipping은 반드시 unscale 이후여야 한다. 안 그러면 스케일 배수가 곱해진 gradient에
    임계값을 적용하게 되어 clip이 사실상 무력화된다.
    """
    start_time = time.perf_counter()
    # policy가 이미 GPU에 올라가 있으므로 device 인자를 따로 받지 않고 파라미터에서 읽는다.
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        # DiTFlowMTPolicy.forward -> 정규화 -> compute_loss(flow matching MSE)
        # 반환: (loss 스칼라 텐서, output_dict | None)
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    # AMP: loss에 큰 수를 곱한 뒤 backward. fp16에서 작은 gradient가 0으로 사라지는 걸 막는다.
    # use_amp=False면 scale 배수가 1이라 사실상 loss.backward()와 같다.
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    # 곱해 뒀던 배수를 gradient에서 되돌린다. 아래 clip이 원래 크기 기준으로 동작해야 하므로 필수.
    grad_scaler.unscale_(optimizer)

    # 전체 파라미터의 L2 norm이 grad_clip_norm을 넘으면 비례 축소한다(방향은 유지).
    # 반환값은 "자르기 전" norm이라 학습 안정성 모니터링 지표로 로깅한다.
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,   # inf/NaN이어도 여기서 죽지 않는다. 아래 scaler.step이 거른다.
    )

    # We also want to log the norm after clipping, so we compute it separately.
    # 자른 뒤 norm. 두 값을 비교하면 clip이 실제로 발동했는지 알 수 있다.
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    grad_norm_after_clip = torch.nn.utils.clip_grad._get_total_norm(grads, 2)

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    # 즉 fp16이 터진 스텝은 조용히 건너뛴다. lock은 분산 학습용이며 이 파일에서는 항상 None.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    # 스텝이 건너뛰어졌으면 배수를 줄이고, 한동안 안정적이면 다시 키운다(적응형).
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    # 이 파일에 epoch 개념이 없으므로 스케줄러도 스텝 단위로 전진시킨다.
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        # 선택적 훅. 정책이 update()를 정의했을 때만 불린다(DiT-Flow에는 없다).
        policy.update()

    # tracker.loss = x 는 내부적으로 metrics["loss"].update(x) -- MetricsTracker.__setattr__ 참조.
    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.grad_norm_after_clip = grad_norm_after_clip.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    # ── [1] 설정 확정 ────────────────────────────────────────────────────────
    # @parser.wrap()이 sys.argv를 파싱해 cfg를 이미 만들어 넣어 줬다. 다만 '.path'류
    # 인자는 그때 일부러 보류됐으므로 여기서 마무리한다. validate()가 하는 일:
    #   --policy.path가 있으면 그 체크포인트의 config.json으로 cfg.policy를 통째로 교체
    #     (아키텍처 하이퍼파라미터는 CLI가 아니라 체크포인트를 따라간다)
    #   --resume이면 저장된 train_config.json 복원
    #   output_dir 확정 + 기존 디렉터리 덮어쓰기 방지 검사
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    # ── [2] 로거 ─────────────────────────────────────────────────────────────
    # 끄면 None. 이후 이 파일 전역에 `if wandb_logger:` 가드가 붙는 이유다.
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # ── [3] 재현성 ───────────────────────────────────────────────────────────
    # random/numpy/torch/cuda 시드를 한 번에 고정. 데이터 셔플 순서와 가중치 초기화가
    # 여기에 걸린다. 단 아래 cudnn.benchmark=True 때문에 완전 결정론은 보장되지 않는다.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # ── [4] 디바이스 ─────────────────────────────────────────────────────────
    # Check device is available
    # cfg.policy.device는 문자열("cuda")이라 torch.device로 바꾸면서 가용성을 검사한다.
    device = get_safe_torch_device(cfg.policy.device, log=True)
    # 입력 크기가 고정일 때 cuDNN이 첫 몇 스텝 동안 알고리즘을 실측해 가장 빠른 걸 고른다.
    # 이 데이터셋은 항상 256x256이라 이득이다(크기가 매번 바뀌면 오히려 손해).
    torch.backends.cudnn.benchmark = True
    # matmul을 TF32로. Ampere 이상에서 fp32 정확도를 조금 내주고 속도를 크게 얻는다.
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── [5] 데이터셋 ─────────────────────────────────────────────────────────
    logging.info("Creating dataset")
    # Make dataset for training and for eval
    # datasets/factory.py: delta_timestamps 계산 + 2단계 로딩 + ImageNet 통계 교체.
    # 데이터가 없으면 여기서 HF Hub로부터 자동 다운로드된다(HF_LEROBOT_HOME 경로).
    #
    # 이 시점 이후 dataset이 들고 있는 것:
    #   dataset.meta                메타데이터(fps, features, tasks, stats) -> 아래 make_policy로
    #   dataset.episode_data_index  {from, to} 에피소드 경계 -> 아래 EpisodeAwareSampler로
    #   dataset.num_frames/episodes 로깅과 MetricsTracker의 epoch 환산에 쓰임
    #   dataset[i]                  (2,3,256,256) / (2,8) / (16,7) 짜리 dict 하나
    dataset = make_dataset(cfg)
    # 평가 전용 데이터셋. delta_timestamps를 일부러 넘기지 않는다 -- 아래 eval_episode_no_loader가
    # 부르는 policy.select_action()이 "현재 프레임 하나"만 받고 과거 프레임은 정책 내부 큐로
    # 관리하기 때문이다. 그래서 여기서 나오는 샘플은 (3,256,256) / (8,) / (7,)로 시간축이 없다.
    # 시간 창을 데이터셋이 만드느냐(학습) 정책이 만드느냐(추론)의 차이이며, 넘기면 shape이 어긋난다.
    if cfg.eval_with_dataset is not None:
        dataset_eval = LeRobotDataset(cfg.eval_with_dataset, video_backend="pyav")

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    # 주의: 조건이 eval_freq > 0이므로, 평가가 실제로 실행되지 않더라도(예: eval_freq가
    # steps보다 큰 경우) 환경은 여기서 만들어진다. gym_libero가 설치돼 있지 않으면
    # 이 줄에서 죽는다. 학습만 할 거면 --eval_freq=0으로 두면 통째로 건너뛴다.
    # ── [6] 평가 환경(선택) ──────────────────────────────────────────────────
    # 반환은 gym.vector.VectorEnv -- 환경 n개를 묶어 한 번에 step하는 래퍼다.
    # n_envs=cfg.eval.batch_size개를 병렬로 굴려 평가 시간을 줄인다.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    # ── [7] 정책 ─────────────────────────────────────────────────────────────
    logging.info("Creating policy")
    # ds_meta를 넘기므로 정책의 input/output feature 차원과 정규화 통계가 데이터셋에서 결정된다.
    # cfg.policy.pretrained_path(--policy.path)가 있으면 체크포인트를 이어받는다.
    #
    # ds_meta로부터 채워지는 것 두 가지:
    #   features -> input_features/output_features (관측 8차원, 액션 7차원 등 실제 차원)
    #   stats    -> Normalize 레이어의 mean/std/min/max 버퍼
    # 두 번째가 중요하다. 이미지 정규화는 데이터셋이 아니라 이 버퍼를 통해 정책 forward에서
    # 일어나며, 버퍼는 체크포인트 state_dict에 함께 저장된다. 그래서 태스크를 바꿔 이어
    # 학습해도 정규화 기준이 흔들리지 않는다.
    # 반환은 nn.Module이며, 이미 cfg.policy.device로 옮겨진 상태다.
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    # ── [8] 옵티마이저 ───────────────────────────────────────────────────────
    # cfg 전체를 넘기는 이유: optimizer/scheduler 설정뿐 아니라 cfg.steps(스케줄러의 총 길이)와
    # cfg.use_policy_training_preset까지 봐야 하기 때문이다.
    # 프리셋이 켜져 있으면 policy.get_optim_params()가 param group을 직접 정한다
    # (예: 사전학습 백본은 낮은 lr). lr_scheduler는 설정에 없으면 None.
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    # use_amp=False면 enabled=False라 이후 scale/unscale/step이 전부 no-op으로 통과한다.
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    # --resume일 때만. 정책 가중치는 이미 make_policy가 불러왔고, 여기서는 optimizer 모멘텀 /
    # 스케줄러 진행도 / rng 상태를 복원하고 중단 지점 step을 돌려받는다. 이 세 가지가 없으면
    # "이어서 학습"이 아니라 "같은 가중치에서 새로 시작"이 되어 lr 스케줄이 처음부터 다시 간다.
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    # requires_grad=True인 것만 센다. 두 값이 다르면 얼린 파라미터가 있다는 뜻이므로,
    # PEFT/어댑터 방식에서 의도대로 얼었는지 확인하는 가장 빠른 지표다.
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    # DiT-Flow는 drop_n_last_frames(=7)를 가지므로 EpisodeAwareSampler가 쓰인다.
    # 각 에피소드의 마지막 7프레임을 샘플 대상에서 제외해, 실제로 실행될 8개 액션이
    # 에피소드 밖으로 넘어가 패딩되는 일을 막는다.
    # (do_mask_loss_for_padding=False이므로 패딩을 손실에서 빼주지 않는다. 그래서
    #  애초에 패딩이 적게 생기도록 하는 이 샘플러가 사실상 유일한 방어책이다.)
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False   # 셔플은 sampler가 담당하므로 여기서는 끈다
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    # DataLoader가 하는 일: sampler가 주는 인덱스로 dataset[i]를 worker 프로세스에서 병렬 호출하고,
    # 나온 dict들을 키별로 stack해 앞에 배치 차원을 붙인다. (2,3,256,256) -> (B,2,3,256,256).
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,   # 0이면 메인 프로세스에서 로드(디버깅 시 유용)
        batch_size=cfg.batch_size,
        shuffle=shuffle,               # sampler와 동시 지정은 금지라 위에서 배타적으로 정했다
        sampler=sampler,
        pin_memory=device.type == "cuda",  # 페이지 고정 메모리 -> GPU 전송 시 non_blocking 가능
        drop_last=False,
        multiprocessing_context="spawn",   # CUDA 초기화 후 fork는 죽는다. 아래 main과 같은 이유.
        persistent_workers=True,       # 반복자가 소진돼도 worker를 살려 둔다. cycle()이 매번
                                       # iter()를 다시 만들므로 이게 없으면 재시작 비용이 계속 든다.
        prefetch_factor=2,             # worker당 2배치 미리 준비
    )
    # epoch 개념 없이 무한 반복자로 감싼다. 학습 길이는 cfg.steps(스텝 수)로만 정해진다.
    # itertools.cycle이 아니라 자체 구현인 이유: 그쪽은 첫 순회 결과를 통째로 캐싱해
    # 메모리를 터뜨리고 셔플도 고정돼 버린다.
    dl_iter = cycle(dataloader)

    policy.train()

    # ── [9] 지표 그릇 ────────────────────────────────────────────────────────
    # 각 AverageMeter는 log_freq 구간 동안 값을 누적 평균한다. 문자열은 로그에 찍히는 짧은 이름,
    # 두 번째 인자는 표시 포맷이다. 여기 항목을 추가하면 로그와 wandb에 자동으로 따라 나온다.
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),                    # clip 전
        "grad_norm_after_clip": AverageMeter("grdn_after_clip", ":.3f"),  # clip 후
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),        # 순전파+역전파 시간
        "dataloading_s": AverageMeter("data_s", ":.3f"),   # 배치 대기 시간.
                                                           # update_s보다 크면 데이터 로딩이 병목.
    }

    # num_frames/num_episodes를 넘기는 건 지표 계산용이 아니라 환산용이다. 스텝 수로부터
    # samples = steps*batch_size, epochs = samples/num_frames 를 파생시켜 함께 로깅한다.
    # initial_step=step이라 --resume 시 스텝 번호가 이어진다.
    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    # ── [10] 학습 루프 ───────────────────────────────────────────────────────
    # epoch이 아니라 스텝 수로만 길이가 정해진다. 루프 변수를 쓰지 않고 아래에서 step을 직접
    # 증가시키는 이유는 --resume 시 range(step, cfg.steps)로 중간부터 이어가기 위해서다.
    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        # 배치 구성(DiT-Flow + LIBERO):
        #   observation.images.image        (B, 2, 3, 256, 256)
        #   observation.images.wrist_image  (B, 2, 3, 256, 256)
        #   observation.state               (B, 2, 8)
        #   observation.state.joint         (B, 2, 7)   <- 로드되지만 모델은 미사용
        #   action                          (B, 16, 7)
        #   action_is_pad                   (B, 16)     <- 이 설정에서는 미사용
        #   task                            list[str]
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # task는 문자열 리스트라 Tensor가 아니므로 GPU로 옮기지 않는다(그대로 통과).
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
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
        train_tracker.step()   # steps/samples/episodes/epochs 파생값 갱신(지표 값과는 무관)
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps  # 마지막 스텝은 무조건 저장
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        # ── 로깅 ─────────────────────────────────────────────────────────────
        if is_log_step:
            logging.info(train_tracker)   # MetricsTracker.__str__ -> "loss:0.123 grdn:1.2 ..."
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                # policy.forward가 추가 지표를 dict로 돌려줬다면 그대로 합쳐 올린다.
                # 본인 방법론의 보조 손실을 그래프로 보고 싶으면 forward에서 dict를 반환하면 끝.
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            # 구간 평균이므로 찍은 뒤 반드시 리셋. 안 하면 학습 전체의 누적 평균이 되어
            # 후반부 변화가 묻힌다.
            train_tracker.reset_averages()

        # ── 체크포인트 ───────────────────────────────────────────────────────
        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            # output_dir/checkpoints/000500 같은 zero-pad 경로. 자릿수를 맞춰 사전순=시간순이 된다.
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            # pretrained_model/(가중치+config) 와 training_state/(optimizer/scheduler/rng) 둘 다 저장.
            # 앞쪽만 있으면 추론은 되지만 --resume은 안 된다.
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            # checkpoints/last 심볼릭 링크를 방금 저장한 곳으로 다시 건다.
            # 순차 파인튜닝 셸 스크립트가 --policy.path=.../checkpoints/last/pretrained_model 로
            # 항상 최신을 가리킬 수 있는 이유가 이 한 줄이다.
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        # ── 평가 A: 시뮬레이터 롤아웃 (env가 있을 때) ────────────────────────
        # 진짜 성능 지표. 정책을 실제로 굴려 과제 성공 여부를 센다.
        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)   # "000500" (파일명 정렬용)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),   # 평가엔 gradient가 필요 없다. 메모리/속도 절약.
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                # 반환 dict:
                #   ["aggregated"]  {"avg_sum_reward", "pc_success", "eval_s"}  <- 아래에서 pop
                #   ["per_episode"] 에피소드별 상세
                #   ["video_paths"] 렌더링된 mp4 경로들
                # 비용 주의: n_episodes번 롤아웃을 끝까지 굴리므로 학습 스텝보다 훨씬 느릴 수 있다.
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,   # 전부 렌더링하면 느리므로 앞 4개만 영상으로
                    start_seed=cfg.seed,       # 평가 조건을 스텝마다 동일하게 고정
                )

            # 평가 지표는 학습과 별도 tracker를 쓴다(누적 평균이 섞이면 안 되므로).
            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

        # ── 평가 B: 데이터셋 기반 액션 비교 (시뮬레이터 없이) ────────────────
        # 시뮬레이터가 없거나 무거울 때 쓰는 저비용 대용품. 성공률을 재는 게 아니라
        # "예측 액션이 시연 액션을 얼마나 따라가는지"를 그래프로 눈으로 확인한다.
        # 평가 A와 배타적이지 않아 둘 다 조건이 맞으면 둘 다 돈다.
        if is_eval_step and cfg.eval_with_dataset is not None:
            # 이 블록 전체가 try로 감싸여 있다. 평가가 실패해도 학습은 계속돼야 하기 때문이며,
            # 아래 except가 예외를 경고 로그로 삼켜 버린다(그래서 조용히 안 그려질 수 있다).
            try:
                logging.info("No env provided, evaluating in an episodes from the dataset instead")
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
                    policy.eval()   # dropout/BN을 추론 모드로. finally에서 반드시 train()으로 되돌린다.
                    # Use configurable episode index for evaluation, defaulting to 2 if not set
                    # getattr로 읽는 이유: eval_episode_index는 TrainPipelineConfig에 정의된 필드가
                    # 아니라서, 이 클래스를 상속한 설정에만 있을 수 있다. 없으면 2번 에피소드.
                    episode_index = getattr(cfg, "eval_episode_index", 2)
                    # 반환: targets_t [T,7] 시연 정답 / preds_t [T,7] 예측 / times_t [T] timestamp
                    # 내부에서 프레임마다 policy.select_action()을 부른다. 단 select_action은
                    # 8스텝에 한 번만 실제 추론하고 나머지는 큐에서 꺼내므로(receding horizon),
                    # preds_t는 매 프레임 새로 계산된 값이 아닌 계단형 궤적이다. 추세 확인용으로만.
                    targets_t, preds_t, times_t = eval_episode_no_loader(
                        dataset_eval, policy, device=device, episode=episode_index
                    )
                    save_path = cfg.output_dir / "images" / f"actions_episode0_{step}.png"
                    if not save_path.parent.exists():
                        save_path.parent.mkdir(parents=True, exist_ok=True)

                    create_episode_plot(
                        targets_t,
                        preds_t,
                        times_t,
                        save_path=save_path,
                    )
                    logging.info(f"Saved action plot to {save_path}")
                    if wandb_logger:
                        wandb_logger.log_figure(save_path, step, mode="eval")
            except Exception as e:
                # 평가 실패로 학습이 죽으면 안 되므로 경고만 남긴다.
                logging.warning(f"Could not evaluate on episode from dataset: {e}")
            finally:
                # 예외가 났든 안 났든 학습 모드 복귀. 이게 없으면 이후 전 스텝이 eval 모드로
                # 학습되어(dropout 비활성) 조용히 잘못된 결과가 나온다.
                policy.train()

    # ── [11] 정리 ────────────────────────────────────────────────────────────
    if eval_env:
        eval_env.close()   # 병렬 환경의 자식 프로세스 종료. 안 하면 좀비로 남는다.
    logging.info("End of training")

    # --policy.push_to_hub=false로 끄는 게 보통이다(기본값이 True라 켜져 있으면
    # 학습이 끝나고 HF Hub 업로드를 시도한다).
    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    # DataLoader worker를 fork가 아닌 spawn으로 띄운다. fork는 부모의 메모리를 그대로
    # 복사하는데, CUDA 컨텍스트는 복사되면 안 되는 자원이라 자식에서 터진다.
    # force=True는 이미 설정돼 있어도 덮어쓴다는 뜻.
    mp.set_start_method("spawn", force=True)
    init_logging()   # 로그 포맷터 설정(시간 + 파일:줄번호)
    # train()에 인자가 없는 건 @parser.wrap()이 sys.argv를 파싱해 cfg를 만들어 주입하기 때문이다.
    train()
