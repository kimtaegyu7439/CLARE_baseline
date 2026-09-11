#!/usr/bin/env python

# Copyright 2025 Nur Muhammad Mahi Shafiullah,
# and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field

from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("ditflow_mt")
@dataclass
class DiTFlowMTConfig(PreTrainedConfig):
    """Configuration class for multi-task DiTFlowPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and `output_shapes`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: DiT-flow model action prediction size as detailed in `DiTFlowPolicy.select_action`.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            See `DiTFlowPolicy.select_action` for more details.
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.


        frequency_embedding_dim: The embedding dimension for the time value embedding in the flow model.
        num_blocks: The number of transformer blocks in the DiT flow model.
        hidden_dim: The hidden dimension for the transformer blocks in the DiT flow model.
        num_heads: The number of attention heads in the transformer blocks.
        dropout: The dropout rate used inside the transformer blocks.
        dim_feedforward: The expanded feedforward dimension in the MLPs used in the transformer block.
        activation: The activation function used in the transformer blocks.
        clip_sample: Whether to clip the sample to [-`clip_sample_range`, +`clip_sample_range`] for each
            denoising step at inference time. WARNING: you will need to make sure your action-space is
            normalized to fit within this range.
        clip_sample_range: The magnitude of the clipping range as described above.
        num_inference_steps: Number of reverse diffusion steps to use at inference time (steps are evenly
            spaced).
        do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions. See
            `LeRobotDataset` and `load_previous_and_future_frames` for mor information. Note, this defaults
            to False as the original Diffusion Policy implementation does the same.
    """

    # ── 입출력 구조 ────────────────────────────────────────────────────────────
    # fps=20이므로 1스텝 = 0.05초. 아래 세 값이 이 정책의 시간 규약 전부를 결정한다.
    n_obs_steps: int = 2      # 보는 과거 관측 프레임 수 -> [t-1, t]    (0.1초)
    horizon: int = 16         # 한 번에 출력하는 액션 개수 (고정 출력)  (0.8초 분량)
    n_action_steps: int = 8   # 그중 실제 실행하는 개수, 이후 재예측     (0.4초)
    # 16개를 예측하고 8개만 쓰는 것은 MPC와 같은 receding horizon 방식이다.
    # 멀리 내다보고 계획해야 앞부분이 일관되지만, 먼 미래는 부정확하므로 절반만 쓰고
    # 새 관측으로 다시 계획한다. temporal ensemble(겹치는 예측 평균)은 하지 않는다 --
    # generate_actions()가 나머지를 그냥 버린다.

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            # 이미지는 MEAN_STD지만 통계는 데이터셋이 아니라 ImageNet 값을 쓴다.
            # DINOv2가 ImageNet 정규화로 사전학습됐기 때문(datasets/factory.py 참조).
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    #
    # EpisodeAwareSampler가 각 에피소드의 마지막 7프레임을 아예 뽑지 않는다.
    # 그 결과 "실제로 실행될 8개 액션(배열 index 1~8)은 절대 패딩이 아님"이 보장되고,
    # 여분의 미리보기 구간(index 9~15)에만 패딩이 허용된다. 실측(에피소드 길이 98):
    #     idx=90(마지막 샘플): action_is_pad = [0]*9 + [1]*7
    #     에피소드 전체로는 액션 슬롯의 약 2%만 패딩
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # ── 아키텍처 ──────────────────────────────────────────────────────────────
    # 언어 백본. CLIP 텍스트 인코더로 자연어 명령("pick up the black bowl...")을 임베딩.
    # HF Hub에서 자동 다운로드되며 HF_HUB_CACHE 경로에 저장된다.
    tokenizer_max_length: int = 48
    language_model_name: str = "openai/clip-vit-base-patch32"
    freeze_language_pretrained: bool = True   # 실제로 forward가 torch.no_grad()로 감싸져 있다

    # 비전 백본. 마찬가지로 동결되며 학습되지 않는다.
    vit_name: str = 'facebook/dinov2-base'

    # Diffusion Transformer (DiT) 파라미터. 여기가 유일하게 학습되는 부분이다
    # (백본 2개는 동결, 학습 대상은 projection 3개 + velocity_net).
    frequency_embedding_dim: int = 256   # flow 시간 t의 사인 임베딩 차원
    hidden_dim: int = 512                # DiT 내부 폭. 조건 벡터도 이 차원으로 투영된다.
    num_blocks: int = 6                  # DiT 디코더 블록 수
    num_heads: int = 16
    dropout: float = 0.1
    dim_feedforward: int = 4096
    activation: str = "gelu"
    # 참고: 조건 벡터 차원 = 512(언어) + 2x8(상태) + 2x2x512(이미지) = 2576.
    # 이 2576이 velocity_net.cond_proj의 입력 크기이고, CLARE 어댑터가 붙는 지점이다.

    # ── 노이즈 스케줄 (flow matching) ────────────────────────────────────────
    # 학습 시 시간 t를 뽑는 분포. uniform이면 U(0,1).
    training_noise_sampling: str = (
        "uniform"  # "uniform" or "beta", from pi0 https://www.physicalintelligence.company/download/pi0.pdf
    )
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # 추론 시 적분 스텝 수. 크면 정확하지만 느리다(실시간 제어에서 병목).
    num_inference_steps: int | None = 100

    # ── 손실 계산 ────────────────────────────────────────────────────────────
    # 주의: 기본값이 False이고 배포된 체크포인트도 False다. 즉 이 설정에서는
    # action_is_pad가 배치에 실려 오지만 compute_loss에서 사용되지 않으며,
    # 에피소드 경계에서 복제된 가짜 액션도 그대로 학습된다.
    # drop_n_last_frames=7이 패딩을 이미 2% 수준으로 억제하므로 영향이 작아
    # 켜지 않은 것으로 보인다. True로 바꾸면 해당 위치가 손실에서 제외된다.
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""

        if self.training_noise_sampling not in ("uniform", "beta"):
            raise ValueError(
                f"`training_noise_sampling` must be either 'uniform' or 'beta'. Got {self.training_noise_sampling}."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")


        # Check that all input images have the same shape.
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

    # ── 시간 규약 ────────────────────────────────────────────────────────────
    # 아래 세 property가 데이터셋 질의로 번역된다(datasets/factory.py의
    # resolve_delta_timestamps가 각 값을 fps로 나눠 초 단위로 바꾼다).
    # 여기를 고치면 학습과 추론 양쪽이 자동으로 함께 움직인다.

    @property
    def observation_delta_indices(self) -> list:
        """관측 창. n_obs_steps=2 -> [-1, 0], 즉 [t-1, t]."""
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        """액션 구간. n_obs_steps=2, horizon=16 -> [-1, 0, 1, ..., 14], 즉 [t-1 .. t+14].

        시작점이 0이 아니라 1-n_obs_steps인 이유는 액션 구간의 원점을 관측 창의
        원점에 맞췄기 때문이다(diffusion policy에서 물려받은 관례).

        그래서 배열 index와 시각의 대응이 한 칸 밀린다:
            index 0 -> t-1  (이미 지나간 시점. 추론 시 버려짐)
            index 1 -> t    (지금. 여기서부터 실행)
            index 8 -> t+7  (실행 마지막)
            index 15 -> t+14 (미리보기)
        따라서 generate_actions()는 [0:8]이 아니라 [1:9]를 슬라이싱해야 한다.
        index 0을 쓰면 로봇이 한 스텝씩 뒤처진다.

        index 0은 관측 두 장 사이에 무슨 액션이 있었는지 맞히는 셈이라 사실상
        inverse dynamics 과제가 된다. 별도 손실 항은 없고 나머지 15개와 동일한 MSE다.
        """
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        """모방학습이므로 보상을 쓰지 않는다. None이면 factory가 해당 분기를 건너뛴다."""
        return None
