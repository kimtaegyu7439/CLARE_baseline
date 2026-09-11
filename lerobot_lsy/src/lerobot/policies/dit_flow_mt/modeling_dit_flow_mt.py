# Copyright 2025 Nur Muhammad Mahi Shafiullah,
# and The HuggingFace Inc. team. All rights reserved.
# Heavy inspiration taken from
# * DETR by Meta AI (Carion et. al.): https://github.com/facebookresearch/detr
# * DiT by Meta AI (Peebles and Xie): https://github.com/facebookresearch/DiT
# * DiT Policy by Dasari et. al. : https://github.com/sudeepdasari/dit-policy

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ============================================================================
# 이 파일 전체 지도 (LIBERO 설정 기준: B=배치, 관측 2스텝, 카메라 2대,
#                    상태 8차원, 액션 7차원, hidden_dim=512, horizon=16)
#
#  [학습 경로]  train.py
#      └─ DiTFlowMTPolicy.forward(batch)                        <- 진입점
#           ├─ normalize_inputs / 카메라 키 stack / normalize_targets
#           └─ DiTFlowModel.compute_loss(batch)
#                ├─ _prepare_global_conditioning(batch) -> (B, 2576)
#                ├─ flow matching: x(t) = (1-t)*noise + t*action
#                └─ velocity_net(x_t, t, cond) 와 (action - noise) 사이 MSE
#
#  [추론 경로]  eval / 실로봇
#      └─ DiTFlowMTPolicy.select_action(obs)                    <- 진입점, 매 스텝 호출
#           ├─ 큐에 관측 누적 (n_obs_steps=2 유지)
#           ├─ 큐가 비었을 때만:
#           │    predict_action_chunk -> DiTFlowModel.generate_actions
#           │        ├─ _prepare_global_conditioning -> (B, 2576)
#           │        ├─ conditional_sample -> _DiTNoiseNet.sample
#           │        │     (노이즈에서 시작해 Euler 100스텝 적분) -> (B,16,7)
#           │        └─ [:, 1:9] 슬라이싱 -> (B, 8, 7)
#           └─ 큐에서 액션 하나 popleft -> (B, 7)
#
#  학습되는 파라미터는 다음 4덩어리뿐이다:
#      language_embedding_projection (512->512)
#      rgb_embedding_projection      (768->512)
#      velocity_net                  (DiT 본체)
#      (USE_STATE_PROJ=True일 때만 state_proj — 현재는 꺼져 있음)
#  CLIP과 DINOv2 백본은 동결(requires_grad_(False) + eval() + no_grad).
# ============================================================================

import copy
from collections import deque

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import CLIPTextModel, CLIPTokenizer, AutoModel

from lerobot.constants import OBS_ENV_STATE, OBS_ROBOT, ACTION, OBS_IMAGES
from lerobot.policies.dit_flow_mt.configuration_dit_flow_mt import DiTFlowMTConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)

# ── 파일 수준 스위치 두 개 ────────────────────────────────────────────────────
# config가 아니라 모듈 상수로 박혀 있다. 즉 체크포인트/설정으로는 못 바꾸고
# 이 파일을 고쳐야 바뀐다. 실험할 때 여기를 건드렸다면 체크포인트 호환이 깨진다.
#
# USE_STATE_PROJ:
#   False(현재) -> 로봇 상태 8차원을 투영 없이 그대로 flatten해서 조건 벡터에 붙인다.
#                  조건 벡터 2576 = 512(언어) + 16(상태) + 2048(이미지) 이 성립하는 이유.
#   True        -> 상태를 hidden_dim(512)로 투영한다. 단, 아래 주의 참조:
#                  DiTFlowModel._prepare_global_conditioning의 True 분기는 (B, 2, 512)
#                  3D 텐서를 append하는데 나머지 항목은 2D라서 마지막 torch.cat에서
#                  차원이 안 맞는다. 켜려면 그 분기에 flatten(start_dim=1)을 넣어야 한다.
USE_STATE_PROJ = False

# NAMING_AS_MLP:
#   True(현재) -> _DiTDecoder의 FFN을 `self.mlp`라는 이름의 서브모듈(MLP 클래스)로 만든다.
#   False      -> linear1/linear2를 블록에 직접 달고 forward에서 펼쳐 쓴다.
#   기능은 완전히 동일하고 파라미터 이름만 다르다. 이름이 중요한 이유는 PEFT/LoRA가
#   정규식으로 대상 모듈을 고르기 때문 (".*velocity_net.cond_proj" 같은 패턴).
#   두 값 사이를 오가면 state_dict 키가 달라져 기존 체크포인트를 못 읽는다.
NAMING_AS_MLP = True


def _get_activation_fn(activation: str):
    """Return an activation function given a string"""
    # config.activation = "gelu" 이므로 실제로는 항상 두 번째 분기가 나간다.
    # 주의: relu/glu는 "함수"를, gelu는 "nn.Module 인스턴스"를 돌려준다.
    # gelu 분기에서 만들어진 nn.GELU는 호출부에서 self.activation에 대입되므로
    # 부모 모듈의 서브모듈로 등록된다(파라미터는 없어서 실질적 영향은 없다).
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN 변조: 조건 벡터로 만든 scale/shift를 특징에 곱하고 더한다.

    shape:
        x     : (T, B, H)   시퀀스 우선(seq-first) 레이아웃. T=horizon=16
        shift : (B, H)  -> unsqueeze(0) -> (1, B, H)   T축으로 브로드캐스트
        scale : (B, H)  -> unsqueeze(0) -> (1, B, H)
        return: (T, B, H)

    (1 + scale)인 이유: scale을 0으로 초기화하면 이 연산이 항등함수가 된다.
    DiT 계열이 학습 초기에 "조건을 아직 안 쓰는 상태"에서 출발하게 만드는 표준 기법.
    """
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)



class LanguageEncoder(nn.Module):
    """
    Language Encoder using pretrained CLIP "Learning Transferable Visual Models From Natural Language Supervision"
    (paper: https://arxiv.org/pdf/2103.00020)

    자연어 명령("pick up the black bowl ...")을 512차원 벡터로 만든다. 이 벡터가
    조건 벡터 2576차원 중 앞의 512를 차지한다.

    동결되어 학습되지 않는다(freeze_language_pretrained=True). 학습되는 건 이 출력을
    받는 DiTFlowModel.language_embedding_projection뿐이다.

    문자열을 키로 임베딩을 캐싱한다. LIBERO는 데이터셋 하나당 과제 문장이 하나뿐이라
    사실상 첫 배치에서 한 번만 계산하고 이후로는 캐시를 재사용한다.

    shape 요약:
        입력 : list[str], 길이 B
        출력 : (B, 512)      512 = clip_model.config.hidden_size
    """

    def __init__(self, config:DiTFlowMTConfig):
        super().__init__()

        self.config = config

        # from_pretrained가 HF Hub에서 받아온다. 저장 위치는 HF_HUB_CACHE 환경변수.
        self.tokenizer = CLIPTokenizer.from_pretrained(config.language_model_name)
        self.clip_model = CLIPTextModel.from_pretrained(config.language_model_name)
        # CLIP 체크포인트에는 비전 타워도 들어 있지만 여기서는 텍스트 타워만 쓴다.
        # 로드 시 나오는 vision_model.* UNEXPECTED 경고는 그 때문이며 정상이다.
        self.cache = {}   # {문장: 임베딩(CPU 텐서)}

        # Get the hidden size of CLIP model
        # clip-vit-base-patch32 -> 512
        self.hidden_size = self.clip_model.config.hidden_size

        # Freeze the base model if specified
        # config.freeze_language_pretrained는 True가 기본값. False로 두면 CLIP이
        # 학습 대상이 되지만, 호출부(_prepare_global_conditioning)가 어차피
        # torch.no_grad()로 감싸고 있어서 그래디언트는 흐르지 않는다.
        # 즉 이 플래그를 False로 바꿔도 CLIP은 사실상 학습되지 않는다.
        if config.freeze_language_pretrained:
            self.clip_model.requires_grad_(False)
            # for param in self.clip_model.parameters():
            #     param.requires_grad = False
            self.clip_model.eval()

    def forward(self, texts):
        """
        Encodes input text into embeddings and projects to specified output dimension.

        Args:
            texts (list[str]): List of text strings to be encoded (batch size B).

        Returns:
            torch.Tensor: The projected text embeddings of shape (B, output_dim).

        동작 흐름 (3단계):
            1) 배치의 각 문장을 캐시 히트/미스로 분류
            2) 미스인 것만 모아서 CLIP 한 번 통과 + 캐시에 저장
            3) 원래 순서대로 재조립해서 stack
        LIBERO 단일 태스크 학습이라면 첫 스텝에서만 2)가 돌고 이후로는
        uncached_texts가 빈 리스트가 되어 CLIP 호출 자체가 사라진다.
        """
        # Check cache first
        cached_embeddings = []    # (주: 아래에서 실제로 쓰이지 않는 잔재 변수)
        uncached_texts = []       # 아직 인코딩 안 한 문장들 (중복 제거는 안 함)
        uncached_indices = []     # 그 문장들이 원래 배치에서 몇 번째였는지

        # ── 1단계: 캐시 히트/미스 분류 ──
        for i, text in enumerate(texts):
            if text in self.cache:
                cached_embeddings.append(self.cache[text])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Process uncached texts
        # ── 2단계: 미스가 하나라도 있을 때만 CLIP 실행 ──
        # 이 if가 False면 (= 전부 캐시 히트) 이 블록 전체를 건너뛴다.
        # 단일 태스크 학습에서는 첫 배치 이후 항상 False.
        if uncached_texts:
            # Tokenize the input texts
            # padding=True -> 배치 내 가장 긴 문장 길이에 맞춰 패딩
            # 반환: {"input_ids": (U, L), "attention_mask": (U, L)}   U=len(uncached_texts)
            inputs = self.tokenizer(
                uncached_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length
            )

            # Move inputs to the same device as the model
            # 주의: 모델의 실제 device가 아니라 config.device 문자열을 신뢰한다.
            device = self.config.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings from CLIP text model
            # ★ 이 조건은 뒤집혀 있다(`not`이 붙지 말았어야 한다). 진리표:
            #     freeze=True  -> eval()  -> training=False -> not False=True  -> grad ON
            #     freeze=False -> train() -> training=True  -> not True =False -> grad OFF
            #   즉 "동결일 때 grad를 켜고, 학습하려 할 때 grad를 끈다". 의도대로라면
            #   set_grad_enabled(self.clip_model.training)이어야 한다.
            #
            # 그런데도 지금 아무 문제가 없는 이유(둘 다 독립적으로 성립):
            #   1) set_grad_enabled(True)는 바깥 no_grad를 실제로 덮어써서 grad를 켠다.
            #      (_prepare_global_conditioning의 with torch.no_grad()보다 이쪽이 이긴다.)
            #      하지만 CLIP 파라미터가 전부 requires_grad=False이고 입력이 정수
            #      token id라, 연쇄 어디에도 grad가 필요한 텐서가 없어 그래프 자체가
            #      만들어지지 않는다. 결과는 requires_grad=False, grad_fn=None.
            #      -> 이 줄은 실질적으로 no-op이고, 아래 캐시의 .detach()도 no-op이다.
            #   2) 어차피 호출부가 no_grad로 감싸므로 의도상으로도 grad는 불필요하다.
            #
            # 진짜 영향은 반대편에 있다: freeze_language_pretrained=False로 두고 CLIP을
            # 미세조정하려 하면, 이 줄이 grad를 꺼 버려서 학습이 안 된다(게다가 호출부
            # no_grad도 막는다). CLIP을 실제로 학습시키려면 (a) 여기 `not`을 빼고
            # (b) _prepare_global_conditioning의 with torch.no_grad()를 풀어야 한다.
            with torch.set_grad_enabled(not self.clip_model.training):
                outputs = self.clip_model(**inputs)

            # Get the EOS token embeddings
            # CLIP 텍스트 타워의 pooler_output = EOS 토큰 위치의 hidden state
            # (U, L, 512) 중 EOS 자리만 뽑은 (U, 512)
            uncached_embeddings = outputs.pooler_output

            # Update cache
            # detach().cpu()로 그래프를 끊고 CPU에 보관 -> GPU 메모리 절약,
            # 그리고 캐시가 학습 그래프를 물고 있지 않게 한다.
            for text, embedding in zip(uncached_texts, uncached_embeddings):
                self.cache[text] = embedding.detach().cpu()  # Store in CPU to save GPU memory

        # Combine cached and uncached embeddings in the original order
        # ── 3단계: 원래 배치 순서로 재조립 ──
        all_embeddings = [None] * len(texts)
        # Process uncached texts
        # 방금 계산한 것들을 원래 자리에 꽂는다 (GPU 텐서 그대로).
        if uncached_texts:
            for i, emb in zip(uncached_indices, uncached_embeddings):
                all_embeddings[i] = emb
        # 나머지 빈 자리는 캐시(CPU)에서 가져와 device로 올린다.
        # `all_embeddings[i] is None` 조건 덕분에 위에서 채운 자리는 덮어쓰지 않는다.
        for i, text in enumerate(texts):
            if text in self.cache and all_embeddings[i] is None:
                # Move cached embedding to same device as model
                all_embeddings[i] = self.cache[text].to(self.config.device)

        # Stack all embeddings into a single tensor
        # list[(512,)] x B  ->  (B, 512)
        return torch.stack(all_embeddings)


class DINOv2Encoder(nn.Module):
    """이미지 한 장 -> 768차원 CLS 토큰. facebook/dinov2-base를 그대로 쓴다.

    이미지도 동결이라 학습되지 않는다. 학습되는 건 이 출력을 512차원으로 줄이는
    DiTFlowModel.rgb_embedding_projection뿐이다.

    호출 횟수에 주의: 배치당 (n_obs_steps=2) x (카메라 2대) = 4장을 인코딩한다.
    이미지가 조건 벡터의 2048/2576 = 79%를 차지하므로 이 정책은 사실상 시각 중심이다.

    shape:
        입력 : (N, 3, H, W)   N = B * n_obs_steps * n_cameras = 4B
        출력 : (N, 768)
    """

    def __init__(self, config: DiTFlowMTConfig):
        super().__init__()
        self.config = config
        self._model = AutoModel.from_pretrained(config.vit_name)
        # nn.Module 서브모듈이므로 policy.to(device)로도 따라가지만, 여기서 먼저 올려둔다.
        self._model.to(config.device)
        # requires_grad_(False) + eval()로 동결. 호출부(_prepare_global_conditioning)도
        # torch.no_grad()로 한 번 더 감싼다.
        # eval()이 중요한 이유: DINOv2 내부 dropout/stochastic depth를 끈다.
        # 다만 policy.train()을 호출하면 이 eval()이 풀린다(부모가 재귀적으로 train 모드로
        # 바꾸므로). requires_grad=False는 유지되어 학습은 안 되지만 dropout은 되살아난다.
        self._model.requires_grad_(False) # hack
        self._model.eval() # hack

        self.hidden_size = self._model.config.hidden_size   # dinov2-base -> 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, H, W) — 이미 ImageNet mean/std로 정규화된 상태로 들어온다
        #                   (Normalize가 VISUAL을 MEAN_STD로 처리)
        outputs = self._model(x)
        # 패치 토큰 전부가 아니라 CLS 토큰 하나만 쓴다. 즉 이미지 한 장이 벡터 하나로
        # 압축되며 공간 정보(어디에 무엇이)는 이 시점에 상당 부분 버려진다.
        # outputs.last_hidden_state는 (N, 1+patches, 768)이지만 여기서는 안 쓴다.
        cls_token = outputs.pooler_output # (B, 768)

        return cls_token


class _TimeNetwork(nn.Module):
    """flow 시간 t(스칼라)를 hidden_dim 벡터로 바꾼다. Transformer 위치 인코딩과 같은 방식.

    ★★ 이 파일에는 "시간"이 두 종류 있다. 절대 헷갈리면 안 된다. ★★

      (1) 환경 시간 = 로봇의 실제 시각.  n_obs_steps=2 -> [t-1, t], horizon=16 -> 액션 16스텝.
          단위는 초(1스텝=0.05s). 이 축은 _TimeNetwork에 오지 않는다.
          관측의 [t-1, t] 두 스텝은 _prepare_global_conditioning에서 flatten되어
          특징 차원으로 흡수된다:  (B,2,8) -> (B,16),  (B,2,2,512) -> (B,2048).
          즉 조건 벡터 (B,2576) 안에 "녹아든" 상태로 들어가지 별도 축으로 남지 않는다.
          액션 쪽 16스텝은 velocity_net 안에서 T축(시퀀스 축)으로 살아 있고,
          순서 정보는 dec_pos(위치 임베딩)가 담당한다.

      (2) flow 시간 τ ∈ [0,1] = 노이즈 -> 데이터로 가는 경로상의 위치.  <-- 이게 여기 들어온다
          환경 시간과 아무 관계가 없다. 샘플 하나당 스칼라 하나뿐이라 shape은 (B,).
            학습: τ ~ U(0,1),  noise_distribution.sample((B,))     -> (B,)  샘플마다 다른 값
            추론: τ = k/100,   t_all[:, k]                          -> (B,)  배치 전체가 같은 값
          forward의 assert len(t.shape) == 1이 이 규약을 강제한다.
          (B,1)이나 (B,2)를 넣으면 거기서 바로 걸린다.

    구조:  t -> [sinusoidal 주파수 임베딩] -> Linear -> SiLU -> Linear
    shape: (B,) -> (B, frequency_embedding_dim) -> (B, hidden_dim)

    두 폭은 시그니처에 기본값이 없는 필수 인자이고, 유일한 생성자인
    _DiTNoiseNet.__init__이 (time_dim, hidden_dim)을 넘긴다. 현 config에서는
    각각 256, 512다  ->  (B,) -> (B, 256) -> (B, 512).
    """

    def __init__(self, frequency_embedding_dim, hidden_dim, learnable_w=False, max_period=1000):
        # cos/sin 두 벌을 concat해서 frequency_embedding_dim을 채우므로 짝수여야 한다.
        assert frequency_embedding_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(frequency_embedding_dim // 2)   # 256 // 2 = 128
        super().__init__()

        # 주파수 사다리를 기하급수로 만든다: w[0]=1 ... w[127]=1/max_period=0.001
        w = np.log(max_period) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()   # (128,)
        # register_parameter + requires_grad=False -> 학습은 안 되지만 state_dict에는 들어간다
        # (buffer가 아니라 Parameter로 등록한 선택. learnable_w=True면 학습 가능해진다.)
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t):
        # t: (B,) — flow matching 시간 τ. 학습 시 U(0,1) 또는 Beta 분포에서 뽑고,
        #           추론 시 0, 0.01, ..., 0.99 로 증가한다.
        #  관측 스텝 수(n_obs_steps=2)와는 무관하다. (B,2)가 아니라 (B,)다.
        #  샘플 하나에 τ 스칼라 하나 -> 그 샘플의 16개 액션 토큰 전부가 같은 τ를 공유한다.
        assert len(t.shape) == 1, "assumes 1d input timestep array"
        # (B,1) * (1,128) -> (B, 128) : 각 주파수로 t를 스케일
        t = t[:, None] * self.w[None]
        # 참고: max_period=1000은 t가 [0,1000] 범위일 때를 상정한 값인데, 여기서 t는 [0,1]이다.
        #       따라서 각도가 최대 1 rad라 cos는 거의 1, sin은 거의 선형이 되어 사인 임베딩의
        #       "여러 해상도" 이점이 크게 살지 않는다. 그래도 t에 대해 단조/단사라 학습은 된다.
        #       (다른 구현들은 보통 t*1000을 넣는다.)
        t = torch.cat((torch.cos(t), torch.sin(t)), dim=1)   # (B, 256)
        return self.out_net(t)                               # (B, 512)


class _ShiftScaleMod(nn.Module):
    """AdaLN의 scale/shift 부분. 조건 c로부터 채널별 아핀 변환을 만들어 x에 적용.

    forward shape:
        x: (T, B, H),  c: (B, H)  ->  (T, B, H)

    reset_parameters로 weight/bias를 전부 0으로 만들면 scale(c)=0, shift(c)=0이 되어
    x * (1+0) + 0 = x, 즉 항등함수에서 학습이 시작된다.
    """

    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)                     # (B, H)
        # [None]은 앞에 축을 하나 붙여 (1, B, H)로 만들어 T축 브로드캐스트를 유도한다.
        # 즉 한 샘플의 조건이 그 샘플의 16개 액션 토큰 전부에 똑같이 적용된다.
        return x * (1 + self.scale(c)[None]) + self.shift(c)[None]

    def reset_parameters(self):
        # DiT 논문의 "zero-init adaLN". 이걸 호출하는 곳은 _DiTDecoder.reset_parameters,
        # 그리고 그걸 호출하는 곳은 _TransformerDecoder.__init__이다(레이어 복제 직후).
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    """AdaLN의 gate 부분. residual 가지에 곱해지는 게이트를 조건으로부터 만든다.

    forward shape:
        x: (T, B, H),  c: (B, H)  ->  (T, B, H)

    _ShiftScaleMod와 달리 (1 + scale)이 아니라 그냥 scale을 곱한다.
    따라서 zero-init 상태에서는 출력이 0 -> residual 가지가 통째로 꺼진 채 시작한다.
    x = x + gate(...) 형태로 쓰이므로 초기 네트워크는 항등 사상에 가깝다.
    """

    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class MLP(nn.Module):
    """Transformer 블록의 FFN. NAMING_AS_MLP=True일 때만 쓰인다.

    shape: (T, B, d_model) -> (T, B, dim_feedforward) -> (T, B, d_model)
    시그니처 기본값(256/2048)은 쓰이지 않는다. _DiTDecoder가 자기 값을 그대로
    넘기고, 그 _DiTDecoder는 _DiTNoiseNet에서 config 값을 받는다.
    현 config 기준 실제 폭:  (T, B, 512) -> (T, B, 4096) -> (T, B, 512)

    별도 클래스로 뺀 이유는 PEFT가 "...mlp.linear1" 같은 이름으로 대상을 찾을 수
    있게 하기 위함이다(아래 _DiTDecoder의 주석 참조).
    """

    def __init__(self, d_model=256, dim_feedforward=2048, dropout=0.0, activation="gelu"):
        super().__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # dropout2/3라는 이름은 _DiTDecoder에서 뽑아온 흔적이다(거기 dropout1이 따로 있다).
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.dropout3(x)

        return x


class _DiTDecoder(nn.Module):
    """DiT 블록 하나. self-attention + FFN, 둘 다 AdaLN-Zero로 조건 변조된다.

    표준 Transformer와 다른 점은 LayerNorm 뒤에 조건 기반 scale/shift가 붙고,
    residual 합류 지점에 조건 기반 gate가 붙는다는 것. 조건이 레이어마다 개별
    Linear를 통해 주입되므로 6개 블록이 서로 다른 방식으로 조건을 해석할 수 있다.

    forward shape (H = d_model. 시그니처 기본값 256이 아니라 _DiTNoiseNet이 넘기는
                   config.hidden_dim = 512가 실제 값. nhead도 기본 6이 아니라 16):
        x    : (T=16, B, H=512)   액션 토큰
        t    : (B, H)             시간 임베딩
        cond : (B, H)             투영된 조건 벡터
        ->     (T, B, H)
    """

    def __init__(self, d_model=256, nhead=6, dim_feedforward=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        # batch_first 인자를 주지 않았으므로 기본값 False = seq-first (T, B, H) 레이아웃.
        # 이 파일 전체가 (T, B, H)를 쓰는 이유가 이것이다.
        # 액션 토큰 16개끼리만 attention한다 (조건/이미지 토큰은 attention에 참여하지 않고
        # AdaLN 경로로만 들어온다). 즉 cross-attention은 없다.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        # 파일 상단 NAMING_AS_MLP 스위치. 현재 True.
        # 두 분기는 수학적으로 동일하고 파라미터 "이름"만 다르다:
        #   True  -> layers.k.mlp.linear1.weight
        #   False -> layers.k.linear1.weight
        # 아래 주석("mlp built upon reference could meet issue with peft")대로,
        # nn.Sequential로 묶으면 같은 모듈이 두 이름으로 등록되어 PEFT가 혼란스러워한다.
        if NAMING_AS_MLP:
            self.mlp = MLP(
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.activation = _get_activation_fn(activation)

        # elementwise_affine 기본값 True라 weight/bias가 있다. 그 위에 다시 AdaLN이
        # scale/shift를 얹는 구조(원 DiT는 여기를 affine=False로 두는 편이다).
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)

        # mlp built upon reference could meet issue with peft
        # create mlp
        # self.mlp = nn.Sequential(
        #     self.linear1,
        #     self.activation,
        #     self.dropout2,
        #     self.linear2,
        #     self.dropout3,
        # )

        # create modulation layers
        # 블록 하나당 조건 주입 지점이 4개. 각각 (B,512)->(B,512) Linear를 갖는다.
        # -> 블록당 Linear 6개(attn_mod 2 + attn_gate 1 + mlp_mod 2 + mlp_gate 1)
        self.attn_modulate = _ShiftScaleMod(d_model)   # norm1 뒤 scale/shift
        self.attn_gate = _ZeroScaleMod(d_model)        # attn residual 게이트
        self.mlp_modulate = _ShiftScaleMod(d_model)    # norm2 뒤 scale/shift
        self.mlp_gate = _ZeroScaleMod(d_model)         # FFN residual 게이트

    def forward(self, x, t, cond):
        # process the conditioning vector first
        # 시간과 조건을 "더해서" 하나로 합친다(concat이 아니라 덧셈).
        # 둘 다 (B, 512)라 그대로 더해지며, 이후 4개 변조 레이어가 이 합을 공유한다.
        cond = cond + t                                  # (B, 512)

        # ── attention 가지 ──
        x2 = self.attn_modulate(self.norm1(x), cond)     # (T,B,H) pre-norm + AdaLN
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)  # self-attn: q=k=v
        # 마스크를 주지 않으므로 인과(causal) 제약이 없다. 16개 액션이 서로 전부 본다.
        x = x + self.attn_gate(self.dropout1(x2), cond)  # gate가 0으로 초기화됨 -> 초기엔 x 그대로

        # ── FFN 가지 ──
        x3 = self.mlp_modulate(self.norm2(x), cond)      # (T,B,H)

        # 위 __init__의 분기와 짝을 이룬다. 현재는 항상 self.mlp 경로.
        if NAMING_AS_MLP:
            x3 = self.mlp(x3)
        else:
            x3 = self.activation(self.linear1(x3))
            x3 = self.dropout2(x3)
            x3 = self.linear2(x3)
            x3 = self.dropout3(x3)

        x3 = self.mlp_gate(x3, cond)                     # 마찬가지로 초기엔 0
        return x + x3

    def reset_parameters(self):
        # _TransformerDecoder.__init__에서 레이어를 deepcopy한 직후 호출된다.
        # 1) 2차원 이상 파라미터(= 모든 Linear/attn weight)를 xavier로 재초기화.
        #    LayerNorm weight와 각종 bias는 1차원이라 건드리지 않는다.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 2) 그다음 변조 레이어만 0으로 덮어쓴다(순서 중요 — 위 xavier를 되돌리는 것).
        #    결과적으로 학습 시작 시점의 블록은 "조건 무시 + residual 통과"에 가깝다.
        for s in (self.attn_modulate, self.attn_gate, self.mlp_modulate, self.mlp_gate):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    """DiT 마지막 층. 토큰(512차원)을 실제 출력(액션 7차원)으로 내린다.

    forward shape:
        x    : (T=16, B, hidden=512)
        t    : (B, 512)
        cond : (B, 512)
        ->     (T, B, out_size=7)
    """

    def __init__(self, hidden_size, out_size):
        super().__init__()
        # 여기는 elementwise_affine=False. 아핀 변환은 아래 adaLN_modulation이 전담한다.
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        # shift와 scale을 한 Linear로 뽑고 chunk로 반으로 가른다(2*hidden 출력).
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = cond + t                                       # (B, 512)

        # (B, 1024) -> chunk(2, dim=1) -> shift (B,512), scale (B,512)
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        # 주의: self.norm_final을 만들어 놓고 여기서 쓰지 않는다. x가 정규화 없이
        #       바로 변조된다. (원 DiT는 modulate(self.norm_final(x), ...) 형태.)
        #       파라미터가 없는 모듈이라 state_dict에는 흔적이 남지 않는다.
        x = modulate(x, shift, scale)                         # (T, B, 512)
        x = self.linear(x)                                    # (T, B, 7)
        return x

    def reset_parameters(self):
        # 전 파라미터를 0으로 -> 초기 출력이 항상 0(= 속도 0)이 된다.
        # 다만 이 메서드는 _DiTNoiseNet.__init__에서 호출되지 않는다.
        # _TransformerDecoder는 자기 layers에 대해서만 reset_parameters를 부르고
        # eps_out은 그 밖에 있기 때문. 즉 현재 eps_out은 PyTorch 기본 초기화 상태다.
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerDecoder(nn.Module):
    """블록 하나를 num_layers번 복제해 쌓는다.

    deepcopy로 복제하므로 각 레이어는 독립된 파라미터를 갖는다(가중치 공유 아님).
    복제 직후 reset_parameters를 돌려 레이어마다 다른 xavier 초기값을 부여한다
    (이걸 안 하면 6개 레이어가 전부 동일한 초기값으로 시작한다).
    """

    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(base_module) for _ in range(num_layers)])

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, src, t, cond):
        # src: (T, B, H). t/cond는 모든 레이어에 "동일하게" 다시 전달된다.
        # 즉 조건은 첫 레이어에서 한 번만 주입되는 게 아니라 6번 반복 주입된다.
        x = src
        for layer in self.layers:
            x = layer(x, t, cond)
        return x


class _DiTNoiseNet(nn.Module):
    def __init__(
        self,
        # ── 아래 기본값들은 전부 "죽은 값"이다 ──────────────────────────────
        # 이 클래스를 만드는 곳은 DiTFlowModel.__init__ 한 군데뿐이고(같은 파일 하단),
        # 거기서 모든 인자를 config 값으로 명시해 넘긴다. 즉 시그니처의 256/8/2048은
        # 절대 쓰이지 않는다. 아래 주석의 "실제:" 값이 런타임에 들어오는 값이다.
        ac_dim,             # 실제: 7    config.action_feature.shape[0]
        ac_chunk,           # 실제: 16   config.horizon
        cond_dim,           # 실제: 2576 language 512 + (state 8 + img 1024) * 2
        time_dim=256,       # 실제: 256  config.frequency_embedding_dim  (우연히 기본값과 같음)
        hidden_dim=256,     # 실제: 512  config.hidden_dim      ★ 기본값과 다르다
        num_blocks=6,       # 실제: 6    config.num_blocks
        dropout=0.1,        # 실제: 0.1  config.dropout
        dim_feedforward=2048,  # 실제: 4096 config.dim_feedforward   ★ 다르다
        nhead=8,            # 실제: 16   config.num_heads             ★ 다르다
        activation="gelu",  # 실제: "gelu"
        clip_sample=False,      # 실제: True   config.clip_sample     ★ 다르다
        clip_sample_range=1.0,  # 실제: 1.0
    ):
        """노이즈 섞인 액션 + flow 시간 t + 조건 벡터 -> 속도장.

        (noisy_actions (B,16,7), time (B,), global_cond (B,2576)) -> (B,16,7)

        조건은 cond_proj를 거쳐 512차원이 된 뒤 6개 디코더 블록 전부에 전달되어
        AdaLN 변조(scale/shift/gate)를 구동한다. 즉 cond_proj 하나가 네트워크 전체의
        조건 반응을 좌우하며, 그래서 CLARE가 정확히 이 레이어만 어댑터 대상으로 삼는다
        (adapter_config.json의 target_modules = ".*velocity_net.cond_proj").
        velocity_net 안에 nn.Linear가 61개 있지만 어댑터가 붙는 건 그중 1개뿐이다.

        (Linear 61개 내역: cond_proj 1 + time_net.out_net 2 + ac_proj 2
                          + 블록당 9개 x 6 = 54 + eps_out 2 = 61.
         블록당 9개 = attn.out_proj 1 + mlp 2 + _ShiftScaleMod 2개x2 + _ZeroScaleMod 2개x1.
         MultiheadAttention의 in_proj는 nn.Linear가 아니라 raw Parameter라 안 세어진다.)

        ※ 아래 본문 주석에 적힌 512는 전부 "런타임의 hidden_dim 값"이지
           시그니처 기본값(256)이 아니다. 시그니처 쪽 주석 참조.
        """
        super().__init__()
        self.ac_dim, self.ac_chunk = ac_dim, ac_chunk   # 7, 16

        # positional encoding blocks
        # 학습되는 위치 임베딩. (ac_chunk=16, 1, hidden_dim=512)
        # 가운데 1은 배치축 브로드캐스트용.
        # 사인 위치 인코딩이 아니라 free parameter라서 "몇 번째 액션인가"를 모델이
        # 직접 배운다. 이게 없으면 self-attention이 순서를 구분하지 못한다.
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # input encoder mlps
        # (B,) -> (B, time_dim=256) -> (B, hidden_dim=512)
        # _TimeNetwork의 두 번째 인자가 출력 폭이다. 여기 넘기는 hidden_dim이 512이므로
        # 출력도 512. (_TimeNetwork 자체의 시그니처 기본값과는 무관하다.)
        self.time_net = _TimeNetwork(time_dim, hidden_dim)
        # 액션 7차원을 토큰으로. 첫 Linear가 7->7이라 폭을 안 늘리고 한 번 섞기만 한다.
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),                 # (…, ac_dim=7) -> (…, 7)
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),             # (…, 7) -> (…, hidden_dim=512)
        )
        # ★ 조건 벡터의 유일한 관문. (B, cond_dim=2576) -> (B, hidden_dim=512).
        #   2576*512 ≈ 132만 파라미터. 언어/상태/이미지가 여기서 처음으로 섞인다.
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # decoder blocks
        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        # decoder_module은 "틀"이고, 실제로는 이 안에서 num_blocks번 deepcopy된다.
        # 원본 decoder_module 자체는 self에 등록되지 않아 파라미터로 잡히지 않는다.
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # turns predicted tokens into epsilons
        # 이름은 eps(노이즈)지만 flow matching이라 실제로 뱉는 값은 "속도"다.
        # (확산 코드에서 옮겨 오며 이름만 남은 것.)
        # (T, B, hidden_dim=512) -> (T, B, ac_dim=7)
        self.eps_out = _FinalLayer(hidden_dim, ac_dim)

        # clip the output samples
        # config: clip_sample=True, clip_sample_range=1.0
        # 액션이 MIN_MAX로 [-1,1] 정규화되어 있으므로 적분 중간값을 그 범위에 가둔다.
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

    def forward(self, noisy_actions, time, global_cond):
        """속도장 v_theta(x_t, t, c) 한 번 평가.

        shape 흐름:
            noisy_actions (B, 16, 7)
            time          (B,)
            global_cond   (B, 2576)
            ->            (B, 16, 7)
        """
        c = self.cond_proj(global_cond)          # (B, 2576) -> (B, 512)
        time_enc = self.time_net(time)           # (B,)      -> (B, 512)

        ac_tokens = self.ac_proj(noisy_actions)  # [B, T, adim] -> [B, T, hidden_dim]
        # nn.MultiheadAttention이 batch_first=False이므로 seq-first로 전치한다.
        ac_tokens = ac_tokens.transpose(0, 1)  # [B, T, hidden_dim] -> [T, B, hidden_dim]

        # Allow variable length action chunks
        # dec_pos는 (16, 1, 512)로 만들어 두고 실제 T만큼만 잘라 쓴다.
        # 학습 때는 T==16이라 슬라이싱이 무의미하지만, 더 짧은 청크로 추론할 때를 대비한 것.
        dec_in = ac_tokens + self.dec_pos[: ac_tokens.size(0)]  # [T, B, hidden_dim]

        # apply decoder
        # 6개 블록 통과. t와 c는 블록마다 반복 주입된다(_TransformerDecoder.forward 참조).
        dec_out = self.decoder(dec_in, time_enc, c)

        # apply final epsilon prediction layer
        eps_out = self.eps_out(dec_out, time_enc, c)  # [T, B, hidden_dim] -> [T, B, adim]
        return eps_out.transpose(0, 1)  # [T, B, adim] -> [B, T, adim]

    @torch.no_grad()
    def sample(
        self, condition: torch.Tensor, timesteps: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """추론: ODE dx/dt = v(x, t, c)를 t=0에서 t=1까지 Euler로 적분한다.

        t=0의 순수 가우시안 노이즈에서 출발해 t=1의 액션 궤적에 도달한다.
        학습에서 배운 경로가 직선이므로 이론상 적은 스텝으로도 되지만
        여기서는 config대로 100스텝을 돈다(= forward를 100번 호출).

        shape:
            condition (B, 2576) -> return (B, 16, 7)
        """
        # Use Euler integration to solve the ODE.
        batch_size, device = condition.shape[0], condition.device
        x_0 = self.sample_noise(batch_size, device, generator)   # (B, 16, 7) ~ N(0, I)
        dt = 1.0 / timesteps                                     # 0.01
        # t_all: (B, 100), 값은 행마다 동일하게 [0, 0.01, ..., 0.99]
        # (배치별로 다른 시간표를 쓸 여지를 남긴 구조지만 실제로는 전부 같다.)
        t_all = (
            torch.arange(timesteps, device=device).float().unsqueeze(0).expand(batch_size, timesteps)
            / timesteps
        )

        # 변수명은 x_0이지만 루프를 돌며 계속 갱신되는 "현재 상태"다(t=0의 값이 아님).
        for k in range(timesteps):
            t = t_all[:, k]                                      # (B,)
            x_0 = x_0 + dt * self.forward(x_0, t, condition)     # Euler 한 스텝
            # config.clip_sample=True이므로 매 스텝 [-1, 1]로 자른다.
            # 액션이 MIN_MAX 정규화되어 있어 그 밖은 정의상 무의미하기 때문.
            if self.clip_sample:
                x_0 = torch.clamp(x_0, -self.clip_sample_range, self.clip_sample_range)
        return x_0                                               # (B, 16, 7), t=1 지점

    def sample_noise(self, batch_size: int, device, generator: torch.Generator | None = None) -> torch.Tensor:
        """(B, 16, 7) 표준정규 노이즈. 학습(compute_loss)과 추론(sample) 양쪽에서 쓴다.

        generator를 주면 결과를 재현할 수 있다(평가 시 시드 고정용).
        """
        return torch.randn(batch_size, self.ac_chunk, self.ac_dim, device=device, generator=generator)


class DiTFlowMTPolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).

    (docstring은 diffusion policy에서 물려받은 것. 실제 구현은 확산이 아니라
     flow matching이고, 백본에 CLIP 언어 인코더가 추가된 멀티태스크 버전이다.)

    이 클래스가 하는 일은 얇다: 정규화 <-> 큐 관리 <-> DiTFlowModel 위임.
    실제 신경망은 전부 self.dit_flow(DiTFlowModel) 안에 있다.
    """

    config_class = DiTFlowMTConfig
    name = "DiTFlowMT"

    def __init__(
        self,
        config: DiTFlowMTConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        # 이미지 또는 env_state 중 최소 하나가 있는지, 이미지들 shape이 같은지 검사.
        config.validate_features()
        self.config = config

        # 정규화 모듈 3종. dataset_stats(mean/std/min/max)를 buffer로 들고 있어
        # 체크포인트에 함께 저장된다 -> 배포 시 통계를 따로 챙길 필요가 없다.
        #   normalize_inputs   : 관측(STATE=MIN_MAX, VISUAL=MEAN_STD)
        #   normalize_targets  : 학습 타깃 action(MIN_MAX -> [-1,1])
        #   unnormalize_outputs: 추론 결과를 원래 단위로 되돌림
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        # 추론 전용 상태. 학습 경로(forward)에서는 전혀 쓰이지 않는다.
        self._queues = None

        self.dit_flow = DiTFlowModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        # 옵티마이저에 넘길 파라미터. dit_flow 전체를 주지만 백본은 requires_grad=False라
        # Adam이 실제로 업데이트하는 건 projection들과 velocity_net뿐이다.
        # (그래도 옵티마이저 state에는 잡히므로, 정말 빼고 싶으면 필터링이 필요하다.)
        return self.dit_flow.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`

        에피소드 경계에서 반드시 불러야 한다. 안 부르면 이전 에피소드의 관측/액션이
        남아 첫 스텝에 잘못된 조건이 들어간다.

        maxlen 덕분에 deque가 알아서 오래된 것부터 밀어낸다:
            관측 큐 : 최근 2스텝 유지
            액션 큐 : 최대 8개 (한 번 생성분)
        """
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        # 아래 두 if는 config에 해당 입력이 있을 때만 큐를 만든다.
        # LIBERO: image_features 있음(카메라 2대), env_state_feature 없음.
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict a chunk of actions given environment observations.

        select_action이 "큐가 비었을 때만" 호출한다. 여기서 실제 100스텝 적분이 돈다.

        shape:
            batch[k]는 (B, ...) 한 스텝짜리로 들어오지만, 아래 stack으로
            (B, n_obs_steps=2, ...) 형태로 바뀐다.
            반환: (B, n_action_steps=8, 7) — 원래 단위(unnormalize 완료)
        """
        # stack n latest observations from the queue
        # 큐에 쌓인 2스텝을 dim=1로 쌓아 시간축을 만든다.
        #   observation.state : deque[(B,8)] x2   -> (B, 2, 8)
        #   observation.images: deque[(B,2,3,H,W)] x2 -> (B, 2, 2, 3, H, W)
        # batch를 제자리에서 수정한다(호출자인 select_action의 batch도 바뀐다).
        # `key in self._queues` 조건 덕분에 task 문자열 같은 비-큐 키는 그대로 통과한다.
        for key in batch:
            if key in self._queues:
                batch[key] = torch.stack(list(self._queues[key]), dim=1)

        # batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.dit_flow.generate_actions(batch)    # (B, 8, 7), 정규화된 값

        # TODO(rcadene): make above methods return output dictionary?
        # [-1,1] -> 실제 로봇 액션 단위로 복원
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]

        return actions

    @torch.no_grad
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying flow model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The flow model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.

        추론 경로의 진입점. 환경이 매 스텝 호출하지만 매번 모델을 돌리지는 않는다.
        큐가 비었을 때만 8개를 새로 생성하고, 그다음 7번은 큐에서 꺼내 쓴다.
        즉 실제 신경망 추론은 8스텝(0.4초)에 한 번만 일어난다 -- receding horizon 방식.

        cf. forward()는 학습용, 이쪽은 배포/평가용이다. 둘은 _prepare_global_conditioning을
        공유하지만 진입점이 다르다.

        @torch.no_grad는 괄호 없이 붙어 있는데(보통 @torch.no_grad()), 최신 PyTorch는
        이 형태도 데코레이터로 받아 준다.

        shape: 입력 batch[k] = (B, ...) 한 스텝  ->  반환 (B, 7) 액션 하나
        """
        batch = self.normalize_inputs(batch)
        # 카메라별로 나뉜 키를 한 텐서로 합친다. dim=-4는 (C,H,W) 앞자리 = 카메라 축.
        #   (B, 3, H, W) x 2대 -> (B, 2, 3, H, W)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        # Note: It's important that this happens after stacking the images into a single key.
        # 큐에 밀어 넣는다. 큐가 비어 있으면(에피소드 첫 스텝) populate_queues가
        # 같은 관측을 maxlen만큼 복제해 채운다 -> 첫 스텝에도 (B,2,...)가 성립.
        self._queues = populate_queues(self._queues, batch)

        # ★ 이 if가 이 정책의 계산량을 결정한다.
        #   비어 있을 때(= 8스텝마다 한 번)만 100스텝 적분이 돈다.
        #   나머지 7스텝은 아래 popleft 한 줄로 끝난다.
        if len(self._queues["action"]) == 0:
            actions = self.predict_action_chunk(batch)        # (B, 8, 7)
            # deque에 시간축 기준으로 하나씩 넣기 위해 (8, B, 7)로 전치한 뒤 extend.
            # -> 큐 원소 하나가 (B, 7)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        # 가장 이른 시점 액션부터 꺼내 쓴다. 8번 꺼내면 큐가 비고 다시 위 if가 참이 된다.
        action = self._queues[ACTION].popleft()               # (B, 7)
        return action

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation.

        학습 경로의 진입점. train.py의 update_policy()가 매 스텝 이걸 호출한다.
        하는 일은 정규화 -> 이미지 합치기 -> 손실 계산 세 단계뿐이고,
        실제 계산은 전부 self.dit_flow(DiTFlowModel)에 위임한다.

        반환값의 두 번째 원소는 wandb에 추가로 남길 지표용 dict인데 여기서는 None이다.
        본인 방법론에서 추가 손실 항의 값을 로깅하고 싶다면 여기에 dict를 채우면 된다.

        입력 batch (데이터로더가 주는 것):
            observation.state          (B, 2, 8)
            observation.images.*       (B, 2, 3, 256, 256)  카메라 키마다 하나씩
            action                     (B, 16, 7)
            action_is_pad              (B, 16)  bool
            task                       list[str], 길이 B
        반환: (스칼라 loss, None)
        """
        # 입력 정규화. STATE는 MIN_MAX, VISUAL은 MEAN_STD(ImageNet 통계).
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            # 카메라별로 따로 있던 키를 하나로 합친다.
            # (B,2,3,256,256) x 2대 -> (B, 2, 2, 3, 256, 256) = (B, 관측스텝, 카메라, C, H, W)
            # dim=-4가 카메라 자리인 이유: 뒤에서 4번째가 (C,H,W) 바로 앞이기 때문.
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        # 타깃(action) 정규화. 데이터셋의 min/max로 [-1,1] 범위에 맞춘다.
        # select_action 쪽에는 이 줄이 없다(거기선 반대로 unnormalize를 한다).
        batch = self.normalize_targets(batch)
        loss = self.dit_flow.compute_loss(batch)
        return loss, None


class DiTFlowModel(nn.Module):
    """정책의 실제 신경망. 인코더 3종 + 투영 + velocity_net(DiT)을 들고 있다.

    __init__의 주된 일은 조건 벡터 차원(cond_dim)을 "누적해서" 계산하는 것이다.
    아래 분기들을 따라가면 최종 2576이 어떻게 나오는지 알 수 있다.
    """

    def __init__(self, config: DiTFlowMTConfig):
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        # ── 언어 (항상 존재) ──
        self.language_encoder = LanguageEncoder(config).to(self.config.device)
        # CLIP hidden(512) -> hidden_dim(512). 차원은 같지만 학습되는 어댑터 역할.
        self.language_embedding_projection = nn.Linear(self.language_encoder.hidden_size, config.hidden_dim)

        # 언어는 시간축이 없다(과제 문장은 에피소드 내내 고정). 그래서 아래에서
        # n_obs_steps를 곱하지 않고 따로 더해진다.
        language_cond_dim = config.hidden_dim                       # 512

        # ── 로봇 상태 ──
        # 파일 상단 USE_STATE_PROJ 스위치. 현재 False이므로 else 분기.
        if USE_STATE_PROJ:
            global_cond_dim = config.hidden_dim                     # 512 (투영 시)
            self.state_proj = nn.Linear(self.config.robot_state_feature.shape[0], config.hidden_dim)
        else:
            global_cond_dim = self.config.robot_state_feature.shape[0]   # 8 (원본 그대로)

        # ── 이미지 ──
        # LIBERO는 카메라 2대이므로 참. 이미지가 아예 없는 설정이면 건너뛴다.
        if self.config.image_features:
            self.pretrained_rgb_encoder = DINOv2Encoder(config)
            # DINOv2 hidden(768) -> hidden_dim(512)
            self.rgb_embedding_projection = nn.Linear(self.pretrained_rgb_encoder.hidden_size, config.hidden_dim)
            global_cond_dim += config.hidden_dim * len(self.config.image_features)   # 8 + 512*2 = 1032

        # ── 환경 상태 (LIBERO에는 없음 -> 이 분기는 실행되지 않는다) ──
        # PushT처럼 env_state를 주는 태스크에서만 더해진다.
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]


        # cond_dim 최종 계산:
        #   language_cond_dim + global_cond_dim * n_obs_steps
        # = 512 + 1032 * 2
        # = 512 + 2064 = 2576
        # global_cond_dim에 n_obs_steps를 곱하는 이유: 상태와 이미지는 관측 스텝마다
        # 하나씩 있고, _prepare_global_conditioning에서 전부 flatten해 이어 붙이기 때문.
        self.velocity_net = _DiTNoiseNet(
            ac_dim=config.action_feature.shape[0],          # 7
            ac_chunk=config.horizon,                        # 16
            cond_dim=language_cond_dim + global_cond_dim * config.n_obs_steps,   # 2576
            time_dim=config.frequency_embedding_dim,        # 256
            hidden_dim=config.hidden_dim,                   # 512
            num_blocks=config.num_blocks,                   # 6
            dropout=config.dropout,                         # 0.1
            dim_feedforward=config.dim_feedforward,         # 4096
            nhead=config.num_heads,                         # 16
            activation=config.activation,                   # "gelu"
            clip_sample=config.clip_sample,                 # True
            clip_sample_range=config.clip_sample_range,     # 1.0
        )

        # config가 None이면 100. 현재 config는 100을 명시하고 있다.
        self.num_inference_steps = config.num_inference_steps or 100
        self.training_noise_sampling = config.training_noise_sampling

        # ── 학습 시 flow 시간 t를 뽑을 분포 ──
        # config.__post_init__에서 "uniform"/"beta" 외의 값은 이미 거부되므로
        # 두 분기 중 하나는 반드시 실행된다(둘 다 안 걸려 noise_distribution이
        # 없는 상태로 남는 일은 생기지 않는다). 현재 설정은 "uniform".
        if config.training_noise_sampling == "uniform":
            # U(0,1). 경로의 모든 구간을 균등하게 학습한다.
            self.noise_distribution = torch.distributions.Uniform(
                low=0,
                high=1,
            )
        elif config.training_noise_sampling == "beta":
            # From the Pi0 paper, https://www.physicalintelligence.company/download/pi0.pdf Appendix B.
            # There, they say the PDF for the distribution they use is the following:
            # $p(t) = Beta((s-t) / s; 1.5, 1)$
            # So, we first figure out the distribution over $t'$ and then transform it to $t = s - s * t'$.
            #
            # 노이즈 쪽(t가 작은 구간)에 표본을 더 많이 배정하는 분포. 그쪽이 더 어렵기
            # 때문. Beta(1.5,1)은 1 근처에 몰리는데, 아핀 변환 t = s - s*t'로 뒤집어
            # 0 근처에 몰리게 만든다.
            s = 0.999  # constant from the paper
            beta_dist = torch.distributions.Beta(
                concentration1=1.5,  # alpha
                concentration0=1.0,  # beta
            )
            affine_transform = torch.distributions.transforms.AffineTransform(loc=s, scale=-s)
            self.noise_distribution = torch.distributions.TransformedDistribution(
                beta_dist, [affine_transform]
            )

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """velocity_net.sample()을 감싸는 얇은 래퍼. device/dtype 정리만 한다.

        shape: global_cond (B, 2576) -> return (B, 16, 7)
        """
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Expand global conditioning to the batch size.
        # generate_actions에서 오는 경우 global_cond는 이미 (batch_size, 2576)이라
        # 이 expand는 사실상 no-op이다. 조건 하나로 여러 샘플을 뽑고 싶을 때
        # (B=1 -> N개 후보) 의미가 생기는 코드.
        if global_cond is not None:
            global_cond = global_cond.expand(batch_size, -1).to(device=device, dtype=dtype)

        # Sample prior.
        # 실제 노이즈 샘플링은 velocity_net.sample() 안에서 일어난다
        # (주석 "Sample prior"와 달리 여기서 직접 뽑지는 않는다).
        sample = self.velocity_net.sample(
            global_cond, timesteps=self.num_inference_steps, generator=generator
        )
        return sample

    def _prepare_global_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode image features and concatenate them all together along with the state vector.

        이 모델의 심장. 언어 + 상태 + 시각을 하나의 조건 벡터로 합친다.
        학습(compute_loss)과 추론(generate_actions) 양쪽이 모두 여기를 거친다.

            언어 (CLIP)     -> (B, 512)              512
            상태            -> (B, 2 x 8)             16    <- flatten, 투영 없음
            이미지 (DINOv2) -> (B, 2 x 2 x 512)     2048    <- 2스텝 x 카메라 2대
                                            concat ------
                                                        2576

        이 2576이 velocity_net.cond_proj의 입력 크기이고, CLARE 어댑터가 붙는 유일한
        지점이며, CLARE 디스크리미네이터(오토인코더)가 감시하는 벡터이기도 하다.

        주의 1: 백본 두 개는 torch.no_grad()로 감싸 학습되지 않는다. 학습되는 건
                투영 레이어들과 velocity_net뿐이다.
        주의 2: batch[OBS_ROBOT]는 "observation.state"만 가리킨다. 데이터셋에 함께
                실려온 observation.state.joint(7차원)는 여기서 아예 참조되지 않는다.
                차원 계산이 2576으로 정확히 맞는 것이 그 증거다.
        주의 3: concat 순서가 [언어, 상태, 이미지]로 고정이다. cond_proj의 가중치가
                이 순서를 전제로 학습되므로, 순서를 바꾸면 기존 체크포인트가 깨진다.
        """
        # (B, 2, 8) 에서 앞 두 축만 꺼낸다
        batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:2]

        # encode text description
        # 동결된 CLIP이지만 no_grad로 한 번 더 막는다(캐시가 있어 보통 즉시 반환).
        with torch.no_grad():
            language_embedding = self.language_encoder(batch["task"])   # (B, 512)
        # CLIP hidden(512) -> hidden_dim(512). 여기는 학습된다.
        language_cond_feats = self.language_embedding_projection(language_embedding)   # (B, 512)
        # language embedding as the first token
        # 이 리스트에 차례로 append한 뒤 마지막에 dim=-1로 concat한다.
        # 따라서 append 순서 = 조건 벡터의 배치 순서.
        global_cond_feats = [language_cond_feats]

        # 상태. USE_STATE_PROJ=False(파일 상단)이므로 아래 else 분기가 실행된다.
        # 즉 8차원 상태는 투영 없이 그대로 flatten되어 (B, 2*8=16)으로 들어간다.
        # 2576 = 512 + 16 + 2048 이 성립하는 이유가 이것이다.
        if USE_STATE_PROJ:
            # (B, 2, 8) -> (2B, 8) 로 펴서 Linear 한 번에 통과시키고 다시 복원.
            # 주의: 결과가 (B, 2, 512) 3D인 채로 append된다. 다른 항목들은 2D라서
            #       마지막 torch.cat(dim=-1)에서 차원 불일치로 터진다.
            #       이 분기를 살리려면 아래 append를 flatten(start_dim=1)로 바꿔야 한다.
            states = einops.rearrange(batch[OBS_ROBOT], "b s ... -> (b s) ...", b=batch_size, s=n_obs_steps)
            states_embedding = self.state_proj(states)
            states_feature = einops.rearrange(states_embedding, "(b s) ... -> b s ...", b=batch_size, s=n_obs_steps)
            global_cond_feats.append(states_feature)
        else:
            # (B, 2, 8) -> (B, 16). 시간 순서가 앞쪽 8개=t-1, 뒤쪽 8개=t 로 보존된다.
            global_cond_feats.append(batch[OBS_ROBOT].flatten(start_dim=1))   # (B, 16)

        # Extract image features.
        # LIBERO에서는 항상 참. 이미지가 없는 설정에서만 건너뛴다.
        if self.config.image_features:
            # 배치/시간/카메라를 하나의 축으로 합쳐 한 번에 인코딩한다.
            # (B, 2, 2, 3, 256, 256) -> (B*2*2, 3, 256, 256) = 배치당 이미지 4B장
            # einops 패턴 "b s n ... -> (b s n) ..." 에서 ...는 (C,H,W)를 그대로 통과시킨다.
            images = einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...", b=batch_size, s=n_obs_steps, n=len(self.config.image_features))
            with torch.no_grad():
                img_cls_tokens = self.pretrained_rgb_encoder(images)      # (B*4, 768)
            img_embeddings = self.rgb_embedding_projection(img_cls_tokens)  # (B*4, 512), 학습됨
            # 다시 펼쳐서 (B, 2, 2*512) -> flatten -> (B, 2048)
            # "(b s n) ... -> b s (n ...)" : 카메라 축 n을 특징 축과 합쳐 (B, 2, 1024)
            img_features = einops.rearrange(
                img_embeddings, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps, n=len(self.config.image_features)
            )
            global_cond_feats.append(img_features.flatten(start_dim=1))   # (B, 2048)

        # LIBERO에서는 env_state_feature가 None이라 이 분기는 실행되지 않는다.
        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE].flatten(start_dim=1))

        # Concatenate features then flatten to (B, global_cond_dim).
        # [ (B,512), (B,16), (B,2048) ] -> (B, 2576)
        return torch.cat(global_cond_feats, dim=-1)

    def generate_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }

        추론 본체. 조건 인코딩 -> 100스텝 적분 -> 슬라이싱 3단계.
        (여기 들어오는 batch에는 "task" 문자열 리스트도 있어야 한다 —
         docstring에는 안 적혀 있지만 _prepare_global_conditioning이 요구한다.)
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        # 큐가 제대로 채워졌는지 확인하는 방어선. reset()을 빼먹어 큐 길이가
        # 어긋나면 여기서 걸린다.
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        # 노이즈에서 시작해 num_inference_steps(=100)번 적분하여 (B, 16, 7)을 만든다.
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        # 16개 중 8개만 취하고 나머지는 버린다(temporal ensemble 아님, 그냥 폐기).
        #
        # start=1인 이유가 중요하다. action_delta_indices가 [-1, 0, ..., 14]이므로
        # 배열 index 0은 "현재"가 아니라 t-1(이미 지나간 시점)이다. index 1이 "지금"이다.
        #     index 0 -> t-1  (버림)
        #     index 1~8 -> t ~ t+7  (실행, 0.4초 분량)
        #     index 9~15 -> t+8 ~ t+14  (미리보기, 버림)
        # 여기서 [0:8]을 쓰면 로봇이 한 스텝(0.05초)씩 뒤처진다.
        start = n_obs_steps - 1                      # = 1
        end = start + self.config.n_action_steps     # = 9
        actions = actions[:, start:end]              # (B, 8, 7)

        return actions

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }

        학습 본체. 반환은 스칼라 하나.
        """
        # Input validation.
        # 필수 키 존재 확인. issuperset이므로 추가 키(task 등)가 있어도 통과한다.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        # 이미지 또는 env_state 중 최소 하나.
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        # 데이터로더의 delta_timestamps 설정과 config가 어긋나면 여기서 잡힌다.
        # (config의 observation_delta_indices / action_delta_indices가 원천.)
        assert horizon == self.config.horizon              # 16
        assert n_obs_steps == self.config.n_obs_steps      # 2

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim) = (B, 2576)

        # ── Flow matching (rectified flow) ──────────────────────────────────
        # 확산모델과 달리 노이즈에서 데이터로 가는 "직선 경로"를 학습한다.
        #   경로:   x(t) = (1-t)*noise + t*action        t=0이면 순수 노이즈, t=1이면 정답
        #   속도:   dx/dt = action - noise               (t와 무관한 상수 = 직선)
        # 네트워크는 임의의 t 지점에서 이 속도를 맞히도록 학습된다.
        trajectory = batch["action"]                            # (B, 16, 7)
        # Sample noise to add to the trajectory.
        noise = self.velocity_net.sample_noise(trajectory.shape[0], trajectory.device)   # (B,16,7)
        # Sample a random noising timestep for each item in the batch.
        # 샘플마다 다른 t를 뽑는다(uniform이면 U(0,1)). 배치 하나로 경로 전 구간을 커버.
        # sample((B,))라 shape는 (B,) — 액션 스텝이나 차원마다 다른 t를 쓰지는 않는다.
        timesteps = self.noise_distribution.sample((trajectory.shape[0],)).to(trajectory.device)
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        # [:, None, None]로 (B,) -> (B,1,1) 브로드캐스트: 한 샘플의 16x7 전체에 같은 t 적용.
        noisy_trajectory = (1 - timesteps[:, None, None]) * noise + timesteps[:, None, None] * trajectory

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        # 여기가 유일하게 그래디언트가 흐르는 forward다(백본 둘은 no_grad 안에 있었다).
        pred = self.velocity_net(noisy_actions=noisy_trajectory, time=timesteps, global_cond=global_cond)
        target = trajectory - noise     # 정답 속도. 이름과 달리 "노이즈 예측"이 아니다.
        # reduction="none"인 이유: 아래 패딩 마스크를 원소별로 곱해야 하기 때문.
        loss = F.mse_loss(pred, target, reduction="none")   # (B, 16, 7)

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        # 주의: do_mask_loss_for_padding의 기본값이 False이고 배포 체크포인트도 False다.
        # 따라서 이 블록은 보통 실행되지 않으며, 에피소드 경계에서 복제된 가짜 액션도
        # 그대로 손실에 포함된다. drop_n_last_frames=7이 패딩을 2% 수준으로 억제하므로
        # 영향이 작다. action_is_pad는 배치에 실려오지만 이 설정에서는 미사용이다.
        if self.config.do_mask_loss_for_padding:
            # 위쪽 assert가 이미 action_is_pad 존재를 보장하지만, 이 메서드를 직접
            # 호출하는 경로를 대비한 방어 코드.
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]          # (B, 16) bool, 유효=True
            loss = loss * in_episode_bound.unsqueeze(-1)        # (B,16,1) 브로드캐스트

        # 주의: 마스킹을 켜도 나누는 값은 여전히 전체 원소 수(B*16*7)다.
        # 즉 masked mean이 아니라 "0을 포함한 평균"이라 패딩이 많을수록 손실이 작아진다.
        return loss.mean()
