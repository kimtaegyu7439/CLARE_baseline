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

"""R9 — 조건 라우팅은 네트워크 어디에서 고장나는가: attn/mlp 분리 방향-분리 지도.

R8의 자매 실험이다. 같은 프로브 지점, 같은 좌표계, 같은 체크포인트 집합을 쓰고
**측정 대상만** 다르다.

    R7  조건을 바꿔도 생성 경로가 겹친다              (입출력 수준)
    R8  조건 대비가 표현 수준에서 감쇠한다             (블록 출력 활성 h_ℓ의 크기)
    R9  그 붕괴가 조건 주입의 **방향** 수준에서 어떻게 무너지는가   ← 여기

무엇을 재는가 — 서브블록별 조건 기여 Δ
    DiT 블록 하나는 attention과 MLP 서브블록을 순차로 갖고, 각자 독립적인 AdaLN
    세트(scale γ, shift β, gate α)와 자기 residual 덧셈을 갖는다.

        h'    = h  + α_attn(c) ⊙ Attn( γ_attn(c) ⊙ LN(h)  + β_attn(c) )
        h_out = h' + α_mlp(c)  ⊙ MLP(  γ_mlp(c)  ⊙ LN(h') + β_mlp(c) )

    조건 c가 residual stream에 더하는 증분이 서브블록마다 하나씩, 블록당 두 개다.

        Δ_attn(ℓ, c) = α_attn(c) ⊙ Attn( AdaLN_attn(h; c) )
        Δ_mlp (ℓ, c) = α_mlp(c)  ⊙ MLP(  AdaLN_mlp(h'; c) )

    ★ Δ는 게이트 벡터가 아니라 **게이트가 곱해진 서브블록 출력 증분**이다. R8 패널 g의
      AdaLN 게이트 대비 G(ℓ)는 α(c) 자체를 재는 것이고 x에 의존하지 않는다. R9의 Δ는
      실제로 residual에 더해진 벡터라 x_t에 의존한다. 두 지표를 섞어 읽으면 안 된다.

방향 분리 점수 cos 와 기여 크기 M
        cos_sub(ℓ,t,p) = ⟨Δ_sub(ℓ,t,c₀,p), Δ_sub(ℓ,t,c₁,p)⟩ / (‖·‖‖·‖)
        M_sub  (ℓ,t,p) = ½( ‖Δ_sub(ℓ,t,c₀,p)‖ + ‖Δ_sub(ℓ,t,c₁,p)‖ )

    ★ 평균의 순서: cos을 프로브마다 먼저 계산하고 그 **스칼라**를 평균한다. Δ 벡터를
      먼저 평균한 뒤 각도를 재면 방향이 제각각인 벡터가 상쇄되어 왜곡된다.
      "각도 먼저, 평균 나중."
    ★ 부호를 유지한다. |cos|을 쓰지 않는다 — cos≈1(정렬=라우팅 없음), cos≈0(직교=라우팅),
      cos<0(반대 방향=강한 분리)은 서로 다른 사건이다.
    ★ cos만 보면 오독한다. cos은 크기 감쇠에 둔감해서, Δ가 거의 0으로 죽어도 남은
      수치 잔여의 방향이 갈라져 cos이 낮게 나올 수 있다. 그래서 M을 병기해 셋을 가른다.
          M 큼 + cos 낮음  살아있는 라우팅
          M 작음           조건 기여 소멸 (크기 붕괴)
          M 큼 + cos 높음  기여는 하나 방향이 정렬 (방향 붕괴)

Shuffle 기준선 (영점)
    "cos이 낮은 게 라우팅인가, 원래 이 아키텍처에서 낮은가"를 가르는 영점.
    조건을 c₀로 **고정**하고 서로 다른 프로브의 Δ끼리 각도를 잰다.

        cos_shuffle_sub(ℓ,t) = mean_p ⟨Δ_sub(ℓ,t,c₀,p), Δ_sub(ℓ,t,c₀,perm(p))⟩ / (‖·‖‖·‖)

    = "조건과 무관한 두 기여의 cos 기대값". 조건간 cos이 이 값보다 낮으면 방향 분리가
    조건에 의한 것이 된다. perm은 고정 시드의 완전순열(교란순열, 고정점 없음)이고,
    c₀·c₁ 양쪽에서 계산해 평균한다.

Δ를 어떻게 뽑는가 — 요청서 §2.2의 (A)와 (B)가 여기서는 같은 텐서다
    modeling_dit_flow_mt.py의 _DiTDecoder.forward는

        x  = x + self.attn_gate(self.dropout1(x2), cond)   # h'    = h  + Δ_attn
        return x + x3   (x3 = self.mlp_gate(...))          # h_out = h' + Δ_mlp

    이므로 **게이트 모듈의 출력이 곧 residual 증분**이다. 게이트가 서브블록 출력
    *뒤에* 곱해지는 구조라 (B) 직접 방식이 정확히 α⊙(서브블록 출력)을 준다. 그래서
    (B)로 잡되, (A) 차분 방식의 항등식

        h_out − h  ==  Δ_attn + Δ_mlp

    을 모델마다 첫 forward에서 assert로 확인한다(§5-1). 두 방식이 어긋날 수 없음을
    코드가 매번 증명하는 셈이다. eval 모드라 dropout1은 항등이다.

두 개의 "시간"을 혼동하지 말 것
    flow time t   생성 과정의 내부 시간. x_t = (1−t)·x₀ + t·a 의 t. 히트맵의 가로축.
    rollout step  로봇이 실제로 움직인 물리 시간. 조건 c를 만드는 관측 s가 언제 것인가.

    ★ 조건 라우팅이 에피소드 **초기 상태에서만** 무너지는지, 아니면 궤적 내내 무너져
      있는지는 서로 다른 주장이다. 그래서 관측을 t=0 한 점이 아니라 rollout_steps까지
      obs_stride 간격으로 떠서 (기본 0, 25, …, 200) 각 지점마다 조건을 새로 만든다.
      히트맵은 이 지점들의 평균이고, 물리시간 축 자체는 패널 m/n·o/p가 보여 준다.
      기본 구간을 200으로 잡은 이유는 아래 "롤아웃 길이" 항목에 있다.

    상태열은 **모델과 무관해야** 한다. 각 모델이 자기 롤아웃으로 만든 상태에서 재면
    "라우팅 차이"와 "상태 분포 차이"가 섞여 비교가 성립하지 않는다. 그래서 기본
    구동자(--obs_driver=demo)는 전문가 데모 액션 재생이다 — 모델에 의존하지 않고,
    a_tgt와 같은 출처라 프로브 설계와도 일관된다. 특정 정책으로 굴리고 싶으면
    --obs_driver=<모델 키>로 하나를 지정한다(그 하나가 세 모델 전부의 상태를 만든다).
    ★ 롤아웃 길이는 에피소드마다 다르다. 데모가 소진됐거나 에피소드가 끝난 뒤의 관측은
      "늦은 물리시간"이 아니라 정지한 장면이므로 **평균에서 뺀다**(--exclude_dead_obs).
      판정은 **짝** 기준이다 — c₀(task A)와 c₁(task B)의 에피소드 길이가 다르므로 한쪽만
      죽어도 그 (에피소드, 물리시간) 짝은 "정지한 장면 vs 움직이는 장면"의 대비가 되어
      조건 대비로 쓸 수 없다. 그래서 늦은 물리시간은 표본이 적어지고, 살아있는 짝이
      하나도 없는 지점은 NaN이 되어 그림에서 빠진다(그 스텝의 판은 아예 만들지 않는다).
      에피소드마다 얼마나 굴렀고 왜 멈췄는지는 패널 q(타임라인)와 콘솔 표에 나온다.
      LIBERO spatial 데모가 대략 150~330스텝이라 rollout_steps를 400으로 두면 후반부가
      통째로 비어 버린다. 그래서 기본값을 200/25로 잡았다 — 지점 수는 같고(9개) 전부
      궤적이 실제로 살아 있는 구간에 들어간다. 벤치마크가 다르면 콘솔의 "usable pairs
      per rollout step" 줄을 보고 맞춰라.

정의되지 않는 cos을 0으로 만들지 않는다
    ★ cos = 0은 "직교 = 라우팅이 가장 살아있음"이라는 강한 의미를 가진 값이다. 기여가
      소멸해 각도를 잴 수 없는 칸을 0으로 채우면 **결론이 정반대로 뒤집힌다**. AdaLN-Zero
      구조상 게이트가 죽으면 Δ가 정확히 0이 되어 norm_floor에 전부 걸릴 수 있으므로
      이건 가상의 위험이 아니다. 그래서 누적은 합과 **유효 표본 수**를 따로 들고 가고,
      유효 표본이 하나도 없는 칸은 NaN으로 남긴다. NaN 칸은 히트맵에서 회색으로,
      단면에서는 선이 끊긴 것으로 나타난다(cmap.set_bad). 평균도 표본 수로 나눈다 —
      에피소드 수로 나누면 유효하지 않은 칸이 값을 희석한다.

그림
    본론은 2×3 히트맵이다. 위 행 = attention 서브블록, 아래 행 = MLP 서브블록,
    열 = pretrained / joint / CL. 색은 cos이 **shuffle 기준선에서 얼마나 벗어났는가**
    (cos − cos_shuffle)이고 흰색이 곧 shuffle 수준이다. 세 모델이 같은 색 스케일을 쓴다.
      파랑 = shuffle보다 직교 (라우팅 있음) · 흰색 = shuffle 수준 (라우팅 없음)
      빨강 = shuffle보다 정렬
    각 칸의 불투명도는 M에 비례한다 — 흐린 칸은 "여기 방향은 무의미"라는 뜻이다.
    이렇게 하면 크기 붕괴(흐려짐)와 방향 붕괴(색이 흰/빨강으로)가 한 그림에서 갈린다.
    흐림과 흰색이 헷갈릴 수 있는데, 그 둘을 최종 판정하는 것이 크기 단면(패널 k/l)이다.

    그림은 두 종류로 나간다.
        R9_full.png            요약. 히트맵이 물리시간을 평균하고, 물리시간 축은
                               패널 m/n(cos)·o/p(‖Δ‖)가 따로 보여 준다.
        R9_full_step0000.png   rollout step 0의 판. 그 스텝 하나만의 2×3 히트맵 + 단면.
        R9_full_step0025.png   … obs_stride 간격으로 한 장씩 (기본 9장).
                               살아있는 짝이 없는 스텝은 판을 만들지 않는다.
    ★ 스텝별 판은 색 스케일·알파 정규화 기준·routing 판정 규칙을 요약본과 **공유**한다
      (전체 (S,L,T)에서 한 번만 정한다). 판마다 자기 범위로 정규화하면 9장을 나란히
      놓고 비교할 수 없어 스텝별로 그리는 의미 자체가 사라진다.

주장 범위
    이 실험은 "sequential 학습에서 조건 라우팅의 방향 분리가 붕괴한다"까지만 말한다.
    "그래서 망각한다"는 이 그림의 주장이 아니다(별도 실험).

이 스크립트는 학습을 하지 않는다. R7/R8이 쓴 통제군 체크포인트를 그대로 읽는다.

사용 예 (인자는 bash/E0/R9.sh 가 채운다 — R8.sh와 같은 값을 쓰는지 거기서 확인해라)
    bash bash/E0/R9.sh
    PLOT_ONLY=1 bash bash/E0/R9.sh          # 캐시로 그림만 다시
    REDO=1 bash bash/E0/R9.sh               # 캐시 무시하고 다시 잰다
    OBS_STRIDE=50 bash bash/E0/R9.sh        # 물리시간 지점을 5개로 줄여 빠르게

직접 부를 때
    python R9.py --policy.path=<any ckpt> --ckpt_root=./outputs/E0/libero_spatial/seed_42/lam0 \
        --pretrain_ckpt=... --joint_ckpt=... --run_tag=libero_spatial_seed42_lam0_task0v1
    python R9.py --plot_only --run_dir=./outputs/R9/libero_spatial_seed42_lam0_task0v1
"""

import hashlib
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from termcolor import colored

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.utils import preprocess_observation
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

# ★ R8의 코드베이스를 그대로 계승한다. 프로브 고정물·조건 구성·모델 표를 두 번 구현하면
#   "같은 s⁰·같은 좌표계·같은 프로브"라는 보장이 깨진다.
from lerobot.scripts.R8 import (
    DIV_HI,
    DIV_LO,
    DIV_MID,
    GRID,
    INK,
    INK2,
    MODEL_COLORS,
    R8Config,
    _style,
    model_specs,
)
from lerobot.scripts.R7 import (
    assert_shared_norm,
    capture_obs,
    demo_chunks,
    exec_range,
    load_policy_at,
    minmax_normalize,
    norm_stats,
    obs_to_cond,
    task_text,
)

SUBS = ("attn", "mlp")
SUB_TITLE = {"attn": "attention sub-block", "mlp": "MLP sub-block"}
# 최종 출력단(_FinalLayer)의 두 지점. 블록 축이 없어 따로 들고 간다.
FINALS = ("512", "7")
FINAL_TITLE = {"512": "final AdaLN increment  (512-d)",
               "7": "after the 512→7 readout  (7-d)"}


# ═════════════════════════════════════════════════════════════════════════════
#  설정
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class R9Config(R8Config):
    """R8Config를 그대로 상속한다 — 프로브 규약(probe_seed, num_probe, t_grid,
    demo_episodes, num_obs, exec_slice, cond_mode)이 R8과 한 글자도 다르면 안 되기 때문.

    상속했지만 R9가 **쓰지 않는** 항목: num_pairs, den_floor_ratio.
    R9의 분모는 "노이즈가 만드는 차이"가 아니라 벡터 자신의 노름이라 노이즈 짝이 필요 없다.
    """

    out_root: str = "outputs/R9"

    # ── 물리시간 축 (R8에는 없다: R8은 초기 관측 한 점만 본다) ───────────────
    # 조건 c를 만드는 관측 s를 롤아웃 스텝 0, obs_stride, 2·obs_stride, ..., rollout_steps
    # 에서 뜬다. num_obs는 이제 "초기 상태(에피소드) 개수"이고, 실제 관측 수는
    # num_obs × len(steps)가 된다 — forward 비용이 그만큼 늘어난다(로그에 찍힌다).
    # ★ 기본값은 LIBERO spatial 데모 길이(대략 150~330스텝)에 맞춘 것이다. 400까지 떠 봐야
    #   후반 지점은 짝이 하나도 안 남아 판이 만들어지지 않는다. 궤적의 실제 구간을 촘촘히
    #   보는 쪽이 낫다: 0, 25, …, 200 (9지점, 지점 수는 같다).
    rollout_steps: int = 200
    obs_stride: int = 25
    # 상태열을 누가 만드는가. 세 모델이 **같은 s**를 봐야 비교가 성립하므로 구동자는
    # 하나뿐이다.
    #   "demo"        전문가 데모 액션 재생 (모델 독립. 기본값)
    #   <모델 키>     그 정책 하나로 굴린다 (예: "joint", "cl"). 상태가 그 모델에 유리하게
    #                 편향되지만 세 모델이 보는 s는 여전히 동일하다.
    obs_driver: str = "demo"
    # ★ 롤아웃 길이는 에피소드마다 다르다. 데모가 소진됐거나(HELD) 에피소드가 끝난(FROZEN)
    #   뒤의 관측은 "늦은 물리시간"이 아니라 정지한 장면이므로 평균에서 뺀다.
    #   판정은 **짝** 기준이다 — task A와 task B의 에피소드 길이가 다르므로, 한쪽만 죽어도
    #   그 (에피소드, 물리시간) 짝은 조건 대비로 쓸 수 없다.
    #   False로 두면 예전처럼 전부 평균에 넣는다(정지 장면 포함).
    exclude_dead_obs: bool = True

    # ── §2.4 shuffle 기준선 ──────────────────────────────────────────────────
    shuffle_seed: int = 20260901        # 프로브 인덱스 교란순열 시드 (고정)

    # ── §5-6 크기 0 방어 ─────────────────────────────────────────────────────
    # ‖Δ‖가 이 값 미만인 (토큰, 프로브)는 cos이 정의되지 않는다 -> cos 평균에서 제외하고
    # 제외 비율을 로그에 남긴다. M에는 그대로 포함한다(0도 크기 정보다).
    # ★ 절대 임계값인 것이 의도다. AdaLN-Zero라 게이트가 완전히 죽으면 Δ가 정확히 0이
    #   되는데, 그 경우를 "직교"로 세면 라우팅이 살아있다는 정반대 결론이 나온다.
    norm_floor: float = 1e-8

    # ── 캡션의 "라우팅 블록 수"를 세는 규칙 (그림에 명시된다) ────────────────
    # 손으로 쓰지 않고 데이터에서 센다. 두 조건을 **모두** 만족해야 routing으로 센다:
    #   (1) t 평균한 (cos − shuffle) 이 −route_gap_thresh 보다 낮다   (방향이 갈렸다)
    #   (2) t 평균한 M 이 전 모델 공통 최대의 route_mag_frac 이상이다  (기여가 살아 있다)
    route_gap_thresh: float = 0.05
    route_mag_frac: float = 0.05

    # ── 그림 ────────────────────────────────────────────────────────────────
    alpha_floor: float = 0.06           # M=0인 칸도 격자 위치는 보이게 남기는 최소 불투명도
    # rollout step마다 판을 한 장씩 더 낸다 (R9_full_step0000.png ...). 색 스케일과
    # 알파 기준은 요약본과 공유하므로 나란히 놓고 비교할 수 있다.
    per_step_figs: bool = True

    # ── §5-2 결정론 ─────────────────────────────────────────────────────────
    # 첫 (oi, t) 격자점에서 같은 forward를 두 번 돌려 Δ가 bit-wise 같은지 확인한다.
    # 전체 2회 실행 비교는 --recompute 로 두 번 돌려 npz를 비교하면 된다(문서에 기록).
    determinism_check: bool = True


def cache_name(cfg: R9Config) -> str:
    """full 모드는 장면이 조건의 일부라 obs_task라는 개념이 없다 (R8과 같은 규약)."""
    return "R9_full.npz" if cfg.cond_mode == "full" else f"R9_lang_obs{cfg.obs_task}.npz"


# ═════════════════════════════════════════════════════════════════════════════
#  서브블록 증분 뽑기
# ═════════════════════════════════════════════════════════════════════════════
class SubBlockTap:
    """블록별 서브블록 residual 증분 Δ_attn, Δ_mlp를 forward hook으로 가로챈다.

    R8의 LayerTap은 블록 **출력**만 잡는다. R9는 residual 덧셈의 증분이 필요하므로
    hook을 서브블록 단위로 내린다.

    걸리는 지점 (블록마다 3개):
        layer.attn_gate  forward hook  -> Δ_attn = α_attn(c) ⊙ Attn(AdaLN_attn(h; c))
        layer.mlp_gate   forward hook  -> Δ_mlp  = α_mlp(c)  ⊙ MLP(AdaLN_mlp(h'; c))
        layer            forward hook  -> (h, h_out)   ← 차분 항등식 검증용(check=True일 때만)

    게이트가 서브블록 출력 **뒤에** 곱해지는 구조(_DiTDecoder.forward 참조)라 게이트
    출력이 곧 residual 증분이다. 그래서 요청서 §2.2의 (A) 차분과 (B) 직접이 같은 텐서가
    되고, verify_residual()이 h_out − h == Δ_attn + Δ_mlp 를 매 모델 첫 forward에서
    확인한다. forward를 손으로 다시 구현하지 않는 이유는 R8과 같다 — 모델이 바뀌면
    조용히 어긋나기 때문이다.

    Δ shape: (T=16, B=num_probe, H=512).
    """

    def __init__(self, net):
        self.d: dict[tuple[int, str], torch.Tensor] = {}
        self.blk: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.fin: dict[str, torch.Tensor] = {}
        self.check = False
        self.handles = []
        self.layers = list(net.decoder.layers)
        for li, layer in enumerate(self.layers):
            self.handles.append(layer.attn_gate.register_forward_hook(self._mk(li, "attn")))
            self.handles.append(layer.mlp_gate.register_forward_hook(self._mk(li, "mlp")))
            self.handles.append(layer.register_forward_hook(self._mk_blk(li)))
        # ── 최종 출력단 (_FinalLayer). 블록과 달리 residual 덧셈이 아니라 x를 통째로
        #    변조하지만, 증분은 똑같이 정의된다:
        #        modulate(x, shift, scale) − x  =  x⊙scale(c) + shift(c)
        #    이게 조건이 **마지막 표현**에 더하는 벡터다. 그리고 그 뒤 linear가 512→7로
        #    내리므로, 505차원이 버려진다 — 방향이 갈려 있어도 사영에서 죽을 수 있다.
        self.eps_out = net.eps_out
        self.W = net.eps_out.linear.weight            # (7, 512)
        self.handles.append(net.eps_out.register_forward_pre_hook(self._fin_pre))
        self.handles.append(net.eps_out.register_forward_hook(self._fin_post))
        self.handles.append(
            net.eps_out.adaLN_modulation.register_forward_hook(self._fin_mod))

    @property
    def n_blocks(self) -> int:
        return len(self.layers)

    def _mk(self, li: int, sub: str):
        def fn(_mod, _args, out):
            self.d[(li, sub)] = out.detach()
        return fn

    def _mk_blk(self, li: int):
        def fn(_mod, args, out):
            if self.check:
                self.blk[li] = (args[0].detach(), out.detach())
        return fn

    def _fin_pre(self, _mod, args):
        self.fin["x"] = args[0].detach()                      # (T, B, H) 최종층 입력

    def _fin_mod(self, _mod, _args, out):
        shift, scale = out.detach().chunk(2, dim=1)           # 각 (B, H)
        self.fin["shift"], self.fin["scale"] = shift, scale

    def _fin_post(self, _mod, _args, out):
        self.fin["v"] = out.detach()                          # (T, B, 7) 속도

    def final_delta(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(Δ_512, Δ_7, AdaLN 변조 벡터).

        Δ_512 = x⊙scale + shift   조건이 최종 표현에 더하는 증분 (블록의 Δ와 같은 의미)
        Δ_7   = W·Δ_512           그 증분이 액션 공간까지 살아남은 부분
                                  (bias는 증분의 차이에서 소거되므로 weight만 곱한다)
        mod   = [shift | scale]   x와 무관한 순수 주입 벡터. 주입 자체가 조건을
                                  구분하는지 보는 용도(프로브 축이 없어 shuffle은 못 잰다).
        """
        x, shift, scale = self.fin["x"], self.fin["shift"], self.fin["scale"]
        d512 = x * scale.unsqueeze(0) + shift.unsqueeze(0)                   # (T, B, H)
        d7 = torch.einsum("tbh,oh->tbo", d512, self.W)                       # (T, B, 7)
        return d512, d7, torch.cat([shift, scale], dim=-1)                   # (B, 2H)

    def verify_final(self, tol: float = 1e-3) -> float:
        """§5-1의 최종층 판. v == linear(x) + W·Δ_512 인지 확인한다.

        블록의 residual 항등식과 같은 역할이다 — Δ_512를 손으로 재구성한 것이 실제
        forward와 일치함을 매번 증명한다. modulate 정의가 바뀌면 여기서 걸린다.
        """
        d512, d7, _ = self.final_delta()
        lhs = self.fin["v"].double()
        rhs = (self.eps_out.linear(self.fin["x"]) + d7).double()
        scale = float(lhs.abs().max()) + 1e-12
        err = float((lhs - rhs).abs().max()) / scale
        if err > tol:
            raise AssertionError(
                f"[R9] 최종층 항등식이 깨졌다: max rel err {err:.2e} > {tol:.0e}\n"
                f"  v ≠ linear(x) + W·(x⊙scale + shift) 라면 _FinalLayer.forward가 "
                f"바뀐 것이다.")
        return err

    def snapshot(self) -> dict[tuple[int, str], torch.Tensor]:
        """지금 버퍼에 있는 Δ를 얕게 복사해 넘긴다. 다음 forward가 덮어쓰기 전에 떠 둔다."""
        return dict(self.d)

    def verify_residual(self, tol: float = 1e-4) -> float:
        """§5-1. (A) 차분 == (B) 직접 을 확인한다. 반환은 관측된 최대 상대 오차."""
        if not self.blk:
            raise RuntimeError("check=True 로 forward를 한 번 돌린 뒤에 호출해야 한다")
        worst = 0.0
        for li in range(self.n_blocks):
            h, h_out = self.blk[li]
            lhs = (h_out - h).double()
            rhs = (self.d[(li, "attn")] + self.d[(li, "mlp")]).double()
            scale = float(lhs.abs().max()) + 1e-12
            worst = max(worst, float((lhs - rhs).abs().max()) / scale)
        if worst > tol:
            raise AssertionError(
                f"[R9] residual 항등식이 깨졌다: max rel err {worst:.2e} > {tol:.0e}\n"
                f"  h_out − h ≠ Δ_attn + Δ_mlp 라면 hook 지점이 틀린 것이다. "
                f"_DiTDecoder.forward 가 바뀌지 않았는지 확인해라.")
        return worst

    def remove(self):
        for handle in self.handles:
            handle.remove()


def derangement(n: int, seed: int) -> np.ndarray:
    """고정점이 없는 순열. §5-3 — 자기 자신과 짝지으면 cos이 항등적으로 1이 된다."""
    if n < 2:
        raise SystemExit(f"[R9] shuffle 기준선에는 프로브가 2개 이상 필요하다 (num_probe={n})")
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    return (np.arange(n) + 1) % n          # 도달할 일이 사실상 없는 안전망


def pair_stats(d0: torch.Tensor, d1: torch.Tensor, tok: slice, floor: float
               ) -> tuple[float, float, int, int]:
    """(T,B,H) 두 개 -> (cos 평균, M 평균, 유효 칸 수, 전체 칸 수).

    ★ 요청서 §2.3의 "각도 먼저, 평균 나중"을 여기서 지킨다. (토큰, 프로브)마다 cos을
      스칼라로 접은 뒤 그 스칼라들을 평균한다. 벡터를 먼저 평균하지 않는다.
    ★ tok은 실행 구간 토큰만 남기는 슬라이스다. 로봇에 나가지 않는 청크 스텝의 토큰까지
      넣으면 "하지도 않는 행동"으로 라우팅을 판정하게 된다 (R7/R8과 같은 규약).
    """
    a = d0[tok].double()
    b = d1[tok].double()
    n0 = a.norm(dim=-1)                                   # (Tk, B)
    n1 = b.norm(dim=-1)
    ok = (n0 > floor) & (n1 > floor)
    cos = (a * b).sum(dim=-1) / (n0 * n1).clamp_min(1e-300)
    n_ok, n_all = int(ok.sum()), int(ok.numel())
    cos_mean = float(cos[ok].mean()) if n_ok else float("nan")
    return cos_mean, float((0.5 * (n0 + n1)).mean()), n_ok, n_all


# ═════════════════════════════════════════════════════════════════════════════
#  물리시간 축 — 롤아웃 도중의 관측을 뜬다
# ═════════════════════════════════════════════════════════════════════════════
# 캡처 지점의 상태. 그림의 회색 띠와 summary.json이 이걸 쓴다.
ALIVE, HELD, FROZEN = 0, 1, 2


def make_task_env(cfg: R9Config, task: int):
    """task 하나의 gym_libero 환경. R3.make_probe_env를 task 인자로 일반화한 것.

    초기 상태를 인덱스로 **직접 지정**해야 짝지은 비교가 되는데, LiberoEnv.reset은
    클래스 변수 카운터로 init_state를 고른다. 그래서 벡터 래퍼 없이 단일 env를 쓴다.
    """
    import importlib

    import gymnasium as gym

    if cfg.env is None:
        raise SystemExit("--env.type=libero --env.benchmark=libero_spatial 가 필요하다.")
    importlib.import_module("gym_libero")
    handle = f"gym_libero/{cfg.env_task_prefix}{task}"
    kwargs = dict(cfg.env.gym_kwargs)
    # 정착 스텝과 캡처 구간이 TimeLimit 예산을 갉아먹지 않게 한도를 늘려 준다.
    kwargs["max_episode_steps"] = cfg.rollout_steps + cfg.settle_steps + 2
    return gym.make(handle, disable_env_checker=True, **kwargs)


def demo_action_seqs(cfg: R9Config, task: int, n_ep: int) -> list[np.ndarray]:
    """태스크의 전문가 데모 액션열. [(T_e, 7)] × n_ep.

    데이터셋의 action은 롤아웃이 env.step에 넣는 것과 **같은 공간**이다(R3가 정책의
    grip_cmd와 데모 action의 마지막 차원을 그대로 비교하는 근거와 같다). 그래서
    정규화 없이 그대로 재생할 수 있다. 이미지 컬럼을 건드리면 수천 장을 디코딩하므로
    select_columns로 잘라 읽는다.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f"{cfg.dataset_prefix}{task}")
    sub = ds.hf_dataset.select_columns(["action", "episode_index"])
    act = np.asarray(sub["action"], dtype=np.float32)
    ep = np.asarray(sub["episode_index"])
    n = min(n_ep, int(ep.max()) + 1)
    seqs = [act[ep == e] for e in range(n)]
    del ds, sub
    return seqs


def _clone_obs(obs: dict) -> dict:
    """env가 준 관측을 R7.capture_obs와 **같은 형태로** 떠 둔다.

    ★ env의 원시 obs를 통째로 복사하면 안 된다. R7.capture_obs는 pixels와 agent_pos만
      남기는데(joint_state·task는 버린다), preprocess_observation이 중첩 dict를 기대하고
      obs_to_cond가 그 위에서 돈다. 형태가 다르면 조건 벡터가 조용히 달라진다 —
      _assert_obs_compatible이 잡아내는 것이 정확히 이 차이다.
    """
    return {
        "pixels": {k: np.array(v) for k, v in obs["pixels"].items()},
        "agent_pos": np.array(obs["agent_pos"]),
    }


def _assert_obs_compatible(mine, theirs) -> None:
    """R7.capture_obs가 주는 것과 같은 모양인지 확인한다.

    obs_to_cond가 어떤 형태를 받는지는 R7이 정한다. 여기서 직접 만든 관측이 그 규약과
    어긋나면 조용히 이상한 조건 벡터가 나오는 대신 여기서 크게 실패해야 한다.
    """
    if type(mine) is not type(theirs):
        raise AssertionError(
            f"[R9] 관측 형식이 R7.capture_obs와 다르다: {type(mine)} vs {type(theirs)}\n"
            f"  capture_obs_traj의 _clone_obs 반환 형태를 R7에 맞춰라.")
    if isinstance(mine, dict):
        a, b = set(mine), set(theirs)
        if a != b:
            raise AssertionError(
                f"[R9] 관측 키가 R7.capture_obs와 다르다.\n"
                f"  여기만 있는 키: {sorted(a - b)}\n  R7에만 있는 키: {sorted(b - a)}")


@torch.no_grad()
def capture_obs_traj(cfg: R9Config, task: int, n_ep: int, steps: np.ndarray,
                     driver=None, device=None) -> tuple[list[list[dict]], np.ndarray, list[dict]]:
    """롤아웃 스텝 steps에서의 관측을 에피소드마다 뜬다.

    반환: (obs[ep][si] 리스트, status (n_ep, len(steps)) int8, 에피소드별 길이 정보)

    ★ 세 모델이 **같은 s**를 봐야 하므로 구동자는 하나뿐이다. driver=None이면 전문가
      데모 액션을 재생한다(모델 독립). driver에 정책을 주면 그 정책 하나가 세 모델
      전부의 상태를 만든다.
    ★ 데모가 끝났거나(HELD) 에피소드가 종료된(FROZEN) 뒤의 캡처 지점은 "늦은 물리시간"이
      아니라 그냥 정지한 장면이다. 상태 배열에 기록해 평균에서 빼고(--exclude_dead_obs),
      에피소드마다 "언제 끝났는가"를 info로 돌려준다.
    """
    env = make_task_env(cfg, task)
    init_states = env.unwrapped._init_states
    task_desc = env.unwrapped.task_description
    demos = None if driver is not None else demo_action_seqs(cfg, task, n_ep)
    if demos is not None and len(demos) < n_ep:
        raise SystemExit(
            f"[R9] task {task}의 데모가 {len(demos)}개뿐인데 num_obs={n_ep}을 요구했다.\n"
            f"  --num_obs 를 {len(demos)} 이하로 줄이거나 --obs_driver 로 정책을 지정해라.")
    null_action = np.zeros(env.action_space.shape, dtype=np.float32)
    null_action[-1] = -1.0                       # OSC_POSE 델타 0, 그리퍼 열림

    steps = np.asarray(steps, dtype=int)
    want = {int(s): i for i, s in enumerate(steps)}
    last = int(steps.max())
    out: list[list[dict]] = []
    status = np.zeros((n_ep, len(steps)), dtype=np.int8)
    info: list[dict] = []

    for e in range(n_ep):
        env.reset()
        raw = env.unwrapped.set_init_state(init_states[e % len(init_states)])
        obs = env.unwrapped._format_raw_obs(raw)
        for _ in range(cfg.settle_steps):        # 물체를 테이블에 내려앉힌다 (R3와 같은 값)
            obs, _r, _t, _tr, _i = env.step(null_action)
        if driver is not None:
            driver.reset()
            # flow matching의 a₀는 전역 RNG에서 나온다. 에피소드로 고정해야 재현된다.
            torch.manual_seed(cfg.probe_seed + e)

        per_ep: dict[int, dict] = {}
        state = ALIVE
        first_dead = -1                           # 관측이 처음으로 '살아있지 않게' 된 스텝
        env_end = -1                              # env가 종료를 알린 스텝
        terminated = False
        for k in range(last + 1):
            if k in want:
                per_ep[k] = _clone_obs(obs)
                status[e, want[k]] = state
            if state != ALIVE and first_dead < 0:
                first_dead = k
            if state == FROZEN:
                continue                          # 종료된 뒤에는 더 굴리지 않는다
            if driver is not None:
                proc = preprocess_observation(obs)
                proc.pop("task", None)
                batch = {kk: v.to(device) for kk, v in proc.items() if isinstance(v, torch.Tensor)}
                batch["task"] = [task_desc]
                act = driver.select_action(batch).squeeze(0).cpu().numpy()
            elif k < len(demos[e]):
                act = demos[e][k]
            else:
                act = null_action                 # 데모 소진 -> 그 자리에서 유지
                state = HELD
            obs, _r, term, trunc, _i = env.step(np.asarray(act, dtype=np.float32))
            if term or trunc:
                state, env_end, terminated = FROZEN, k, bool(term)
        out.append([per_ep[int(s)] for s in steps])
        live = first_dead if first_dead >= 0 else last + 1
        dlen = None if demos is None else int(len(demos[e]))
        info.append({
            "episode": e,
            "live_len": int(live),                # 이 길이까지의 관측만 평균에 쓴다
            "demo_len": dlen,
            "env_end": int(env_end),
            "terminated": terminated,             # env 기준 성공 종료인가
            "reason": ("still running" if first_dead < 0 else
                       "demo exhausted" if (dlen is not None and live >= dlen and
                                            (env_end < 0 or live <= env_end)) else
                       ("episode succeeded" if terminated else "episode truncated")),
        })

    env.close()
    n_alive = int((status == ALIVE).sum())
    logging.info(f"[R9] task {task}: 관측 {n_ep}ep × {len(steps)}step = {n_ep * len(steps)}개  "
                 f"(alive {n_alive} · held {int((status == HELD).sum())} · "
                 f"frozen {int((status == FROZEN).sum())})")
    for it in info:                               # 에피소드마다 "언제 끝났는가"
        logging.info(f"[R9]   task {task} ep {it['episode']}: live {it['live_len']} steps  "
                     f"(demo {it['demo_len']} · env end {it['env_end']}) — {it['reason']}")
    if n_alive < 0.5 * status.size:
        logging.warning(colored(
            f"[R9] task {task}: 캡처 지점의 절반 이상이 롤아웃 종료 뒤다. rollout_steps="
            f"{cfg.rollout_steps}가 실제 에피소드 길이보다 길다는 뜻이다. 그 지점들은 "
            f"평균에서 빠지므로(--exclude_dead_obs) 늦은 물리시간은 표본이 적거나 없다.",
            "yellow"))
    return out, status, info


def _digest(obj) -> str:
    """고정물 해시 (§5-5). 세 모델이 같은 프로브를 봤는지 로그로 비교할 수 있게 남긴다."""
    h = hashlib.sha1()

    def walk(x):
        if isinstance(x, torch.Tensor):
            h.update(np.ascontiguousarray(x.detach().float().cpu().numpy()).tobytes())
        elif isinstance(x, np.ndarray):
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                walk(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        else:
            h.update(repr(x).encode())

    walk(obj)
    return h.hexdigest()[:12]


# ═════════════════════════════════════════════════════════════════════════════
#  본 실험
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_probe(cfg: R9Config, run_dir: Path) -> Path:
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    a, b = cfg.task_a, cfg.task_b
    specs = model_specs(cfg)                      # R8과 같은 표·같은 순서 규약

    meta_a = LeRobotDatasetMetadata(f"{cfg.dataset_prefix}{a}")
    ref = load_policy_at(cfg, specs[0]["ckpt"], meta_a, device)
    pol_cfg = ref.config
    stats = norm_stats(ref)
    horizon = int(pol_cfg.horizon)
    e0, e1 = exec_range(pol_cfg, cfg.exec_slice)
    tok = slice(e0, e1)
    logging.info(colored(f"[R9] 실행 구간 = 청크 index {e0}..{e1 - 1} 토큰만 잰다", "green"))

    text = {0: task_text(cfg.dataset_prefix, a), 1: task_text(cfg.dataset_prefix, b)}
    logging.info(colored(f"[R9] c₀ = {text[0]!r}", "cyan"))
    logging.info(colored(f"[R9] c₁ = {text[1]!r}", "cyan"))

    # ── [1] 프로브 고정물 ────────────────────────────────────────────────────
    # ★★ 이 블록은 R8.run_probe의 같은 블록과 **한 글자도 달라선 안 된다**. 난수 소비
    #    순서가 바뀌면 x₀와 a가 달라져 두 그림이 다른 지점을 본 것이 된다. R8은 여기서
    #    x0 다음에 x0b(노이즈 짝)를 더 뽑지만, 그건 x0/a_tgt가 확정된 **뒤**라 R9가
    #    안 뽑아도 두 고정물은 동일하다. 아래 해시를 R8 로그와 대조하면 확인된다.
    rng = np.random.default_rng(cfg.probe_seed)
    chunks = np.concatenate([
        minmax_normalize(demo_chunks(cfg, pol_cfg, a, cfg.demo_episodes), stats),
        minmax_normalize(demo_chunks(cfg, pol_cfg, b, cfg.demo_episodes), stats)])
    pick = rng.choice(len(chunks), size=cfg.num_probe, replace=False)
    a_tgt = torch.from_numpy(chunks[pick]).float().to(device)                  # (N, 16, 7)
    gen = torch.Generator(device="cpu").manual_seed(cfg.probe_seed)
    x0 = torch.randn(cfg.num_probe, horizon, 7, generator=gen).to(device)      # (N, 16, 7)
    t_grid = np.linspace(0.0, float(cfg.t_max), cfg.t_steps, dtype=np.float32)
    perm_np = derangement(cfg.num_probe, cfg.shuffle_seed)
    assert not np.any(perm_np == np.arange(cfg.num_probe)), "교란순열에 고정점이 있다"
    perm = torch.from_numpy(perm_np).long().to(device)
    hash_x0, hash_a = _digest(x0), _digest(a_tgt)
    logging.info(f"[R9] 프로브 {cfg.num_probe}개 × t {cfg.t_steps}격자(0..{cfg.t_max})  "
                 f"x₀#{hash_x0} a#{hash_a}  perm#{_digest(perm_np)}")

    # ── [2] 두 조건을 만들 상황 — 물리시간 축을 따라 뜬다 ────────────────────
    # R8은 초기 관측 한 점(rollout step 0)만 봤다. R9는 0..rollout_steps를 obs_stride
    # 간격으로 훑어 "라우팅 붕괴가 초기 상태에서만인가, 궤적 내내인가"를 가른다.
    obs_steps = np.arange(0, cfg.rollout_steps + 1, cfg.obs_stride, dtype=int)
    n_step = len(obs_steps)
    driver = None
    if cfg.obs_driver != "demo":
        pick_spec = next((s for s in specs if s["key"] == cfg.obs_driver), None)
        if pick_spec is None:
            raise SystemExit(f"--obs_driver 는 'demo' 또는 {[s['key'] for s in specs]} 중 "
                             f"하나여야 한다 ({cfg.obs_driver!r})")
        driver = load_policy_at(cfg, pick_spec["ckpt"], meta_a, device)
        logging.info(colored(
            f"[R9] 상태열 구동자 = 정책 '{cfg.obs_driver}'. 세 모델이 보는 s는 동일하지만 "
            f"그 상태 분포는 이 모델에 편향된다.", "yellow"))
    else:
        logging.info(colored("[R9] 상태열 구동자 = 전문가 데모 재생 (모델 독립)", "green"))
    logging.info(f"[R9] 물리시간 {n_step}지점 {list(obs_steps)} × 초기상태 {cfg.num_obs}개")

    status, ep_info = {}, {}
    if cfg.cond_mode == "full":
        obs_sets = {}
        for ci, task in ((0, a), (1, b)):
            obs_sets[ci], status[ci], ep_info[ci] = capture_obs_traj(
                cfg, task, cfg.num_obs, obs_steps, driver, device)
        logging.info(colored("[R9] cond_mode=full — 장면과 지시문을 함께 바꾼다 (본 실험)",
                             "green"))
    elif cfg.cond_mode == "language":
        fixed, st, inf = capture_obs_traj(cfg, cfg.obs_task, cfg.num_obs, obs_steps,
                                          driver, device)
        obs_sets, status, ep_info = {0: fixed, 1: fixed}, {0: st, 1: st}, {0: inf, 1: inf}
        logging.info(colored(f"[R9] cond_mode=language — 장면을 task {cfg.obs_task}로 고정 (부록)",
                             "green"))
    else:
        raise SystemExit(f"--cond_mode 는 full 또는 language 여야 한다 ({cfg.cond_mode!r})")

    # ★ 짝 기준 생존. 조건 대비는 (c₀ 관측, c₁ 관측) 한 쌍에서 나오므로 **둘 다** 살아
    #   있어야 그 (에피소드, 물리시간) 짝을 쓸 수 있다. 두 태스크의 에피소드 길이가 다르니
    #   한쪽만 죽어도 그 짝은 "정지한 장면 vs 움직이는 장면"의 대비가 되어 버린다.
    pair_alive = (status[0] == ALIVE) & (status[1] == ALIVE)          # (n_ep, S) — 사실
    pair_used = pair_alive                                            # 실제로 평균에 넣는 것
    if not cfg.exclude_dead_obs:
        logging.warning(colored("[R9] --exclude_dead_obs=false — 종료된 롤아웃의 정지 장면도 "
                                "평균에 넣는다.", "yellow"))
        pair_used = np.ones_like(pair_alive)
    n_live = pair_used.sum(axis=0)
    logging.info("[R9] 물리시간별 유효 에피소드 수: " +
                 "  ".join(f"{int(s)}:{int(c)}/{cfg.num_obs}"
                           for s, c in zip(obs_steps, n_live)))
    if not n_live.any():
        raise SystemExit("[R9] 살아있는 (에피소드, 물리시간) 짝이 하나도 없다. "
                         "--rollout_steps 를 줄여라.")
    # ★ 여기서 만든 관측이 R7.capture_obs의 규약과 같은 모양인지 확인한다. 어긋나면
    #   obs_to_cond가 조용히 이상한 조건 벡터를 만든다 (§5 파이프라인 정합성).
    _assert_obs_compatible(obs_sets[0][0][0], capture_obs(cfg, a, 1)[0])
    hash_obs = {ci: _digest(obs_sets[ci]) for ci in (0, 1)}
    logging.info(f"[R9] obs 고정물 해시  c₀#{hash_obs[0]}  c₁#{hash_obs[1]}")
    if driver is not None:
        del driver
    del ref
    torch.cuda.empty_cache()

    # ── [3] 모델 × 관측 × t × 조건 격자 ──────────────────────────────────────
    blob: dict[str, np.ndarray] = {}
    diag: dict[str, dict] = {}
    n_block = None
    for spec in specs:
        policy = load_policy_at(cfg, spec["ckpt"], meta_a, device)
        assert_shared_norm(stats, norm_stats(policy), spec["key"])            # §5-4
        net = policy.dit_flow.velocity_net
        tap = SubBlockTap(net)
        n_block = tap.n_blocks
        logging.info(colored(f"[R9] {spec['key']}: {spec['ckpt']}", "cyan", attrs=["bold"]))

        # ★ 합과 **유효 표본 수**를 따로 들고 간다. NaN(= 그 칸에서 cos이 정의되지 않음)을
        #   0으로 더하면 안 된다 — cos=0은 "완벽한 직교 = 라우팅이 가장 살아있음"이라는
        #   강한 의미를 가진 값이라, 기여가 소멸한 칸이 정반대 결론으로 색칠된다.
        #   AdaLN-Zero라 게이트가 죽으면 Δ가 정확히 0이 되어 실제로 일어날 수 있는 일이다.
        #   나눗셈도 에피소드 수가 아니라 유효 표본 수로 한다. 유효 표본이 0이면 NaN으로
        #   남겨 그림이 회색으로 처리하게 둔다.
        shape = (n_step, n_block, cfg.t_steps)
        cos_sum = {s: np.zeros(shape) for s in SUBS}
        cos_cnt = {s: np.zeros(shape) for s in SUBS}
        shuf_sum = {s: np.zeros(shape) for s in SUBS}
        shuf_cnt = {s: np.zeros(shape) for s in SUBS}
        mag_sum = {s: np.zeros(shape) for s in SUBS}
        mag_cnt = {s: np.zeros(shape) for s in SUBS}
        ok_s = {s: 0.0 for s in SUBS}
        all_s = {s: 0.0 for s in SUBS}
        # ── 최종 출력단 (_FinalLayer). 블록 축이 없으므로 (S, T)다. ──────────────
        #   512  : Δ_512 = x⊙scale + shift          조건이 마지막 표현에 더하는 증분
        #   7    : Δ_7   = W·Δ_512                  액션 공간까지 살아남은 부분
        #   mod  : [shift|scale]                    x와 무관한 순수 주입 벡터
        # 512는 갈렸는데 7이 정렬이면 = 512→7 사영의 null space가 라우팅을 버린 것.
        fshape = (n_step, cfg.t_steps)
        fin_sum = {k: np.zeros(fshape) for k in FINALS}
        fin_cnt = {k: np.zeros(fshape) for k in FINALS}
        fsh_sum = {k: np.zeros(fshape) for k in FINALS}
        fsh_cnt = {k: np.zeros(fshape) for k in FINALS}
        fmag_sum = {k: np.zeros(fshape) for k in FINALS}
        fmod_sum = np.zeros(fshape)
        fpair_cnt = np.zeros(fshape)
        resid_err = fin_err = 0.0
        first = True

        for oi in range(cfg.num_obs):
            for si in range(n_step):
                # 롤아웃이 이미 끝난 짝은 건너뛴다 — 그 관측은 정지한 장면이다.
                # 유효 표본 수(cos_cnt/mag_cnt)가 그만큼 줄고, 살아있는 짝이 하나도 없는
                # 물리시간은 자연히 NaN이 되어 그림에서 회색/끊긴 선으로 나온다.
                if not pair_used[oi, si]:
                    continue
                # 조건 하나는 (그 물리시간의 장면, 지시문) 한 쌍에서 통째로 만들어진다.
                cond = {ci: obs_to_cond(policy, obs_sets[ci][oi][si], text[ci], device)
                        for ci in (0, 1)}
                for ti, t in enumerate(t_grid):
                    tf = float(t)
                    xt = (1.0 - tf) * x0 + tf * a_tgt                          # (N, 16, 7)
                    tt = torch.full((cfg.num_probe,), tf, device=device)

                    tap.check = first                   # 첫 격자점에서만 (h, h_out)도 뜬다
                    d, f = {}, {}
                    for ci in (0, 1):
                        net(xt, tt, cond[ci].expand(cfg.num_probe, -1))
                        d[ci] = tap.snapshot()
                        f[ci] = tap.final_delta()
                        if first and ci == 0:
                            resid_err = tap.verify_residual()                  # §5-1
                            fin_err = tap.verify_final()                       # 최종층 판
                            tap.check = False
                    if first and cfg.determinism_check:                        # §5-2
                        net(xt, tt, cond[0].expand(cfg.num_probe, -1))
                        again = tap.snapshot()
                        for key in d[0]:
                            assert torch.equal(d[0][key], again[key]), (
                                f"[R9] 같은 입력에 Δ가 달라진다 ({key}) — eval 모드/드롭아웃 확인")
                        logging.info(f"[R9]   결정론 OK · residual 항등식 {resid_err:.2e} · "
                                     f"최종층 항등식 {fin_err:.2e}")
                    first = False

                    for li in range(n_block):
                        for s in SUBS:
                            d0, d1 = d[0][(li, s)], d[1][(li, s)]
                            c, mm, nok, nall = pair_stats(d0, d1, tok, cfg.norm_floor)
                            if not np.isnan(c):
                                cos_sum[s][si, li, ti] += c
                                cos_cnt[s][si, li, ti] += 1
                            mag_sum[s][si, li, ti] += mm
                            mag_cnt[s][si, li, ti] += 1
                            ok_s[s] += nok
                            all_s[s] += nall
                            # §2.4 영점: 조건을 고정하고 프로브 인덱스만 섞어 각도를 잰다.
                            # c₀·c₁ 양쪽에서 재되, 정의된 것만 세어 평균한다.
                            for dd in (d0, d1):
                                v = pair_stats(dd, dd.index_select(1, perm), tok,
                                               cfg.norm_floor)[0]
                                if not np.isnan(v):
                                    shuf_sum[s][si, li, ti] += v
                                    shuf_cnt[s][si, li, ti] += 1

                    # ── 최종 출력단 ────────────────────────────────────────────
                    # 블록과 **똑같은** cos·M·shuffle을 512차원 증분과 7차원 사영
                    # 각각에 대해 잰다. 두 값을 나란히 놔야 "방향이 죽었나, 사영이
                    # 버렸나"가 갈린다.
                    for k, idx in (("512", 0), ("7", 1)):
                        g0, g1 = f[0][idx], f[1][idx]
                        c, mm, _nok, _nall = pair_stats(g0, g1, tok, cfg.norm_floor)
                        if not np.isnan(c):
                            fin_sum[k][si, ti] += c
                            fin_cnt[k][si, ti] += 1
                        fmag_sum[k][si, ti] += mm
                        for gg in (g0, g1):
                            v = pair_stats(gg, gg.index_select(1, perm), tok,
                                           cfg.norm_floor)[0]
                            if not np.isnan(v):
                                fsh_sum[k][si, ti] += v
                                fsh_cnt[k][si, ti] += 1
                    # 주입 벡터 자체 [shift|scale]. x에 의존하지 않아 프로브 축이 없고
                    # 따라서 shuffle 영점도 없다 — "주입이 조건을 구분하나"만 본다.
                    m0, m1 = f[0][2][0].double(), f[1][2][0].double()
                    fmod_sum[si, ti] += float(
                        (m0 @ m1) / (m0.norm() * m1.norm()).clamp_min(1e-30))
                    fpair_cnt[si, ti] += 1

        def _mean(num, cnt):
            """유효 표본 수로 나눈다. 표본이 없으면 0이 아니라 NaN이다."""
            return np.where(cnt > 0, num / np.maximum(cnt, 1.0), np.nan)

        for s in SUBS:
            cos_s = _mean(cos_sum[s], cos_cnt[s])
            mag_s = _mean(mag_sum[s], mag_cnt[s])
            shuf_s = _mean(shuf_sum[s], shuf_cnt[s])
            key = f"{spec['key']}_{s}"
            blob[f"{key}_cos"] = cos_s.astype(np.float32)                 # (S, L, T)
            blob[f"{key}_M"] = mag_s.astype(np.float32)
            blob[f"{key}_cos_shuffle"] = shuf_s.astype(np.float32)
            blob[f"{key}_cos_n"] = cos_cnt[s].astype(np.int32)            # 유효 표본 수
        for k in FINALS:
            key = f"{spec['key']}_final{k}"
            blob[f"{key}_cos"] = _mean(fin_sum[k], fin_cnt[k]).astype(np.float32)
            blob[f"{key}_M"] = _mean(fmag_sum[k], fpair_cnt).astype(np.float32)
            blob[f"{key}_cos_shuffle"] = _mean(fsh_sum[k], fsh_cnt[k]).astype(np.float32)
            blob[f"{key}_cos_n"] = fin_cnt[k].astype(np.int32)
        blob[f"{spec['key']}_finalmod_cos"] = _mean(fmod_sum, fpair_cnt).astype(np.float32)
        excl = {s: 1.0 - float(ok_s[s] / max(all_s[s], 1.0)) for s in SUBS}
        undef = {s: float(np.mean(cos_cnt[s] == 0)) for s in SUBS}
        diag[spec["key"]] = {"excluded_frac": excl, "cos_undefined_frac": undef,
                             "residual_max_rel_err": resid_err,
                             "final_max_rel_err": fin_err, "ckpt": spec["ckpt"]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for k in FINALS:
                fc = _mean(fin_sum[k], fin_cnt[k])
                fs = _mean(fsh_sum[k], fsh_cnt[k])
                logging.info(f"[R9]  final{k:>4}: ⟨cos⟩ {np.nanmean(fc):+.3f}  "
                             f"⟨shuffle⟩ {np.nanmean(fs):+.3f}  "
                             f"gap {np.nanmean(fc - fs):+.3f}  "
                             f"⟨‖Δ‖⟩ {np.nanmean(_mean(fmag_sum[k], fpair_cnt)):.3e}")
            logging.info(f"[R9]  final mod: ⟨cos([shift|scale](c₀), (c₁))⟩ "
                         f"{np.nanmean(_mean(fmod_sum, fpair_cnt)):+.3f}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # 전부 NaN인 칸은 정상이다
            for s in SUBS:
                cos_s = _mean(cos_sum[s], cos_cnt[s])
                shuf_s = _mean(shuf_sum[s], shuf_cnt[s])
                mag_s = _mean(mag_sum[s], mag_cnt[s])
                logging.info(
                    f"[R9]   {s:>4}: ⟨cos⟩ {np.nanmean(cos_s):+.3f}  "
                    f"⟨shuffle⟩ {np.nanmean(shuf_s):+.3f}  "
                    f"gap {np.nanmean(cos_s - shuf_s):+.3f}  "
                    f"⟨‖Δ‖⟩ {np.nanmean(mag_s):.3e}  "
                    f"제외 {excl[s] * 100:.2f}%  cos 미정의 칸 {undef[s] * 100:.2f}%")
                if excl[s] > 0.5 or undef[s] > 0.05:
                    logging.warning(colored(
                        f"[R9]   {spec['key']}/{s}: 기여가 사실상 소멸한 구간이 있다. "
                        f"cos이 아니라 M(크기 단면)으로 읽어라 — 미정의 칸은 그림에서 "
                        f"회색으로 나온다.", "yellow"))
        tap.remove()
        del policy
        torch.cuda.empty_cache()

    blob["t_grid"] = t_grid
    blob["obs_steps"] = obs_steps.astype(np.int32)
    blob["obs_status_c0"] = status[0].astype(np.int8)      # ALIVE / HELD / FROZEN
    blob["obs_status_c1"] = status[1].astype(np.int8)
    blob["obs_pair_alive"] = pair_alive.astype(bool)       # 두 롤아웃이 다 살아 있었나
    blob["obs_pair_used"] = pair_used.astype(bool)         # 실제로 평균에 들어갔나
    blob["meta"] = np.array(json.dumps({
        "task_a": a, "task_b": b, "text_c0": text[0], "text_c1": text[1],
        "n_block": n_block, "subs": list(SUBS),
        "num_probe": cfg.num_probe, "probe_seed": cfg.probe_seed,
        "shuffle_seed": cfg.shuffle_seed, "norm_floor": cfg.norm_floor,
        "t_steps": cfg.t_steps, "t_max": cfg.t_max, "num_obs": cfg.num_obs,
        "obs_task": cfg.obs_task, "cond_mode": cfg.cond_mode, "exec_slice": [e0, e1],
        "demo_episodes": cfg.demo_episodes,
        "rollout_steps": cfg.rollout_steps, "obs_stride": cfg.obs_stride,
        "obs_driver": cfg.obs_driver, "settle_steps": cfg.settle_steps,
        "obs_status_legend": {"0": "alive", "1": "held (demo exhausted)",
                              "2": "frozen (episode ended)"},
        "exclude_dead_obs": cfg.exclude_dead_obs,
        "episodes": {"c0": ep_info[0], "c1": ep_info[1]},
        "live_pairs_by_step": [int(v) for v in n_live],
        "route_gap_thresh": cfg.route_gap_thresh, "route_mag_frac": cfg.route_mag_frac,
        "alpha_floor": cfg.alpha_floor,
        "delta_method": "B-direct (gate module output) == A-difference (asserted every model)",
        "hook_points": ["layers[ℓ].attn_gate output", "layers[ℓ].mlp_gate output",
                        "layers[ℓ] input/output (residual identity check)"],
        "hash_x0": hash_x0, "hash_a_tgt": hash_a, "hash_perm": _digest(perm_np),
        "hash_obs_c0": hash_obs[0], "hash_obs_c1": hash_obs[1],
        "diagnostics": diag,
        "specs": [{k: s[k] for k in ("key", "ckpt", "title")} for s in specs],
    }))
    cache = run_dir / cache_name(cfg)
    np.savez_compressed(cache, **blob)
    logging.info(colored(f"[R9] saved -> {cache}", "green", attrs=["bold"]))
    write_method_doc(cache)
    return cache


def write_method_doc(cache: Path) -> Path:
    """§6 문서: 실제로 쓴 Δ 추출 방식·hook 지점·시드·해시를 남긴다."""
    m = json.loads(str(np.load(cache, allow_pickle=False)["meta"]))
    lines = [
        "# R9 — method record (auto-generated)", "",
        f"cache: `{cache.name}`", "",
        "## Δ extraction", "",
        f"- method: **{m['delta_method']}**",
        "- `_DiTDecoder.forward` 가 `x = x + attn_gate(...)`, `return x + mlp_gate(...)` 이므로",
        "  게이트 모듈의 출력이 곧 residual 증분이다. (A) 차분과 (B) 직접이 같은 텐서다.",
        "- 항등식 `h_out − h == Δ_attn + Δ_mlp` 를 모델마다 첫 forward에서 assert 했다:",
    ]
    for k, v in m["diagnostics"].items():
        lines.append(f"  - `{k}`: max rel err {v['residual_max_rel_err']:.2e}, "
                     f"cos 제외 비율 attn {v['excluded_frac']['attn'] * 100:.2f}% / "
                     f"mlp {v['excluded_frac']['mlp'] * 100:.2f}%, "
                     f"cos 미정의 칸 attn {v['cos_undefined_frac']['attn'] * 100:.2f}% / "
                     f"mlp {v['cos_undefined_frac']['mlp'] * 100:.2f}%")
    lines += [
        "- hook 지점: " + ", ".join(f"`{h}`" for h in m["hook_points"]), "",
        "## Probe / geometry", "",
        f"- flow time: `x_t = (1−t)·x₀ + t·a`, t ∈ linspace(0, {m['t_max']}, {m['t_steps']})",
        f"- num_probe {m['num_probe']} · probe_seed {m['probe_seed']} · "
        f"num_obs {m['num_obs']} (초기 상태) · demo_episodes {m['demo_episodes']}",
        f"- cond_mode `{m['cond_mode']}` · task_a {m['task_a']} vs task_b {m['task_b']}",
        f"- exec_slice: chunk index {m['exec_slice'][0]}..{m['exec_slice'][1] - 1}",
        f"- shuffle_seed {m['shuffle_seed']} (derangement, 고정점 없음) · "
        f"norm_floor {m['norm_floor']:.0e}", "",
        "## Physical time (rollout step)", "",
        f"- 조건 관측을 rollout step 0..{m['rollout_steps']} 에서 {m['obs_stride']} 간격으로 "
        f"떴다 (settle_steps {m['settle_steps']} 이후 기준).",
        f"- 상태열 구동자: `{m['obs_driver']}`" + (
            "  — 전문가 데모 액션 재생. 모델에 의존하지 않는다."
            if m["obs_driver"] == "demo" else
            "  — 이 정책 하나가 세 모델 전부의 상태를 만든다(모델이 셋 다 같은 s를 보되, "
            "상태 분포는 이 모델에 편향된다)."),
        "- 캡처 지점 상태는 `obs_status_c0` / `obs_status_c1` (0 alive · 1 held(데모 소진) · "
        "2 frozen(에피소드 종료))에 기록되어 있다.", "",
        f"- `exclude_dead_obs = {m.get('exclude_dead_obs', True)}` — 롤아웃이 이미 끝난 "
        f"(에피소드, 물리시간) 짝은 평균에서 뺐다.",
        "- 판정은 **짝** 기준이다: c₀와 c₁의 에피소드 길이가 다르므로 한쪽만 죽어도 그 짝은 "
        "'정지한 장면 vs 움직이는 장면'의 대비가 되어 쓸 수 없다.",
        "- 물리시간별 유효 짝 수: " +
        "  ".join(f"{s}:{c}" for s, c in zip(range(0, m["rollout_steps"] + 1, m["obs_stride"]),
                                             m.get("live_pairs_by_step", []))), "",
        "### 에피소드별 롤아웃 길이", "",
        "| rollout | episode | live_len | demo_len | env_end | why it stopped |",
        "|---|---|---|---|---|---|",
    ]
    for ci, tag in (("c0", f"c₀ (task {m['task_a']})"), ("c1", f"c₁ (task {m['task_b']})")):
        for it in m.get("episodes", {}).get(ci, []):
            lines.append(f"| {tag} | {it['episode']} | {it['live_len']} | {it['demo_len']} | "
                         f"{it['env_end']} | {it['reason']} |")
    lines += ["",
        "## Undefined cos", "",
        "- ‖Δ‖가 norm_floor 미만이면 cos이 정의되지 않는다. 이 칸은 **0으로 채우지 않는다** — "
        "cos=0은 '완벽한 직교 = 라우팅 최상'이라는 강한 의미를 가진 값이라 결론이 뒤집힌다.",
        "- 누적은 합과 유효 표본 수를 따로 들고, 표본 수로 나눈다. 유효 표본이 0인 칸은 NaN으로 "
        "남아 히트맵에서 회색으로, 단면에서는 선이 끊긴 것으로 나타난다.",
        "- 칸별 유효 표본 수는 npz의 `<model>_<sub>_cos_n` 에 그대로 들어 있다.", "",
        "## Fixture hashes (R8 로그와 대조할 것)", "",
        f"- x₀ `{m['hash_x0']}` · a_tgt `{m['hash_a_tgt']}` · perm `{m['hash_perm']}`",
        f"- obs c₀ `{m['hash_obs_c0']}` · obs c₁ `{m['hash_obs_c1']}`", "",
        "## Checkpoints", "",
    ]
    lines += [f"- `{s['key']}` — {s['title']}\n  - `{s['ckpt']}`" for s in m["specs"]]
    lines += [
        "", "## Routing count rule (그림 캡션의 숫자)", "",
        f"- `mean_t(cos − cos_shuffle) < −{m['route_gap_thresh']}` **그리고** "
        f"`mean_t(M) ≥ {m['route_mag_frac']} × (전 모델 공통 M 최대)`", "",
        "## Determinism", "",
        "- 첫 격자점에서 같은 forward를 두 번 돌려 Δ가 bit-wise 동일함을 확인했다.",
        "- 전체 2회 실행 비교: `--recompute` 로 한 번 더 돌린 뒤 두 npz의 "
        "`*_cos` / `*_M` 배열을 `np.allclose` 로 비교하면 된다.",
    ]
    out = cache.with_name(cache.stem + ".method.md")
    out.write_text("\n".join(lines) + "\n")
    print(f"saved doc    -> {out}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  그림
# ═════════════════════════════════════════════════════════════════════════════
def _nanmean(x, axis=None):
    """전부 NaN인 축(= 그 칸에서 cos이 한 번도 정의되지 않음)은 경고 없이 NaN을 준다.

    그건 병리가 아니라 "기여가 소멸해 각도를 잴 수 없었다"는 관측 결과다. 0으로 바꾸면
    '완벽한 직교 = 라우팅 최상'이 되어 정반대로 읽힌다.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


def _shade_dead(ax, steps, dead_frac, thresh: float = 0.5) -> bool:
    """살아있는 에피소드가 절반도 안 남은 물리시간 구간에 회색 띠를 깐다.

    죽은 짝은 이미 평균에서 빠졌으므로(남은 값 자체는 유효하다) 이 띠는 "틀린 구간"이
    아니라 **표본이 적은 구간**이라는 뜻이다. 정확한 개수는 타임라인 패널이 말한다.
    """
    bad = np.flatnonzero(dead_frac > thresh)
    if not len(bad):
        return False
    x0 = float(steps[bad[0]])
    ax.axvspan(x0, float(steps[-1]) + 1e-9, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.text(x0, 1.0, " fewer than half the episodes still running ",
            transform=ax.get_xaxis_transform(), fontsize=6.5, color=INK2, ha="left", va="top")
    return True


def _draw_timeline(ax, ctx: dict, si: int | None) -> None:
    """에피소드마다 롤아웃이 얼마나 길었고 언제 끝났는지.

    조건 대비는 (c₀ 관측, c₁ 관측) 짝에서 나오므로 두 태스크의 에피소드를 나란히 그린다.
    두 막대가 서로 다른 곳에서 끝나는 것이 곧 "짝 기준으로 걸러야 하는 이유"다.
    채운 마커 = 평균에 실제로 들어간 캡처 지점, 빈 마커 = 롤아웃이 끝나 빠진 지점.
    """
    m, steps = ctx["m"], ctx["obs_steps"]
    eps = m.get("episodes", {})
    n_ep, alive = ctx["n_ep"], ctx["pair_used"]
    rows, labels = [], []
    for ci, tag in ((1, "c₁"), (0, "c₀")):        # 위에서부터 c₀가 오도록 뒤집어 쌓는다
        for e in range(n_ep - 1, -1, -1):
            rows.append((ci, e))
            labels.append(f"{tag} ep{e}")
    last = float(steps[-1])
    # ★ 여기서는 모델 색을 쓰지 않는다. 이 패널의 행은 체크포인트가 아니라 **조건(태스크)**
    #   이라서, 다른 패널의 joint/CL 색을 재사용하면 같은 그림 안에서 같은 색이 두 가지를
    #   뜻하게 된다. 중립 잉크 두 톤 + 그룹 구분선으로 가른다.
    COND_COL = {0: INK, 1: INK2}
    for y, (ci, e) in enumerate(rows):
        it = (eps.get(f"c{ci}") or [{}] * n_ep)[e]
        live = float(it.get("live_len", 0))
        col = COND_COL[ci]
        # 살아 있던 구간 -> 진한 선, 그 뒤 정지 구간 -> 옅은 선
        ax.plot([0, min(live, last)], [y, y], color=col, lw=3.2, solid_capstyle="butt", zorder=3)
        if live < last:
            ax.plot([live, last], [y, y], color=GRID, lw=3.2, solid_capstyle="butt", zorder=2)
        ax.text(min(live, last), y + 0.30, f"{int(live)}  {it.get('reason', '')}", fontsize=6.5,
                color=INK2, va="bottom", ha="left", zorder=5)
        for sj, st in enumerate(steps):
            used = bool(alive[e, sj])
            ax.plot([st], [y], marker="o", ms=4.2, zorder=4, mfc=col if used else "white",
                    mec=col if used else INK2, mew=0.9)
    ax.axhline(n_ep - 0.5, color=GRID, lw=1.0, zorder=1)      # c₀ / c₁ 그룹 구분선
    if si is not None:
        ax.axvline(float(steps[si]), color=DIV_LO, lw=1.6, alpha=0.85, zorder=6)
        ax.text(float(steps[si]), len(rows) - 0.35, f" step {int(steps[si])} ", fontsize=7.5,
                color=DIV_LO, ha="left", va="bottom", fontweight="bold")
    _style(ax)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7, axis="x")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_xlim(-6, last * 1.06)
    ax.set_xticks(steps)
    ax.set_xlabel("rollout step", color=INK2, fontsize=8.5)
    live_txt = " ".join(f"{int(s)}:{int(c)}" for s, c in zip(steps, ctx["n_live"]))
    ax.set_title(f"t   episode timelines — how long each rollout actually ran "
                 f"(c₀ = task {m['task_a']} black, c₁ = task {m['task_b']} grey)   ·   "
                 f"usable pairs per step   {live_txt}   (of {n_ep})",
                 fontsize=10, color=INK, pad=8, loc="left")


def _prepare(cache: Path) -> dict:
    """npz 하나를 읽어 그림이 쓸 모든 것을 만든다.

    ★ 색 스케일(norm)과 알파 정규화 기준(m_max)은 **전체 (S,L,T)** 에서 한 번만 정한다.
      스텝별 그림이 각자 자기 범위로 정규화하면 9장을 나란히 놓고 비교할 수 없다 —
      그러면 스텝별로 그리는 의미가 사라진다.
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    z = {k: v for k, v in np.load(cache, allow_pickle=False).items()}
    m = json.loads(str(z["meta"]))
    keys = [s["key"] for s in m["specs"]]
    titles = {s["key"]: s["title"] for s in m["specs"]}

    # 배열은 (S, L, T) — S = 물리시간(rollout step) 지점 수.
    cos_s = {(k, s): z[f"{k}_{s}_cos"].astype(np.float64) for k in keys for s in SUBS}
    mag_s = {(k, s): z[f"{k}_{s}_M"].astype(np.float64) for k in keys for s in SUBS}
    shuf_s = {(k, s): z[f"{k}_{s}_cos_shuffle"].astype(np.float64) for k in keys for s in SUBS}

    # ── 색: shuffle 기준선을 중심(흰색)에 놓는다 ─────────────────────────────
    # ★ cos을 그대로 칠하고 중심만 옮기는 대신 (cos − shuffle)을 칠한다. shuffle 기준선은
    #   (ℓ, t)마다 값이 다르므로, 단일 vcenter로는 "칸마다 다른 영점"을 표현할 수 없다.
    #   차이를 칠하고 0에 중심을 두면 흰색이 정확히 "그 칸의 shuffle 수준"이 된다.
    # 방향은 R8의 발산 팔레트를 **뒤집어** 쓴다: 낮은 쪽(직교=라우팅)이 파랑이다.
    cmap = LinearSegmentedColormap.from_list("route", [DIV_HI, DIV_MID, DIV_LO])
    cmap.set_bad("#eceae5")
    gap_s = {ks: cos_s[ks] - shuf_s[ks] for ks in cos_s}
    # 최종 출력단 (S, T). 블록 축이 없다.
    fcos_s = {(k, w): z[f"{k}_final{w}_cos"].astype(np.float64) for k in keys for w in FINALS}
    fmag_s = {(k, w): z[f"{k}_final{w}_M"].astype(np.float64) for k in keys for w in FINALS}
    fshuf_s = {(k, w): z[f"{k}_final{w}_cos_shuffle"].astype(np.float64)
               for k in keys for w in FINALS}
    fmod_s = {k: z[f"{k}_finalmod_cos"].astype(np.float64) for k in keys}
    fgap_s = {kw: fcos_s[kw] - fshuf_s[kw] for kw in fcos_s}
    # 색 스케일은 블록과 최종층이 **공유**한다. 따로 잡으면 "블록에선 파랑인데
    # 최종층에선 흰색"이 색의 문제인지 값의 문제인지 알 수 없게 된다.
    flat = np.concatenate([g.ravel() for g in list(gap_s.values()) + list(fgap_s.values())])
    flat = flat[np.isfinite(flat)]
    lo = min(float(np.percentile(flat, 1)), -0.02)
    hi = max(float(np.percentile(flat, 99)), 0.02)

    return {
        "cache": cache, "z": z, "m": m, "keys": keys, "titles": titles,
        "short": {k: titles[k].split("  ")[0] for k in keys},
        "t_grid": z["t_grid"], "obs_steps": z["obs_steps"],
        "L": m["n_block"], "nt": len(z["t_grid"]), "ns": len(z["obs_steps"]),
        "cos_s": cos_s, "mag_s": mag_s, "shuf_s": shuf_s, "gap_s": gap_s,
        "fcos_s": fcos_s, "fmag_s": fmag_s, "fshuf_s": fshuf_s, "fgap_s": fgap_s,
        "fmod_s": fmod_s,
        # 실제로 평균에 들어간 (에피소드, 물리시간) 짝. 죽은 짝은 run_probe가 이미 뺐다.
        "pair_alive": z["obs_pair_alive"].astype(bool),
        "pair_used": z["obs_pair_used"].astype(bool),
        "n_live": z["obs_pair_used"].astype(bool).sum(axis=0),
        "dead_frac": 1.0 - z["obs_pair_used"].astype(bool).mean(axis=0),
        "n_ep": int(z["obs_pair_alive"].shape[0]),
        "cmap": cmap, "norm": TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi),
        "m_max": max(float(np.nanmax(v)) for v in mag_s.values()),
        "a_floor": float(m.get("alpha_floor", 0.06)),
        "gth": m["route_gap_thresh"], "mfrac": m["route_mag_frac"],
    }


def _slice(ctx: dict, si: int | None) -> dict:
    """si=None이면 물리시간 평균, 정수면 그 스텝 하나. 둘 다 (L, T) 배열을 준다."""
    if si is None:
        pick = lambda d: {ks: _nanmean(v, axis=0) for ks, v in d.items()}   # noqa: E731
    else:
        pick = lambda d: {ks: v[si] for ks, v in d.items()}                 # noqa: E731
    cos, mag, shuf = pick(ctx["cos_s"]), pick(ctx["mag_s"]), pick(ctx["shuf_s"])
    gap = {ks: cos[ks] - shuf[ks] for ks in cos}
    fcos, fmag, fshuf = pick(ctx["fcos_s"]), pick(ctx["fmag_s"]), pick(ctx["fshuf_s"])
    fmod = pick(ctx["fmod_s"])
    fgap = {kw: fcos[kw] - fshuf[kw] for kw in fcos}
    gap_layer = {ks: _nanmean(gap[ks], axis=1) for ks in gap}
    mag_layer = {ks: _nanmean(mag[ks], axis=1) for ks in mag}
    with warnings.catch_warnings():
        # NaN 블록(그 블록에서 cos이 한 번도 정의되지 않음)은 routing으로 세지 않는다.
        # NaN 비교는 False라 자동으로 빠지지만, 그게 의도라는 것을 여기 적어 둔다.
        warnings.simplefilter("ignore", RuntimeWarning)
        routes = {ks: (gap_layer[ks] < -ctx["gth"])
                  & (mag_layer[ks] >= ctx["mfrac"] * ctx["m_max"]) for ks in gap}
    return {"cos": cos, "mag": mag, "shuf": shuf, "gap": gap, "gap_layer": gap_layer,
            "mag_layer": mag_layer, "routes": routes,
            "fcos": fcos, "fmag": fmag, "fshuf": fshuf, "fgap": fgap, "fmod": fmod,
            "n_route": {ks: int(routes[ks].sum()) for ks in routes}}


def _maybe_log(ax, vals) -> None:
    """크기 축은 모델 간 비율이 크면 로그로. 0/NaN이 섞여도 안전하게 판정한다."""
    allv = np.concatenate([np.asarray(v, dtype=float).ravel() for v in vals])
    allv = allv[np.isfinite(allv) & (allv > 0)]
    if allv.size and allv.max() / allv.min() > 30:
        ax.set_yscale("log")


def _draw_magnitude(ctx: dict, out: Path) -> Path:
    """크기 전용 그림. 본 그림에서 뺀 ‖Δ‖를 여기서 전부 숫자축으로 본다.

    본 그림이 불투명도로 크기를 나타내던 것을 버린 이유는 단순하다 — 흰 배경에서
    "gap이 0이라 흰 칸"과 "‖Δ‖가 작아 흐린 칸"이 똑같이 보여, 무엇 때문에 흰지 알 수
    없었다. 색은 방향만 말하게 두고, 크기는 축이 있는 그래프로 분리한다.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    m, keys, titles, short = ctx["m"], ctx["keys"], ctx["titles"], ctx["short"]
    t_grid, obs_steps, L = ctx["t_grid"], ctx["obs_steps"], ctx["L"]
    mfrac, m_max = ctx["mfrac"], ctx["m_max"]
    d = _slice(ctx, None)
    rows = np.arange(1, L + 1)
    floor = mfrac * m_max

    fig = plt.figure(figsize=(14.0, 12.6))
    gs = fig.add_gridspec(4, 2, hspace=0.46, wspace=0.22,
                          left=0.075, right=0.975, top=0.905, bottom=0.055)

    def _fin(ax, xs, vals, xlabel, ylabel, title, xticks=None, floor_line=True):
        if floor_line:
            ax.axhline(floor, color=INK2, lw=0.9, ls="--", zorder=2)
            ax.text(xs[0], floor, f" routing floor ({mfrac:g}×max)", fontsize=7, color=INK2,
                    ha="left", va="bottom", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        _style(ax)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
        _maybe_log(ax, vals)
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
        ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)
        ax.set_title(title, fontsize=10, color=INK, pad=8, loc="left")

    # ── a/b: 블록별 ─────────────────────────────────────────────────────────
    for i, sub in enumerate(SUBS):
        ax = fig.add_subplot(gs[0, i]); vals = []
        for ki, k in enumerate(keys):
            prof = d["mag_layer"][(k, sub)]; vals.append(prof)
            ax.plot(rows, prof, color=MODEL_COLORS[k], lw=2.0, marker="o", ms=4, zorder=4)
            j = min(L - 1, ki + 1)
            if np.isfinite(prof[j]):
                ax.annotate(short[k], (rows[j], prof[j]), textcoords="offset points",
                            xytext=(0, 7), fontsize=8, color=MODEL_COLORS[k],
                            fontweight="bold", ha="center")
        _fin(ax, rows, vals, "DiT block", f"‖Δ‖  ({sub})",
             f"{'ab'[i]}   contribution magnitude by depth — {sub}", xticks=rows)

    # ── c/d: flow time별 ────────────────────────────────────────────────────
    for i, sub in enumerate(SUBS):
        ax = fig.add_subplot(gs[1, i]); vals = []
        for ki, k in enumerate(keys):
            prof = _nanmean(d["mag"][(k, sub)], axis=0); vals.append(prof)
            ax.plot(t_grid, prof, color=MODEL_COLORS[k], lw=2.0, marker="o", ms=3.5, zorder=4)
            j = len(t_grid) - 1 - 3 * ki
            if np.isfinite(prof[j]):
                ax.annotate(short[k], (t_grid[j], prof[j]), textcoords="offset points",
                            xytext=(0, 7), fontsize=8, color=MODEL_COLORS[k],
                            fontweight="bold", ha="center")
        _fin(ax, t_grid, vals, "flow time  t", f"‖Δ‖  ({sub}, blocks averaged)",
             f"{'cd'[i]}   magnitude over flow time — {sub}")

    # ── e/f: 물리시간(rollout step)별 ───────────────────────────────────────
    for i, sub in enumerate(SUBS):
        ax = fig.add_subplot(gs[2, i]); vals = []
        _shade_dead(ax, obs_steps, ctx["dead_frac"])
        for ki, k in enumerate(keys):
            prof = _nanmean(ctx["mag_s"][(k, sub)], axis=(1, 2)); vals.append(prof)
            ax.plot(obs_steps, prof, color=MODEL_COLORS[k], lw=2.0, marker="o", ms=4, zorder=4)
            j = max(0, len(obs_steps) - 1 - ki)
            if np.isfinite(prof[j]):
                ax.annotate(short[k], (obs_steps[j], prof[j]), textcoords="offset points",
                            xytext=(0, 7), fontsize=8, color=MODEL_COLORS[k],
                            fontweight="bold", ha="center")
        _fin(ax, obs_steps, vals, "rollout step  (physical time)",
             f"‖Δ‖  ({sub}, blocks and t averaged)",
             f"{'ef'[i]}   magnitude over physical time — {sub}", xticks=obs_steps)

    # ── g/h: 최종 출력단 ────────────────────────────────────────────────────
    for i, w in enumerate(FINALS):
        ax = fig.add_subplot(gs[3, i]); vals = []
        for ki, k in enumerate(keys):
            prof = d["fmag"][(k, w)]; vals.append(prof)
            ax.plot(t_grid, prof, color=MODEL_COLORS[k], lw=2.0, marker="o", ms=3.5, zorder=4)
            j = len(t_grid) - 1 - 3 * ki
            if np.isfinite(prof[j]):
                ax.annotate(short[k], (t_grid[j], prof[j]), textcoords="offset points",
                            xytext=(0, 7), fontsize=8, color=MODEL_COLORS[k],
                            fontweight="bold", ha="center")
        # 최종층은 블록과 스케일이 달라(7차원은 사영이라 훨씬 작다) routing floor를 안 긋는다.
        _fin(ax, t_grid, vals, "flow time  t",
             f"‖Δ‖  ({'512-d Δ' if w == '512' else '7-d W·Δ'})",
             f"{'gh'[i]}   final layer magnitude — {FINAL_TITLE[w]}", floor_line=False)

    handles = [Line2D([0], [0], color=MODEL_COLORS[k], lw=2.4, marker="o", ms=5,
                      label=titles[k]) for k in keys]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 0.006))
    a, b = m["task_a"], m["task_b"]
    fig.suptitle(
        f"R9 magnitudes: how big is each sub-block's conditional contribution?   "
        f"task {a} vs task {b}\n"
        f"‖Δ‖ = ½(‖Δ(c₀)‖ + ‖Δ(c₁)‖), the size of the vector the sub-block adds to the "
        f"residual stream — the companion to the direction map.\n"
        f"Split out of the main figure on purpose: encoding it as opacity there made a "
        f"chance-level cell and a collapsed cell look identical.\n"
        f"Below the routing floor ({floor:.2g}) a block's direction carries no meaning, so "
        f"the main figure's colour should not be read there.",
        fontsize=10.5, color=INK, y=0.992, linespacing=1.45)
    fig.savefig(out, dpi=170, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return out


def _draw(ctx: dict, si: int | None, out: Path) -> Path:
    """그림 한 장. si=None이면 물리시간을 평균한 요약본(+ 물리시간 단면 패널),
    정수면 그 rollout step 하나만 담은 판이다.

    두 경우가 같은 함수를 쓰는 것이 핵심이다 — 색 스케일·알파 기준·routing 규칙이
    갈라지면 9장을 나란히 놓고 읽을 수 없다.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    m, keys, titles, short = ctx["m"], ctx["keys"], ctx["titles"], ctx["short"]
    t_grid, obs_steps = ctx["t_grid"], ctx["obs_steps"]
    L, nt, ns = ctx["L"], ctx["nt"], ctx["ns"]
    cmap, norm, m_max, a_floor = ctx["cmap"], ctx["norm"], ctx["m_max"], ctx["a_floor"]
    gth, mfrac = ctx["gth"], ctx["mfrac"]
    # 완전 불투명이 되는 기준선. routing floor(= mfrac × m_max)의 2배.
    d = _slice(ctx, si)
    cos, mag, shuf, gap = d["cos"], d["mag"], d["shuf"], d["gap"]
    n_route, mag_layer = d["n_route"], d["mag_layer"]
    a, b = m["task_a"], m["task_b"]
    summary_fig = si is None

    # ── 캡션의 판정 줄은 손으로 쓰지 않고 데이터에서 센다 ────────────────────
    # 손으로 쓰면 조건을 바꿨을 때 캡션만 옛말이 되어 남는다.
    route_txt = " ;   ".join(
        f"{short[k]}: " + " · ".join(f"{s} {n_route[(k, s)]}/{L}" for s in SUBS)
        for k in keys)
    full = m.get("cond_mode", "full") == "full"
    what = ("the whole conditioning vector (scene + instruction)" if full
            else f"the instruction alone, scene held at task {m['obs_task']}")
    drv = m.get("obs_driver", "demo")
    drv_txt = ("expert demo replay, so the states are model-independent" if drv == "demo"
               else f"a rollout of '{drv}', one fixed driver for all {len(keys)} checkpoints")
    # 롤아웃 길이가 에피소드마다 다르다는 사실은 캡션이 직접 말해야 한다.
    lens = [it["live_len"] for ci in ("c0", "c1") for it in m.get("episodes", {}).get(ci, [])]
    n_alive_pairs = int(ctx["pair_alive"].sum())
    if m.get("exclude_dead_obs", True) and lens:
        dead_txt = (f"Rollout length differs per episode ({min(lens)}–{max(lens)} steps; panel q "
                    f"lists every one), so a (episode, step) pair is averaged in\n"
                    f"only while BOTH task rollouts are still running — {n_alive_pairs} "
                    f"of {ctx['n_ep'] * ns} pairs qualified.  Late steps therefore rest on fewer "
                    f"episodes, not on frozen scenes.\n")
    elif lens:
        dead_txt = (f"--exclude_dead_obs=false: every capture point is averaged in, including "
                    f"{ctx['n_ep'] * ns - n_alive_pairs} taken after the rollout had "
                    f"already ended (frozen scenes).\n")
    else:
        dead_txt = ""
    # 최종 출력단 요약도 손으로 쓰지 않고 센다. 512는 갈렸는데 7이 정렬이면
    # "라우팅은 살아 있었고 512→7 readout이 버렸다"가 된다.
    f512 = {k: float(_nanmean(d["fgap"][(k, "512")])) for k in keys}
    f7 = {k: float(_nanmean(d["fgap"][(k, "7")])) for k in keys}
    final_txt = ("Panels q/r/s add the readout that the block hooks never see: "
                 "gap of Δ (512-d) → W·Δ (7-d)   " +
                 " · ".join(f"{short[k]} {f512[k]:+.2f}→{f7[k]:+.2f}" for k in keys) +
                 ".\n")
    n_undef = sum(int(np.sum(~np.isfinite(cos[ks]))) for ks in cos)
    undef_txt = (f"{n_undef} of {len(cos) * L * nt} cells are grey: ‖Δ‖ fell below "
                 f"{m['norm_floor']:.0e} there, so cos is undefined — never filled with 0, "
                 f"which would read as perfect orthogonality.\n" if n_undef else "")
    if summary_fig:
        when = (f"the conditioning observation is taken at rollout steps 0 to "
                f"{int(obs_steps[-1])} every {m['obs_stride']} ({ns} points)\n"
                f"across {m['num_obs']} episodes, driven by {drv_txt}.  Heat maps average over "
                f"that axis; panels m/n/o/p keep it.\n")
        title0 = (f"R9: where does conditional routing break?   task {a} vs task {b}   —   "
                  f"{what} is swapped\n")
        rule = "averaged over t and rollout step"
    else:
        dead = ctx["dead_frac"][si]
        warn = ("" if dead <= 0 else
                f"WARNING: {dead * 100:.0f}% of the episodes had already ended by this step, "
                f"so those observations are a frozen scene.\n")
        when = (f"the conditioning observation is taken at rollout step "
                f"{int(obs_steps[si])} only ({si + 1} of {ns} sampled),\n"
                f"across {m['num_obs']} episodes, driven by {drv_txt}.\n"
                f"{warn}"
                f"Colour scale, opacity normalisation and routing rule are shared with every "
                f"other step figure, so the {ns} panels can be read side by side.\n")
        title0 = (f"R9 · rollout step {int(obs_steps[si])} of {int(obs_steps[-1])}:   "
                  f"where does conditional routing break?   task {a} vs task {b}   —   "
                  f"{what} is swapped\n")
        rule = "averaged over t"
    caption = (
        title0 +
        f"Δ_sub(ℓ,c) = the vector each sub-block adds to the residual stream, α(c) ⊙ sub-block "
        f"output, read at the gate module\n"
        f"(identical to the residual difference h_out − h, asserted on every run).\n"
        f"Colour = cos(Δ(c₀), Δ(c₁)) minus that cell's own shuffle baseline, so white is "
        f"exactly chance level:\n"
        f"blue = the two conditions push in more orthogonal directions than chance (routing)  ·  "
        f"red = more aligned than chance.\n"
        f"Colour carries direction only — no opacity encoding, because a white cell "
        f"(gap≈0) and a faded cell (small ‖Δ‖) are indistinguishable\n"
        f"on a white ground. Contribution magnitudes ‖Δ‖ live in a separate figure, "
        f"<name>_magnitude.png.\n"
        f"{undef_txt}"
        f"Probe points are R8's (x_t = (1−t)·x₀ + t·a, {m['num_probe']} points), and " + when +
        f"{dead_txt}"
        f"{final_txt}"
        f"Counting a block as routing when its (cos − shuffle) {rule} is below −{gth:g} and its "
        f"‖Δ‖ is at least\n"
        f"{mfrac:g}× the global maximum:   {route_txt}   (out of {L} blocks each).\n"
        f"This figure claims only that the directional separation of conditional routing "
        f"changes across these checkpoints — not that it explains forgetting.")

    # 캡션 줄 수가 그림마다 다르다(회색 칸 안내·정지 경고가 붙었다 말았다 한다).
    # 위 여백을 고정하면 긴 캡션이 패널 제목을 덮으므로, 실제 줄 수에서 역산한다.
    fig_h = 18.0 if summary_fig else 15.0   # 크기 패널은 별도 그림으로 뺐다
    n_line = caption.count("\n") + 1
    top = min(0.86, max(0.60, 0.995 - n_line * 10 * 1.45 / 72 / fig_h - 0.045))

    n = len(keys)
    # 마지막 행은 에피소드 타임라인. 어떤 물리시간에 표본이 몇 개였는지를 이 행이 말한다.
    nrow = 5 if summary_fig else 4
    ratios = ([1.0, 1.0, 0.66, 0.55, 0.62] if summary_fig
              else [1.0, 1.0, 0.66, 0.62])
    fig = plt.figure(figsize=(4.1 * n + 3.4, fig_h))
    fig.suptitle(caption, fontsize=10, color=INK, y=0.995, linespacing=1.45)
    gs = fig.add_gridspec(nrow, n + 2, width_ratios=[1] * n + [0.075, 1.05],
                          height_ratios=ratios, hspace=0.40, wspace=0.20,
                          left=0.088, right=0.978, top=top, bottom=0.062)

    # ── a~f: 2×3 히트맵 ─────────────────────────────────────────────────────
    for ri, s in enumerate(SUBS):
        for ci, k in enumerate(keys):
            ax = fig.add_subplot(gs[ri, ci])
            bad = ~np.isfinite(gap[(k, s)])
            rgba = cmap(norm(np.where(bad, np.nan, gap[(k, s)])))
            rgba[bad] = to_rgb("#eceae5") + (1.0,)
            # ★ 불투명도로 크기를 나타내지 않는다. gap≈0(흰색)과 ‖Δ‖ 작음(투명)이
            #   흰 배경에서 똑같이 희게 보여 무엇 때문에 흰지 알 수 없기 때문이다.
            #   색은 **방향만** 말하고, 크기는 별도 그림(R9_full_magnitude.png)이 맡는다.
            rgba[..., 3] = 1.0
            ax.imshow(rgba, aspect="auto", origin="lower", interpolation="nearest",
                      extent=(-0.5, nt - 0.5, 0.5, L + 0.5))
            tix = np.unique(np.linspace(0, nt - 1, 5).round().astype(int))
            ax.set_xticks(tix)
            ax.set_xticklabels([f"{t_grid[i]:.2f}" for i in tix])
            ax.set_yticks(range(1, L + 1))
            ax.set_yticklabels([f"block {i}" for i in range(1, L + 1)] if ci == 0
                               else [""] * L, fontsize=8)
            ax.tick_params(colors=INK2, labelsize=8, length=3)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            if ri == len(SUBS) - 1:
                ax.set_xlabel("flow time  t", color=INK2, fontsize=8.5)
            tag = "abcdef"[ri * n + ci]
            head = f"{tag}   {titles[k]}" if ri == 0 else f"{tag}   {short[k]}"
            ax.set_title(head, fontsize=10, color=INK, pad=8, loc="left")
            if ci == 0:
                # 행이 무엇인지(attn / mlp)를 ylabel로 단다. fig.text로 좌표를 손으로 잡으면
                # 눈금 라벨 폭에 따라 캔버스 밖으로 밀려난다 — ylabel은 자동으로 피해 준다.
                ax.set_ylabel(SUB_TITLE[s].upper(), color=INK, fontsize=9.5,
                              fontweight="bold", labelpad=8)

    # 컬러바(색 = 방향) 위, 알파 램프(불투명도 = 크기) 아래. 둘 다 세로.
    cax = fig.add_subplot(gs[0:2, n])
    pos = cax.get_position()
    w, x = pos.width * 0.42, pos.x0 - 0.012
    h_c = pos.height * 0.62
    cax.set_position([x, pos.y0 + pos.height - h_c, w, h_c])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    # 눈금과 라벨을 컬러바 **왼쪽**에 둔다. 오른쪽에 두면 옆 단면 패널과 부딪힌다.
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.set_label("cos − shuffle   (cell colour)", color=INK2, fontsize=8)
    cb.ax.tick_params(colors=INK2, labelsize=7.5)
    # ★ 여기 끝을 그냥 "aligned"/"orthogonal"이라 쓰면 안 된다. 축은 cos이 아니라
    #   cos − shuffle 이고, 아래 끝은 cos=0(직교)이 아니라 "우연보다 훨씬 덜 정렬"이다.
    #   실제 cos은 0.4~0.7 대역이라 직교와는 거리가 멀다.
    cb.ax.text(0.5, 1.02, "more aligned\nthan chance", transform=cb.ax.transAxes,
               ha="center", va="bottom", fontsize=6.5, color=DIV_LO, linespacing=1.2)
    cb.ax.text(0.5, -0.02, "more orthogonal\nthan chance", transform=cb.ax.transAxes,
               ha="center", va="top", fontsize=6.5, color=DIV_HI, linespacing=1.2)
    cb.outline.set_edgecolor(GRID)

    over = "averaged over t and rollout step" if summary_fig else "averaged over t"

    # ── g/h: depth 단면 (히트맵과 y축 공유) ──────────────────────────────────
    rows = np.arange(1, L + 1)
    for ri, s in enumerate(SUBS):
        ax = fig.add_subplot(gs[ri, n + 1])
        for ki, k in enumerate(keys):
            prof = _nanmean(cos[(k, s)], axis=1)
            ax.plot(prof, rows, color=MODEL_COLORS[k], lw=2.0, marker="o", ms=4, zorder=4)
            ax.plot(_nanmean(shuf[(k, s)], axis=1), rows, color=MODEL_COLORS[k], lw=1.3,
                    ls="--", alpha=0.85, zorder=3)
            # 색만으로 식별하게 두지 않는다(relief 규칙). 곡선마다 다른 행에 라벨을 붙인다.
            # NaN 지점에는 라벨을 못 붙이므로 유효한 행 중에서 고른다.
            ok = np.flatnonzero(np.isfinite(prof))
            if not len(ok):
                continue
            r = ok[max(0, len(ok) - 1 - ki)]
            ax.annotate(short[k], (prof[r], rows[r]), textcoords="offset points",
                        xytext=(5, 3), fontsize=8, color=MODEL_COLORS[k], fontweight="bold")
        ax.axvline(0.0, color=INK2, lw=0.9, ls=":", zorder=2)
        _style(ax)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.7, axis="x")
        ax.set_ylim(0.5, L + 0.5)
        ax.set_yticks(rows)
        ax.set_yticklabels([f"block {i}" for i in rows], fontsize=8)
        xl = ax.get_xlim()
        ax.set_xlim(xl[0], xl[1] + 0.22 * (xl[1] - xl[0]))
        ax.set_xlabel(f"cos   ({s}, {over})", color=INK2, fontsize=8.5)
        ax.set_title(f"{'gh'[ri]}   depth cross-section — {s}", fontsize=10, color=INK,
                     pad=8, loc="left")

    # ── i/j: flow-time 단면 ─────────────────────────────────────────────────
    row2 = gs[2, :].subgridspec(1, 4, wspace=0.30)
    for i, s in enumerate(SUBS):
        ax = fig.add_subplot(row2[0, i])
        for ki, k in enumerate(keys):
            prof = _nanmean(cos[(k, s)], axis=0)
            ax.plot(t_grid, prof, color=MODEL_COLORS[k], lw=2.0, marker="o", ms=3.5, zorder=4)
            ax.plot(t_grid, _nanmean(shuf[(k, s)], axis=0), color=MODEL_COLORS[k], lw=1.3,
                    ls="--", alpha=0.85, zorder=3)
            j = nt - 1 - 3 * ki
            if np.isfinite(prof[j]):
                ax.annotate(short[k], (t_grid[j], prof[j]), textcoords="offset points",
                            xytext=(0, 7), fontsize=8, color=MODEL_COLORS[k],
                            fontweight="bold", ha="center")
        ax.axhline(0.0, color=INK2, lw=0.9, ls=":", zorder=2)
        _style(ax)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
        ax.set_xlim(-0.02, 1.08)
        ax.set_xlabel("flow time  t      (early = mode selection · late = refinement)",
                      color=INK2, fontsize=8)
        ax.set_ylabel(f"cos   ({s}, averaged over blocks)", color=INK2, fontsize=8.5)
        ax.set_title(f"{'ij'[i]}   flow-stage cross-section — {s}", fontsize=10, color=INK,
                     pad=8, loc="left")

    # ── k/l: 물리시간 단면. 요약본에만 (스텝별 판은 그 축이 한 점이라 무의미) ──
    if summary_fig:
        for i, s in enumerate(SUBS):
            ax = fig.add_subplot(row2[0, 2 + i])
            _shade_dead(ax, obs_steps, ctx["dead_frac"])
            for ki, k in enumerate(keys):
                prof = _nanmean(ctx["cos_s"][(k, s)], axis=(1, 2))
                ax.plot(obs_steps, prof, color=MODEL_COLORS[k], lw=2.0, marker="o", ms=4,
                        zorder=4)
                ax.plot(obs_steps, _nanmean(ctx["shuf_s"][(k, s)], axis=(1, 2)),
                        color=MODEL_COLORS[k], lw=1.3, ls="--", alpha=0.85, zorder=3)
                j = max(0, ns - 1 - ki)
                if np.isfinite(prof[j]):
                    ax.annotate(short[k], (obs_steps[j], prof[j]), textcoords="offset points",
                                xytext=(0, 7), fontsize=8, color=MODEL_COLORS[k],
                                fontweight="bold", ha="center")
            ax.axhline(0.0, color=INK2, lw=0.9, ls=":", zorder=2)
            _style(ax)
            ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
            ax.set_xticks(obs_steps)
            ax.set_xlabel("rollout step      (physical time the observation was taken)",
                          color=INK2, fontsize=8)
            ax.set_ylabel(f"cos   ({s}, averaged over blocks and t)", color=INK2, fontsize=8.5)
            ax.set_title(f"{'kl'[i]}   physical-time cross-section — {s}", fontsize=10,
                         color=INK, pad=8, loc="left")

    

    # ── q/r/s: 최종 출력단 — R9의 블록 측정이 닿지 않는 7번째 조건 주입 지점 ──
    # 여기가 핵심이다. 블록 안에서 라우팅이 살아 있어도, 조건이 만든 방향 차이가
    # 512→7 사영의 null space(505차원)로 들어가면 액션에는 아무것도 안 남는다.
    # 왼쪽 스트립 = 모델별로 512차원 증분과 7차원 사영을 위아래로 놓은 것.
    fin_gs = gs[nrow - 2, :].subgridspec(1, n + 2, wspace=0.30)
    for ci, k in enumerate(keys):
        ax = fig.add_subplot(fin_gs[0, ci])
        arr = np.stack([d["fgap"][(k, "7")], d["fgap"][(k, "512")]])      # 아래=7, 위=512
        bad = ~np.isfinite(arr)
        rgba = cmap(norm(np.where(bad, np.nan, arr)))
        rgba[bad] = to_rgb("#eceae5") + (1.0,)
        # ★ 알파는 최종층 안에서만 정규화한다. 512차원 증분과 7차원 사영은 노름의
        #   스케일이 애초에 다르고(사영은 차원이 1/73), 블록 Δ와도 다르다. 전역
        #   최대로 재면 이 행 전체가 통째로 흐려져 아무것도 안 보인다.
        rgba[..., 3] = 1.0                       # 위와 같은 이유로 알파를 쓰지 않는다
        ax.imshow(rgba, aspect="auto", origin="lower", interpolation="nearest",
                  extent=(-0.5, nt - 0.5, -0.5, 1.5))
        tix = np.unique(np.linspace(0, nt - 1, 5).round().astype(int))
        ax.set_xticks(tix)
        ax.set_xticklabels([f"{t_grid[i]:.2f}" for i in tix])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["W·Δ  (7-d)", "Δ  (512-d)"] if ci == 0 else ["", ""], fontsize=8)
        ax.tick_params(colors=INK2, labelsize=8, length=3)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_xlabel("flow time  t", color=INK2, fontsize=8.5)
        head = f"q   final layer — {short[k]}" if ci == 0 else f"    {short[k]}"
        ax.set_title(head, fontsize=10, color=INK, pad=8, loc="left")

    # 오른쪽 두 패널: 512차원과 7차원을 각각 블록 단면과 같은 규약으로(실선 cos,
    # 점선 shuffle) 그린다. 두 패널을 견주는 것이 "사영이 버렸나"의 판정이다.
    for wi, w in enumerate(FINALS):
        ax = fig.add_subplot(fin_gs[0, n + wi])
        for ki, k in enumerate(keys):
            prof = d["fcos"][(k, w)]
            ax.plot(t_grid, prof, color=MODEL_COLORS[k], lw=2.0, marker="o", ms=3.5, zorder=4)
            ax.plot(t_grid, d["fshuf"][(k, w)], color=MODEL_COLORS[k], lw=1.3, ls="--",
                    alpha=0.85, zorder=3)
            j = nt - 1 - 3 * ki
            if np.isfinite(prof[j]):
                ax.annotate(short[k], (t_grid[j], prof[j]), textcoords="offset points",
                            xytext=(0, 7), fontsize=8, color=MODEL_COLORS[k],
                            fontweight="bold", ha="center")
        ax.axhline(0.0, color=INK2, lw=0.9, ls=":", zorder=2)
        _style(ax)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
        ax.set_xlim(-0.02, 1.08)
        ax.set_xlabel("flow time  t", color=INK2, fontsize=8.5)
        ax.set_ylabel(f"cos   ({'512-d Δ' if w == '512' else '7-d W·Δ'})",
                      color=INK2, fontsize=8.5)
        ax.set_title(f"{'rs'[wi]}   {FINAL_TITLE[w]}", fontsize=9.5, color=INK,
                     pad=8, loc="left")

    # ── t: 에피소드 타임라인 (마지막 행 전체) ───────────────────────────────
    _draw_timeline(fig.add_subplot(gs[nrow - 1, :]), ctx, si)

    handles = [Line2D([0], [0], color=MODEL_COLORS[k], lw=2.4, marker="o", ms=5,
                      label=titles[k]) for k in keys]
    handles += [Line2D([0], [0], color=INK2, lw=1.4, ls="--",
                       label="shuffle baseline (same condition, different probe)")]
    handles += [Line2D([0], [0], color=INK2, lw=0, marker="o", ms=6, mfc="white", mec=INK2,
                       label="capture point dropped (that rollout had already ended)")]
    if (ctx["dead_frac"] > 0.5).any():
        handles += [Patch(facecolor=GRID, alpha=0.45,
                          label="fewer than half the episodes still running")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 0.004))

    fig.savefig(out, dpi=170, facecolor="white")
    if summary_fig:
        fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return out


def _write_tables(ctx: dict) -> None:
    """콘솔 표와 summary.json. 요약(물리시간 평균)과 스텝별 값을 모두 남긴다."""
    m, keys, short, L = ctx["m"], ctx["keys"], ctx["short"], ctx["L"]
    obs_steps, diag = ctx["obs_steps"], ctx["m"].get("diagnostics", {})
    d = _slice(ctx, None)
    cos, shuf, gap, mag = d["cos"], d["shuf"], d["gap"], d["mag"]

    # ── 롤아웃 길이: 에피소드마다 다르므로 눈으로 확인할 수 있게 먼저 찍는다 ──
    eps = m.get("episodes", {})
    if eps:
        print("")
        print(f"{'rollout':<10}{'episode':<9}{'live_len':>10}{'demo_len':>10}{'env_end':>9}"
              f"   {'why it stopped'}")
        print("-" * 76)
        for ci, tag in (("c0", f"c₀ task {m['task_a']}"), ("c1", f"c₁ task {m['task_b']}")):
            for it in eps.get(ci, []):
                print(f"{tag:<10}{it['episode']:<9}{it['live_len']:>10}"
                      f"{str(it['demo_len']):>10}{it['env_end']:>9}   {it['reason']}")
        print(f"{'usable pairs per rollout step':<40}" +
              "  ".join(f"{int(s)}:{int(c)}" for s, c in zip(obs_steps, ctx["n_live"])) +
              f"   (of {ctx['n_ep']})")

    print("")
    print(f"{'model':<12}{'sub':<6}{'⟨cos⟩':>9}{'⟨shuffle⟩':>11}{'gap':>9}"
          f"{'route':>9}{'⟨‖Δ‖⟩':>12}{'excl%':>8}{'undef%':>8}")
    print("-" * 84)
    summary = {}
    for k in keys:
        summary[k] = {"title": ctx["titles"][k], "ckpt": diag.get(k, {}).get("ckpt", ""),
                      "subs": {}}
        for s in SUBS:
            ks = (k, s)
            ex = diag.get(k, {}).get("excluded_frac", {}).get(s, float("nan")) * 100
            ud = diag.get(k, {}).get("cos_undefined_frac", {}).get(s, float("nan")) * 100
            print(f"{short[k]:<12}{s:<6}{_nanmean(cos[ks]):>+9.3f}"
                  f"{_nanmean(shuf[ks]):>+11.3f}{_nanmean(gap[ks]):>+9.3f}"
                  f"{d['n_route'][ks]:>6}/{L:<2}{_nanmean(mag[ks]):>12.3e}{ex:>8.2f}{ud:>8.2f}")
            summary[k]["subs"][s] = {
                "cos_by_layer": _nanmean(cos[ks], axis=1).tolist(),
                "cos_by_t": _nanmean(cos[ks], axis=0).tolist(),
                "cos_by_rollout_step": _nanmean(ctx["cos_s"][ks], axis=(1, 2)).tolist(),
                "shuffle_by_layer": _nanmean(shuf[ks], axis=1).tolist(),
                "shuffle_by_t": _nanmean(shuf[ks], axis=0).tolist(),
                "shuffle_by_rollout_step": _nanmean(ctx["shuf_s"][ks], axis=(1, 2)).tolist(),
                "gap_by_layer": d["gap_layer"][ks].tolist(),
                "M_by_layer": d["mag_layer"][ks].tolist(),
                "M_by_rollout_step": _nanmean(ctx["mag_s"][ks], axis=(1, 2)).tolist(),
                "routing_blocks": [int(i + 1) for i in np.flatnonzero(d["routes"][ks])],
                "n_routing_blocks": d["n_route"][ks],
                # 스텝별 판정도 남긴다 — 그림 9장이 말하는 것을 숫자로 따라갈 수 있게.
                "n_routing_blocks_by_rollout_step": [
                    _slice(ctx, i)["n_route"][ks] for i in range(ctx["ns"])],
                "cos_mean": float(_nanmean(cos[ks])),
                "shuffle_mean": float(_nanmean(shuf[ks])),
                "gap_mean": float(_nanmean(gap[ks])),
                "M_mean": float(_nanmean(mag[ks])),
                "excluded_frac": diag.get(k, {}).get("excluded_frac", {}).get(s, None),
                "cos_undefined_frac": diag.get(k, {}).get("cos_undefined_frac", {}).get(s, None),
            }
    # ── 최종 출력단: 블록 측정이 닿지 않는 곳 ──────────────────────────────
    print("")
    print(f"{'model':<12}{'final':<8}{'⟨cos⟩':>9}{'⟨shuffle⟩':>11}{'gap':>9}{'⟨‖Δ‖⟩':>12}"
          f"   해석")
    print("-" * 84)
    for k in keys:
        for w in FINALS:
            g = float(_nanmean(d["fgap"][(k, w)]))
            note = ("갈림(라우팅)" if g < -0.05 else "정렬(라우팅 없음)" if g > 0.05 else "우연 수준")
            print(f"{short[k]:<12}{('Δ 512-d' if w == '512' else 'W·Δ 7-d'):<8}"
                  f"{_nanmean(d['fcos'][(k, w)]):>+9.3f}{_nanmean(d['fshuf'][(k, w)]):>+11.3f}"
                  f"{g:>+9.3f}{_nanmean(d['fmag'][(k, w)]):>12.3e}   {note}")
        summary[k]["final"] = {
            w: {"cos_by_t": _nanmean(d["fcos"][(k, w)], axis=0).tolist()
                if d["fcos"][(k, w)].ndim > 1 else d["fcos"][(k, w)].tolist(),
                "cos_mean": float(_nanmean(d["fcos"][(k, w)])),
                "shuffle_mean": float(_nanmean(d["fshuf"][(k, w)])),
                "gap_mean": float(_nanmean(d["fgap"][(k, w)])),
                "M_mean": float(_nanmean(d["fmag"][(k, w)]))} for w in FINALS}
        summary[k]["final"]["adaln_mod_cos"] = float(_nanmean(d["fmod"][k]))
        print(f"{short[k]:<12}{'AdaLN':<8}{_nanmean(d['fmod'][k]):>+9.3f}"
              f"{'—':>11}{'—':>9}{'—':>12}   주입 벡터 [shift|scale] 자체의 방향")
    print("")
    ctx["cache"].with_suffix(".summary.json").write_text(json.dumps({
        "t_grid": ctx["t_grid"].tolist(), "obs_steps": obs_steps.tolist(),
        "obs_dead_frac_by_step": ctx["dead_frac"].tolist(),
        "episodes": eps, "usable_pairs_by_step": [int(v) for v in ctx["n_live"]],
        "n_episodes": ctx["n_ep"], "exclude_dead_obs": m.get("exclude_dead_obs", True),
        "n_block": L, "M_max_all_models": ctx["m_max"],
        "route_rule": {"gap_thresh": ctx["gth"], "mag_frac": ctx["mfrac"]},
        "models": summary, "meta": m}, indent=2, ensure_ascii=False))


def plot_r9(cache: str | Path, out_png: str | Path | None = None,
            per_step: bool = True) -> None:
    """요약 한 장 + rollout step마다 한 장.

    스텝별 판은 `<cache>_step0000.png` … 로 나가고, 색 스케일·알파 기준·routing 규칙을
    요약본과 **공유**한다. 판마다 자기 범위로 정규화하면 9장을 비교할 수 없기 때문이다.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ModuleNotFoundError:
        print("matplotlib 없음 -> 그림 생략")
        return

    cache = Path(cache)
    ctx = _prepare(cache)
    _write_tables(ctx)

    out = Path(out_png) if out_png else cache.with_suffix(".png")
    _draw(ctx, None, out)
    print(f"saved figure -> {out}  (+ .pdf)")
    mg = _draw_magnitude(ctx, out.with_name(out.stem + "_magnitude.png"))
    print(f"saved figure -> {mg}  (+ .pdf)   크기 전용")
    if not per_step:
        return
    stem = out.with_suffix("")
    for si, step in enumerate(ctx["obs_steps"]):
        # 살아있는 짝이 하나도 없는 물리시간은 그릴 것이 없다(전 칸이 NaN). 통째로 회색인
        # 판을 내놓으면 "여기서 라우팅이 사라졌다"로 오독되므로 건너뛰고 이유를 남긴다.
        if not int(ctx["n_live"][si]):
            print(f"skipped step {int(step):>4}: 살아있는 롤아웃이 없다 "
                  f"(모든 에피소드가 이 물리시간 전에 끝났다)")
            continue
        p = _draw(ctx, si, Path(f"{stem}_step{int(step):04d}.png"))
        print(f"saved figure -> {p}   ({int(ctx['n_live'][si])}/{ctx['n_ep']} episodes live)")


# ═════════════════════════════════════════════════════════════════════════════
#  메인 (R8과 같은 순서)
# ═════════════════════════════════════════════════════════════════════════════
@parser.wrap()
def main(cfg: R9Config):
    cfg.validate()
    cfg.save_checkpoint = False
    logging.info(pformat(cfg.to_dict()))
    if not cfg.ckpt_root:
        raise SystemExit("--ckpt_root 가 필요하다 (예: outputs/E0/libero_spatial/seed_42/lam0).")
    run_dir = Path(cfg.out_root) / (cfg.run_tag or "run")
    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.seed is not None:
        set_seed(cfg.seed)

    cache = run_dir / cache_name(cfg)
    if cache.exists() and not cfg.recompute:
        logging.info(f"[R9] 캐시 재사용: {cache}")
    else:
        cache = run_probe(cfg, run_dir)
    if not cfg.no_plot:
        plot_r9(cache, per_step=cfg.per_step_figs)


if __name__ == "__main__":
    init_logging()
    if "--plot_only" in sys.argv:
        kv = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
        # --per_step=0 이면 요약 한 장만 낸다 (스텝별 9장을 건너뛴다).
        per_step = kv.get("per_step", "1").lower() not in ("0", "false", "no")
        if "cache" in kv:
            plot_r9(kv["cache"], per_step=per_step)
        elif "run_dir" in kv:
            found = sorted(Path(kv["run_dir"]).glob("R9_*.npz"))
            if not found:
                raise SystemExit(f"[R9] npz 캐시가 없다: {kv['run_dir']}")
            for c in found:
                plot_r9(c, per_step=per_step)
        else:
            raise SystemExit("--plot_only 에는 --run_dir=<...> 또는 --cache=<...npz> 가 필요하다")
    else:
        main()
