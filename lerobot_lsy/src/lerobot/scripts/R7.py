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

"""R7 — 조건이 다르면 flow 경로가 갈라지는가: 생성 경로의 조건 붕괴.

무엇을 묻는가
    R6은 조건을 α로 끌고 가도 **롤아웃 궤적**이 한 다발로 뭉친다는 것을 보였다.
    R7은 같은 질문을 한 단계 안쪽, **행동을 만들어 내는 생성 과정 자체**에서 묻는다.

      x_0 ~ N(0, I)  ──[ v_θ(x_t, t, c) 로 t=0→1 Euler 적분 ]──>  x_1 = action chunk

    조건 c만 c₀ ↔ c₁로 바꾸고 노이즈 x_0를 고정한 채 두 경로 다발을 겹쳐 그린다.

      건강한 모델   두 다발이 t가 커질수록 벌어져 서로 다른 종점으로 간다
      붕괴한 모델   c와 무관하게 한 다발 — 조건이 속도장을 바꾸지 못한다

무엇을 c라고 부르는가 — 조건 **전체**다 (cond_mode=full, 기본)
    이 정책이 받는 조건 벡터는 세 조각이 이어 붙은 것이다.

        global_cond (2576) = [ CLIP 언어 512 | 로봇 상태 16 | DINOv2 이미지 2048 ]

    c₀ = global_cond(task A 장면, task A 지시문)
    c₁ = global_cond(task B 장면, task B 지시문)

    ★ 언어 512만 바꾸면 안 된다. 그건 "언어에 무감각해진다"는 **다른 논문의 명제**이고,
      실제로 이 모델들의 언어 채널은 살아 있다(먼 지시문에는 반응한다). 이 논문의 주장은
      조건 **전체**에 무감각해진다는 것이므로, 장면과 지시문을 함께 바꾼 상황 전체를
      조건으로 놓는다. R6이 full global_cond를 α로 끌고 간 것과 같은 조작이다.
      cond_mode=language는 "조건의 어느 부분을 무시하는가"를 보는 부록 분석으로만 쓴다.

방어 지점 세 가지 (리뷰어가 때릴 순서대로)

  1. "사영이 아티팩트 아니냐"
     x_t는 (horizon 16) × (action 7) = 112차원이다. 사영 기저는 **데모 데이터에서
     1회 고정**되고 모든 조건·모든 체크포인트·모든 패널에 **동일하게** 적용된다.
     조건마다/모델마다 PCA를 다시 맞추는 일은 없다. 두 가지를 준비했다.

       --basis=demo_diff (기본, 분리 우선)
           1축 = normalize(mean(task B 데모 청크) − mean(task A 데모 청크))
           2축 = 두 task 데모를 합친 집합의 최대 분산 방향 중 1축과 직교인 성분.
       --basis=ee (해석 우선)
           청크 첫 스텝의 EE 변위 두 축. {Δx, Δy, Δz} 중 데모 통계로 두 task가 가장
           잘 갈리는 두 축을 고른다. 캡션에 "이 평면은 EE 변위 평면"이라고 쓸 수 있다.

     좌표계는 **정규화 액션 공간**([-1,1], MIN_MAX)이다. flow ODE가 실제로 도는 공간이
     거기이고, 체크포인트들의 정규화 통계가 bit 단위로 같다는 것을 실행 시 검사한다
     (assert_shared_norm). 다르면 즉시 죽는다 — 조용히 사과가 오렌지와 비교되지 않는다.

  2. "입력이 달라서 갈라진 것 아니냐"
     x_0 100개를 시드 고정으로 한 번 뽑아 저장하고 **모든 패널이 같은 100개**를 쓴다.
     그래서 다발 간 차이에서 노이즈 요인이 소거되고, t=0에서 발산 곡선이 정확히 0에서
     출발한다. 경로는 첫 action chunk의 open-loop 적분 한 번이라 롤아웃이 아니고,
     "환경 피드백이 궤적을 오염시켰다"는 반론이 원천적으로 성립하지 않는다.

  3. "한 모델이 두 task를 담으면 원래 합쳐지는 것 아니냐"
     joint 통제군이 막는다. joint과 CL은 **같은 사전학습에서 출발해, 같은 데이터를,
     같은 스텝 수만큼** 본다. 다른 것은 섞어서 봤는가 순서대로 봤는가 하나뿐이다.
     그래서 둘의 차이는 그 하나에 귀속된다. pretrain은 "학습 전" 기준선이다.

     ★ 개별 FT(--models 에 ft_a/ft_b)는 기본에서 뺀다. 학습 중 조건이 상수였던 모델에
       다른 task의 조건을 넣는 것은 정의상 OOD이고, 이 논문이 다루는 명제가 아니다.

패널
    위줄  모델마다 한 패널. c₀ 빨강 다발 vs c₁ 파랑 다발, 축은 전부 공유.
          색 = 조건, 명도 = flow 시간(t=0 옅음 → t=1 진함), 종점에 점,
          회색 ×/+ = 데모 액션 청크의 사영("어디로 갔어야 했는가"의 참조점)
    d  종점 분리   d_between = ⟨‖x₁(c₁) − x₁(c₀)‖⟩ (같은 x₀ 짝끼리)
                  d_within  = 같은 조건에서 노이즈만 다른 종점 쌍의 평균 거리
    e  발산 시점   ‖x_t(c₁) − x_t(c₀)‖ 를 flow 시간의 함수로. 짝지었으므로 0에서 출발한다.
    f  분리비      d_between / d_within. 1보다 작으면 조건이 만드는 차이가 노이즈가
                  만드는 차이보다도 작다는 뜻이다.

이 스크립트가 하는 일
    --mode=trace       (기본) 위 그림/숫자를 만든다. 학습하지 않는다.
    --mode=train_joint joint 통제군을 학습한다(체크포인트가 없을 때 한 번).
    --plot_only        캐시 npz로 그림만 다시 그린다.

사용 예
    python R7.py --mode=train_joint --policy.path=<pretrain> --output_dir=outputs/R7/joint01 ...
    python R7.py --policy.path=<any ckpt> --ckpt_root=outputs/E0/libero_spatial/seed_42/lam0 \
        --pretrain_ckpt=... --joint_ckpt=... --cond_mode=full --run_tag=seq_seed42
    python R7.py --plot_only --run_dir=outputs/R7/seq_seed42
"""

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from termcolor import colored
from torch.amp import GradScaler

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle
from lerobot.envs.utils import preprocess_observation
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint
from lerobot.utils.utils import get_safe_torch_device, init_logging

# 조건 인코딩은 R1의 것을 그대로 쓴다. select_action은 내부 큐가 조건 인코딩부터 실행까지를
# 감싸고 있어 조건만 갈아끼울 틈이 없고, 애초에 여기서는 실행이 아니라 생성 과정을 본다.
from lerobot.scripts.E0 import episode_sampler, split_episodes, to_device, update_policy
from lerobot.scripts.R1 import encode_global_cond
from lerobot.scripts.R3 import make_probe_env, stage_ckpt

# ── 색 ────────────────────────────────────────────────────────────────────────
# 조건 = hue. c₀ 빨강 / c₁ 파랑 (R6의 데모 색과 같은 값 — 논문 안에서 "빨강=task 0"이 유지된다).
# 시간 = 명도. t=0 옅음 -> t=1 진함. 각 hue마다 3점 램프를 만들어 보간한다.
C0_RAMP = ["#f3c9cb", "#dc6a70", "#8e181e"]   # 빨강: 옅음 -> 진함
C1_RAMP = ["#c3dbf5", "#5b9ae4", "#12457f"]   # 파랑: 옅음 -> 진함
C0_INK, C1_INK = "#c9252d", "#2a78d6"
DEMO_GRAY = "#9a9894"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
# 모델 세 그룹의 막대 색 (순서형 아님 -> hue로 가른다). 개별FT=회녹, joint=청록, CL=보라.
# 모델 hue. 문서 팔레트 슬롯이고 all-pairs 검증 통과(worst normal ΔE 27.6, CVD ΔE 9.2;
# OKLab×100, Machado 2009 severity 1.0). aqua는 흰 배경 대비 2.82:1이라 3:1을 못 넘으므로
# relief 규칙에 따라 막대에 값 라벨을 붙이고 곡선에는 직접 라벨을 단다.
MODEL_COLORS = {"ft_pair": "#eb6834", "joint": "#1baf7a", "cl": "#4a3aa7",
                "pretrain": "#8a8a86", "ft_a": "#2a78d6", "ft_b": "#eb6834"}


# ═════════════════════════════════════════════════════════════════════════════
#  설정
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class R7Config(TrainPipelineConfig):
    """train.py 인자 전부 + flow 경로 추적 인자. trace 모드에서 학습 인자는 무시된다."""

    mode: str = "trace"                     # trace | train_joint

    # ── 무엇을 바꾸는가 ──────────────────────────────────────────────────────
    # full     조건 벡터 **전체**를 바꾼다. c₀ = global_cond(task A 장면, task A 지시문),
    #          c₁ = global_cond(task B 장면, task B 지시문). 주장이 "조건 전체에 무감각"
    #          이므로 이쪽이 본 실험이다. 장면이 함께 바뀌므로 s를 따로 고정하지 않는다.
    # language 장면을 한 task에 고정하고 지시문만 바꾼다. 부록용 — "조건의 어느 부분을
    #          무시하는가"를 가르는 보조 분석이지, 주 주장이 아니다.
    #          ★ 언어만 보는 것은 다른 논문의 명제(언어 무감각)라 주 실험이 될 수 없다.
    cond_mode: str = "full"                 # full | language

    # ── 어떤 체크포인트를 볼 것인가 ──────────────────────────────────────────
    ckpt_root: str = ""                     # 순차 CL 트리 (task_0 = FT_A, task_1 = CL)
    pretrain_ckpt: str = ""                 # 사전학습 = "학습 전" 기준선
    ft1_ckpt: str = ""                      # task B만 단독 학습한 체크포인트
    joint_ckpt: str = ""                    # task A+B 동시 학습 체크포인트 (통제군)
    # 그림에 넣을 모델. 개별 FT는 기본에서 뺀다 — 학습된 적 없는 조건을 넣는 것이라
    # 정의상 OOD이고, 이 논문이 다루는 명제가 아니다.
    # ft_a와 ft_b가 함께 있으면 **한 패널**로 합쳐 그린다: FT_A에 c₀, FT_B에 c₁.
    # 각 모델이 자기가 학습한 조건만 받으므로 OOD가 없고, 두 다발의 간격이 곧
    # "제대로 학습되면 조건이 갈라 놓는 거리" = 분리의 상한선이 된다.
    models: str = "ft_a,ft_b,joint,cl"     # pretrain | ft_a | ft_b | joint | cl
    task_a: int = 0                         # 먼저 배운 task (조건 c₀)
    task_b: int = 1                         # 나중에 배운 task (조건 c₁)

    # ── 짝짓기 ───────────────────────────────────────────────────────────────
    num_noise: int = 100                    # 모든 패널이 공유하는 x₀ 샘플 수
    noise_seed: int = 20260813              # x₀를 뽑는 시드. 이 값이 그림 전체를 재현한다
    num_obs: int = 5                        # 기준 관측 s의 개수 (초기 상태 인덱스 0..N-1)
    obs_task: int = 0                       # 기준 관측을 어느 task 환경에서 딸 것인가 (s⁰)
    settle_steps: int = 5                   # R3/R6과 같은 값. 물체를 테이블에 내려앉힌다

    # ── flow 적분 ────────────────────────────────────────────────────────────
    # 정책의 실제 추론값(config.num_inference_steps=100)을 기본으로 쓴다. 스텝 수를 바꾸면
    # 종점이 미세하게 달라지므로 "실제 추론과 같은 경로"를 보고 싶으면 건드리지 않는다.
    flow_steps: int = 0                     # 0 = 정책 config의 num_inference_steps
    trace_stride: int = 1                   # 경로를 몇 스텝마다 기록할지 (1 = 전부)

    # ── 사영 ─────────────────────────────────────────────────────────────────
    basis: str = "demo_diff"                # demo_diff | ee
    # ★ 청크 16스텝 중 **실제로 실행되는 구간만** 재고 그린다.
    #   generate_actions가 actions[:, n_obs_steps-1 : n_obs_steps-1+n_action_steps] 만 로봇에
    #   내보내고(=index 1..8), index 0은 t-1이라 버리며 9..15는 미리보기라 버린다. 버려지는
    #   스텝까지 거리에 넣으면 로봇이 하지도 않는 행동으로 "조건 민감도"를 재게 된다.
    #   auto = 정책 config에서 유도(권장) · "a:b" = 직접 지정 · "all" = 16스텝 전부(옛 동작)
    exec_slice: str = "auto"
    basis_episodes: int = 10                # 기저를 만들 때 쓰는 task당 데모 에피소드 수
    demo_scatter_episodes: int = 5          # 그림에 회색 ×로 깔 데모 에피소드 수

    # ── 그림 ─────────────────────────────────────────────────────────────────
    draw_noise: int = 24                    # 한 패널에 그릴 경로 수 (숫자는 전부로 낸다)
    draw_obs: int = 0                       # 그림에 쓸 기준 관측 인덱스

    # ── joint 학습 (mode=train_joint) ────────────────────────────────────────
    # 두 task 로더를 스텝마다 번갈아 먹인다. joint_steps=10000, batch=32면 task당 5000스텝으로
    # 순차 학습(태스크당 5000스텝)과 **task당 노출량**이 맞는다.
    joint_steps: int = 10000
    holdout_episodes: int = 5               # E0와 같은 분할. 뒤 5 에피소드는 학습에서 제외

    # ── 데이터 / 환경 ────────────────────────────────────────────────────────
    dataset_prefix: str = "continuallearning/libero_spatial_image_task_"
    env_task_prefix: str = "Libero_Spatial_Task_"
    probe_task: int = 0                     # make_probe_env가 읽는다. obs_task로 덮어쓴다
    max_steps: int = 50                     # make_probe_env의 TimeLimit용. 롤아웃은 안 한다

    # ── 출력 / 제어 ──────────────────────────────────────────────────────────
    out_root: str = "outputs/R7"
    run_tag: str = ""
    # 그림 파일명에 끼울 꼬리표. ""이면 R7_full_demo_diff.png, "ER"이면
    # R7_ER_full_demo_diff.png. npz 캐시와 summary.json은 이름을 그대로 둔다 —
    # 캐시 재사용 로직(cache_name)이 그 이름에 걸려 있어서 건드리면 다시 적분한다.
    plot_tag: str = ""
    recompute: bool = False
    no_plot: bool = False

    def validate(self):
        """R1/R3/R6과 같은 이유로 output_dir 존재 검사만 우회한다(캐시 재사용).

        단 mode=train_joint에서는 그 검사가 산출물 덮어쓰기를 막는 안전장치이므로 살린다.
        """
        out = self.output_dir
        if self.mode != "train_joint" and isinstance(out, Path) and out.is_dir():
            self.output_dir = None
            super().validate()
            self.output_dir = out
        else:
            super().validate()


# ═════════════════════════════════════════════════════════════════════════════
#  공통 유틸
# ═════════════════════════════════════════════════════════════════════════════
def task_text(prefix: str, task: int) -> str:
    meta = LeRobotDatasetMetadata(f"{prefix}{task}")
    return str(list(meta.tasks.values())[0])


def load_policy_at(cfg: R7Config, ckpt: str | Path, ds_meta, device):
    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"체크포인트가 없다: {ckpt}")
    pcfg = PreTrainedConfig.from_pretrained(ckpt)
    pcfg.pretrained_path = ckpt
    pcfg.device = cfg.policy.device
    policy = make_policy(cfg=pcfg, ds_meta=ds_meta)
    policy.eval()
    assert not policy.training, "policy가 eval 모드가 아니다"
    return policy


def norm_stats(policy) -> dict[str, np.ndarray]:
    """액션 MIN_MAX 통계. 모델 간 공통 좌표계인지 검사하는 데 쓴다."""
    buf = policy.unnormalize_outputs.buffer_action
    return {"min": buf["min"].detach().cpu().numpy().copy(),
            "max": buf["max"].detach().cpu().numpy().copy()}


def assert_shared_norm(ref: dict[str, np.ndarray], cur: dict[str, np.ndarray], name: str) -> None:
    """네 체크포인트가 같은 정규화 통계를 쓰는지 확인한다.

    ★ 이게 깨지면 정규화 공간이 모델마다 다른 좌표계가 되어 "겹친다/갈라진다"가 전부
      무의미해진다. 조용히 넘어가지 않고 여기서 죽는다.
    """
    for k in ("min", "max"):
        if not np.allclose(ref[k], cur[k], rtol=0, atol=0):
            raise SystemExit(
                f"[R7] {name}의 액션 정규화 통계가 기준과 다르다 ({k}).\n"
                f"      정규화 공간이 모델마다 다른 좌표계가 되므로 사영 비교가 성립하지 않는다.\n"
                f"      기준={ref[k]}\n      {name}={cur[k]}")


def minmax_normalize(a: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """원 단위 액션 -> [-1, 1]. normalize.py의 MIN_MAX 분기와 **같은 식**(1e-8 포함)."""
    lo, hi = stats["min"], stats["max"]
    return 2.0 * ((a - lo) / (hi - lo + 1e-8)) - 1.0


# ═════════════════════════════════════════════════════════════════════════════
#  데모 액션 청크 — 사영 기저와 참조점의 재료
# ═════════════════════════════════════════════════════════════════════════════
def demo_chunks(cfg: R7Config, policy_cfg, task: int, n_episodes: int) -> np.ndarray:
    """데모 에피소드의 action 청크를 (N, horizon, 7)로 모은다.

    ★ ds[i]를 부르지 않는다. __getitem__은 비디오까지 디코딩하는데 여기서 필요한 것은
      action뿐이라 수천 프레임을 디코딩할 이유가 없다. 대신 action 열을 통째로 읽고
      _get_query_indices와 **같은 규칙**(에피소드 경계에서 잘라 복제)으로 청크를 만든다.
      delta는 정책 config의 action_delta_indices([-1, 0, ..., 14])이므로 학습이 본 것과
      같은 모양·같은 정렬의 청크다. x₁(생성된 청크)과 같은 공간에 놓이는 이유가 이것이다.
    """
    repo = f"{cfg.dataset_prefix}{task}"
    ds = LeRobotDataset(repo)
    delta = np.asarray(policy_cfg.action_delta_indices, dtype=np.int64)
    # ★ hf_dataset[...]나 with_format("numpy")를 거치면 안 된다. 포맷터가 열 전체를
    #   텐서화하려다 비디오 열 때문에 torchvision VideoReader를 부른다. arrow 테이블에서
    #   action 열만 직접 꺼내면 그 경로를 아예 타지 않는다.
    acts = np.asarray(ds.hf_dataset.data.column("action").to_pylist(), dtype=np.float32)
    n_ep = min(n_episodes, int(ds.meta.total_episodes))
    out = []
    for ep in range(n_ep):
        i0 = int(ds.episode_data_index["from"][ep])
        i1 = int(ds.episode_data_index["to"][ep])
        idx = np.clip(np.arange(i0, i1)[:, None] + delta[None, :], i0, i1 - 1)
        out.append(acts[idx])
    del ds
    arr = np.concatenate(out).astype(np.float32)
    assert arr.shape[1:] == (policy_cfg.horizon, 7), f"청크 모양이 이상하다: {arr.shape}"
    logging.info(f"[R7] task {task} 데모 청크 {arr.shape} ({n_ep} 에피소드)")
    return arr


def build_basis(chunks_a: np.ndarray, chunks_b: np.ndarray, mode: str) -> dict:
    """모든 패널이 공유할 2D 사영 기저. **데모에서 1회만** 계산되고 이후 동결된다.

    입력은 정규화 액션 공간의 청크 (N, H, 7). 반환 B는 (2, H*7).
    """
    fa = chunks_a.reshape(len(chunks_a), -1)
    fb = chunks_b.reshape(len(chunks_b), -1)
    d = fa.shape[1]

    if mode == "demo_diff":
        # 1축: 두 task를 가르는 방향을 명시적으로 축에 박는다.
        u = fb.mean(0) - fa.mean(0)
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            raise SystemExit("[R7] 두 task 데모의 평균 액션이 같다 — demo_diff 기저를 만들 수 없다.")
        u = u / nu
        # 2축: 두 task를 합친 집합의 최대 분산 방향에서 1축 성분을 뺀 것.
        pooled = np.concatenate([fa, fb], 0)
        cen = pooled - pooled.mean(0)
        cen = cen - np.outer(cen @ u, u)                       # 1축 제거 후의 잔차
        # SVD 대신 공분산 power iteration이면 충분하지만, d=112라 SVD가 더 싸고 정확하다.
        _, s, vt = np.linalg.svd(cen, full_matrices=False)
        v = vt[0]
        v = v - (v @ u) * u
        v = v / max(np.linalg.norm(v), 1e-9)
        labels = ("task-separating axis  (mean$_1$ − mean$_0$)",
                  "top orthogonal variance axis")
        info = {"axis1_norm_raw": float(nu), "axis2_singular": float(s[0])}
    elif mode == "ee":
        # 청크 첫 스텝의 EE 변위 축 {Δx, Δy, Δz} 중 두 task가 가장 잘 갈리는 두 개.
        names = ["Δx", "Δy", "Δz"]
        score = []
        for ax in range(3):
            a, b = chunks_a[:, 0, ax], chunks_b[:, 0, ax]
            pooled_sd = np.sqrt(0.5 * (a.var() + b.var())) + 1e-9
            score.append(abs(a.mean() - b.mean()) / pooled_sd)
        order = list(np.argsort(score)[::-1][:2])
        order.sort()                                            # 축 순서는 x<y<z로 고정
        u, v = np.zeros(d), np.zeros(d)
        u[order[0]] = 1.0                                       # 첫 스텝 = flat index 0..6
        v[order[1]] = 1.0
        labels = (f"first-step EE {names[order[0]]}  (normalized)",
                  f"first-step EE {names[order[1]]}  (normalized)")
        info = {"separability": {names[i]: float(score[i]) for i in range(3)},
                "picked": [names[i] for i in order]}
    else:
        raise SystemExit(f"--basis 는 demo_diff 또는 ee 여야 한다 (받은 값: {mode!r})")

    B = np.stack([u, v]).astype(np.float32)
    logging.info(colored(f"[R7] 사영 기저 = {mode}  {labels}  info={info}", "green"))
    return {"B": B, "labels": labels, "mode": mode, "info": info}


def exec_range(policy_cfg, spec: str = "auto") -> tuple[int, int]:
    """청크에서 실제로 실행되는 구간 [start, end).

    generate_actions와 **같은 식**으로 유도한다. 여기가 어긋나면 로봇이 하지 않는 행동을
    재게 되므로, 상수로 박지 않고 config에서 뽑는다.
    """
    if spec == "all":
        return 0, int(policy_cfg.horizon)
    if spec != "auto":
        lo, hi = (int(x) for x in spec.split(":"))
        return lo, hi
    start = int(policy_cfg.n_obs_steps) - 1
    return start, start + int(policy_cfg.n_action_steps)


def project(x: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(..., H, 7) -> (..., 2). 기저는 어디서 부르든 같은 하나다."""
    flat = x.reshape(*x.shape[:-2], -1)
    return flat @ B.T


# ═════════════════════════════════════════════════════════════════════════════
#  기준 관측 s — 모든 패널이 공유한다
# ═════════════════════════════════════════════════════════════════════════════
def capture_obs(cfg: R7Config, task: int, n_obs: int) -> list[dict]:
    """task 환경의 초기 상태 0..n_obs-1에서 정착 후 관측을 딴다.

    정책을 태우기 전의 **원시 관측**을 캐시한다. 네 모델이 문자 그대로 같은 픽셀·같은
    상태를 보게 하려는 것이다(모델별로 다시 굴리면 시뮬레이터 난수가 끼어든다).
    """
    cfg.probe_task = task
    env = make_probe_env(cfg)
    init_states = env.unwrapped._init_states
    if n_obs > len(init_states):
        raise SystemExit(f"num_obs={n_obs} > 초기 상태 {len(init_states)}개")
    null_action = np.zeros(env.action_space.shape, dtype=np.float32)
    null_action[-1] = -1.0

    out = []
    for i in range(n_obs):
        env.reset()
        raw = env.unwrapped.set_init_state(init_states[i])
        obs = env.unwrapped._format_raw_obs(raw)
        for _ in range(cfg.settle_steps):
            obs, _r, _t, _tr, _i = env.step(null_action)
        # pixels는 카메라별 중첩 dict, task는 문자열이다. 구조를 그대로 두고 복사만 한다
        # (preprocess_observation이 중첩 dict를 기대한다).
        out.append({
            "pixels": {k: np.array(v) for k, v in obs["pixels"].items()},
            "agent_pos": np.array(obs["agent_pos"]),
        })
    env.close()
    logging.info(f"[R7] task {task} 기준 관측 {len(out)}개 확보 (settle={cfg.settle_steps})")
    return out


def obs_to_cond(policy, obs: dict, text: str, device) -> torch.Tensor:
    """원시 관측 + 지시문 -> global_cond (1, D).

    R6과 같은 t=0 정렬: 히스토리가 없으므로 첫 관측을 n_obs_steps번 복제한다.
    """
    proc = preprocess_observation({k: v for k, v in obs.items()})
    proc.pop("task", None)
    hist = [proc] * policy.config.n_obs_steps
    return encode_global_cond(policy, hist, text, device)


# ═════════════════════════════════════════════════════════════════════════════
#  flow 경로 추적 — sample()과 같은 적분, 중간 x_t를 전부 남긴다
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def flow_trace(policy, cond: torch.Tensor, x0: torch.Tensor, steps: int, stride: int) -> torch.Tensor:
    """dx/dt = v_θ(x, t, c) 를 t=0→1 Euler로 적분하며 경로를 남긴다.

    _DiTNoiseNet.sample과 **같은 식**이다(dt=1/steps, 매 스텝 clip). 다른 점은 두 가지뿐:
      - x₀를 밖에서 받는다 (패널 간 노이즈 짝짓기)
      - 중간 x_t를 버리지 않는다

    반환 (B, S+1, H, 7). S = ceil(steps/stride), 0번은 x₀.
    """
    net = policy.dit_flow.velocity_net
    B = x0.shape[0]
    c = cond.expand(B, -1)
    x = x0.clone()
    keep = [x.clone()]
    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full((B,), k / steps, device=x.device, dtype=x.dtype)
        x = x + dt * net(x, t, c)
        if net.clip_sample:
            x = torch.clamp(x, -net.clip_sample_range, net.clip_sample_range)
        if (k + 1) % stride == 0 or k == steps - 1:
            keep.append(x.clone())
    return torch.stack(keep, dim=1)


# ═════════════════════════════════════════════════════════════════════════════
#  통제군 학습 — joint multitask (mode=train_joint)
# ═════════════════════════════════════════════════════════════════════════════
def train_joint(cfg: R7Config):
    """task A와 B를 **동시에** 학습한다. "두 task를 한 모델에 담으면 원래 합쳐진다"의 통제군.

    두 로더를 스텝마다 번갈아 먹인다. ConcatDataset을 안 쓰는 이유는 LeRobotDataset의
    episode_data_index가 데이터셋 로컬 인덱스라 합치면 EpisodeAwareSampler가 어긋나기
    때문이다(E0.episode_sampler의 주석과 같은 함정). 번갈아 먹이면 각 task가 자기
    샘플러를 그대로 쓴다.

    나머지 하이퍼파라미터(옵티마이저, 배치, holdout 분할, 시작 체크포인트)는 E0의 순차
    학습과 같게 둔다. 다른 것은 "데이터를 섞었는가" 하나뿐이어야 통제군이 된다.
    """
    logging.info(colored(f"[R7] joint multitask 학습: task {cfg.task_a} + {cfg.task_b}",
                         "green", attrs=["bold"]))
    if cfg.seed is not None:
        set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    tasks = [cfg.task_a, cfg.task_b]
    datasets, loaders = [], []
    for k in tasks:
        repo = f"{cfg.dataset_prefix}{k}"
        meta = LeRobotDatasetMetadata(repo)
        ds = LeRobotDataset(repo, delta_timestamps=resolve_delta_timestamps(cfg.policy, meta),
                            video_backend=cfg.dataset.video_backend)
        train_eps, holdout_eps = split_episodes(repo, None, cfg.holdout_episodes)
        logging.info(f"[R7] task {k}: train {len(train_eps)} ep / held-out {len(holdout_eps)} ep")
        datasets.append(ds)
        loaders.append(torch.utils.data.DataLoader(
            ds,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            sampler=episode_sampler(cfg, ds, train_eps),
            pin_memory=device.type == "cuda",
            drop_last=False,
            multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
            persistent_workers=cfg.num_workers > 0,
        ))

    policy = make_policy(cfg=cfg.policy, ds_meta=datasets[0].meta)
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    iters = [cycle(dl) for dl in loaders]

    policy.train()
    tracker = MetricsTracker(
        cfg.batch_size,
        sum(d.num_frames for d in datasets),
        sum(d.num_episodes for d in datasets),
        {"loss": AverageMeter("loss", ":.3f"), "mse": AverageMeter("mse", ":.3f"),
         "penalty": AverageMeter("pen", ":.3e"), "grad_norm": AverageMeter("grdn", ":.3f"),
         "lr": AverageMeter("lr", ":0.1e"), "update_s": AverageMeter("updt_s", ":.3f"),
         "dataloading_s": AverageMeter("data_s", ":.3f")},
        initial_step=0,
    )

    logging.info(f"[R7] joint 학습 {cfg.joint_steps} 스텝 (task당 {cfg.joint_steps // len(tasks)})")
    for step in range(cfg.joint_steps):
        t0 = time.perf_counter()
        batch = to_device(next(iters[step % len(tasks)]), device)   # ★ 번갈아
        tracker.dataloading_s = time.perf_counter() - t0
        tracker, out = update_policy(tracker, policy, batch, optimizer,
                                     cfg.optimizer.grad_clip_norm, grad_scaler=grad_scaler,
                                     lr_scheduler=lr_scheduler, use_amp=cfg.policy.use_amp)
        tracker.step()
        if cfg.log_freq > 0 and (step + 1) % cfg.log_freq == 0:
            logging.info(tracker)
            tracker.reset_averages()

    ckpt = get_step_checkpoint_dir(cfg.output_dir, cfg.joint_steps, cfg.joint_steps)
    save_checkpoint(ckpt, cfg.joint_steps, cfg, policy, optimizer, lr_scheduler)
    update_last_checkpoint(ckpt)
    (Path(cfg.output_dir) / ".done").write_text(
        f"joint_steps={cfg.joint_steps}\ntasks={tasks}\n")
    logging.info(colored(f"[R7] joint 체크포인트 -> {ckpt}", "green", attrs=["bold"]))


# ═════════════════════════════════════════════════════════════════════════════
#  본 실험 (mode=trace)
# ═════════════════════════════════════════════════════════════════════════════
def cache_name(cfg: R7Config) -> str:
    """full 모드는 장면이 조건의 일부라 obs_task라는 개념이 없다. 이름에도 그걸 반영한다."""
    if cfg.cond_mode == "full":
        return f"R7_full_{cfg.basis}.npz"
    return f"R7_lang_obs{cfg.obs_task}_{cfg.basis}.npz"


def tagged_png(cache: str | Path, plot_tag: str) -> Path | None:
    """캐시 이름에서 그림 이름을 만든다. R7_full_x.npz + "ER" -> R7_ER_full_x.png.

    plot_tag가 비면 None을 돌려주고, plot_r7이 기존대로 cache.with_suffix(".png")를 쓴다.
    """
    if not plot_tag:
        return None
    cache = Path(cache)
    return cache.with_name(cache.stem.replace("R7_", f"R7_{plot_tag}_", 1) + ".png")


def model_specs(cfg: R7Config) -> list[dict]:
    """--models 가 고른 체크포인트들. key는 그림/요약에서 계속 쓰인다.

    순서는 이야기 순서다: 학습 전(pretrain) -> 섞어 학습(joint) -> 순서대로 학습(CL).
    joint과 CL은 같은 사전학습에서 출발해 같은 데이터를 같은 스텝만큼 본다. 다른 것은
    섞었는가 순서대로인가 하나뿐이므로, 둘의 차이는 그 하나에 귀속된다.
    """
    a, b = cfg.task_a, cfg.task_b
    table = {
        "pretrain": {"ckpt": cfg.pretrain_ckpt, "title": "pretrained  (before either task)"},
        "ft_a": {"ckpt": str(stage_ckpt(cfg.ckpt_root, a)), "title": f"FT{a}  (task {a} only)"},
        "ft_b": {"ckpt": cfg.ft1_ckpt, "title": f"FT{b}  (task {b} only)"},
        "joint": {"ckpt": cfg.joint_ckpt, "title": f"joint  (task {a} + {b} mixed)"},
        "cl": {"ckpt": str(stage_ckpt(cfg.ckpt_root, b)), "title": f"CL  (task {a} → task {b})"},
    }
    specs = []
    for key in [k.strip() for k in cfg.models.split(",") if k.strip()]:
        if key not in table:
            raise SystemExit(f"[R7] 모르는 모델 이름: {key!r} (가능: {list(table)})")
        if not table[key]["ckpt"]:
            raise SystemExit(f"[R7] {key} 체크포인트 경로가 비어 있다. "
                             f"--pretrain_ckpt / --ft1_ckpt / --joint_ckpt / --ckpt_root 확인.")
        specs.append({"key": key, **table[key]})
    return specs


@torch.no_grad()
def run_trace(cfg: R7Config, run_dir: Path) -> Path:
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    a, b = cfg.task_a, cfg.task_b
    specs = model_specs(cfg)

    # ── 정책 골격 하나로 config / 정규화 통계 / 데모 청크를 먼저 만든다 ──────────
    meta_a = LeRobotDatasetMetadata(f"{cfg.dataset_prefix}{a}")
    ref_policy = load_policy_at(cfg, specs[0]["ckpt"], meta_a, device)
    pol_cfg = ref_policy.config
    stats = norm_stats(ref_policy)
    steps = cfg.flow_steps or int(pol_cfg.num_inference_steps)
    horizon, adim = int(pol_cfg.horizon), 7
    e0, e1 = exec_range(pol_cfg, cfg.exec_slice)
    logging.info(colored(f"[R7] 실행 구간 = 청크 index {e0}..{e1 - 1} "
                         f"({e1 - e0}스텝 × {adim} = {(e1 - e0) * adim}차원). "
                         f"나머지는 로봇에 나가지 않으므로 재지 않는다.", "green"))

    text = {0: task_text(cfg.dataset_prefix, a), 1: task_text(cfg.dataset_prefix, b)}
    logging.info(colored(f"[R7] c₀ = {text[0]!r}", "cyan"))
    logging.info(colored(f"[R7] c₁ = {text[1]!r}", "cyan"))

    # ── [1] 사영 기저: 데모에서 1회 고정 ────────────────────────────────────────
    raw_a = demo_chunks(cfg, pol_cfg, a, cfg.basis_episodes)[:, e0:e1]
    raw_b = demo_chunks(cfg, pol_cfg, b, cfg.basis_episodes)[:, e0:e1]
    nrm_a, nrm_b = minmax_normalize(raw_a, stats), minmax_normalize(raw_b, stats)
    basis = build_basis(nrm_a, nrm_b, cfg.basis)
    B = basis["B"]

    # 그림에 깔 참조점(데모 액션 청크의 사영). 기저와 같은 것을 쓴다.
    n_sc = cfg.demo_scatter_episodes
    sc_a = project(minmax_normalize(demo_chunks(cfg, pol_cfg, a, n_sc)[:, e0:e1], stats), B)
    sc_b = project(minmax_normalize(demo_chunks(cfg, pol_cfg, b, n_sc)[:, e0:e1], stats), B)

    # ── [2] 짝지을 노이즈: 시드 고정으로 한 번만 ────────────────────────────────
    gen = torch.Generator(device="cpu").manual_seed(cfg.noise_seed)
    x0_cpu = torch.randn(cfg.num_noise, horizon, adim, generator=gen)
    x0 = x0_cpu.to(device)
    logging.info(f"[R7] x₀ {tuple(x0.shape)} 고정 (seed={cfg.noise_seed}) — 모든 패널이 공유한다")

    # ── [3] 두 조건을 만들 상황 ────────────────────────────────────────────────
    # full     c₀ = (task A 장면, task A 지시문) · c₁ = (task B 장면, task B 지시문)
    #          조건 벡터 전체가 바뀐다. 주장이 "조건 전체에 무감각"이므로 이게 본 실험이다.
    # language 장면을 obs_task 하나로 고정하고 지시문만 바꾼다(부록).
    if cfg.cond_mode == "full":
        obs_sets = {0: capture_obs(cfg, a, cfg.num_obs), 1: capture_obs(cfg, b, cfg.num_obs)}
        logging.info(colored("[R7] cond_mode=full — 장면과 지시문을 함께 바꾼다", "green"))
    elif cfg.cond_mode == "language":
        fixed = capture_obs(cfg, cfg.obs_task, cfg.num_obs)
        obs_sets = {0: fixed, 1: fixed}
        logging.info(colored(f"[R7] cond_mode=language — 장면을 task {cfg.obs_task}로 고정하고 "
                             f"지시문만 바꾼다 (부록)", "green"))
    else:
        raise SystemExit(f"--cond_mode 는 full 또는 language 여야 한다 (받은 값: {cfg.cond_mode!r})")
    del ref_policy
    torch.cuda.empty_cache()

    # ── [4] 모델 × 조건 × 관측 격자를 돈다 ─────────────────────────────────────
    # FT 쌍은 서로 **다른 모델**의 경로를 비교하므로, 그 둘만 경로를 들고 있는다.
    # (5 관측 × 100 노이즈 × 101 시점 × 56) float32 ≈ 11MB/개.
    keys_all = [sp["key"] for sp in specs]
    pair_on = "ft_a" in keys_all and "ft_b" in keys_all
    want = {("ft_a", 0), ("ft_b", 1)} if pair_on else set()
    pair_keep: dict[tuple, list] = {w: [] for w in want}

    blob: dict[str, np.ndarray] = {}
    for spec in specs:
        policy = load_policy_at(cfg, spec["ckpt"], meta_a, device)
        assert_shared_norm(stats, norm_stats(policy), spec["key"])
        logging.info(colored(f"[R7] {spec['key']}: {spec['ckpt']}", "cyan", attrs=["bold"]))

        key = spec["key"]
        proj = {0: [], 1: []}       # (obs, noise, S+1, 2)
        ends = {0: [], 1: []}       # (obs, noise, H*7)  전차원 종점
        div = []                    # (obs, noise, S+1)  ‖x_t(c₁) − x_t(c₀)‖
        for oi in range(cfg.num_obs):
            tr = {}
            for ci in (0, 1):
                # ★ 조건 하나는 (장면, 지시문) 한 쌍에서 통째로 만들어진다.
                cond = obs_to_cond(policy, obs_sets[ci][oi], text[ci], device)
                # ★ 적분은 16스텝 전부로 돈다(모델이 그렇게 동작한다). 자르는 것은
                #   재고 그리는 단계뿐이다.
                tr[ci] = flow_trace(policy, cond, x0, steps, cfg.trace_stride)   # (N,S+1,H,7)
                t_np = tr[ci].cpu().numpy()[:, :, e0:e1, :]
                proj[ci].append(project(t_np, B))
                ends[ci].append(t_np[:, -1].reshape(len(t_np), -1))
                if (key, ci) in want:
                    pair_keep[(key, ci)].append(t_np.astype(np.float32))
            # 발산 거리는 사영이 아니라 **전차원**에서 잰다. 사영은 보여 주기 위한 것이고
            # 숫자는 112차원 그대로여야 "사영 때문에 겹쳐 보인다"는 반론이 닿지 않는다.
            d = (tr[1] - tr[0])[:, :, e0:e1, :].flatten(2).norm(dim=2)             # (N, S+1)
            div.append(d.cpu().numpy())
            del tr
        for ci in (0, 1):
            blob[f"{key}_c{ci}_proj"] = np.stack(proj[ci]).astype(np.float32)
            blob[f"{key}_c{ci}_end"] = np.stack(ends[ci]).astype(np.float32)
        blob[f"{key}_div"] = np.stack(div).astype(np.float32)
        logging.info(f"[R7]   proj {blob[f'{key}_c0_proj'].shape}  "
                     f"종점거리 평균 {blob[f'{key}_div'][:, :, -1].mean():.4f}")
        del policy
        torch.cuda.empty_cache()

    if pair_on:
        # FT_A|c₀  vs  FT_B|c₁ — 같은 x₀ 짝끼리 이어 붙여 비교한다.
        A = np.stack(pair_keep[("ft_a", 0)])        # (obs, noise, S+1, 8, 7)
        Bp = np.stack(pair_keep[("ft_b", 1)])
        blob["ft_pair_div"] = np.linalg.norm((Bp - A).reshape(*A.shape[:3], -1),
                                             axis=-1).astype(np.float32)
        blob["ft_pair_c0_end"] = A[:, :, -1].reshape(*A.shape[:2], -1).astype(np.float32)
        blob["ft_pair_c1_end"] = Bp[:, :, -1].reshape(*Bp.shape[:2], -1).astype(np.float32)
        logging.info(f"[R7] FT 쌍 (FT{a}|c₀ vs FT{b}|c₁) 종점거리 평균 "
                     f"{blob['ft_pair_div'][:, :, -1].mean():.4f}")
        del A, Bp, pair_keep

    blob["basis_B"] = B
    blob["demo_scatter_a"] = sc_a.reshape(-1, 2).astype(np.float32)
    blob["demo_scatter_b"] = sc_b.reshape(-1, 2).astype(np.float32)
    blob["x0"] = x0_cpu.numpy().astype(np.float32)
    blob["meta"] = np.array(json.dumps({
        "task_a": a, "task_b": b, "text_c0": text[0], "text_c1": text[1],
        "cond_mode": cfg.cond_mode, "exec_slice": [e0, e1], "horizon": horizon,
        "basis_mode": basis["mode"], "basis_labels": list(basis["labels"]),
        "basis_info": basis["info"], "flow_steps": steps, "trace_stride": cfg.trace_stride,
        "num_noise": cfg.num_noise, "noise_seed": cfg.noise_seed,
        "num_obs": cfg.num_obs, "obs_task": cfg.obs_task, "settle_steps": cfg.settle_steps,
        "draw_noise": cfg.draw_noise, "draw_obs": cfg.draw_obs,
        "specs": [{k: s[k] for k in ("key", "ckpt", "title")} for s in specs],
        "pair_on": bool(pair_on),
        "action_min": stats["min"].tolist(), "action_max": stats["max"].tolist(),
    }))

    cache = run_dir / cache_name(cfg)
    np.savez_compressed(cache, **blob)
    logging.info(colored(f"[R7] saved -> {cache}", "green", attrs=["bold"]))
    return cache


# ═════════════════════════════════════════════════════════════════════════════
#  요약 숫자
# ═════════════════════════════════════════════════════════════════════════════
def summarize(z: dict, m: dict) -> dict:
    """전 관측 · 전 노이즈로 낸 분리 지표. 그림의 다발이 아니라 이쪽이 주장이다.

    ft_pair는 특수하다. 한 모델의 두 조건이 아니라 **두 모델**을 비교한다:
    FT_A|c₀ vs FT_B|c₁. 각자 자기 조건만 받은 전문가 둘의 간격이므로
    "조건이 갈라 놓을 수 있는 최대치"의 기준선이 된다.
    """
    entries = [(s["key"], s["title"]) for s in m["specs"]]
    if m.get("pair_on") and {"ft_a", "ft_b"} <= {k for k, _ in entries}:
        a_, b_ = m["task_a"], m["task_b"]
        entries = ([("ft_pair", f"individual FTs  (FT{a_}|c₀ vs FT{b_}|c₁)")]
                   + [e for e in entries if e[0] not in ("ft_a", "ft_b")])
    out = {}
    for k, title in entries:
        e0 = z[f"{k}_c0_end"]                       # (O, N, D)
        e1 = z[f"{k}_c1_end"]
        between = np.linalg.norm(e1 - e0, axis=-1)  # 짝지은 종점 거리 (O, N)

        # 같은 조건 안에서 노이즈만 다른 종점 쌍의 평균 거리 = "노이즈가 만드는 차이"
        within = []
        for e in (e0, e1):
            for o in range(e.shape[0]):
                d = np.linalg.norm(e[o][:, None, :] - e[o][None, :, :], axis=-1)
                iu = np.triu_indices(len(d), k=1)
                within.append(d[iu])
        within = np.concatenate(within)

        div = z[f"{k}_div"]                          # (O, N, S+1)
        out[k] = {
            "title": title,
            "d_between_mean": float(between.mean()), "d_between_sd": float(between.std()),
            "d_within_mean": float(within.mean()), "d_within_sd": float(within.std()),
            "ratio": float(between.mean() / max(within.mean(), 1e-9)),
            "div_curve_mean": div.mean(axis=(0, 1)).tolist(),
            "div_curve_q25": np.percentile(div, 25, axis=(0, 1)).tolist(),
            "div_curve_q75": np.percentile(div, 75, axis=(0, 1)).tolist(),
        }
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  그림
# ═════════════════════════════════════════════════════════════════════════════
def _ramp(colors, n):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("r", colors)(np.linspace(0.0, 1.0, n))


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def _draw_bundle(ax, proj, ramp, n_draw, oi, lw=0.9):
    """한 다발. 선 하나가 x₀ 하나의 flow 경로이고, 명도가 flow 시간이다."""
    from matplotlib.collections import LineCollection

    P = proj[oi][:n_draw]                                # (n, S+1, 2)
    S = P.shape[1]
    cols = _ramp(ramp, S - 1)
    segs, cs = [], []
    for p in P:
        segs.extend([[p[i], p[i + 1]] for i in range(S - 1)])
        cs.extend(cols)
    ax.add_collection(LineCollection(segs, colors=cs, linewidths=lw, alpha=0.75, zorder=4,
                                     capstyle="round"))
    ax.scatter(P[:, -1, 0], P[:, -1, 1], s=11, color=ramp[-1], edgecolor="white",
               linewidth=0.35, zorder=6)
    ax.scatter(P[:, 0, 0], P[:, 0, 1], s=5, color=ramp[0], zorder=5)


def _traj_panel(ax, z, m, entries, title, subtitle, n_draw, oi, lim):
    # ★ 데모 액션 산점도는 그리지 않는다. 데모 청크는 에피소드 전 구간(접근·파지·놓기)에서
    #   나온 것이라 "첫 결정 하나"인 궤적 종점과 시점이 맞지 않는다. 비교군이 어긋난
    #   참조점을 배경에 깔면 없느니만 못하다.
    for key, ci, ramp in entries:
        _draw_bundle(ax, z[f"{key}_c{ci}_proj"], ramp, n_draw, oi)
    _style(ax)
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])
    ax.set_title(title, fontsize=10.5, color=INK, pad=22, loc="left")
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2,
            va="bottom", ha="left")
    ax.set_xlabel(m["basis_labels"][0], color=INK2, fontsize=8.5)
    ax.set_ylabel(m["basis_labels"][1], color=INK2, fontsize=8.5)


def plot_r7(cache: str | Path, out_png: str | Path | None = None, plot_tag: str = "") -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError:
        print("matplotlib 없음 -> 그림 생략")
        return

    cache = Path(cache)
    z = {k: v for k, v in np.load(cache, allow_pickle=False).items()}
    m = json.loads(str(z["meta"]))
    # 팔을 여러 개 낼 때 "CL"만으로는 어느 팔의 순차 학습인지 알 수 없다. 꼬리표가 있으면
    # CL 항목의 이름을 CL(ER)처럼 바꾼다. specs를 고치면 아래 titles와 summarize가 모두
    # 여기서 읽으므로 그림 제목·막대 축 라벨·summary.json이 한 번에 따라온다.
    # 다른 항목(ft_pair, joint)은 팔과 무관한 통제군이라 이름을 바꾸지 않는다.
    if plot_tag:
        for s in m.get("specs", []):
            if s["key"] == "cl":
                s["title"] = s["title"].replace("CL", f"CL({plot_tag})", 1)
    a, b = m["task_a"], m["task_b"]
    stats = summarize(z, m)
    (cache.with_suffix(".summary.json")).write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if not kk.startswith("div_curve")}
         for k, v in stats.items()}, indent=2, ensure_ascii=False))

    keys = [s["key"] for s in m["specs"]]
    titles = {s["key"]: s["title"] for s in m["specs"]}
    a_, b_ = m["task_a"], m["task_b"]
    # ft_a/ft_b는 서로 다른 모델을 한 패널에 겹치는 특수 항목이다. FT_A는 c₀만, FT_B는
    # c₁만 받으므로 어느 모델도 학습한 적 없는 조건을 받지 않는다(OOD 없음).
    if m.get("pair_on") and {"ft_a", "ft_b"} <= set(keys):
        keys = ["ft_pair"] + [k for k in keys if k not in ("ft_a", "ft_b")]
        titles["ft_pair"] = f"individual FTs  (FT{a_}|c₀ vs FT{b_}|c₁)"
    draw_keys = [k for k in keys if k != "ft_pair"] or ["ft_a"]
    if "ft_pair" in keys:
        draw_keys = ["ft_a", "ft_b"] + draw_keys
    n_draw = min(int(m["draw_noise"]), z[f"{draw_keys[0]}_c0_proj"].shape[1])
    oi = int(m["draw_obs"])

    # 궤적 패널의 축은 **공유**한다. 패널마다 축이 다르면 "겹친다"가 눈속임이 된다.
    # 범위는 경로가 정한다. 데모 산점도는 참조점이라 화면 밖으로 나가도 되지만, 경로가
    # 데모 전체 범위에 눌려 점 하나로 보이면 그림이 아무것도 말하지 못한다.
    pts = []
    for k in draw_keys:
        for ci in (0, 1):
            pts.append(z[f"{k}_c{ci}_proj"][oi, :n_draw].reshape(-1, 2))
    allp = np.concatenate(pts)
    lo, hi = allp.min(0), allp.max(0)
    pad = (hi - lo) * 0.16 + 1e-6
    lim = [(lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])]

    n_top = len(keys)
    fig = plt.figure(figsize=(5.2 * n_top, 10.0))
    gs = fig.add_gridspec(2, n_top, height_ratios=[1.28, 1.0], hspace=0.40, wspace=0.235,
                          left=0.062, right=0.985, top=0.795, bottom=0.10)

    # ── 위: 모델마다 궤적 한 패널. 같은 축, 같은 x₀, 조건만 c₀/c₁ ────────────────
    sub = {"pretrain": "before either task is learned",
           "joint": "both tasks in one training run",
           "cl": "both tasks, one after the other",
           "ft_pair": "two experts, each given only the condition it was trained on"}
    for i, k in enumerate(keys):
        ent = ([("ft_a", 0, C0_RAMP), ("ft_b", 1, C1_RAMP)] if k == "ft_pair"
               else [(k, 0, C0_RAMP), (k, 1, C1_RAMP)])
        _traj_panel(fig.add_subplot(gs[0, i]), z, m, ent,
                    f"{'abcde'[i]}   {titles[k]}", sub.get(k, ""), n_draw, oi, lim)

    # ── 아래 왼쪽: 종점 분리 막대 ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    xs = np.arange(len(keys))
    bet = [stats[k]["d_between_mean"] for k in keys]
    wit = [stats[k]["d_within_mean"] for k in keys]
    cols = [MODEL_COLORS[k] for k in keys]
    ax.bar(xs - 0.19, bet, 0.36, color=cols, zorder=3)
    ax.bar(xs + 0.19, wit, 0.36, color="white", edgecolor=cols, linewidth=1.2, hatch="///",
           zorder=3)
    for x, v in zip(xs, bet):
        ax.text(x - 0.19, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    short = {k: titles[k].split("  ")[0] for k in keys}
    _style(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels([short[k] for k in keys], fontsize=8.5, color=INK2)
    ax.set_ylabel("‖·‖ in normalized action space", color=INK2, fontsize=8.5)
    ax.set_title("d   endpoint separation", fontsize=10.5, color=INK, pad=22, loc="left")
    ax.text(0, 1.015, "solid = between conditions (paired) · hatched = within condition (noise only)",
            transform=ax.transAxes, fontsize=8, color=INK2, va="bottom", ha="left")

    # ── 아래 가운데: 발산 곡선 ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    S = len(stats[keys[0]]["div_curve_mean"])
    tt = np.linspace(0, 1, S)
    for k in keys:
        s = stats[k]
        c = MODEL_COLORS[k]
        ax.plot(tt, s["div_curve_mean"], color=c, lw=2.0, zorder=4, label=short[k])
        ax.fill_between(tt, s["div_curve_q25"], s["div_curve_q75"], color=c, alpha=0.13,
                        linewidth=0, zorder=2)
        ax.annotate(short[k], (tt[-1], s["div_curve_mean"][-1]), textcoords="offset points",
                    xytext=(5, 0), fontsize=8, color=c, fontweight="bold", va="center")
    _style(ax)
    ax.set_xlim(0, 1.14)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("flow time  t   (t=0 noise → t=1 action chunk)", color=INK2, fontsize=8.5)
    ax.set_ylabel("‖x$_t$(c₁) − x$_t$(c₀)‖", color=INK2, fontsize=8.5)
    ax.set_title("e   when do the two conditions part?", fontsize=10.5, color=INK, pad=22,
                 loc="left")
    ax.text(0, 1.015, "paired x₀, so every curve starts at exactly 0",
            transform=ax.transAxes, fontsize=8, color=INK2, va="bottom", ha="left")

    # ── 아래 오른쪽: 분리비 ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2] if n_top > 2 else gs[1, 1])
    ratios = [stats[k]["ratio"] for k in keys]
    ax.bar(xs, ratios, 0.55, color=cols, zorder=3)
    ax.axhline(1.0, color=INK2, lw=0.9, ls="--", zorder=4)
    ax.text(-0.45, 1.0, "conditioning = noise ", fontsize=7.5, color=INK2, va="center",
            ha="right", clip_on=False)
    for x, v in zip(xs, ratios):
        ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    _style(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels([short[k] for k in keys], fontsize=8.5, color=INK2)
    ax.set_ylabel("d$_{between}$ / d$_{within}$", color=INK2, fontsize=8.5)
    ax.set_title("f   separation ratio", fontsize=10.5, color=INK, pad=22, loc="left")
    ax.text(0, 1.015, "below 1 = the conditioning moves the sample less than the noise does",
            transform=ax.transAxes, fontsize=8, color=INK2, va="bottom", ha="left")

    # ── 범례 / 제목 ────────────────────────────────────────────────────────────
    full = m.get("cond_mode", "full") == "full"
    c_desc = ("scene + instruction" if full else "instruction only")
    handles = [
        Line2D([0], [0], color=C0_INK, lw=2.4, label=f"c₀ = task {a} {c_desc}"),
        Line2D([0], [0], color=C1_INK, lw=2.4, label=f"c₁ = task {b} {c_desc}"),
        Line2D([0], [0], color=C0_RAMP[0], lw=2.4, label="light → dark = flow time t: 0 → 1"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.006))

    es = m.get("exec_slice", [0, 16])
    ex = (f"only the {es[1] - es[0]} chunk steps the robot actually executes are measured "
          f"(index {es[0]}..{es[1] - 1} of {m.get('horizon', 16)})")
    if full:
        head = (f"the whole conditioning vector is swapped — DINOv2 image features, robot state "
                f"and the CLIP instruction together ({m['num_obs']} initial states per task)")
        # ★ 결론 문장은 이 그림의 막대에서 만든다.
        #   전에는 seq 팔에서 관측한 문장이 그대로 박혀 있었는데, 팔마다 답이 다르다.
        #   ER 팔은 CL 6.19 / joint 6.42 로 순차 학습이 조건을 유지하는데도 그림은
        #   "does not"이라고 적혀 나왔다. 그림이 자기 데이터와 어긋나는 주장을 하지
        #   않도록 두 비율을 직접 비교하고, 판단 근거로 쓴 수치를 함께 적는다.
        r_cl = stats.get("cl", {}).get("ratio")
        r_joint = stats.get("joint", {}).get("ratio")
        if r_cl is None or r_joint is None:
            tail = ("mixing the two tasks in one training run keeps the conditions apart; "
                    "seeing them one after the other — same data, same number of steps — does not")
        elif r_cl < 0.6 * r_joint:
            # seq 팔에서 보던 문장. 수치를 덧붙여 눈으로 확인할 수 있게 한다.
            tail = ("mixing the two tasks in one training run keeps the conditions apart; "
                    "seeing them one after the other — same data, same number of steps — does not "
                    f"(separation ratio {r_cl:.2f} vs {r_joint:.2f})")
        else:
            tail = ("mixing the two tasks in one training run keeps the conditions apart, and here "
                    "seeing them one after the other keeps them apart too "
                    f"(separation ratio {r_cl:.2f} vs {r_joint:.2f})")
    else:
        head = (f"APPENDIX: the scene is held at task {m['obs_task']} and only the CLIP instruction "
                f"is swapped, to ask which part of the conditioning is being ignored")
        tail = ("this isolates the language channel alone — the headline claim is about the whole "
                "conditioning vector, see the main figure")
    fig.suptitle(
        f"R7: does the generated action path follow the conditioning?   "
        f"open-loop flow integration, task {a} vs task {b}\n"
        f"{head}.\n"
        f"Every panel shares one projection basis fixed once on demo data ({m['basis_mode']}) and "
        f"the same {m['num_noise']} noise draws x₀ (seed {m['noise_seed']}).\n"
        f"{ex[0].upper()}{ex[1:]} — the rest is discarded by generate_actions and never reaches "
        f"the robot.\n"
        f"{tail}",
        fontsize=11.5, color=INK, y=0.988, linespacing=1.5)

    out = Path(out_png) if out_png else cache.with_suffix(".png")
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"saved figure -> {out}")


# ═════════════════════════════════════════════════════════════════════════════
#  메인
# ═════════════════════════════════════════════════════════════════════════════
@parser.wrap()
def main(cfg: R7Config):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.mode == "train_joint":
        train_joint(cfg)
        return

    cfg.save_checkpoint = False
    if not cfg.ckpt_root:
        raise SystemExit("--ckpt_root 가 필요하다 (예: outputs/E0/libero_spatial/seed_42/lam0).")
    run_dir = Path(cfg.out_root) / (cfg.run_tag or "run")
    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.seed is not None:
        set_seed(cfg.seed)

    cache = run_dir / cache_name(cfg)
    if cache.exists() and not cfg.recompute:
        logging.info(f"[R7] 캐시 재사용: {cache}")
    else:
        cache = run_trace(cfg, run_dir)
    if not cfg.no_plot:
        plot_r7(cache, tagged_png(cache, cfg.plot_tag), cfg.plot_tag)


if __name__ == "__main__":
    init_logging()
    if "--plot_only" in sys.argv:
        kv = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
        tag = kv.get("plot_tag", "")
        if "cache" in kv:
            plot_r7(kv["cache"], tagged_png(kv["cache"], tag), tag)
        elif "run_dir" in kv:
            found = sorted(Path(kv["run_dir"]).glob("R7_*.npz"))
            if not found:
                raise SystemExit(f"[R7] npz 캐시가 없다: {kv['run_dir']}")
            for c in found:
                plot_r7(c, tagged_png(c, tag), tag)
        else:
            raise SystemExit("--plot_only 에는 --run_dir=<...> 또는 --cache=<...npz> 가 필요하다")
    else:
        main()
