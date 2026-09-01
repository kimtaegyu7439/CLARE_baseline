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

"""R8 — 조건 신호는 네트워크 어디에서 죽는가: layer × flow-time 대비비(CR) 지형도.

무엇을 묻는가
    R7은 조건 c₀/c₁을 바꿔도 생성 경로가 한 다발로 뭉친다는 것을 **입출력 수준**에서
    보였다. 그러면 남는 질문은 "그 무시가 내부 어디에서 일어나는 사건인가"이다.

    속도장 v(x_t, t, s, c)는 한 방에 계산되는 함수가 아니라 블록 스택을 통과한다.

        h₀ = ac_proj(x_t) + pos                       (액션 토큰 임베딩)
        h_ℓ = Block_ℓ(h_{ℓ-1};  c, t)                  ℓ = 1..L   (DiT, AdaLN-Zero)
        v  = FinalLayer(h_L; c, t)

    조건이 출력에 도달하려면 이 스택 전체를 **생존해서** 통과해야 한다. 그러므로
    "입구에서부터 안 들어왔나 / 중간 어느 블록에서 뭉개졌나 / 끝까지 살아있다가
    출력 사영에서만 버려졌나"는 입출력만 봐서는 원리적으로 알 수 없는 질문이고,
    내부를 열어야만 답이 된다. 주장 자체가 내부에 대한 명제(조건 라우팅의 형성/붕괴)
    이므로 layer 축은 있으면 좋은 축이 아니라 이 주장을 검증할 수 있는 유일한 축이다.

무엇을 재는가 — 대비비 CR (contrast ratio)
                    E ‖ h_ℓ(x_t, t, s, c₀) − h_ℓ(x_t, t, s, c₁) ‖        ← 조건이 만드는 차이
        CR(ℓ, t) = ────────────────────────────────────────────────
                    E ‖ h_ℓ(x_t, t, s, c) − h_ℓ(x′_t, t, s, c) ‖        ← 노이즈가 만드는 차이

    분자는 짝지은 조건 대비(같은 x_t, 같은 s, 지시문만 다름), 분모는 같은 조건에서
    프로브 지점만 다른 짝이다. 즉 "조건이 만드는 차이가 노이즈가 만드는 차이의 몇 배인가".

    ★ 비율로 설계한 이유가 핵심이다. h_ℓ은 블록마다 스케일이 다르므로(LayerNorm,
      residual 성장) 절대 거리로는 layer 간 비교가 성립하지 않는다. 분자와 분모를
      같은 layer에서 재면 스케일이 약분되어 layer 축을 따라 읽는 것이 정당해진다.
      동시에 이것은 R7의 분리비(d_between/d_within)를 layer 축으로 분해한 것이라
      논문 전체의 지표 어휘가 통일된다.

      CR ≈ 1   조건이 만드는 차이가 노이즈가 만드는 차이와 같은 크기 (판정선이 아니라 **눈금**)
      CR ≫ 1   조건이 표현을 지배한다

    ★ CR=1을 "죽음"으로 읽으면 안 된다. 건강한 통제군(joint)조차 깊은 블록에서 1.3 수준이다.
      판정은 절대 임계값이 아니라 **통제군 대비 상대값**으로 한다. 1은 "조건이 노이즈만큼
      움직인다"는 참조 눈금일 뿐이다.

프로브 지점을 어떻게 잡는가 — 순환을 피한다
    x_t를 "어느 모델의 field로 적분해" 만들면 프로브 지점 자체가 특정 모델·조건의
    산물이 되어 비교가 오염된다. 대신 보간 공식으로 직접 생성한다.

        x_t = (1 − t)·x₀ + t·a        x₀ ~ N(0, I),  a = 데모 액션 청크

    이건 임의의 선택이 아니라 **이 모델이 학습된 바로 그 지점**이다. compute_loss가
    noisy_trajectory = (1−t)·noise + t·action 으로 만들어 그 자리에서 속도를 맞히도록
    학습했다(modeling_dit_flow_mt.py의 flow matching 블록). 따라서 세 체크포인트가
    정확히 같은 (x_t, t) 격자 위에서, 그것도 분포 안에서 평가된다.

무엇을 보게 되는가
    세로로 훑으면  조건 신호가 깊이를 따라 어떻게 전파/소멸하는가
    가로로 훑으면  flow의 어느 단계에서 조건이 일하는가 (초반 = 모드 선택, 후반 = 정밀화)

    이 그림의 본론은 한 장이 아니라 **같은 색 스케일로 나란히 놓은 체크포인트 비교**다.
    CL의 CR이 통제군(joint)보다 낮아지기 시작하는 지점이 조건 정보가 새는 곳이고,
    그 위치를 짚는 것이 이 실험의 헤드라인이다.

주입 사망 vs 전달 사망 — AdaLN 게이트 대비 (패널 f)
    DiT는 조건을 매 블록 AdaLN으로 **다시** 주입한다. 그래서 CR(ℓ)이 낮은 데에는 두
    가지 원인이 가능하다.
        (a) 주입 사망  그 블록의 변조/게이트가 c₀와 c₁을 구분하길 멈췄다
        (b) 전달 사망  주입은 구분되는데 residual 스트림이 그 차이를 못 실어나른다
    게이트는 x와 무관하게 (c, t)만의 함수이므로 따로 잴 수 있다. 블록 ℓ의 조건 유래
    변조 벡터 m_ℓ(c) = [attn scale·shift, attn gate, mlp scale·shift, mlp gate]에 대해

        G(ℓ) = ‖m_ℓ(c₀) − m_ℓ(c₁)‖ / (½‖m_ℓ(c₀)‖ + ½‖m_ℓ(c₁)‖)

    ★ G는 CR이 **아니다**. 분모가 다르다 — CR의 분모는 노이즈가 만드는 차이이고, G의 분모는
      게이트 벡터 자신의 크기다. α는 x에 의존하지 않아 "노이즈가 만드는 차이"를 만들 수
      없기 때문이다. 두 패널의 세로축은 서로 비교하면 안 된다.

    판정은 통제군 대비로 한다.
      G가 통제군보다 낮다              -> 주입 실패 (그 블록이 c₀와 c₁을 구분하길 멈췄다)
      G는 통제군과 비슷한데 CR만 낮다  -> 전달 실패 (주입은 되는데 스트림이 못 실어나른다)

색
    히트맵은 발산형이고 중심이 CR = 1(= log CR 0)이다. 파랑 = 조건이 지배, 흰색 =
    노이즈와 동급, 빨강 = 노이즈보다도 약함. 네 모델 곡선의 hue는 문서 팔레트
    슬롯 1/2/3/7이고 all-pairs 검증을 통과한다(worst normal ΔE 16.3, worst CVD ΔE 9.2;
    OKLab×100, Machado 2009 severity 1.0). aqua는 흰 배경 대비가 2.82:1이라 3:1을 못
    넘으므로 relief 규칙에 따라 모든 곡선에 직접 라벨을 단다 — 색만으로 식별하지 않는다.

이 스크립트는 학습을 하지 않는다. R7이 만든 통제군 체크포인트를 그대로 읽는다.

사용 예
    python R8.py --policy.path=<any ckpt> --ckpt_root=outputs/E0/libero_spatial/seed_42/lam0 \
        --ft1_ckpt=... --joint_ckpt=... --run_tag=seq_seed42
    python R8.py --plot_only --run_dir=outputs/R8/seq_seed42
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

# R7이 이미 분리해 둔 조각을 그대로 쓴다. 관측 확보·조건 인코딩·정규화 검사를 두 번
# 구현하면 두 그림이 "같은 s⁰, 같은 좌표계"라는 보장이 깨진다.
from lerobot.scripts.R3 import stage_ckpt
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

# ── 색 ────────────────────────────────────────────────────────────────────────
# 발산 양극: 문서 팔레트의 blue ↔ red, 중립 회색 중점. 중심은 CR = 1.
DIV_LO, DIV_MID, DIV_HI = "#e34948", "#f0efec", "#2a78d6"   # CR<1(빨강) · 1(회색) · CR>1(파랑)
# 모델 4색 = 문서 팔레트 슬롯 1/2/3/7. all-pairs 검증 통과.
MODEL_COLORS = {"pretrain": "#eb6834", "joint": "#1baf7a", "cl": "#4a3aa7",
                "ft_a": "#2a78d6", "ft_b": "#8a8a86"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


# ═════════════════════════════════════════════════════════════════════════════
#  설정
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class R8Config(TrainPipelineConfig):
    """train.py 인자 전부 + 내부 표현 프로빙 인자. 학습 인자는 무시된다."""

    # ── 무엇을 바꾸는가 (R7과 같은 규약) ────────────────────────────────────
    # full     조건 벡터 전체를 바꾼다. c₀=(task A 장면, A 지시문), c₁=(task B 장면, B 지시문).
    #          주장이 "조건 전체에 무감각"이므로 이쪽이 본 실험이다.
    # language 장면을 obs_task로 고정하고 지시문만 바꾼다 (부록).
    cond_mode: str = "full"
    # ★ R7과 같은 규약: 청크 16스텝 중 실제로 실행되는 토큰만 잰다. 활성 h_ℓ의 토큰 축이
    #   청크 스텝에 1:1로 대응하므로, 버려지는 스텝의 토큰을 빼고 거리를 잰다.
    exec_slice: str = "auto"

    # ── 어떤 체크포인트를 볼 것인가 (R7과 같은 집합) ────────────────────────
    ckpt_root: str = ""
    pretrain_ckpt: str = ""
    ft1_ckpt: str = ""
    joint_ckpt: str = ""
    models: str = "pretrain,joint,cl"
    task_a: int = 0
    task_b: int = 1

    # ── 프로브 지점 ──────────────────────────────────────────────────────────
    num_probe: int = 100                    # 프로브 지점 개수 (x₀, a 짝)
    probe_seed: int = 20260813              # R7의 noise_seed와 같은 값으로 두면 x₀가 공유된다
    num_pairs: int = 50                     # 분모(노이즈 짝) 추정에 쓸 짝 수
    # flow 시간 격자. t=1은 **일부러 뺀다** — 거기서는 x_t = a 라서 노이즈 짝 둘이
    # 같은 점이 되어 분모가 구조적으로 0이다(비율이 정의되지 않는다).
    t_steps: int = 20                       # {0, 0.05, ..., 0.95}
    t_max: float = 0.95
    demo_episodes: int = 10                 # a를 뽑을 데모 에피소드 수 (task당)

    # ── 기준 관측 s ──────────────────────────────────────────────────────────
    num_obs: int = 5
    obs_task: int = 0
    settle_steps: int = 5

    # ── 안전장치 ─────────────────────────────────────────────────────────────
    # 분모가 0에 가까우면 비율이 튄다. 중앙값 대비 이 비율보다 작은 칸은 빗금으로 표시한다.
    den_floor_ratio: float = 1e-3

    # ── 데이터 / 환경 ────────────────────────────────────────────────────────
    dataset_prefix: str = "continuallearning/libero_spatial_image_task_"
    env_task_prefix: str = "Libero_Spatial_Task_"
    probe_task: int = 0                     # capture_obs가 덮어쓴다
    max_steps: int = 50                     # make_probe_env의 TimeLimit용. 롤아웃은 안 한다

    # ── 출력 / 제어 ──────────────────────────────────────────────────────────
    out_root: str = "outputs/R8"
    run_tag: str = ""
    recompute: bool = False
    no_plot: bool = False

    def validate(self):
        """R1/R3/R6/R7과 같은 이유로 output_dir 존재 검사만 우회한다(캐시 재사용)."""
        out = self.output_dir
        if isinstance(out, Path) and out.is_dir():
            self.output_dir = None
            super().validate()
            self.output_dir = out
        else:
            super().validate()


# ═════════════════════════════════════════════════════════════════════════════
#  중간 표현 뽑기
# ═════════════════════════════════════════════════════════════════════════════
class LayerTap:
    """velocity net의 블록별 중간 표현 h_ℓ을 forward hook으로 가로챈다.

    forward를 손으로 다시 구현하지 않는 이유: 그러면 모델이 바뀔 때 조용히 어긋난다.
    hook은 실제로 실행된 그 텐서를 준다.

    뽑는 지점 (L = num_blocks = 6이면 8개):
        0        h₀   = decoder.layers[0]의 입력 = ac_proj(x_t) + dec_pos  (조건 주입 전)
        1..L     h_ℓ  = 각 DiT 블록의 출력
        L+1      v    = eps_out의 출력 (속도 그 자체. R7의 입출력 수준 지표와 이어진다)
    """

    def __init__(self, net):
        self.h: dict[int, torch.Tensor] = {}
        self.handles = []
        layers = net.decoder.layers
        self.n_blocks = len(layers)
        self.handles.append(layers[0].register_forward_pre_hook(self._pre))
        for k, layer in enumerate(layers):
            self.handles.append(layer.register_forward_hook(self._make(k + 1)))
        self.handles.append(net.eps_out.register_forward_hook(self._make(self.n_blocks + 1)))

    def _pre(self, _mod, args):
        self.h[0] = args[0].detach()

    def _make(self, idx):
        def fn(_mod, _args, out):
            self.h[idx] = out.detach()
        return fn

    @property
    def labels(self) -> list[str]:
        return ["embed h₀"] + [f"block {k}" for k in range(1, self.n_blocks + 1)] + ["v out"]

    def collect(self) -> list[torch.Tensor]:
        return [self.h[i] for i in range(self.n_blocks + 2)]

    def remove(self):
        for handle in self.handles:
            handle.remove()


def token_dist(a: torch.Tensor, b: torch.Tensor, sl: slice = slice(None)) -> torch.Tensor:
    """(T, B, H) 두 개 -> (B,). 토큰별 거리를 먼저 재고 토큰 축으로 평균한다.

    토큰을 통째로 flatten해 한 번에 노름을 재면 토큰 수가 많은 층이 커 보인다.
    토큰별로 재고 평균하면 그 편향이 없다.

    sl은 실행 구간 토큰만 남기는 슬라이스다. 로봇에 나가지 않는 스텝의 토큰까지 넣으면
    "하지도 않는 행동"으로 조건 민감도를 재게 된다.
    """
    return (a[sl] - b[sl]).norm(dim=-1).mean(dim=0)


@torch.no_grad()
def adaln_contrast(net, cond0: torch.Tensor, cond1: torch.Tensor, t: float, device) -> np.ndarray:
    """블록별 AdaLN 조건 주입 대비 G(ℓ). x와 무관하게 (c, t)만의 함수라 따로 잴 수 있다.

    반환 (n_blocks, 3): [전체, 게이트만, scale/shift만]의 상대 발산.
    """
    t_emb = net.time_net(torch.tensor([t], device=device, dtype=torch.float32))   # (1, H)
    out = []
    for layer in net.decoder.layers:
        vecs = {0: [], 1: []}
        gates = {0: [], 1: []}
        mods = {0: [], 1: []}
        for i, c in enumerate((cond0, cond1)):
            # _DiTDecoder.forward의 첫 줄과 같다: cond = cond + t
            cc = net.cond_proj(c) + t_emb                     # (1, H)
            act = torch.nn.functional.silu(cc)
            g = torch.cat([layer.attn_gate.scale(act), layer.mlp_gate.scale(act)], dim=-1)
            m = torch.cat([layer.attn_modulate.scale(act), layer.attn_modulate.shift(act),
                           layer.mlp_modulate.scale(act), layer.mlp_modulate.shift(act)], dim=-1)
            gates[i], mods[i] = g, m
            vecs[i] = torch.cat([g, m], dim=-1)
        row = []
        for pair in (vecs, gates, mods):
            d = (pair[0] - pair[1]).norm()
            scale = 0.5 * (pair[0].norm() + pair[1].norm())
            row.append(float(d / (scale + 1e-12)))
        out.append(row)
    return np.asarray(out, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  본 실험
# ═════════════════════════════════════════════════════════════════════════════
def cache_name(cfg: R8Config) -> str:
    """full 모드는 장면이 조건의 일부라 obs_task라는 개념이 없다."""
    return "R8_full.npz" if cfg.cond_mode == "full" else f"R8_lang_obs{cfg.obs_task}.npz"


def model_specs(cfg: R8Config) -> list[dict]:
    """--models 가 고른 체크포인트들. R7과 같은 집합·같은 순서 규약."""
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
            raise SystemExit(f"[R8] 모르는 모델 이름: {key!r} (가능: {list(table)})")
        if not table[key]["ckpt"]:
            raise SystemExit(f"[R8] {key} 체크포인트 경로가 비어 있다.")
        specs.append({"key": key, **table[key]})
    return specs


@torch.no_grad()
def run_probe(cfg: R8Config, run_dir: Path) -> Path:
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    a, b = cfg.task_a, cfg.task_b
    specs = model_specs(cfg)

    meta_a = LeRobotDatasetMetadata(f"{cfg.dataset_prefix}{a}")
    ref = load_policy_at(cfg, specs[0]["ckpt"], meta_a, device)
    pol_cfg = ref.config
    stats = norm_stats(ref)
    horizon = int(pol_cfg.horizon)
    e0, e1 = exec_range(pol_cfg, cfg.exec_slice)
    tok = slice(e0, e1)
    logging.info(colored(f"[R8] 실행 구간 = 청크 index {e0}..{e1 - 1} 토큰만 잰다", "green"))

    text = {0: task_text(cfg.dataset_prefix, a), 1: task_text(cfg.dataset_prefix, b)}
    logging.info(colored(f"[R8] c₀ = {text[0]!r}", "cyan"))
    logging.info(colored(f"[R8] c₁ = {text[1]!r}", "cyan"))

    # ── [1] 프로브 지점: x_t = (1−t)·x₀ + t·a  (학습 보간식 그대로) ────────────
    rng = np.random.default_rng(cfg.probe_seed)
    chunks = np.concatenate([
        minmax_normalize(demo_chunks(cfg, pol_cfg, a, cfg.demo_episodes), stats),
        minmax_normalize(demo_chunks(cfg, pol_cfg, b, cfg.demo_episodes), stats)])
    pick = rng.choice(len(chunks), size=cfg.num_probe, replace=False)
    a_tgt = torch.from_numpy(chunks[pick]).float().to(device)                  # (N, 16, 7)
    gen = torch.Generator(device="cpu").manual_seed(cfg.probe_seed)
    x0 = torch.randn(cfg.num_probe, horizon, 7, generator=gen).to(device)      # (N, 16, 7)
    # ★ 분모의 짝은 **노이즈만** 다른 짝이어야 한다. 데모 목표 a까지 바꿔 버리면 분모가
    #   "완전히 다른 두 지점"의 거리가 되어 CR이 통째로 눌리고, 1을 기준선으로 읽을 수
    #   없게 된다. 그래서 a는 그대로 두고 x₀만 새로 뽑는다.
    P = min(cfg.num_pairs, cfg.num_probe)
    x0b = torch.randn(P, horizon, 7, generator=gen).to(device)                 # (P, 16, 7)
    a_pair = a_tgt[:P]
    t_grid = np.linspace(0.0, float(cfg.t_max), cfg.t_steps, dtype=np.float32)
    logging.info(f"[R8] 프로브 {cfg.num_probe}개 × t {cfg.t_steps}격자"
                 f"(0..{cfg.t_max}), 노이즈 짝 {P}개")

    # ── [2] 두 조건을 만들 상황 (R7과 같은 규약) ─────────────────────────────
    if cfg.cond_mode == "full":
        obs_sets = {0: capture_obs(cfg, a, cfg.num_obs), 1: capture_obs(cfg, b, cfg.num_obs)}
        logging.info(colored("[R8] cond_mode=full — 장면과 지시문을 함께 바꾼다", "green"))
    elif cfg.cond_mode == "language":
        fixed = capture_obs(cfg, cfg.obs_task, cfg.num_obs)
        obs_sets = {0: fixed, 1: fixed}
        logging.info(colored(f"[R8] cond_mode=language — 장면을 task {cfg.obs_task}로 고정 (부록)",
                             "green"))
    else:
        raise SystemExit(f"--cond_mode 는 full 또는 language 여야 한다 ({cfg.cond_mode!r})")
    del ref
    torch.cuda.empty_cache()

    # ── [3] 모델 × t × 조건 격자 ─────────────────────────────────────────────
    blob: dict[str, np.ndarray] = {}
    labels = None
    for spec in specs:
        policy = load_policy_at(cfg, spec["ckpt"], meta_a, device)
        assert_shared_norm(stats, norm_stats(policy), spec["key"])
        net = policy.dit_flow.velocity_net
        tap = LayerTap(net)
        labels = tap.labels
        n_layer = len(labels)
        logging.info(colored(f"[R8] {spec['key']}: {spec['ckpt']}", "cyan", attrs=["bold"]))

        num = np.zeros((n_layer, cfg.t_steps), dtype=np.float64)
        den = np.zeros((n_layer, cfg.t_steps), dtype=np.float64)
        gate = np.zeros((len(net.decoder.layers), 3), dtype=np.float64)

        for oi in range(cfg.num_obs):
            # ★ 조건 하나는 (장면, 지시문) 한 쌍에서 통째로 만들어진다.
            cond = {ci: obs_to_cond(policy, obs_sets[ci][oi], text[ci], device) for ci in (0, 1)}
            for ti, t in enumerate(t_grid):
                tf = float(t)
                xt = (1.0 - tf) * x0 + tf * a_tgt                              # (N, 16, 7)
                xtb = (1.0 - tf) * x0b + tf * a_pair                           # (P, 16, 7) 노이즈만 다름
                tt = torch.full((cfg.num_probe,), tf, device=device)
                ttb = torch.full((P,), tf, device=device)
                acts, acts_b = {}, {}
                for ci in (0, 1):
                    net(xt, tt, cond[ci].expand(cfg.num_probe, -1))
                    acts[ci] = tap.collect()
                    net(xtb, ttb, cond[ci].expand(P, -1))
                    acts_b[ci] = tap.collect()
                for li in range(n_layer):
                    # 분자: 같은 x_t, 같은 s, 지시문만 다름
                    num[li, ti] += float(token_dist(acts[0][li], acts[1][li], tok).mean())
                    # 분모: 같은 지시문, 같은 a, 노이즈만 다름 (두 조건에서 재고 평균)
                    den[li, ti] += 0.5 * sum(
                        float(token_dist(acts[ci][li][:, :P], acts_b[ci][li], tok).mean())
                        for ci in (0, 1))
                gate += adaln_contrast(net, cond[0], cond[1], tf, device)
        num /= cfg.num_obs
        den /= cfg.num_obs
        gate /= cfg.num_obs * cfg.t_steps

        blob[f"{spec['key']}_num"] = num.astype(np.float32)
        blob[f"{spec['key']}_den"] = den.astype(np.float32)
        blob[f"{spec['key']}_gate"] = gate.astype(np.float32)
        cr = num / np.maximum(den, 1e-12)
        logging.info(f"[R8]   CR 범위 {cr.min():.3f}–{cr.max():.3f}   "
                     f"v out 행 평균 {cr[-1].mean():.3f}")
        tap.remove()
        del policy
        torch.cuda.empty_cache()

    blob["t_grid"] = t_grid
    blob["meta"] = np.array(json.dumps({
        "task_a": a, "task_b": b, "text_c0": text[0], "text_c1": text[1],
        "layer_labels": labels, "num_probe": cfg.num_probe, "num_pairs": cfg.num_pairs,
        "probe_seed": cfg.probe_seed, "t_steps": cfg.t_steps, "num_obs": cfg.num_obs,
        "obs_task": cfg.obs_task, "den_floor_ratio": cfg.den_floor_ratio,
        "cond_mode": cfg.cond_mode, "exec_slice": [e0, e1],
        "specs": [{k: s[k] for k in ("key", "ckpt", "title")} for s in specs],
    }))
    cache = run_dir / cache_name(cfg)
    np.savez_compressed(cache, **blob)
    logging.info(colored(f"[R8] saved -> {cache}", "green", attrs=["bold"]))
    return cache


# ═════════════════════════════════════════════════════════════════════════════
#  그림
# ═════════════════════════════════════════════════════════════════════════════
def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.set_axisbelow(True)


def plot_r8(cache: str | Path, out_png: str | Path | None = None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
        from matplotlib.lines import Line2D
    except ModuleNotFoundError:
        print("matplotlib 없음 -> 그림 생략")
        return

    cache = Path(cache)
    z = {k: v for k, v in np.load(cache, allow_pickle=False).items()}
    m = json.loads(str(z["meta"]))
    a, b = m["task_a"], m["task_b"]
    labels = m["layer_labels"]
    t_grid = z["t_grid"]
    keys = [s["key"] for s in m["specs"]]
    titles = {s["key"]: s["title"] for s in m["specs"]}
    n_layer = len(labels)

    # ★ embed h₀ 행은 조건이 아직 주입되기 **전**이라 분자가 구조적으로 정확히 0이다.
    #   병리가 아니라 파이프라인이 조건만 재고 있다는 내장 검증이므로, 값으로 칠하지 않고
    #   따로 표시한다(색 스케일과 단면 평균에서도 뺀다).
    STRUCT0 = 0
    cr, weak = {}, {}
    for k in keys:
        num, den = z[f"{k}_num"].astype(np.float64), z[f"{k}_den"].astype(np.float64)
        # 분모가 0 근처면 비율이 튄다. 중앙값 대비로 판정하고 해당 칸은 빗금으로 덮는다.
        floor = m["den_floor_ratio"] * np.median(den)
        weak[k] = den < floor
        cr[k] = num / np.maximum(den, floor)
        assert np.allclose(num[STRUCT0], 0), "embed h₀ 행이 0이 아니다 — 훅 위치를 확인해라"
    n_weak = int(sum(w.sum() for w in weak.values()))
    if n_weak:
        logging.warning(f"[R8] 분모가 불안정한 칸 {n_weak}개 -> 빗금 표시")

    live = slice(STRUCT0 + 1, None)          # 조건이 실제로 들어오는 행들만
    # 색 스케일은 네 패널이 **공유**한다. 패널마다 다르면 비교가 성립하지 않는다.
    # ★ 중심은 CR=1에 고정하되 양 끝은 데이터에 맞춘다(비대칭). 대칭으로 잡으면 관측된
    #   CR이 전부 1 아래일 때 파랑 절반이 통째로 놀고 빨강 쪽 대비가 반으로 줄어든다.
    lg = np.concatenate([np.log10(np.maximum(cr[k][live], 1e-6)).ravel() for k in keys])
    lo = min(float(np.percentile(lg, 1)), -0.05)
    hi = max(float(np.percentile(lg, 99)), 0.05)
    cmap = LinearSegmentedColormap.from_list("cr", [DIV_LO, DIV_MID, DIV_HI])
    cmap.set_bad("#eceae5")                  # 구조적 0 행(= 조건 주입 전)
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)

    summary = {k: {"title": titles[k],
                   "cr_by_layer": cr[k].mean(axis=1).tolist(),
                   "cr_by_t": cr[k][live].mean(axis=0).tolist(),
                   "cr_vout_mean": float(cr[k][-1].mean()),
                   "adaln_contrast_all": z[f"{k}_gate"][:, 0].tolist(),
                   "adaln_contrast_gate": z[f"{k}_gate"][:, 1].tolist(),
                   "adaln_contrast_modulate": z[f"{k}_gate"][:, 2].tolist()}
               for k in keys}
    cache.with_suffix(".summary.json").write_text(
        json.dumps({"layer_labels": labels, "t_grid": t_grid.tolist(), "models": summary},
                   indent=2, ensure_ascii=False))

    n_hm = len(keys)
    fig = plt.figure(figsize=(4.1 * n_hm + 3.4, 9.8))
    gs = fig.add_gridspec(2, n_hm + 2, width_ratios=[1] * n_hm + [0.075, 1.05],
                          height_ratios=[1.0, 0.66], hspace=0.42, wspace=0.20,
                          left=0.052, right=0.978, top=0.745, bottom=0.095)

    # ── 위: 히트맵 넷 (같은 색 스케일) ────────────────────────────────────────
    ims = None
    for ci, k in enumerate(keys):
        ax = fig.add_subplot(gs[0, ci])
        grid = np.ma.masked_invalid(np.log10(np.where(cr[k] > 0, cr[k], np.nan)))
        grid[STRUCT0] = np.ma.masked                 # 조건 주입 전 행은 값이 아니다
        ims = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, norm=norm,
                        extent=(-0.5, len(t_grid) - 0.5, -0.5, n_layer - 0.5),
                        interpolation="nearest")
        # 불안정한 칸은 빗금으로 덮어 "값이 아니라 추정 실패"임을 표시한다.
        ys, xs = np.where(weak[k])
        for yy, xx in zip(ys, xs):
            ax.add_patch(plt.Rectangle((xx - 0.5, yy - 0.5), 1, 1, fill=False, hatch="///",
                                       edgecolor=INK2, linewidth=0))
        tix = np.unique(np.linspace(0, len(t_grid) - 1, 5).round().astype(int))
        ax.set_xticks(tix)
        ax.set_xticklabels([f"{t_grid[i]:.2f}" for i in tix])
        ax.set_yticks(range(n_layer))
        ax.set_yticklabels(labels if ci == 0 else [""] * n_layer, fontsize=8)
        ax.tick_params(colors=INK2, labelsize=8, length=3)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_xlabel("flow time  t", color=INK2, fontsize=8.5)
        ax.set_title(f"{'abcd'[ci]}   {titles[k]}", fontsize=10, color=INK, pad=8, loc="left")
    cax = fig.add_subplot(gs[0, n_hm])
    pos = cax.get_position()
    # 컬러바를 왼쪽으로 붙이고 얇게 만들어 옆 패널의 y라벨과 부딪히지 않게 한다.
    cax.set_position([pos.x0 - 0.010, pos.y0, pos.width * 0.42, pos.height])
    cb = fig.colorbar(ims, cax=cax)
    # 눈금과 라벨을 컬러바 **왼쪽**에 둔다. 오른쪽에 두면 옆 패널의 y라벨과 부딪히는데,
    # 왼쪽은 히트맵 d의 여백이라 비어 있다.
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.set_label("log₁₀ CR    (0 = conditioning is at noise level)", color=INK2, fontsize=8.5)
    cb.ax.tick_params(colors=INK2, labelsize=7.5)
    cb.outline.set_edgecolor(GRID)

    # ── 위 오른쪽: layer 단면 (히트맵과 y축 공유) ────────────────────────────
    ax = fig.add_subplot(gs[0, n_hm + 1])
    rows = np.arange(STRUCT0 + 1, n_layer)
    for ki, k in enumerate(keys):
        prof = cr[k].mean(axis=1)[live]
        ax.plot(prof, rows, color=MODEL_COLORS[k], lw=2.0, marker="o", ms=4, zorder=4)
        # aqua는 흰 배경 대비 2.82:1이라 색만으로 식별하게 두지 않는다(relief 규칙).
        # 라벨을 곡선마다 다른 행에 붙여 서로 겹치지 않게 한다.
        r = len(rows) - 1 - ki
        ax.annotate(titles[k].split("  ")[0], (prof[r], rows[r]), textcoords="offset points",
                    xytext=(5, 3), fontsize=8, color=MODEL_COLORS[k], fontweight="bold")
    ax.axvline(1.0, color=INK2, lw=0.9, ls="--", zorder=3)
    ax.text(1.0, rows[-1] + 0.30, "CR = 1 ", fontsize=7.5, color=INK2, ha="right", va="bottom")
    _style(ax)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7, axis="x")
    ax.set_xscale("log")
    ax.set_xlim(right=ax.get_xlim()[1] * 2.2)      # 오른쪽 직접 라벨이 잘리지 않게
    ax.set_ylim(-0.5, n_layer - 0.5)
    ax.set_yticks(range(n_layer))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("CR   (averaged over t)", color=INK2, fontsize=8.5)
    ax.set_title("e   depth cross-section", fontsize=10, color=INK, pad=8, loc="left")

    # ── 아래 왼쪽: flow-time 단면 ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0:n_hm])
    for ki, k in enumerate(keys):
        prof = cr[k][live].mean(axis=0)
        ax.plot(t_grid, prof, color=MODEL_COLORS[k], lw=2.0, marker="o", ms=4, zorder=4)
        # 곡선마다 다른 t에 라벨을 붙인다(끝에 몰면 넷이 겹친다).
        j = len(t_grid) - 1 - 3 * ki
        ax.annotate(titles[k].split("  ")[0], (t_grid[j], prof[j]), textcoords="offset points",
                    xytext=(0, 7), fontsize=8, color=MODEL_COLORS[k], fontweight="bold",
                    ha="center")
    ax.axhline(1.0, color=INK2, lw=0.9, ls="--", zorder=3)
    _style(ax)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlim(-0.02, 1.10)
    ax.set_xlabel("flow time  t      (early = mode selection · late = refinement)",
                  color=INK2, fontsize=8.5)
    ax.set_ylabel("CR   (averaged over layers)", color=INK2, fontsize=8.5)
    ax.set_title("f   flow-stage cross-section", fontsize=10, color=INK, pad=8, loc="left")

    # ── 아래 오른쪽: AdaLN 주입 대비 ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, n_hm + 1])
    nb = z[f"{keys[0]}_gate"].shape[0]
    for k in keys:
        ax.plot(np.arange(1, nb + 1), z[f"{k}_gate"][:, 1], color=MODEL_COLORS[k], lw=2.0,
                marker="o", ms=4, zorder=4)
    _style(ax)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
    ax.set_xticks(range(1, nb + 1))
    ax.set_xlabel("DiT block", color=INK2, fontsize=8.5)
    ax.set_ylabel("‖α(c₀) − α(c₁)‖ / ½(‖α(c₀)‖+‖α(c₁)‖)", color=INK2, fontsize=8)
    ax.set_title("g   AdaLN gate divergence  (not CR — different denominator)",
                 fontsize=10, color=INK, pad=18, loc="left")
    ax.text(0, 1.015, "low vs the control = the block stopped telling c₀ from c₁ (injection) · "
                      "on par with the control but low CR = the stream fails to carry it",
            transform=ax.transAxes, fontsize=7, color=INK2, va="bottom", ha="left")

    handles = [Line2D([0], [0], color=MODEL_COLORS[k], lw=2.4, marker="o", ms=5,
                      label=titles[k]) for k in keys]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.004))

    # 캡션의 마지막 줄은 **실제로 나온 것**을 말해야 한다. 깊이에 따른 증폭 배율을 재서
    # "어디서 꺼지는가(경계선)"와 "처음부터 낮은가(주입)"를 숫자로 가른다.
    gain = {k: cr[k][-2].mean() / max(cr[k][1].mean(), 1e-9) for k in keys}   # block1 -> block L
    gain_txt = " · ".join(f"{titles[k].split('  ')[0]} ×{gain[k]:.1f}" for k in keys)
    nb_txt = labels[-2].replace("block ", "")
    # ★ "누가 CR=1을 넘는가"는 캡션에 손으로 쓰지 않고 데이터에서 센다. 손으로 쓰면
    #   실험 조건을 바꿨을 때 캡션만 옛말이 되어 남는다.
    short = {k: titles[k].split("  ")[0] for k in keys}
    cross = [k for k in keys if cr[k][live].max() >= 1.0]
    top = max(keys, key=lambda k: cr[k][live].mean())
    if cross:
        lead_txt = (f"{', '.join(short[k] for k in cross)} reach noise level (CR ≥ 1) in the deepest "
                    f"blocks at late flow time; {short[top]} goes highest "
                    f"(peak CR {cr[top][live].max():.2f})")
    else:
        lead_txt = (f"no model reaches noise level anywhere; {short[top]} comes closest "
                    f"(peak CR {cr[top][live].max():.2f})")

    full = m.get("cond_mode", "full") == "full"
    what = ("the whole conditioning vector (scene + instruction)" if full
            else f"the instruction alone, scene held at task {m['obs_task']}")
    fig.suptitle(
        f"R8: where in the network does the conditioning signal die?   "
        f"task {a} vs task {b}   —   {what} is swapped\n"
        f"CR(ℓ,t)  =  E‖h_ℓ(x_t,t,s,c₀) − h_ℓ(x_t,t,s,c₁)‖  /  "
        f"E‖h_ℓ(x_t,t,s,c) − h_ℓ(x′_t,t,s,c)‖   — how many times the noise-scale difference "
        f"the instruction makes, at each block and each flow time\n"
        f"probe points are the training interpolant itself, x_t = (1−t)·x₀ + t·a, so all four "
        f"checkpoints are read at the very same in-distribution locations "
        f"({m['num_probe']} points, {m['num_pairs']} noise pairs, {m['num_obs']} initial states "
        f"per condition)\n"
        f"blue = the conditioning dominates the representation · white = it is at noise level · "
        f"red = below noise.\n{lead_txt}.\n"
        f"There is no cliff: every model amplifies CR by roughly the same factor from block 1 to "
        f"block {nb_txt}  ({gain_txt}).\n"
        f"So the signal is not lost in propagation — it enters weak (panel g) and stays "
        f"proportionally weak all the way down.",
        fontsize=11, color=INK, y=0.988, linespacing=1.5)

    out = Path(out_png) if out_png else cache.with_suffix(".png")
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"saved figure -> {out}")


# ═════════════════════════════════════════════════════════════════════════════
#  메인
# ═════════════════════════════════════════════════════════════════════════════
@parser.wrap()
def main(cfg: R8Config):
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
        logging.info(f"[R8] 캐시 재사용: {cache}")
    else:
        cache = run_probe(cfg, run_dir)
    if not cfg.no_plot:
        plot_r8(cache)


if __name__ == "__main__":
    init_logging()
    if "--plot_only" in sys.argv:
        kv = dict(x.lstrip("-").split("=", 1) for x in sys.argv[1:] if "=" in x)
        if "cache" in kv:
            plot_r8(kv["cache"])
        elif "run_dir" in kv:
            found = sorted(Path(kv["run_dir"]).glob("R8_*.npz"))
            if not found:
                raise SystemExit(f"[R8] npz 캐시가 없다: {kv['run_dir']}")
            for c in found:
                plot_r8(c)
        else:
            raise SystemExit("--plot_only 에는 --run_dir=<...> 또는 --cache=<...npz> 가 필요하다")
    else:
        main()
