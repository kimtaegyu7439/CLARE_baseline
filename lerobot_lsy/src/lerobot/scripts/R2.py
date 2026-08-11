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

"""R2 — 모드 결어긋남(mode decoherence): 망각은 무엇을 할지가 아니라 어느 것을 할지를 잃는다.

배경 가설
    Flow matching 정책은 결정적 수송 사상 Φ_θ(o, a₀)다. 노이즈 a₀가 노이즈 공간의
    어느 영역에 떨어지느냐가 "이번엔 어느 방식으로 할까"(모드)를 결정하고, 정책은
    재계획할 때마다 새 a₀를 뽑는다.

    가설: **망각은 모드의 중심보다 영역 경계를 먼저 침식한다.** 중심은 데이터가 두꺼워
    안정적이지만, 모드 사이 경계는 데이터가 희박하고 손실이 평평해 작은 파라미터
    변화에도 크게 밀린다. 경계가 밀리면 같은 a₀가 다른 모드로 배정되고, 재계획마다
    다른 계획이 실행되어 **개별 행동은 유효한데 시간적으로 일관되지 않게** 된다.

이 가설이 설명하는 R1 관찰
    - held-out loss 불변 + SR 붕괴 (중심 보존 / 경계 이동. 경계는 노이즈 측도로 얇아
      loss 평균에 거의 잡히지 않는다)
    - Seq stage3/4: dwell 0.82~0.89인데 SR=0 (각 계획은 데모 근처라 d는 작지만 서로
      달라 진행이 없다)
    - Δa 크기가 SR을 못 가름 (EWC stage3 Δa≈0.7 SR60% vs stage4 Δa≈1.1 SR0%)
    - Δa의 심한 시간 진동 (재계획마다 같은 모드/다른 모드가 번갈아)
    - 지도교수 관찰 "단순한 궤적 하나를 주면 오히려 잘 배운다" (모드가 하나면 경계가
      없어 침식할 대상이 없다)

실험 A — 끈끈한 노이즈 (인과 개입)
    R1의 롤아웃과 **완전히 동일**하되, 재계획 시 flow matching 초기 노이즈 a₀를 뽑는
    방식만 바꾼다. 학습도, 파라미터도, 환경도 건드리지 않는다.
        fresh   재계획마다 새 a₀ (= R1. baseline이자 파이프라인 검산)
        sticky  롤아웃 시작에 한 번 뽑아 에피소드 내내 재사용
        ou      a₀_new = ρ·a₀_old + sqrt(1−ρ²)·ε  (주변분포 N(0,I) 유지)
    세 모드의 rollout_id별 첫 a₀는 **동일**하다. 그래야 "노이즈 재사용 여부"만이
    유일한 차이가 된다.

    해석
      sticky에서 SR이 fresh 대비 유의하게 오르면 → 결어긋남이 SR 붕괴의 인과적 원인.
      동시에 "학습 없이 망각을 완화하는 추론 시점 개입"이라는 부수 결과가 된다.
      오르지 않으면 → 가설 약화. 단 switch_count(fresh)가 애초에 낮았다면 개입이 걸릴
      여지가 없었던 것이므로, switch_count(fresh)를 반드시 함께 보고해 구분한다.

실험 B — 모드 센서스 (기제 확인, 시뮬레이터 불필요)
    고정된 관측 N개 × 고정된 a₀ M개 격자 위에서 사상 Φ_θ(o, a₀_j)를 직접 샘플링하고,
    stage1에서 정한 클러스터 구조를 기준으로 후속 스테이지를 평가한다.
        center_shift    stage1 클러스터 중심 대비 그 클러스터 a₀들이 만드는 행동 중심의 이동
                        = **모드 중심이 얼마나 움직였나**
        assign_change   stage1의 클러스터 중심에 최근접 배정했을 때 라벨이 바뀐 비율
                        = **경계가 얼마나 밀렸나**
        weight_entropy  배정 비율의 엔트로피
        demo_match_mass 실제 데모 행동에 충분히 가까운 a₀의 비율 = **정답 모드의 질량**
        n_modes         관측당 클러스터 개수 (probe task가 실제로 다봉인지 — 대부분
                        단봉이면 가설의 전제가 성립하지 않으므로 먼저 보고한다)

    가설의 예측: center_shift는 작고 assign_change는 크다(비대칭). 그리고
    demo_match_mass가 held-out loss보다 SR을 잘 예측한다.

R1과의 관계
    지표 함수와 φ/τ/z-정규화 통계는 **R1에서 import하고, demo_ref.npz는 R1이 만든 것을
    그대로 읽는다**. 자가 달라지면 R1과 R2의 d(t)를 나란히 놓을 수 없다. R2가 새로
    하는 일은 "a₀를 어떻게 뽑는가"와 "노이즈 격자 위의 모드 구조" 둘뿐이다.

이 스크립트는 **학습을 하지 않는다.**

전제
    R1이 만든 run_dir (demo_ref.npz + r1_results.jsonl). 없으면 즉시 에러.
    E0가 만든 체크포인트 트리 + gym_libero.

사용 예
    python R2.py --r1_run_dir=outputs/R1/libero_spatial_seed42_probe0 \
        --ckpt_roots="seq=outputs/E0/.../lam0,ewc=outputs/E0/.../lam100,frozen=outputs/E0/.../laminf" \
        --policy.path=<ref ckpt> --env.type=libero --env.benchmark=libero_spatial \
        --targets="ewc@2,ewc@3,seq@2,seq@3" --run_tag=libero_spatial_seed42_probe0
    python R2.py --plot_only --run_dir=outputs/R2/libero_spatial_seed42_probe0
"""

import hashlib
import json
import logging
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
import torch.multiprocessing as mp
from termcolor import colored

from lerobot.configs import parser
from lerobot.constants import ACTION
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.envs.utils import preprocess_observation
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.scripts.E0 import episode_sampler, split_episodes

# ★ 지표/φ/롤아웃 부품은 전부 R1에서 가져온다. 복사본을 두면 R1 대 R2 비교가 성립하지
#   않는다. R2가 새로 만드는 것은 a₀ 생성 방식과 모드 센서스뿐이다.
from lerobot.scripts.R1 import (
    METHOD_LABEL,
    STAGE_MARKER,
    R1Config,
    chunk_seed,
    encode_global_cond,
    executed_slice,
    load_demo_ref,
    load_policy_at,
    make_probe_env,
    method_color,
    nearest_dist,
    norm_stats_hash,
    parse_kv,
    phi_from_sim,
    phi_spec,
    rollout_metrics,
    spearman_r2,
    stage_ckpt,
    write_csv,
)

NOISE_LABEL = {"fresh": "fresh (R1)", "sticky": "sticky", "ou": "OU"}
NOISE_HATCH = {"fresh": "", "sticky": "//", "ou": ".."}


@dataclass
class R2Config(R1Config):
    """R1Config 전부 + 노이즈 개입/센서스 인자.

    R1Config를 상속하는 이유: make_probe_env/phi_spec/load_policy_at 등이 cfg의 필드를
    이름으로 읽는다. 필드를 따로 선언하면 하나라도 어긋났을 때 "R1과 같은 롤아웃"이라는
    전제가 조용히 깨진다.
    """

    # ── R1 산출물 (필수) ─────────────────────────────────────────────────────
    # demo_ref.npz(φ 정규화 통계와 τ)와 r1_results.jsonl을 여기서 읽는다. 절대 새로
    # 만들지 않는다 — 자가 달라지면 R1과 R2의 d(t)를 나란히 놓을 수 없다.
    r1_run_dir: str = ""

    # ── 무엇을 돌릴 것인가 ───────────────────────────────────────────────────
    # "" -> ckpt_roots × stage 0..num_stages-1 전부. "ewc@2,ewc@3" 처럼 부분 지정 가능
    # (stage는 R1과 같은 0-based. 그림 라벨의 stage3/4가 여기서는 @2,@3이다).
    targets: str = ""
    noise_modes: str = "fresh,sticky,ou"
    ou_rho: float = 0.9

    # switch 임계. 0 이하 -> reference 체크포인트(fresh)의 계획 거리 분포에서 자동 보정.
    # ★ 임계는 **하나만** 쓴다. 체크포인트마다 자기 분포로 보정하면 "많이 흔들리는
    #   체크포인트"가 자기 기준으로는 안 흔들리는 것으로 보여 비교가 무너진다.
    switch_thresh: float = 0.0
    switch_pct: float = 90.0

    # ── 실험 B: 모드 센서스 ──────────────────────────────────────────────────
    census_obs: int = 200          # held-out에서 뽑는 고정 관측 수
    census_k: int = 64             # 고정 a₀ 개수 M
    census_batch: int = 8          # 한 번에 인코딩할 관측 수 (M개는 항상 한 배치)
    census_seed: int = 20260807
    census_kmax: int = 8           # 실루엣으로 훑을 최대 클러스터 수
    census_sil_min: float = 0.5    # 이 값을 못 넘으면 그 관측은 단봉(k=1)으로 본다
    demo_match_pct: float = 50.0   # demo_match 거리 임계를 stage1 분포의 몇 퍼센타일로 둘지

    # ── 제어 ─────────────────────────────────────────────────────────────────
    skip_sticky: bool = False      # 실험 A 건너뛰기 (센서스만)
    skip_census: bool = False      # 실험 B 건너뛰기 (롤아웃만)
    recompute_census: bool = False
    right_key: str = "demo_match_mass"   # 그림 C 오른쪽 축

    # ★ env.reset() 앞에서 전역 numpy RNG를 고정한다. R1에는 없던 것이고, 없으면
    #   같은 rollout_id도 실행할 때마다 다른 물리 상태에서 출발한다 (아래 실측 참조).
    #   그러면 fresh/sticky 비교가 "노이즈 재사용" 대신 "초기 상태 차이"를 재게 된다.
    deterministic_reset: bool = True
    # fresh SR이 R1과 이만큼 넘게 다르면 중단. 0 이하면 검사만 하고 넘어간다.
    # ★ 왜 "정확히 같음"이 아닌가: R1은 이 시드 고정이 없어 **자기 자신도 재현하지
    #   못한다**. 같은 설정으로 두 번 돌리면 d(0)부터 갈린다(실측 max|Δφ|≈5e-3,
    #   48스텝 뒤 max|Δd|≈1.3, τ=3.83 기준). robosuite의 placement initializer가
    #   전역 numpy RNG를 소모하는데 set_init_state는 sim 상태만 덮어쓰고 그 RNG에
    #   의존하는 로봇/컨트롤러 상태는 되돌리지 않기 때문이다. 그래서 R1과의 일치는
    #   "표본오차 안"까지가 원리적 상한이다. 0.2 = 30 롤아웃 중 6개.
    fresh_sr_tol: float = 0.2


# ═════════════════════════════════════════════════════════════════════════════
#  [N] 노이즈 개입 — R2가 R1에 더하는 유일한 것
# ═════════════════════════════════════════════════════════════════════════════
def make_a0(policy, k_samples: int, seed: int, device) -> torch.Tensor:
    """R1의 sample_chunks가 내부에서 만드는 것과 **바이트 단위로 같은** a₀.

    R1: Generator(device).manual_seed(seed % (2**31-1)) -> velocity_net.sample_noise(K,...)
    여기서 그 값을 밖으로 꺼내 우리가 들고 다닌다. 시드 처리가 한 글자라도 다르면
    fresh 모드가 R1을 재현하지 못하고, 그러면 파이프라인 검산이 무의미해진다.
    """
    net = policy.dit_flow.velocity_net
    gen = torch.Generator(device=device).manual_seed(int(seed) % (2**31 - 1))
    return net.sample_noise(k_samples, device, gen)


def a0_fingerprint(a0: torch.Tensor) -> str:
    """a₀의 지문. 세 모드의 출발점이 같은지 검사하는 데만 쓴다."""
    return hashlib.sha1(
        np.ascontiguousarray(a0.detach().float().cpu().numpy()).tobytes()).hexdigest()[:12]


@contextmanager
def fixed_noise(policy, a0: torch.Tensor):
    """velocity_net.sample()이 내부에서 뽑는 a₀를 주어진 텐서로 갈아끼운다.

    ★ Euler 적분 루프를 R2가 다시 구현하지 않는 이유가 이것이다. clip_sample,
      timesteps, dt 규칙이 하나라도 어긋나면 "노이즈만 바꿨다"가 거짓이 된다.
      sample_noise만 인스턴스 속성으로 잠깐 덮어써서 **적분 경로는 원본 그대로** 둔다.
    """
    net = policy.dit_flow.velocity_net
    original = net.sample_noise

    def _fixed(batch_size: int, device, generator=None):
        assert a0.shape[0] >= batch_size, f"a0가 모자란다: {a0.shape[0]} < {batch_size}"
        return a0[:batch_size].to(device=device, dtype=a0.dtype)

    net.sample_noise = _fixed
    try:
        yield
    finally:
        net.sample_noise = original


@torch.no_grad()
def sample_chunks_with(policy, global_cond, a0: torch.Tensor):
    """주어진 a₀에서 출발해 액션 청크를 뽑는다. 반환 (K, horizon, adim) 정규화 공간."""
    k = a0.shape[0]
    cond = global_cond.expand(k, -1)
    with fixed_noise(policy, a0):
        return policy.dit_flow.velocity_net.sample(
            cond, timesteps=policy.dit_flow.num_inference_steps, generator=None)


def advance_a0(mode: str, a0: torch.Tensor, policy, seed: int, rho: float, device):
    """재계획 시점의 a₀ 갱신 규칙. 세 모드의 유일한 차이가 여기 있다."""
    if mode == "fresh":
        return make_a0(policy, a0.shape[0], seed, device)
    if mode == "sticky":
        return a0                                     # 에피소드 내내 그대로
    if mode == "ou":
        # 주변분포가 N(0,I)로 유지되는 형태. ρ=0이면 fresh와 같고 ρ=1이면 sticky와 같다.
        eps = make_a0(policy, a0.shape[0], seed, device)
        return rho * a0 + math.sqrt(max(1.0 - rho * rho, 0.0)) * eps
    raise SystemExit(f"모르는 noise_mode: {mode!r} (fresh|sticky|ou)")


# ═════════════════════════════════════════════════════════════════════════════
#  [P] 계획 겹침 — 연속 재계획이 같은 것을 말하는가
# ═════════════════════════════════════════════════════════════════════════════
def plan_windows(policy) -> tuple[slice, slice, int]:
    """연속 두 재계획이 **같은 env 스텝**을 가리키는 청크 구간.

    horizon=16, n_obs_steps=2, n_action_steps=8 기준:
        executed_slice = chunk[1:9]  -> env 스텝 t..t+7 (실제로 실행되는 부분)
        계획 i의 꼬리   = chunk[9:16] -> env 스텝 t+8..t+14  (실행되지 않고 버려지는 예고)
        계획 i+1의 머리 = chunk[1:8]  -> env 스텝 t+8..t+14  (같은 시각을 다시 말한 것)
    둘의 거리가 "재계획이 말을 바꿨는가"다. executed_slice끼리는 시간이 겹치지 않아
    이 질문에 답할 수 없다 — 그래서 버려지는 꼬리를 쓴다.
    """
    start = policy.config.n_obs_steps - 1
    n_act = policy.config.n_action_steps
    horizon = policy.config.horizon
    overlap = horizon - start - n_act              # 7
    if overlap <= 0:
        raise SystemExit(
            f"horizon({horizon})이 짧아 연속 재계획이 겹치지 않는다 "
            f"(start={start}, n_action_steps={n_act}). plan_consistency를 정의할 수 없다.")
    tail = slice(start + n_act, start + n_act + overlap)   # [9, 16)
    head = slice(start, start + overlap)                   # [1, 8)
    return tail, head, overlap


def unnorm(policy, chunk: torch.Tensor, sl: slice) -> torch.Tensor:
    """청크의 한 구간을 실제 단위로. executed_slice와 같은 역정규화를 쓴다."""
    return policy.unnormalize_outputs({ACTION: chunk[..., sl, :]})[ACTION]


# ═════════════════════════════════════════════════════════════════════════════
#  [A] 롤아웃 — R1.rollout_checkpoint와 같은 절차, a₀ 생성만 다르다
# ═════════════════════════════════════════════════════════════════════════════
def rollout_noise_mode(cfg: R2Config, env, spec: dict, ref: dict, policy, ref_policy,
                       device, method: str, stage: int, ckpt: Path, mode: str) -> dict:
    """체크포인트 하나 × 노이즈 모드 하나로 num_rollouts개를 돌린다.

    R1.rollout_checkpoint와 다른 점은 세 곳뿐이다.
      1) a₀를 밖에서 들고 다니며 advance_a0로 갱신한다 (R1은 매번 시드에서 새로 뽑는다)
      2) 연속 재계획의 겹침 거리(plan_dist)를 기록한다
      3) 첫 a₀가 세 모드에서 같은지 self-check
    나머지(초기 상태 지정, 정착, K샘플, 0번 샘플 실행, 패딩)는 전부 R1 그대로다.
    """
    max_steps = cfg.max_steps or cfg.env.episode_length
    n_act = policy.config.n_action_steps
    stride = cfg.action_eval_stride or n_act
    # ★ R1은 재계획이 아닌 평가 스텝에서도 a₀를 새로 뽑는다. R2는 a₀를 들고 다니므로
    #   두 눈금이 어긋나면 fresh가 R1을 재현하지 못한다. 같을 때만 허용한다.
    if stride != n_act:
        raise SystemExit(
            f"R2는 action_eval_stride(={stride})가 n_action_steps(={n_act})와 같아야 한다. "
            f"a₀를 재계획 시점에만 갱신하기 때문이다 (--action_eval_stride=0 이 기본).")
    da_steps = np.arange(0, max_steps, stride)
    n_eval = len(da_steps)
    n_replan = max_steps // n_act + 1
    D = len(spec["labels"])
    R = cfg.num_rollouts
    tail, head, _ = plan_windows(policy)

    init_states = env.unwrapped._init_states
    if R > len(init_states):
        raise SystemExit(f"num_rollouts={R} > 사용 가능한 초기 상태 {len(init_states)}개")
    init_hash = hashlib.sha1(
        np.ascontiguousarray(np.asarray(init_states)[:R], dtype=np.float64).tobytes()).hexdigest()[:12]

    phi_all = np.zeros((R, max_steps, D), dtype=np.float32)
    da_all = np.full((R, n_eval), np.nan, dtype=np.float32)
    var_cur_all = np.full((R, n_eval), np.nan, dtype=np.float32)
    var_ref_all = np.full((R, n_eval), np.nan, dtype=np.float32)
    plan_dist = np.full((R, n_replan), np.nan, dtype=np.float32)
    a0_sig = np.empty(R, dtype="<U12")
    lengths = np.zeros(R, dtype=np.int32)
    success = np.zeros(R, dtype=bool)

    task_text = env.unwrapped.task_description
    null_action = np.zeros(env.action_space.shape, dtype=np.float32)
    null_action[-1] = -1.0

    for rid in range(R):
        # ★ R1에 없던 한 줄. robosuite의 placement initializer가 전역 numpy RNG를
        #   소모하는데, set_init_state는 sim 상태만 덮어쓰고 그 RNG에 의존하는
        #   로봇/컨트롤러 상태는 되돌리지 않는다. 그래서 시드를 안 고정하면 같은
        #   rollout_id도 매번 다른 물리 상태에서 출발하고(실측: settle 5스텝 뒤
        #   max|Δφ|≈5e-3, 48스텝 뒤 max|Δd|≈1.3), fresh 대 sticky 비교가
        #   "노이즈 재사용"이 아니라 "초기 상태 차이"를 재게 된다.
        if cfg.deterministic_reset:
            np.random.seed(cfg.rollout_seed_base + rid)
        env.reset()
        # ★ robosuite는 reset마다 MjSim을 새로 만든다. 루프 밖에서 캐시하면 죽은 객체를 본다.
        sim = env.unwrapped._env.sim
        spec = phi_spec(env, spec["objects"])
        raw = env.unwrapped.set_init_state(init_states[rid])
        obs = env.unwrapped._format_raw_obs(raw)
        for _ in range(cfg.settle_steps):
            obs, _r, _term, _trunc, _i = env.step(null_action)
        policy.reset()
        ref_policy.reset()

        # ★ 세 모드의 출발점을 같게 만드는 한 줄. sticky의 a₀ = fresh의 t=0 a₀.
        a0 = make_a0(policy, cfg.num_samples, chunk_seed(cfg, rid, 0), device)
        a0_sig[rid] = a0_fingerprint(a0)
        prev_tail = None

        hist: list[dict] = []
        queue: list[np.ndarray] = []
        t = 0
        for t in range(max_steps):
            proc = preprocess_observation(obs)
            proc.pop("task", None)
            if not hist:
                hist = [proc] * policy.config.n_obs_steps
            else:
                hist = (hist + [proc])[-policy.config.n_obs_steps:]

            phi_all[rid, t] = phi_from_sim(sim, spec)

            replan = len(queue) == 0
            evaluate = (t % stride) == 0
            if replan and t > 0:
                a0 = advance_a0(mode, a0, policy, chunk_seed(cfg, rid, t), cfg.ou_rho, device)

            if replan or evaluate:
                cond_cur = encode_global_cond(policy, hist, task_text, device)
                chunk_cur = sample_chunks_with(policy, cond_cur, a0)
                exec_cur = executed_slice(policy, chunk_cur)      # (K, n_act, adim) 실제 단위

            if evaluate:
                # θ*₁은 **같은 a₀**로 평가만 한다. 그래야 Δa가 샘플링 잡음이 아니라 정책 차이다.
                cond_ref = encode_global_cond(ref_policy, hist, task_text, device)
                chunk_ref = sample_chunks_with(ref_policy, cond_ref, a0)
                exec_ref = executed_slice(ref_policy, chunk_ref)
                j = t // stride
                da_all[rid, j] = float(torch.linalg.vector_norm(
                    exec_cur.mean(dim=0) - exec_ref.mean(dim=0)))
                var_cur_all[rid, j] = float(exec_cur.var(dim=0, unbiased=False).mean())
                var_ref_all[rid, j] = float(exec_ref.var(dim=0, unbiased=False).mean())

            if replan:
                # 실행 계획(0번 샘플)의 머리가 직전 계획의 꼬리와 같은 말을 하는가.
                cur_head = unnorm(policy, chunk_cur[0], head)
                if prev_tail is not None:
                    plan_dist[rid, t // n_act] = float(
                        torch.linalg.vector_norm(cur_head - prev_tail))
                prev_tail = unnorm(policy, chunk_cur[0], tail)
                queue = list(exec_cur[0].cpu().numpy())

            action = queue.pop(0)
            obs, _reward, terminated, truncated, _info = env.step(
                np.asarray(action, dtype=np.float32))
            if terminated or truncated:
                success[rid] = bool(terminated)
                break

        lengths[rid] = t + 1
        # 조기 종료분은 마지막 값으로 패딩(생존 편향 방지). R1과 같은 규칙이어야 d/Δa가 비교된다.
        if lengths[rid] < max_steps:
            phi_all[rid, lengths[rid]:] = phi_all[rid, lengths[rid] - 1]
        last = np.where(np.isfinite(da_all[rid]))[0]
        if len(last):
            da_all[rid, last[-1] + 1:] = da_all[rid, last[-1]]
            var_cur_all[rid, last[-1] + 1:] = var_cur_all[rid, last[-1]]
            var_ref_all[rid, last[-1] + 1:] = var_ref_all[rid, last[-1]]
        # plan_dist는 패딩하지 않는다. "재계획이 몇 번 있었나"가 롤아웃 길이에 따라
        # 달라지는 것이 사실이고, switch_count는 아래에서 비율이 아니라 개수로 센다.

        pd_alive = plan_dist[rid][np.isfinite(plan_dist[rid])]
        logging.info(
            f"[R2][A] {method} stage{stage} {mode} rollout {rid + 1}/{R}: "
            f"len={lengths[rid]} success={bool(success[rid])} "
            f"plan_d_med={np.median(pd_alive) if len(pd_alive) else float('nan'):.3f}")

    d_all = np.stack([
        nearest_dist(phi_all[r], ref["ref_z"], ref["mean"], ref["std"], device) for r in range(R)
    ]).astype(np.float32)

    return {
        "d": d_all,
        "da": da_all,
        "da_steps": da_steps.astype(np.int32),
        "var_cur": var_cur_all,
        "var_ref": var_ref_all,
        "plan_dist": plan_dist,
        "a0_sig": a0_sig,
        "lengths": lengths,
        "success": success,
        "init_hash": np.array(init_hash),
        "meta": np.array(json.dumps({
            "method": method, "stage": stage, "noise_mode": mode, "ckpt": str(ckpt),
            "probe_task": cfg.probe_task, "num_rollouts": R, "max_steps": max_steps,
            "num_samples": cfg.num_samples, "stride": stride, "settle_steps": cfg.settle_steps,
            "seed_base": cfg.rollout_seed_base, "ou_rho": cfg.ou_rho,
            "norm_hash": norm_stats_hash(policy), "ref_norm_hash": norm_stats_hash(ref_policy),
            "n_action_steps": n_act, "labels": spec["labels"],
        })),
    }


def plan_metrics(rec: dict, thresh: float) -> tuple[list[float], list[int]]:
    """롤아웃별 (plan_consistency, switch_count).

    plan_consistency = 겹침 구간 L2 거리의 중앙값 (임계 없는 연속 지표)
    switch_count     = 그 거리가 thresh를 넘은 재계획 횟수 (임계 지표)
    """
    pc, sc = [], []
    for i in range(rec["plan_dist"].shape[0]):
        v = rec["plan_dist"][i]
        v = v[np.isfinite(v)]
        pc.append(float(np.median(v)) if len(v) else float("nan"))
        sc.append(int((v > thresh).sum()) if len(v) else 0)
    return pc, sc


# ═════════════════════════════════════════════════════════════════════════════
#  [B] 모드 센서스 — 고정 관측 × 고정 a₀ 격자
# ═════════════════════════════════════════════════════════════════════════════
def build_census_probe(cfg: R2Config, policy, device, out_path: Path) -> dict:
    """고정 관측 집합 + 고정 a₀ 세트 + 데모 행동을 한 번만 만들어 굳힌다.

    ★ 관측은 **인덱스로** 저장한다. 이미지를 저장하면 200개만 해도 수백 MB인데,
      데이터셋은 결정적이므로 인덱스가 곧 관측이다. 해시로 동일성을 검사한다.
    """
    repo_id = f"{cfg.dataset_prefix}{cfg.probe_task}"
    _, holdout_eps = split_episodes(repo_id, None, cfg.holdout_episodes)
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps=resolve_delta_timestamps(policy.config, LeRobotDatasetMetadata(repo_id)),
        video_backend=cfg.dataset.video_backend,
    )
    # E0/R1과 같은 샘플러로 "액션 청크가 에피소드 밖으로 나가지 않는" 프레임만 고른다.
    pool = np.array(sorted(episode_sampler(cfg, dataset, holdout_eps, shuffle=False)), dtype=np.int64)
    if len(pool) < cfg.census_obs:
        logging.warning(colored(
            f"[R2][B] held-out 가용 프레임 {len(pool)}개 < census_obs={cfg.census_obs}. 전부 쓴다.",
            "yellow"))
    n = min(cfg.census_obs, len(pool))
    # 균등 샘플(무작위가 아니라 등간격). 에피소드 초반/후반이 고르게 들어간다.
    idx = pool[np.linspace(0, len(pool) - 1, n).round().astype(int)]

    start = policy.config.n_obs_steps - 1
    end = start + policy.config.n_action_steps
    demo_act = np.stack([
        np.asarray(dataset[int(i)][ACTION][start:end], dtype=np.float32) for i in idx
    ])                                                     # (n, n_act, adim) 실제 단위

    # a₀의 모양은 velocity_net에서 직접 읽는다(config 필드명이 바뀌어도 따라간다).
    net = policy.dit_flow.velocity_net
    gen = torch.Generator(device="cpu").manual_seed(cfg.census_seed)
    a0 = torch.randn(cfg.census_k, net.ac_chunk, net.ac_dim, generator=gen)

    probe = {
        "obs_idx": idx,
        "a0": a0.numpy().astype(np.float32),
        "demo_act": demo_act,
        "repo_id": np.array(repo_id),
        "holdout_episodes": np.array(holdout_eps),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **probe)
    logging.info(colored(
        f"[R2][B] census probe 저장 -> {out_path}  (관측 {n}, a₀ {cfg.census_k}, "
        f"hash={census_hash(probe)})", "green"))
    return probe


def census_hash(probe: dict) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(probe["obs_idx"], dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(probe["a0"], dtype=np.float32).tobytes())
    return h.hexdigest()[:12]


@torch.no_grad()
def run_census(cfg: R2Config, policy, probe: dict, device) -> np.ndarray:
    """각 관측에서 M개 a₀로 행동 청크를 만든다. 반환 (n_obs, M, n_act*adim) 실제 단위."""
    repo_id = str(probe["repo_id"])
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps=resolve_delta_timestamps(policy.config, LeRobotDatasetMetadata(repo_id)),
        video_backend=cfg.dataset.video_backend,
    )
    idx = probe["obs_idx"]
    a0 = torch.as_tensor(probe["a0"], dtype=torch.float32, device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, [int(i) for i in idx]),
        batch_size=cfg.census_batch, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    out = []
    seen = 0
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        batch = policy.normalize_inputs(batch)
        if policy.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[k] for k in policy.config.image_features], dim=-4)
        cond = policy.dit_flow._prepare_global_conditioning(batch)     # (B, 2576)
        for b in range(cond.shape[0]):
            chunk = sample_chunks_with(policy, cond[b: b + 1], a0)     # (M, horizon, adim)
            exe = executed_slice(policy, chunk)                        # (M, n_act, adim)
            out.append(exe.reshape(exe.shape[0], -1).cpu().numpy().astype(np.float32))
        seen += cond.shape[0]
        if seen % (10 * cfg.census_batch) == 0:
            logging.info(f"[R2][B]   census {seen}/{len(idx)} obs")
    return np.stack(out)                                               # (n_obs, M, F)


# ── 군집화: sklearn 없이. M=64라 O(M³)도 즉시 끝난다 ─────────────────────────
def average_linkage(dist: np.ndarray) -> list[np.ndarray]:
    """평균연결 응집 군집화. k=M..1 각각의 라벨 배열을 반환(인덱스 k-1이 k개 클러스터)."""
    m = dist.shape[0]
    labels = np.arange(m)
    out = [labels.copy()]
    active = [{i} for i in range(m)]
    d = dist.astype(np.float64).copy()
    np.fill_diagonal(d, np.inf)
    alive = list(range(m))
    for _ in range(m - 1):
        sub = d[np.ix_(alive, alive)]
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        a, b = alive[i], alive[j]
        na, nb = len(active[a]), len(active[b])
        for c in alive:
            if c in (a, b):
                continue
            d[a, c] = d[c, a] = (na * d[a, c] + nb * d[b, c]) / (na + nb)
        active[a] = active[a] | active[b]
        alive.remove(b)
        lab = np.empty(m, dtype=int)
        for new, cl in enumerate(alive):
            for member in active[cl]:
                lab[member] = new
        out.append(lab)
    return out[::-1]     # out[0] = 1개 클러스터, out[k-1] = k개


def silhouette(dist: np.ndarray, labels: np.ndarray) -> float:
    """평균 실루엣. 클러스터가 1개거나 크기 1짜리만 있으면 nan."""
    k = labels.max() + 1
    if k < 2:
        return float("nan")
    scores = []
    for i in range(len(labels)):
        same = labels == labels[i]
        if same.sum() <= 1:
            scores.append(0.0)
            continue
        a = dist[i, same & (np.arange(len(labels)) != i)].mean()
        b = min(dist[i, labels == c].mean() for c in range(k) if c != labels[i])
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


def cluster_reference(X: np.ndarray, kmax: int, sil_min: float) -> np.ndarray:
    """관측 하나의 M개 행동을 군집화. 클러스터 수는 실루엣으로 고르고, 못 넘으면 단봉.

    ★ 클러스터 수를 관측 간에 고정하지 않는다. 어떤 관측은 진짜로 단봉이고(집어 든 뒤
      갈 곳이 하나), 어떤 관측은 다봉이다(어느 그릇부터 잡을까). 고정하면 없는 경계를
      만들어 assign_change가 잡음만 재게 된다.
    """
    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    parts = average_linkage(dist)
    best_k, best_s = 1, -np.inf
    for k in range(2, min(kmax, len(X)) + 1):
        s = silhouette(dist, parts[k - 1])
        if np.isfinite(s) and s > best_s:
            best_k, best_s = k, s
    return parts[0] if best_s < sil_min else parts[best_k - 1]


def mode_scales(ref_X: np.ndarray, labels: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """관측별 (모드 간 중심 거리, 모드 내 반경). 단봉 관측은 nan.

    ★ center_shift를 이 눈금으로 나눠야 "중심이 움직였다"가 뜻을 갖는다. 행동 단위의
      raw 거리만 보면 크고 작음을 판단할 기준이 없다. 중심 이동이 모드 간격보다 크면
      경계 침식이 아니라 사상 전체가 옮겨간 것이고, 모드 내 반경 수준이면 제자리다.
    """
    n = ref_X.shape[0]
    sep = np.full(n, np.nan)
    rad = np.full(n, np.nan)
    for o in range(n):
        lab = labels[o]
        k = lab.max() + 1
        if k < 2:
            continue
        C = np.stack([ref_X[o][lab == c].mean(axis=0) for c in range(k)])
        dd = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)
        sep[o] = dd[np.triu_indices(k, 1)].mean()
        rad[o] = np.mean([np.linalg.norm(ref_X[o][lab == c] - C[c], axis=-1).mean()
                          for c in range(k)])
    return sep, rad


def census_metrics(ref_X: np.ndarray, cur_X: np.ndarray, demo: np.ndarray,
                   labels: list[np.ndarray], demo_tau: float) -> dict:
    """관측별 지표를 계산해 배열로 돌려준다. 군집 구조는 **stage1 것만** 쓴다."""
    n_obs = ref_X.shape[0]
    center_shift = np.full(n_obs, np.nan, dtype=np.float64)
    assign_change = np.full(n_obs, np.nan, dtype=np.float64)
    weight_entropy = np.full(n_obs, np.nan, dtype=np.float64)
    demo_mass = np.full(n_obs, np.nan, dtype=np.float64)
    n_modes = np.zeros(n_obs, dtype=int)

    for o in range(n_obs):
        lab = labels[o]
        k = lab.max() + 1
        n_modes[o] = k
        # stage1 클러스터 중심 (기준점. 이후 스테이지는 여기에 최근접 배정된다)
        centers = np.stack([ref_X[o][lab == c].mean(axis=0) for c in range(k)])

        # center_shift: 같은 a₀ 집합이 이 스테이지에서 만드는 중심의 이동 (크기 가중 평균)
        shifts, sizes = [], []
        for c in range(k):
            m = lab == c
            shifts.append(np.linalg.norm(cur_X[o][m].mean(axis=0) - centers[c]))
            sizes.append(m.sum())
        center_shift[o] = float(np.average(shifts, weights=sizes))

        # assign_change: 새로 군집화하지 않고 stage1 중심에 최근접 배정한다.
        # (다시 군집화하면 라벨 대응 문제가 생겨 "바뀐 비율"을 정의할 수 없다)
        dd = np.linalg.norm(cur_X[o][:, None, :] - centers[None, :, :], axis=-1)
        new_lab = dd.argmin(axis=1)
        assign_change[o] = float((new_lab != lab).mean())

        p = np.bincount(new_lab, minlength=k) / len(new_lab)
        nz = p[p > 0]
        weight_entropy[o] = float(-(nz * np.log(nz)).sum())

        demo_mass[o] = float(
            (np.linalg.norm(cur_X[o] - demo[o][None, :], axis=-1) <= demo_tau).mean())

    return {
        "center_shift": center_shift,
        "assign_change": assign_change,
        "weight_entropy": weight_entropy,
        "demo_match_mass": demo_mass,
        "n_modes": n_modes,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  메인 (train.py / E0 / R1과 같은 [1]~ 순서)
# ═════════════════════════════════════════════════════════════════════════════
def r1_checkpoint_rows(r1_dir: Path) -> dict[tuple[str, int], dict]:
    p = r1_dir / "r1_results.jsonl"
    if not p.exists():
        raise SystemExit(
            f"R1 결과가 없다: {p}\n  --r1_run_dir 가 R1 산출물을 가리켜야 한다. "
            f"R2는 demo_ref/τ/정규화 통계를 절대 새로 만들지 않는다.")
    uniq: dict[tuple[str, int], dict] = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "checkpoint":
            uniq[(r["method"], r["stage"])] = r
    return uniq


@parser.wrap()
def main(cfg: R2Config):
    # ── [1] 설정 ─────────────────────────────────────────────────────────────
    cfg.validate()
    cfg.save_checkpoint = False
    if not cfg.ckpt_roots:
        raise SystemExit('--ckpt_roots 가 필요하다 (예: "seq=...,ewc=...,frozen=...")')
    if not cfg.r1_run_dir:
        raise SystemExit(
            "--r1_run_dir 가 필요하다. demo_ref.npz(φ 정규화 통계와 τ)는 R1이 만든 것을 "
            "그대로 읽는다 — 자가 달라지면 R1과 R2의 d(t)를 나란히 놓을 수 없다.")
    roots = parse_kv(cfg.ckpt_roots)
    r1_dir = Path(cfg.r1_run_dir)
    run_tag = cfg.run_tag or f"{getattr(cfg.env, 'benchmark', 'libero')}_probe{cfg.probe_task}"
    run_dir = Path(cfg.out_root) / run_tag
    (run_dir / "cache").mkdir(parents=True, exist_ok=True)
    results = run_dir / "r2_results.jsonl"
    modes = [m.strip() for m in cfg.noise_modes.split(",") if m.strip()]
    logging.info(pformat(cfg.to_dict()))
    logging.info(colored(
        f"[R2] probe_task={cfg.probe_task}  methods={list(roots)}  modes={modes}  "
        f"rollouts={cfg.num_rollouts}  -> {run_dir}", "green", attrs=["bold"]))

    # ── [2] 로거: 스칼라 표와 그림만 낸다 (wandb 없음) ────────────────────────
    # ── [3] 재현성 ───────────────────────────────────────────────────────────
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # ── [4] 디바이스 ─────────────────────────────────────────────────────────
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── [5] 데이터셋 메타 ────────────────────────────────────────────────────
    ds_meta = LeRobotDatasetMetadata(f"{cfg.dataset_prefix}{cfg.probe_task}")

    # ── [6] [위생] demo_ref는 R1 것을 읽는다. 절대 새로 만들지 않는다 ────────
    ref_path = r1_dir / "demo_ref.npz"
    if not ref_path.exists():
        raise SystemExit(
            f"demo_ref.npz가 없다: {ref_path}\n"
            f"  R1을 먼저 돌려라. R2는 τ와 z-정규화 통계를 재계산하지 않는다.")
    ref = load_demo_ref(ref_path)
    tau = float(ref["tau"])
    logging.info(colored(
        f"[R2] demo_ref (R1 산출물) 재사용: {ref_path}  τ={tau:.3f}  "
        f"참조 {ref['ref_z'].shape[0]} 프레임", "cyan"))
    r1_rows = r1_checkpoint_rows(r1_dir)

    # ── [7] reference 정책 θ*₁ ───────────────────────────────────────────────
    ref_ckpt = Path(cfg.ref_ckpt) if cfg.ref_ckpt \
        else stage_ckpt(next(iter(roots.values())), cfg.probe_task)
    logging.info(colored(f"[R2] reference policy θ*₁ = {ref_ckpt}", "cyan"))
    ref_policy = load_policy_at(cfg, ref_ckpt, ds_meta, device)

    # ── [8] 볼 체크포인트 목록 ───────────────────────────────────────────────
    # reference stage는 모든 상대 지표의 기준점이라 --targets가 뭐든 항상 포함한다.
    all_targets = [(m, k) for m in roots for k in range(cfg.num_stages)]
    if cfg.targets:
        want = set()
        for item in cfg.targets.split(","):
            item = item.strip()
            if not item:
                continue
            m, _, s = item.partition("@")
            want.add((m.strip(), int(s)))
        ref_method = next(iter(roots))
        want.add((ref_method, cfg.probe_task))
        targets = [t for t in all_targets if t in want]
        missing = want - set(all_targets)
        if missing:
            raise SystemExit(f"--targets 에 없는 조합이 있다: {sorted(missing)}")
    else:
        targets = all_targets
    ref_key = (next(iter(roots)), cfg.probe_task)
    logging.info(colored(
        f"[R2] targets: {[f'{m}@{k}' for m, k in targets]}  (reference={ref_key[0]}@{ref_key[1]})",
        "cyan"))

    def emit(row: dict):
        with results.open("a") as f:
            f.write(json.dumps(row) + "\n")

    env = None
    init_hashes: dict[str, str] = {}
    recs: dict[tuple[str, int, str], dict] = {}

    # ═══ 실험 A: 끈끈한 노이즈 ════════════════════════════════════════════════
    if not cfg.skip_sticky:
        env = make_probe_env(cfg)
        objects = list(ref["objects"])
        spec = phi_spec(env, objects)
        logging.info(f"[R2] φ 차원 {len(spec['labels'])} (R1과 동일: {objects})")

        for method, stage in targets:
            ckpt = stage_ckpt(roots[method], stage)
            if not Path(ckpt).exists():
                logging.warning(colored(f"[R2] 체크포인트 없음, 건너뜀: {ckpt}", "yellow"))
                continue
            policy = None
            for mode in modes:
                cache = run_dir / "cache" / f"sticky_{method}_stage{stage}_{mode}.npz"
                if cache.exists() and not cfg.recompute_rollouts:
                    z = np.load(cache, allow_pickle=False)
                    rec = {k: z[k] for k in z.files}
                    logging.info(colored(f"[R2][A] 캐시 재사용: {cache}", "cyan"))
                else:
                    if policy is None:
                        policy = load_policy_at(cfg, ckpt, ds_meta, device)
                    logging.info(colored(
                        f"[R2][A] rollout {method} stage{stage} noise={mode}: {ckpt}",
                        "cyan", attrs=["bold"]))
                    rec = rollout_noise_mode(cfg, env, spec, ref, policy, ref_policy, device,
                                             method, stage, Path(ckpt), mode)
                    np.savez_compressed(cache, **rec)
                    logging.info(f"[R2][A] 원시 기록 저장 -> {cache}")
                recs[(method, stage, mode)] = rec

                # ── 위생 체크 ────────────────────────────────────────────────
                meta = json.loads(str(rec["meta"]))
                want = {"num_samples": cfg.num_samples,
                        "max_steps": cfg.max_steps or cfg.env.episode_length,
                        "settle_steps": cfg.settle_steps,
                        "seed_base": cfg.rollout_seed_base,
                        "probe_task": cfg.probe_task,
                        "noise_mode": mode}
                bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
                assert not bad, (f"캐시와 설정이 다르다 (캐시값, 현재값): {bad} — "
                                 f"--recompute_rollouts 로 다시 돌려라 ({cache})")
                # 초기 상태가 모든 체크포인트·모드에서 같아야 짝지은 비교가 성립한다.
                ih = str(rec["init_hash"])
                init_hashes[f"{method}@{stage}:{mode}"] = ih
                first_key, first_hash = next(iter(init_hashes.items()))
                assert ih == first_hash, (
                    f"초기 상태가 다르다: {method}@{stage}:{mode}={ih} vs {first_key}={first_hash}")
                # sticky/ou의 첫 a₀가 fresh의 t=0 a₀와 같은가 (개입의 유일성 검사)
                if mode != "fresh" and (method, stage, "fresh") in recs:
                    assert np.array_equal(rec["a0_sig"], recs[(method, stage, "fresh")]["a0_sig"]), (
                        f"{mode}의 첫 a₀가 fresh와 다르다 — 노이즈 재사용 외의 차이가 섞였다.")

            # fresh SR이 R1과 일치하는가 — 파이프라인 검산
            if "fresh" in modes and (method, stage, "fresh") in recs:
                sr = float(recs[(method, stage, "fresh")]["success"].mean())
                r1sr = r1_rows.get((method, stage), {}).get("sr")
                # ★ 이 검산은 "정확히 같은가"가 아니라 "표본오차 안인가"를 묻는다.
                #   R1은 env.reset()의 전역 RNG를 고정하지 않아 자기 자신도 재현하지
                #   못하기 때문이다(R2Config.fresh_sr_tol 주석 참조). 대신 이 차이를
                #   전부 JSONL에 남겨, R1 숫자에 붙은 실행 간 변동을 12개 체크포인트에
                #   대해 사후에 정량화할 수 있게 한다.
                if r1sr is None:
                    logging.warning(colored(
                        f"[R2] R1에 {method}@{stage} 행이 없어 fresh 검산을 건너뛴다.", "yellow"))
                else:
                    gap = abs(sr - float(r1sr))
                    emit({"kind": "fresh_check", "method": method, "stage": stage,
                          "sr_r2": sr, "sr_r1": float(r1sr), "gap": gap,
                          "tol": cfg.fresh_sr_tol, "num_rollouts": cfg.num_rollouts})
                    msg = (f"[R2] fresh 검산 {method}@{stage}: R2={sr:.3f} vs R1={float(r1sr):.3f} "
                           f"(차이 {gap:.3f})")
                    if cfg.fresh_sr_tol > 0 and gap > cfg.fresh_sr_tol:
                        raise SystemExit(colored(
                            msg + f" — 허용치 {cfg.fresh_sr_tol:.2f}를 넘었다. 노이즈 개입 외의 "
                            f"차이가 파이프라인에 섞였을 수 있다. 시뮬레이터 재현성 문제라면 "
                            f"--fresh_sr_tol 을 올려라.", "red"))
                    logging.info(colored(msg, "green" if gap <= cfg.fresh_sr_tol else "yellow"))
            del policy
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ── switch 임계 보정: reference 체크포인트(fresh)의 계획 거리 분포에서 한 번만 ──
        thresh = cfg.switch_thresh
        if thresh <= 0:
            base = recs.get((*ref_key, "fresh"))
            if base is None:
                raise SystemExit(
                    f"switch 임계를 보정할 reference({ref_key[0]}@{ref_key[1]}, fresh)가 없다. "
                    f"--switch_thresh 로 직접 주거나 reference를 함께 돌려라.")
            v = base["plan_dist"][np.isfinite(base["plan_dist"])]
            thresh = float(np.percentile(v, cfg.switch_pct))
        logging.info(colored(
            f"[R2][A] switch 임계 = {thresh:.4f} "
            f"({'수동' if cfg.switch_thresh > 0 else f'{ref_key[0]}@{ref_key[1]} fresh의 {cfg.switch_pct:g}퍼센타일'})",
            "green", attrs=["bold"]))

        for (method, stage, mode), rec in recs.items():
            per = rollout_metrics(rec, tau)
            pc, sc = plan_metrics(rec, thresh)
            for i, p in enumerate(per):
                emit({"kind": "rollout", "method": method, "stage": stage, "noise_mode": mode,
                      "probe_task": cfg.probe_task, "seed": cfg.seed,
                      "plan_consistency": pc[i], "switch_count": sc[i], **p})
            n = len(per)
            sr = float(np.mean([p["success"] for p in per]))
            emit({"kind": "checkpoint_sticky", "method": method, "stage": stage,
                  "noise_mode": mode, "probe_task": cfg.probe_task, "seed": cfg.seed,
                  "ckpt": str(stage_ckpt(roots[method], stage)),
                  "num_rollouts": n, "sr": sr,
                  "sr_se": float(math.sqrt(max(sr * (1 - sr), 0.0) / max(n, 1))),
                  "switch_count": float(np.median(sc)),
                  "switch_thresh": thresh,
                  "plan_consistency": float(np.nanmedian(pc)),
                  "dwell": float(np.median([p["dwell"] for p in per])),
                  "dauc": float(np.median([p["dauc"] for p in per])),
                  "t_star": float(np.median([p["t_star"] for p in per])),
                  "da_mean": float(np.nanmedian([p["da_mean"] for p in per
                                                 if p["da_mean"] is not None] or [np.nan])),
                  "tau": tau})
            logging.info(colored(
                f"[R2][A] {method} stage{stage} {mode}: SR={sr:.3f}  "
                f"switch={np.median(sc):.1f}  plan_d={np.nanmedian(pc):.3f}  "
                f"dwell={np.median([p['dwell'] for p in per]):.2f}", "green"))
        env.close()
        env = None

    # ═══ 실험 B: 모드 센서스 ══════════════════════════════════════════════════
    if not cfg.skip_census:
        probe_path = run_dir / "cache" / "census_probe.npz"
        if probe_path.exists() and not cfg.recompute_census:
            z = np.load(probe_path, allow_pickle=False)
            probe = {k: z[k] for k in z.files}
            logging.info(colored(
                f"[R2][B] census probe 재사용: {probe_path} (hash={census_hash(probe)})", "cyan"))
        else:
            probe = build_census_probe(cfg, ref_policy, device, probe_path)
        phash = census_hash(probe)

        cen: dict[tuple[str, int], np.ndarray] = {}
        for method, stage in targets:
            ckpt = stage_ckpt(roots[method], stage)
            if not Path(ckpt).exists():
                continue
            cache = run_dir / "cache" / f"census_{method}_stage{stage}.npz"
            if cache.exists() and not cfg.recompute_census:
                z = np.load(cache, allow_pickle=False)
                assert str(z["probe_hash"]) == phash, (
                    f"census 캐시의 관측/a₀ 집합이 다르다 ({cache}): "
                    f"{str(z['probe_hash'])} vs {phash}. --recompute_census 로 다시 돌려라.")
                cen[(method, stage)] = z["chunks"]
                logging.info(colored(f"[R2][B] 캐시 재사용: {cache}", "cyan"))
                continue
            logging.info(colored(
                f"[R2][B] census {method} stage{stage}: {ckpt}", "cyan", attrs=["bold"]))
            policy = load_policy_at(cfg, ckpt, ds_meta, device)
            X = run_census(cfg, policy, probe, device)
            np.savez_compressed(cache, chunks=X, probe_hash=np.array(phash))
            cen[(method, stage)] = X
            logging.info(f"[R2][B] 저장 -> {cache}  shape={X.shape}")
            del policy
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if ref_key not in cen:
            raise SystemExit(
                f"reference({ref_key[0]}@{ref_key[1]})의 census가 없다 — 모든 상대 지표의 기준점이다.")
        ref_X = cen[ref_key]
        demo = probe["demo_act"].reshape(probe["demo_act"].shape[0], -1)

        logging.info("[R2][B] stage1 군집 구조를 만든다 (이후 스테이지는 이 구조로 평가된다)")
        labels = [cluster_reference(ref_X[o], cfg.census_kmax, cfg.census_sil_min)
                  for o in range(ref_X.shape[0])]
        n_modes = np.array([lab.max() + 1 for lab in labels])
        # ★ 가설의 전제 검사. 대부분 단봉이면 "경계 침식"이라는 이야기 자체가 성립하지 않는다.
        uni = float((n_modes == 1).mean())
        logging.info(colored(
            f"[R2][B] n_modes: 평균 {n_modes.mean():.2f}  중앙값 {np.median(n_modes):.0f}  "
            f"단봉 비율 {uni:.1%}", "green" if uni < 0.5 else "yellow", attrs=["bold"]))
        if uni >= 0.5:
            logging.warning(colored(
                "[R2][B] 관측의 절반 이상이 단봉이다. 모드 경계가 애초에 드물다는 뜻이므로 "
                "assign_change를 '경계 침식'으로 읽기 전에 이 사실을 먼저 보고해야 한다.", "yellow"))

        # demo_match 거리 임계: stage1 분포에서 보정 (기준 정책의 baseline이 곧 눈금)
        ref_demo_d = np.linalg.norm(ref_X - demo[:, None, :], axis=-1)
        demo_tau = float(np.percentile(ref_demo_d, cfg.demo_match_pct))
        logging.info(f"[R2][B] demo_match 거리 임계 = {demo_tau:.4f} "
                     f"(stage1 분포의 {cfg.demo_match_pct:g}퍼센타일)")

        # ★ 모드 구조의 자연 눈금. center_shift를 여기에 견줘야 크고 작음을 말할 수 있다.
        sep, rad = mode_scales(ref_X, labels)
        multi = n_modes > 1
        logging.info(colored(
            f"[R2][B] 다봉 관측 {int(multi.sum())}개: 모드 간 거리 중앙 {np.nanmedian(sep):.3f}, "
            f"모드 내 반경 중앙 {np.nanmedian(rad):.3f}", "cyan"))

        for (method, stage), X in sorted(cen.items()):
            met = census_metrics(ref_X, X, demo, labels, demo_tau)
            row = {"kind": "census", "method": method, "stage": stage,
                   "probe_task": cfg.probe_task, "seed": cfg.seed,
                   "ckpt": str(stage_ckpt(roots[method], stage)),
                   "ref_key": f"{ref_key[0]}@{ref_key[1]}",
                   "n_obs": int(X.shape[0]), "census_k": int(X.shape[1]),
                   "probe_hash": phash, "demo_tau": demo_tau,
                   "unimodal_frac": uni, "n_modes_mean": float(n_modes.mean()),
                   "n_multimodal": int(multi.sum()),
                   "mode_sep": float(np.nanmedian(sep)), "mode_radius": float(np.nanmedian(rad))}
            for k, v in met.items():
                row[k] = float(np.median(v))
                row[f"{k}_q25"] = float(np.percentile(v, 25))
                row[f"{k}_q75"] = float(np.percentile(v, 75))
                # ★ 다봉 관측만 따로. 단봉 관측은 assign_change가 정의상 0이라
                #   전체 중앙값이 단봉 다수에 눌려 구조적으로 0이 된다(실측 78.5% 단봉).
                #   그림과 결론은 이쪽을 쓴다.
                if multi.any():
                    vm = v[multi]
                    row[f"{k}_multi"] = float(np.median(vm))
                    row[f"{k}_multi_q25"] = float(np.percentile(vm, 25))
                    row[f"{k}_multi_q75"] = float(np.percentile(vm, 75))
            # 중심 이동을 모드 간격으로 정규화. >1이면 "경계가 밀렸다"가 아니라
            # "사상 전체가 다른 모드보다 멀리 옮겨갔다"는 뜻이다.
            if multi.any():
                rel = met["center_shift"][multi] / sep[multi]
                row["center_shift_rel"] = float(np.nanmedian(rel))
                row["center_shift_rel_q25"] = float(np.nanpercentile(rel, 25))
                row["center_shift_rel_q75"] = float(np.nanpercentile(rel, 75))
            emit(row)
            logging.info(colored(
                f"[R2][B] {method} stage{stage}: center_shift={row['center_shift']:.4f}  "
                f"assign_change={row['assign_change']:.3f}  "
                f"demo_match_mass={row['demo_match_mass']:.3f}  "
                f"H={row['weight_entropy']:.3f}", "green"))

    logging.info(colored(f"[R2] done -> {results}", "green", attrs=["bold"]))
    if not cfg.no_plot:
        plot_r2(str(run_dir), str(r1_dir), right_key=cfg.right_key)


# ═════════════════════════════════════════════════════════════════════════════
#  그림 (--plot_only에서도 여기만 돈다)
# ═════════════════════════════════════════════════════════════════════════════
def load_rows(run_dir: Path, kind: str, key) -> dict:
    """r2_results.jsonl의 행. append-only이므로 같은 키는 뒤쪽(최신)만 남긴다."""
    p = run_dir / "r2_results.jsonl"
    if not p.exists():
        raise SystemExit(f"결과가 없다: {p}")
    uniq, n = {}, 0
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") != kind:
            continue
        n += 1
        uniq[key(r)] = r
    if n != len(uniq):
        print(f"deduped ({kind}): dropped {n - len(uniq)} stale row(s)")
    return uniq


def plot_A(run_dir: Path, sticky: dict, plt):
    """R2-A — 노이즈를 고정하면 SR이 오르는가."""
    if not sticky:
        print("[R2-A] checkpoint_sticky 행이 없다")
        return
    keys = sorted({(r["method"], r["stage"]) for r in sticky.values()})
    modes = [m for m in ("fresh", "sticky", "ou")
             if any(r["noise_mode"] == m for r in sticky.values())]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    x = np.arange(len(keys), dtype=float)
    w = 0.8 / max(len(modes), 1)
    csv_rows = []
    for i, m in enumerate(modes):
        srs, ses = [], []
        for k in keys:
            r = sticky.get((*k, m))
            srs.append(np.nan if r is None else r["sr"])
            ses.append(0.0 if r is None else r["sr_se"])
        ax_l.bar(x + (i - (len(modes) - 1) / 2) * w, srs, w, yerr=ses, capsize=3,
                 label=NOISE_LABEL.get(m, m), hatch=NOISE_HATCH.get(m, ""),
                 edgecolor="k", linewidth=0.6, alpha=0.9)
        for k, s, e in zip(keys, srs, ses, strict=True):
            csv_rows.append([k[0], k[1] + 1, m, f"{s:.4f}", f"{e:.4f}"])
    ax_l.set(xticks=x, ylim=(0, 1.05), xlabel="checkpoint", ylabel="success rate",
             title="(a) does freezing the noise rescue the policy?")
    ax_l.set_xticklabels([f"{METHOD_LABEL.get(m, m).split(' ')[0]}\nstage{s + 1}"
                          for m, s in keys], fontsize=9)
    ax_l.grid(alpha=0.3, axis="y")
    ax_l.legend(fontsize=9, title="init noise $a_0$", title_fontsize=9)

    # 오른쪽: 원래 모드가 많이 흔들리던 체크포인트일수록 개입 효과가 크다 (가설의 예측)
    methods = [k[0] for k in keys]
    pts = []
    for k in keys:
        f, s = sticky.get((*k, "fresh")), sticky.get((*k, "sticky"))
        if f is None or s is None:
            continue
        pts.append((k[0], k[1], f["switch_count"], s["sr"] - f["sr"]))
    if pts:
        for m, s, sw, dsr in pts:
            ax_r.scatter(sw, dsr, s=110, color=method_color(m, methods),
                         marker=STAGE_MARKER[s % len(STAGE_MARKER)],
                         edgecolors="k", linewidths=0.7, zorder=3)
        rho, r2 = spearman_r2(np.array([p[2] for p in pts]), np.array([p[3] for p in pts]))
        stat = ("n/a (a variable is constant)" if not np.isfinite(rho)
                else f"Spearman $\\rho$ = {rho:+.2f}\n$R^2$ = {r2:.2f}")
        ax_r.text(0.03, 0.95, f"{stat}\nn = {len(pts)} checkpoints", transform=ax_r.transAxes,
                  fontsize=10, va="top",
                  bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"})
    ax_r.axhline(0, color="k", ls="--", lw=1)
    ax_r.set(xlabel="median switch_count under fresh noise",
             ylabel=r"$\Delta$SR = SR(sticky) $-$ SR(fresh)",
             title="(b) the more the modes flickered, the more the fix helps")
    ax_r.grid(alpha=0.3)

    fig.suptitle("R2-A: if merely freezing the sampling noise raises SR, the collapse was never about "
                 "inaccurate actions —\nit was about successive replans disagreeing on WHICH mode to run",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = run_dir / "R2_A_sticky.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out}")
    plt.close(fig)
    write_csv(run_dir / "R2_A_sticky.csv", ["method", "stage", "noise_mode", "sr", "sr_se"], csv_rows)


def plot_B(run_dir: Path, census: dict, plt):
    """R2-B — 중심은 그대로인데 배정만 바뀌는가."""
    if not census:
        print("[R2-B] census 행이 없다")
        return
    methods = list(dict.fromkeys(r["method"] for r in census.values()))
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    csv_rows = []
    # ★ 두 지표 모두 **다봉 관측만** 쓴다. 단봉 관측은 assign_change가 정의상 0이라
    #   전체 중앙값이 단봉 다수(실측 78.5%)에 눌려 구조적으로 0이 된다.
    #   center_shift는 모드 간 거리로 정규화한다 — 1.0을 넘으면 "경계가 밀렸다"가 아니라
    #   "사상 전체가 옆 모드보다 멀리 옮겨갔다"는 뜻이고, 가설이 반증되는 쪽이다.
    for ax, key, ylab, title in (
        (ax_l, "center_shift_rel",
         "center_shift / distance between modes",
         "(a) did the mode centres stay put?"),
        (ax_r, "assign_change_multi",
         "assign_change: fraction of $a_0$ reassigned",
         "(b) did the boundaries move?"),
    ):
        for m in methods:
            rs = sorted([r for r in census.values() if r["method"] == m], key=lambda r: r["stage"])
            if not rs:
                continue
            xs = [r["stage"] + 1 for r in rs]
            med = [r[key] for r in rs]
            q25 = [r[f"{key}_q25"] for r in rs]
            q75 = [r[f"{key}_q75"] for r in rs]
            c = method_color(m, methods)
            ax.fill_between(xs, q25, q75, color=c, alpha=0.25, linewidth=0)
            ax.plot(xs, med, "-o", ms=6, lw=2, color=c, label=METHOD_LABEL.get(m, m))
            for r in rs:
                csv_rows.append([m, r["stage"] + 1, key, f"{r[key]:.5f}",
                                 f"{r[f'{key}_q25']:.5f}", f"{r[f'{key}_q75']:.5f}",
                                 r["n_obs"], r["census_k"]])
        ax.set(xlabel="CL stage k (tasks 0..k-1 learned)", ylabel=ylab, title=title,
               xticks=sorted({r["stage"] + 1 for r in census.values()}))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    ax_r.set_ylim(bottom=0)
    # 두 기준선. 이게 없으면 "0.5가 큰가 작은가"를 그림에서 판단할 수 없다.
    ax_l.axhline(1.0, color="crimson", ls="--", lw=1.4)
    ax_l.text(0.02, 1.0, " centres moved further than the gap between modes",
              transform=ax_l.get_yaxis_transform(), color="crimson", fontsize=8, va="bottom")
    ax_r.axhline(0.5, color="crimson", ls="--", lw=1.4)
    ax_r.text(0.02, 0.5, " chance (2 modes: assignments fully scrambled)",
              transform=ax_r.get_yaxis_transform(), color="crimson", fontsize=8, va="bottom")
    any_row = next(iter(census.values()))
    fig.suptitle(
        "R2-B: the hypothesis predicts (a) stays well below 1 while (b) climbs — centres held, boundaries eroded.\n"
        f"median + IQR over the {any_row.get('n_multimodal', '?')} MULTIMODAL of "
        f"{any_row['n_obs']} held-out observations (M={any_row['census_k']} fixed $a_0$; "
        f"{any_row['unimodal_frac']:.0%} of observations are unimodal and carry no boundary to erode)",
        fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = run_dir / "R2_B_census.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out}")
    plt.close(fig)
    write_csv(run_dir / "R2_B_census.csv",
              ["method", "stage", "metric", "median", "q25", "q75", "n_obs", "census_k"], csv_rows)


def plot_C(run_dir: Path, r1_dir: Path, sticky: dict, census: dict, plt, right_key: str):
    """R2-C — R1-C의 확장. 왼쪽은 R1의 held-out loss 그대로, 오른쪽은 새 지표."""
    r1 = r1_checkpoint_rows(r1_dir)
    # 오른쪽 축이 census 지표면 census 키를, sticky 지표면 sticky 키를 돈다.
    keys = sorted(census) if census else sorted({(m, s) for m, s, _ in sticky})
    pts = []
    for m, s in keys:
        c = census.get((m, s), {})
        f = sticky.get((m, s, "fresh"))
        loss = r1.get((m, s), {}).get("heldout_loss")
        # SR은 fresh(=R1과 같은 프로토콜)를 쓴다. 없으면 R1의 값을 그대로.
        sr = f["sr"] if f else r1.get((m, s), {}).get("sr")
        right = c.get(right_key, (f or {}).get(right_key))
        if loss is None or sr is None or right is None:
            continue
        pts.append({"method": m, "stage": s, "heldout_loss": loss, "sr": sr, "right": right})
    if len(pts) < 3:
        print(f"[R2-C] 점이 부족하다 ({len(pts)}개)")
        return

    right_label = {
        "demo_match_mass": "demo_match_mass: fraction of $a_0$ landing on the demo's mode",
        "assign_change": "assign_change: fraction of $a_0$ reassigned to another mode",
        "switch_count": "median switch_count (fresh noise)",
        "center_shift": "center_shift: how far the mode centres moved",
    }.get(right_key, right_key)

    methods = list(dict.fromkeys(p["method"] for p in pts))
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for ax, key, xlabel, title in (
        (ax_l, "heldout_loss", "held-out demo FM loss (probe task, fixed grid) — from R1",
         "graded where the EXPERT was"),
        (ax_r, "right", right_label, "graded on the MODE STRUCTURE"),
    ):
        for p in pts:
            ax.scatter(p[key], p["sr"], s=95, color=method_color(p["method"], methods),
                       marker=STAGE_MARKER[p["stage"] % len(STAGE_MARKER)],
                       edgecolors="k", linewidths=0.7, zorder=3)
        rho, r2 = spearman_r2(np.array([p[key] for p in pts]), np.array([p["sr"] for p in pts]))
        stat = ("n/a (a variable is constant)" if not np.isfinite(rho)
                else f"Spearman $\\rho$ = {rho:+.2f}\n$R^2$ = {r2:.2f}")
        ax.text(0.03, 0.05, f"{stat}\nn = {len(pts)} checkpoints", transform=ax.transAxes,
                fontsize=11, va="bottom",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"})
        ax.set(xlabel=xlabel, title=title)
        ax.grid(alpha=0.3)
    ax_l.set(ylabel="measured SR on the same rollouts", ylim=(-0.05, 1.05))

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=method_color(m, methods),
                      markeredgecolor="k", label=METHOD_LABEL.get(m, m)) for m in methods]
    handles += [Line2D([], [], marker=STAGE_MARKER[s % len(STAGE_MARKER)], ls="", color="0.5",
                       markeredgecolor="k", label=f"stage{s + 1}")
                for s in sorted({p["stage"] for p in pts})]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=len(handles), fontsize=9, frameon=False)
    fig.suptitle(
        "R2-C: does the mode structure predict SR better than the loss?   "
        "(R1 reference: dwell $\\rho$=+0.82, $t^*$ $\\rho$=+0.72)",
        fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    out = run_dir / f"R2_C_predictor_{right_key}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out}")
    plt.close(fig)
    write_csv(run_dir / f"R2_C_predictor_{right_key}.csv",
              ["method", "stage", "heldout_loss", "sr", right_key],
              [[p["method"], p["stage"] + 1, p["heldout_loss"], p["sr"], p["right"]] for p in pts])


def plot_r2(run_dir_str: str, r1_dir_str: str, right_key: str = "demo_match_mass"):
    run_dir = Path(run_dir_str)
    r1_dir = Path(r1_dir_str)
    sticky = load_rows(run_dir, "checkpoint_sticky",
                       lambda r: (r["method"], r["stage"], r["noise_mode"]))
    census = load_rows(run_dir, "census", lambda r: (r["method"], r["stage"]))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib 없음 -> 그림 생략 (pip install matplotlib 후 --plot_only 다시)")
        return

    plot_A(run_dir, sticky, plt)
    plot_B(run_dir, census, plt)
    if census:
        plot_C(run_dir, r1_dir, sticky, census, plt, right_key)
        if right_key != "assign_change":
            plot_C(run_dir, r1_dir, sticky, census, plt, "assign_change")
    if sticky:
        plot_C(run_dir, r1_dir, sticky, census, plt, "switch_count")


if __name__ == "__main__":
    if "--plot_only" in sys.argv:
        kv = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
        init_logging()
        plot_r2(kv.get("run_dir", "outputs/R2"),
                kv.get("r1_run_dir", "outputs/R1"),
                right_key=kv.get("right_key", "demo_match_mass"))
    else:
        mp.set_start_method("spawn", force=True)
        init_logging()
        main()
