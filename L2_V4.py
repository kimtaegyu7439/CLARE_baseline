#!/usr/bin/env python
"""L2_codebook **v4** — LOTUS 셋팅 lifelong 학습, 단일 파일 실행본.

이 파일 하나로 v4 를 전부 볼 수 있다. 다른 실험 파일을 import 하지 않는다
(의존은 lerobot 패키지뿐). 알고리즘·정책 재현·코드북·드라이버가 모두 여기 있다.

════════════════════════════════════════════════════════════════════════════
v4 가 v3 에서 바꾸는 것 — 정확히 두 가지
════════════════════════════════════════════════════════════════════════════
FM(flow matching) 손실은 **건드리지 않는다**. 기존 MSE 그대로다. 속도 정규화도
EMA 도 FM 쪽에는 없다. 바뀌는 것은 **앵커 경로뿐**이다.

(1) 샘플러  soft_kernel -> hard_linear      [sample_codebook_hard]

    v3:  w_j = softmax(-||z(s~) - c_j||^2 / h^2)
         o~  = SUM_j w_j (m_j + A_j d_j) + sqrt(SUM_j w_j sig_o,j^2) (*) eps'
    v4:  k*  ~ Cat(pi)
         s~  = mu_{s,k*} + sig_{s,k*} (*) eps
         z~  = (s~ - mean_s) / std_s
         o~  = m_{k*} + A_{k*} (z~ - zbar_{k*})

    커널 가중 w, 대역폭 h, 관측 잡음 sig_o 를 쓰지 않는다. 코드북 **생성**은
    그대로라 v3 가 만든 코드북 파일과 호환된다.

(2) 앵커 손실에 속도 역제곱 가중       [speed_weights, weighted_sq]

    v~_T   = v_T + eps_1                    de-noised 프레임
    rho    = ||v~_T[..., 0:6]||             pose 성분의 스텝 속도
    w      = 1 / (rho^2 + c),   c = kappa * speed_med2_j
    w      = min(w, median(w) * w_clip)     위로만 자른다
    w~     = w / mean(w)                    배치 평균 1

    L_j = (1/(H*A)) * mean_b SUM_s [ w~[b,s] * SUM_{d<6} r^2 + r[...,6]^2 ]

    r = v_S - v_T. **pose 6 차원에만** w~ 를 곱하고 gripper 는 그대로 둔다.
    speed_med2_j = median_{데모,스텝} ||a_s[0:6]||^2 로, 코드북을 만들 때 같이
    계산해 cb["speed_med2"] 에 넣는다.

    mean(w~)=1 이라 평균 스케일이 기존 MSE 와 같다. lambda_level 을 다시
    훑지 않아도 된다.

    왜: 로봇이 느리게 움직이는 구간(rho 작음)에 큰 가중이 가서 "정지에 가까운
    구간의 미세한 속도 차이"가 앵커에 반영된다. 가중이 없으면 빠른 구간의 큰
    잔차가 손실을 지배해 정밀 구간이 묻힌다.

★ 두 변경 모두 **플래그 뒤에 있다**. 기본값(soft_kernel / mse)으로 돌리면
  v3 와 같은 경로를 탄다. 이식이 맞는지 확인할 때 이 등가성을 먼저 보라.

    v4-  = --sampler_mode hard_linear                     (샘플러만)
    v4   = --sampler_mode hard_linear --anchor_weight speedinv

════════════════════════════════════════════════════════════════════════════
코드북 셀 수 K — 여기서 조용히 망가진다
════════════════════════════════════════════════════════════════════════════
LOTUS 셋팅은 base 가 태스크당 50 demo, lifelong 이 10 demo 로 5 배 차이가 난다.
둘에 같은 K(예: 96)를 쓰면 lifelong 쪽 셀당 표본이 10 개 안팎으로 떨어지고
grad_min_frames(기본 24)에 전부 걸려 **A_k 가 모든 셀에서 0 으로 강등**된다.
그러면 이름만 v4 이고 실제로는 0 차 앵커(v1)가 된다. 에러가 나지 않아 결과를
다 뽑고 나서야 알게 된다. 실측: 997 프레임 / K=96 -> 88 셀 전부 강등.

    K = max(k_min, k_per_demo * demo)      k_per_demo=2, k_min=8
    50 demo -> K=100,   10 demo -> K=20

코드북 로그의 `강등 N셀` 이 K 의 절반을 넘으면 멈추고 확인하라.
정상 범위는 K=20 에서 0~5 셀이다.

════════════════════════════════════════════════════════════════════════════
파일 구성
════════════════════════════════════════════════════════════════════════════
  A. 정책 재현    prep_batch / encode_lang / rgb_cls / cond_tail / make_cond
                  sample_fm / fm_loss / snapshot
                  -> 저장소 DiTFlowMTPolicy.forward 를 그대로 복제한 것.
                     verify() 가 실행 시작에 원본과 일치하는지 확인한다.
  B. 코드북       kmeans / build_codebook / sample_codebook / to_device_cb
  C. **v4 변경**  POSE_DIMS / sample_codebook_hard / speed_weights /
                  weighted_sq / action_speed_med2
  D. LOTUS 드라이버  스플릿 로드 · 코드북 생성 · 앵커 손실 · 학습 · 롤아웃 · 리포트

════════════════════════════════════════════════════════════════════════════
실행
════════════════════════════════════════════════════════════════════════════
    # 스모크 (본 run 전에 반드시)
    python L2_V4.py --suite 50 --steps 60 --batch_size 8 --num_workers 4 \
        --eval_episodes 2 --eval_batch_size 2 --n_stages 1 \
        --sampler_mode hard_linear --anchor_weight speedinv --out results/_smoke

    # 본 run
    python L2_V4.py --suite 50 --steps 10000 \
        --seed 1000 --eval_seed 7 --eval_episodes 20 --eval_batch_size 20 \
        --num_workers 6 --sampler_mode hard_linear --anchor_weight speedinv \
        --out results/l2_codebook_v4_50_seed1000

경로는 환경변수로 덮어쓴다.
    LOTUS_ROOT=/경로/LOTUS_setting  CLARE_MODEL_ROOT=/경로/Models  python L2_V4.py ...

CUDA_VISIBLE_DEVICES 는 **torch import 전에** 잡혀야 한다. --gpu 는 그 뒤라
효과가 없으니 셸에서 export 하고 띄워라.
"""
from __future__ import annotations

import argparse
import json
import os
import copy
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# ★ 다른 실험 파일을 import 하지 않는다. 알고리즘은 아래 A/B/C 절에 전부 있다.

from lerobot.configs.default import DatasetConfig, EvalConfig          # noqa: E402
from lerobot.configs.policies import PreTrainedConfig                  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig                  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps          # noqa: E402
from lerobot.datasets.lerobot_dataset import (                         # noqa: E402
    LeRobotDataset, LeRobotDatasetMetadata,
)
from lerobot.datasets.utils import cycle                               # noqa: E402
from lerobot.envs.configs import LiberoEnv                             # noqa: E402
from lerobot.optim.factory import make_optimizer_and_scheduler         # noqa: E402
from lerobot.policies.factory import make_policy                       # noqa: E402
from lerobot.scripts.E0 import to_device                               # noqa: E402
from lerobot.utils.random_utils import set_seed                        # noqa: E402
from lerobot.utils.train_utils import (                                # noqa: E402
    get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint,
)
from lerobot.utils.utils import get_safe_torch_device, init_logging    # noqa: E402

# ===========================================================================
#  A. 정책 재현 — 저장소 DiTFlowMTPolicy.forward 를 그대로 복제한 것
# ===========================================================================
def prep_batch(policy, batch: dict) -> dict:
    """DiTFlowMTPolicy.forward (modeling_dit_flow_mt.py:984-996) 의 전처리 3단계."""
    # 들어오는 batch: observation.state       (B,S,Ds)   = (B,2,8)
    #                 observation.images.* 각 (B,S,C,H,W) = (B,2,3,224,224)
    #                 action                  (B,H,A)    = (B,16,7)
    #                 task                    길이 B 의 문자열 리스트
    batch = policy.normalize_inputs(batch)          # 관측을 통계로 정규화 (shape 불변)
    if policy.config.image_features:
        batch = dict(batch)
        # 카메라 여러 대를 한 축으로 쌓는다: (B,S,M,C,H,W) = (B,2,2,3,224,224)
        batch["observation.images"] = torch.stack(
            [batch[k] for k in policy.config.image_features], dim=-4   # 각 (B,S,C,H,W)
        )
    # batch["action"] 을 MIN_MAX 로 [-1,1] 에 넣는다. 이후 모든 수식이 이 스케일이다.
    return policy.normalize_targets(batch)          # action (B,H,A) — shape 은 그대로


def encode_lang(policy, texts: list[str]) -> torch.Tensor:
    """명령어 -> 조건 벡터의 앞 512차원. LanguageEncoder 는 문자열 캐시가 있어 싸다."""
    dit = policy.dit_flow                           # texts: 길이 B 의 문자열 리스트
    with torch.no_grad():
        emb = dit.language_encoder(texts)           # (B,512) CLIP 텍스트 임베딩. 동결.
    # language_embedding_projection 은 학습 대상이므로 no_grad 밖에 둔다.
    return dit.language_embedding_projection(emb)   # (B,512) -> (B,512)  cond 의 앞부분


def rgb_cls(policy, batch: dict) -> torch.Tensor | None:
    """DINOv2 CLS 토큰. cond_tail 에서 가장 비싼 부분이고 **동결**이라 값이 안 변한다.

    teacher 스냅샷은 pretrained_rgb_encoder 를 student 와 **객체째 공유**하므로
    (snapshot 참조) 같은 배치에 대해 CLS 토큰이 문자 그대로 같다. 앵커가 과거
    태스크 여러 개를 도는 동안 이걸 한 번만 계산해 돌려 쓰면 DINOv2 호출이
    스테이지당 k번 -> 1번으로 줄어든다.
    """
    import einops

    dit = policy.dit_flow
    cfg = policy.config
    if not cfg.image_features:
        return None
    bsz, n_obs = batch["observation.state"].shape[:2]         # B, S(=2)
    images = einops.rearrange(
        batch["observation.images"], "b s n ... -> (b s n) ...",   # (B,S,M,C,H,W)
        b=bsz, s=n_obs, n=len(cfg.image_features),
    )                                    # -> (B*S*M, C,H,W) = (B*4, 3,224,224)
    with torch.no_grad():
        return dit.pretrained_rgb_encoder(images)             # (B*S*M, Dc) = (B*4,768)


def cond_tail(policy, batch: dict, cls: torch.Tensor | None = None) -> torch.Tensor:
    """조건 벡터에서 언어를 뺀 나머지 (상태 + 이미지). 명령어와 무관하다.

    cls 를 주면 DINOv2 를 건너뛴다. rgb_embedding_projection 은 **학습 대상**이라
    정책마다 다르므로 투영은 매번 다시 한다.
    """
    import einops

    dit = policy.dit_flow
    cfg = policy.config
    # bsz = 배치 크기 B,  n_obs = 관측 프레임 수 (n_obs_steps=2, 즉 [t-1, t])
    bsz, n_obs = batch["observation.state"].shape[:2]         # B, S(=2)

    # 8차원 상태 x 2프레임 = 16
    parts = [batch["observation.state"].flatten(start_dim=1)]  # (B,S,Ds) -> (B,16) = s
    if cfg.image_features:
        if cls is None:
            # 배치·프레임·카메라를 한 축으로 접어 인코더에 한 번에 넣는다
            images = einops.rearrange(
                batch["observation.images"], "b s n ... -> (b s n) ...",   # (B,S,M,C,H,W)
                b=bsz, s=n_obs, n=len(cfg.image_features),
            )                                                 # -> (B*4, 3,224,224)
            with torch.no_grad():
                cls = dit.pretrained_rgb_encoder(images)      # (B*4,768) CLS 토큰. 동결.
        cls_tokens = cls                                      # (B*S*M, Dc) = (B*4,768)
        emb = dit.rgb_embedding_projection(cls_tokens)        # (B*4,768) -> (B*4,512)
        # 다시 펼친다: 512 x 2프레임 x 2카메라 = 2048
        feats = einops.rearrange(
            emb, "(b s n) ... -> b s (n ...)", b=bsz, s=n_obs, n=len(cfg.image_features)
        )                                     # (B*4,512) -> (B,S,M*512) = (B,2,1024)
        parts.append(feats.flatten(start_dim=1))                        # (B,2048)
    if cfg.env_state_feature:                                # libero 에서는 안 쓴다
        parts.append(batch["observation.environment_state"].flatten(start_dim=1))
    return torch.cat(parts, dim=-1)                  # (B,16) ⊕ (B,2048) -> (B,2064)


def make_cond(lang_vec: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """(B,512) + (B,2064) -> (B,2576). concat 순서가 원본과 같아야 한다(verify 가 확인)."""
    return torch.cat([lang_vec, tail], dim=-1)               # (B,2576)


def sample_fm(policy, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x_t, t, target) 을 만든다. 앵커 항이 x_t, t 를 직접 잡아야 해서 복제했다."""
    dit = policy.dit_flow
    traj = batch["action"]                        # a: (B,H,A) = (B,16,7) 정답 청크, [-1,1]
    noise = dit.velocity_net.sample_noise(traj.shape[0], traj.device)   # ε: (B,H,A) ~ N(0,I)
    t = dit.noise_distribution.sample((traj.shape[0],)).to(traj.device)  # t: (B,) ~ U(0,1)
    # 직선 경로 위의 점. t=0 이면 순수 노이즈, t=1 이면 정답.
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * traj  # (B,1,1)*(B,H,A)->(B,H,A)
    # 그 직선의 속도. t 와 무관한 상수 벡터라 목표가 "직선을 따라가라"가 된다.
    return x_t, t, traj - noise                   # (B,H,A), (B,), (B,H,A) = a - ε


def fm_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """do_mask_loss_for_padding=False 경로. modeling_dit_flow_mt.py:1298-1320."""
    # 패딩 마스크를 쓰지 않는 경로. (B,H,A) 전 원소 평균이라 청크 길이로도 나뉜다.
    # pred, target: (B,H,A) -> 원소별 제곱오차 (B,H,A) -> 스칼라 ()
    return F.mse_loss(pred, target, reduction="none").mean()


def snapshot(policy, bf16: bool = False):
    """teacher 스냅샷. 동결 백본(DINOv2/CLIP)은 student 것을 **참조로 공유**한다.

    측정: 전체 193.6M(739 MiB) 중
        pretrained_rgb_encoder (DINOv2)  86.6M  330 MiB   동결
        language_encoder       (CLIP)    63.2M  241 MiB   동결
        ------------------------------------------------ 소계 571 MiB (77%)
        velocity_net                     43.2M  165 MiB   학습 (cond_proj 포함)
        rgb_embedding_projection          0.4M    2 MiB   학습
        language_embedding_projection     0.3M    1 MiB   학습
        ------------------------------------------------ 소계 167 MiB
    동결 백본은 requires_grad=False 이고 학습 내내 값이 변하지 않으므로 teacher 마다
    복사할 이유가 없다. 공유하면 teacher 하나가 739 -> 167 MiB (4.4x) 가 되어
    frozen teachers 가 40태스크에서도 24GB 안에 들어온다(30GB -> 7.8GB).

    ★ 수치는 완전히 동일하다. 동결 모듈이라 student 쪽 값이 변하지 않기 때문이다.
    ★ bf16 과는 함께 쓸 수 없다 — .to(bfloat16) 이 공유된 백본까지 바꿔 student 를
      오염시킨다. bf16 을 요청하면 공유하지 않고 전체를 복사한다.
    """
    SHARED = ("pretrained_rgb_encoder", "language_encoder")   # 동결 백본 = 복사하지 않을 모듈
    if bf16:
        return copy.deepcopy(policy).eval().requires_grad_(False).to(torch.bfloat16)

    d = policy.dit_flow
    saved = {}                     # 잠시 떼어낸 백본 모듈들. finally 에서 되돌린다.
    for name in SHARED:                      # 복사 전에 떼어내 peak 메모리도 아낀다
        if hasattr(d, name):
            saved[name] = getattr(d, name)
            setattr(d, name, None)
    try:
        snap = copy.deepcopy(policy).eval().requires_grad_(False)
    finally:
        for name, m in saved.items():        # student 원상복구
            setattr(d, name, m)
    for name, m in saved.items():            # teacher 는 같은 객체를 가리킨다
        setattr(snap.dit_flow, name, m)
    torch.cuda.empty_cache()
    return snap

# ===========================================================================
#  B. 코드북 — k-means + 셀별 (pi, mu_s, sig_s, c, m, sig_o, A_k, h)
# ===========================================================================
def kmeans(x: torch.Tensor, K: int, n_init: int = 10, iters: int = 100,
           seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, float]:
    """k-means++ 초기화 x n_init. (centers(K,D), labels(N,), inertia) 를 돌려준다.

    x: (N,D) — 여기서는 표준화된 s 공간이라 D = Dz = 16, N <= n_pairs = 8000.
    """
    N, D = x.shape                                # x: (N,D) = (N,16)
    best = None
    for init in range(n_init):
        g = torch.Generator(device="cpu").manual_seed(seed * 1000 + init)
        # k-means++
        idx = [int(torch.randint(N, (1,), generator=g))]   # 중심 인덱스 리스트, 최종 길이 K
        d2 = (x - x[idx[0]]).pow(2).sum(1)                 # (N,) 최근접 중심까지 거리²
        for _ in range(K - 1):
            p = (d2 / d2.sum().clamp_min(1e-12)).cpu()     # (N,) 다음 중심 추첨 확률
            idx.append(int(torch.multinomial(p, 1, generator=g)))
            d2 = torch.minimum(d2, (x - x[idx[-1]]).pow(2).sum(1))    # (N,)
        c = x[idx].clone()                                 # (K,D) 중심
        prev = None
        for _ in range(iters):
            lab = torch.cdist(x, c).argmin(1)              # (N,K) -> (N,) 소속 라벨
            cnt = torch.bincount(lab, minlength=K).clamp_min(1)       # (K,)
            cnew = torch.zeros_like(c).index_add_(0, lab, x) / cnt[:, None]   # (K,D)
            empty = torch.bincount(lab, minlength=K) == 0  # (K,) bool
            if empty.any():                       # 빈 셀은 가장 먼 점으로 재배치
                far = torch.cdist(x, c).min(1).values.argsort(descending=True)   # (N,)
                cnew[empty] = x[far[:int(empty.sum())]]    # (n_empty,D) 를 덮어쓴다
            inertia = float((x - cnew[lab]).pow(2).sum())  # (N,D) -> 스칼라
            if prev is not None and abs(prev - inertia) < 1e-7 * max(prev, 1.0):
                c = cnew
                break
            prev, c = inertia, cnew
        lab = torch.cdist(x, c).argmin(1)                  # (N,)
        inertia = float((x - c[lab]).pow(2).sum())         # 스칼라
        if best is None or inertia < best[2]:
            best = (c, lab, inertia)
    return best                                   # ((K,D), (N,), float)


def build_codebook(s: torch.Tensor, o: torch.Tensor, K: int, seed: int,
                   min_n: int = 5, h_scale: float = 1.0, ridge_rho: float = 0.05,
                   grad_min_frames: int = 24) -> dict:
    """(s(N,16), o(N,3072)) -> 코드북 dict. §2 그대로.

    입력  s: (N,Dz) = (N,16)      같은 프레임의 상태
          o: (N,Do) = (N,3072)    같은 프레임의 DINOv2 CLS 를 펼친 것
    출력  dict — 각 항목의 shape 은 함수 끝의 out 조립부에 적어 두었다.
    """
    N = s.shape[0]
    mean_s, std_s = s.mean(0), s.std(0).clamp_min(1e-8)   # 각 (16,)
    zs = (s - mean_s) / std_s                             # (N,16) 표준화된 s
    c, lab, inertia = kmeans(zs, K, n_init=10, seed=seed) # c (K,16), lab (N,), float

    # n_k < min_n 셀은 최근접 셀로 병합
    cnt = torch.bincount(lab, minlength=K)     # (K,) — 원소 j = j번 셀에 속한 표본 수
    keep = (cnt >= min_n).nonzero().squeeze(1) # (n_keep,) 살아남는 셀 인덱스
    drop = (cnt < min_n).nonzero().squeeze(1)  # (n_drop,) 병합될 셀 인덱스
    n_merged = int(drop.numel())
    if n_merged and keep.numel():
        # 버려지는 셀의 소속 점들을 남는 셀 중 최근접으로
        remap = torch.arange(K, device=s.device)   # (K,) 옛 라벨 -> 새 라벨 표
        d = torch.cdist(c[drop], c[keep])          # (n_drop, n_keep)
        remap[drop] = keep[d.argmin(1)]            # (n_drop,)
        lab = remap[lab]                           # (N,)
        # 라벨을 0..K_eff-1 로 압축
        uniq, lab = torch.unique(lab, return_inverse=True)   # uniq (Ke,), lab (N,)
        c = c[uniq]                                # (Ke,16)
    K_eff = int(c.shape[0])                        # Ke

    pi = torch.bincount(lab, minlength=K_eff).float()   # (Ke,) — 원소 j = j번 셀의 표본 수
    n_k = pi.clone()                                    # (Ke,)
    pi = pi / pi.sum()                                  # (Ke,) 합이 1 인 셀 확률

    def per_cell(v, D):
        """v: (N,D) -> (평균, 표준편차) 각 (Ke,D). 누적은 float64 로 한다.

        N 개 행을 lab 로 그룹지어 Ke 개 행으로 접는다. 반환 텐서의 j번째 행은
        "j번 셀에 속한 n_k[j] 개 표본의 평균/표준편차" 이지 표본 하나가 아니다.
        """
        m = torch.zeros(K_eff, D, device=v.device, dtype=torch.float64)   # (Ke,D) Σv
        sq = torch.zeros_like(m)                                          # (Ke,D) Σv²
        m.index_add_(0, lab, v.double())
        sq.index_add_(0, lab, v.double() ** 2)
        cnt = n_k.double().clamp_min(1.0)[:, None]      # (Ke,1) 브로드캐스트용
        mu = m / cnt                                    # (Ke,D)
        sd = (sq / cnt - mu ** 2).clamp_min(0.0).sqrt() # (Ke,D)
        return mu.float(), sd.float()

    # ★ 여기서 N 이 사라진다: (N,16) -> (Ke,16). 행 하나가 표본이 아니라 **셀** 이다.
    mu_s, sig_s = per_cell(s, s.shape[1])          # 각 (Ke,16)   = (셀 개수, Dz)
    m_o, sig_o = per_cell(o, o.shape[1])           # 각 (Ke,3072) = (셀 개수, Do)

    # ── v2: 셀별 선형 기울기 A_k (3072x16) ──────────────────────────────────
    # v1 은 셀 안에서 o 가 s 를 따라 움직이는 성분을 m_k 하나로 뭉개고, 그 변동을
    # σ_o,k 에 분산으로 흡수한다(스미어). A_k 를 두면 õ 의 중심이 s̃ 를 서브셀
    # 스케일로 따라가고, σ_o,k 는 **선형 잔차** 기준이 되어 얇아진다.
    n_demoted = 0
    stats_v1_var = sig_o.pow(2).sum(1).clone()      # (Ke,) 셀별 0차 총분산(로그용)
    D = o.shape[1]                                          # Do = 3072
    zbar = (mu_s - mean_s) / std_s                          # (Ke,16) 셀 평균의 표준화 좌표
    A = torch.zeros(K_eff, D, s.shape[1], device=o.device)  # (Ke,3072,16) 셀별 기울기
    sig_lin = sig_o.clone()                                 # (Ke,3072)
    for kk in range(K_eff):
        sel = (lab == kk).nonzero().squeeze(1)              # (n_k,) 이 셀의 표본 인덱스
        if sel.numel() == 0:
            continue
        dl = zs[sel] - zbar[kk]                             # (n_k,16)  δ = z − z̄_k
        Y = o[sel] - m_o[kk]                                # (n_k,3072) 중심화된 o
        if int(sel.numel()) < grad_min_frames:
            n_demoted += 1                                  # A_k = 0 (0차 강등)
        else:
            S = dl.T @ dl                                   # (16,n_k)@(n_k,16) -> (16,16)
            lam_r = ridge_rho * float(S.diagonal().sum()) / s.shape[1]   # 스칼라 ρ
            G = Y.T @ dl                                    # (3072,n_k)@(n_k,16) -> (3072,16)
            A[kk] = G @ torch.linalg.inv(                   # (3072,16)@(16,16) -> (3072,16)
                S + lam_r * torch.eye(s.shape[1], device=o.device))      # (16,16)
            Y = Y - dl @ A[kk].T             # (n_k,16)@(16,3072) -> (n_k,3072) 선형 잔차
        sig_lin[kk] = Y.std(0, unbiased=False)              # (n_k,3072) -> (3072,)
    sig_o = sig_lin                                         # (Ke,3072) 잔차 기준으로 교체

    # σ 하한 (§2)
    sig_s = sig_s.clamp_min(1e-3 * float(s.std()))          # (Ke,16)
    sig_o = sig_o.clamp_min(1e-3 * float(o.std()))          # (Ke,3072)
    # 대역폭 h = median(각 zs -> 최근접 중심 거리)
    # 스펙: h = median(각 zs -> 최근접 중심 거리).
    # 그런데 softmax(−d²/h²) 에 이 값을 그대로 쓰면 2등 셀도 거의 같은 가중을 받아
    # w 가 20여 개 셀에 퍼진다(합성 테스트 실측: 참여 셀 median 20.7/96, 최대가중 0.147).
    # 그러면 w@m 이 여러 중심의 평균이 되어 구조가 뭉개지고, 남은 분산은 대각 잡음이
    # 채워 d_eff 가 폭발한다 — R13 의 등방 실패 모드로 되돌아간다.
    # h_scale 로 대역폭을 좁힐 수 있게 열어 둔다(1.0 = 스펙 그대로).
    # ★ c 와 h 는 커널 가중(softmax(−‖zs−c‖²/h²))용이다. L2_codebook_bayes 는 가중을
    #   GMM 사후확률로 바꾸므로 그 팔에서는 **deprecated** — 파일에는 남긴다
    #   (스키마 호환 + sanity 의 구가중 대비용).
    # cdist (N,Ke) -> 행별 최소 (N,) -> 중앙값 -> 스칼라 h
    h = float(torch.cdist(zs, c).min(1).values.median().clamp_min(1e-6)) * h_scale

    # 코드북 dict — 이 파일에서 cb[...] 로 읽히는 모든 항목의 shape 일람 (Ke = K_eff)
    out = {"h_scale": h_scale,             # float   대역폭 배율
            "pi": pi.cpu(),                # (Ke,)   셀 확률 (합 1)
            "mu_s": mu_s.cpu(),            # (Ke,16) 셀별 s 평균
            "sig_s": sig_s.cpu(),          # (Ke,16) 셀별 s 표준편차 (대각 가우시안)
            "c": c.cpu(),                  # (Ke,16) 표준화 s 공간의 중심
            "m": m_o.cpu(),                # (Ke,3072) 셀별 o 평균
            "sig_o": sig_o.cpu(),          # (Ke,3072) 선형 잔차 표준편차
            "mean_s": mean_s.cpu(),        # (16,)   s 표준화 통계
            "std_s": std_s.cpu(),          # (16,)
            "h": h,                        # float   softmax 대역폭
            "K_eff": K_eff, "N": N,        # int, int
            "n_k": n_k.cpu(),              # (Ke,)   원소 j = j번 셀의 표본 수 (Σ = N)
            "inertia": inertia,            # float
            "n_merged": n_merged}          # int
    out["A"] = A.half().cpu()              # (Ke,3072,16) fp16 저장(~10MB), 연산은 fp32
    out["zbar"] = ((mu_s - mean_s) / std_s).cpu()   # (Ke,16) 셀 평균의 표준화 좌표
    out["ridge_rho"] = ridge_rho                    # float
    out["n_demoted"] = n_demoted                    # int
    # (Ke,)/(Ke,) -> (Ke,)  셀별 0차분산 / 1차잔차분산
    out["var_ratio_cells"] = (stats_v1_var / sig_o.pow(2).sum(1).clamp_min(1e-12)).cpu()
    out["A_fro"] = A.flatten(1).norm(dim=1).cpu()   # (Ke,3072*16)->(Ke,)  ‖A_k‖_F
    return out


def sample_codebook(cb: dict, n: int, gen=None) -> tuple[torch.Tensor, torch.Tensor]:
    """§3. (s̃(n,16), õ(n,3072))."""
    # 아래 첫 축은 전부 **셀 인덱스** j = 0..Ke-1 이다 (표본 축이 아니다).
    pi, mu_s, sig_s = cb["pi"], cb["mu_s"], cb["sig_s"]    # (Ke,), (Ke,16), (Ke,16)
    c, m, sig_o = cb["c"], cb["m"], cb["sig_o"]            # (Ke,16), (Ke,3072), (Ke,3072)
    k = torch.multinomial(pi, n, replacement=True, generator=gen)   # (n,) 셀 인덱스
    eps = torch.randn(n, mu_s.shape[1], device=mu_s.device, generator=gen)   # (n,16)
    s = mu_s[k] + sig_s[k] * eps              # (n,16)  p(s|j) 는 대각 가우시안
    z = (s - cb["mean_s"]) / cb["std_s"]      # (n,16)  표준화 좌표
    d2 = torch.cdist(z, c).pow(2)                          # (n,Ke)
    w = torch.softmax(-d2 / (cb["h"] ** 2), dim=1)         # (n,Ke) 행마다 합 1
    o_mu = w @ m                              # (n,Ke)@(Ke,3072) -> (n,3072)
    # õ_mean = Σ_j w_j (m_j + A_j δ_j)  —  δ_j = z − z̄_j
    # 두 번째 항을 (B,K,16) -> (B,K·16) @ (K·16, D) 한 번의 matmul 로 접는다.
    # (B,K,D) 중간 텐서를 만들지 않으므로 top-m 절단이 필요 없다.
    A = cb["A"].float()                                   # (Ke,3072,16)
    d = z[:, None, :] - cb["zbar"][None]                  # (n,1,16)-(1,Ke,16) -> (n,Ke,16)
    u = (w[:, :, None] * d).reshape(z.shape[0], -1)       # (n,Ke,16) -> (n,Ke*16)
    # A.permute -> (Ke,16,3072) -> (Ke*16,3072);  (n,Ke*16)@(Ke*16,3072) -> (n,3072)
    o_mu = o_mu + u @ A.permute(0, 2, 1).reshape(-1, A.shape[1])
    o_sd = (w @ sig_o.pow(2)).clamp_min(0.0).sqrt()       # (n,Ke)@(Ke,3072) -> (n,3072)
    o = o_mu + o_sd * torch.randn(n, m.shape[1], device=m.device, generator=gen)  # (n,3072)
    return s, o                                           # (n,16), (n,3072)


def to_device_cb(cb: dict, device) -> dict:
    return {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in cb.items()}

# ===========================================================================
#  C. ★ v4 변경점 — 이 파일에서 v3 와 다른 전부
# ===========================================================================
POSE_DIMS = 6      # action = [x,y,z,roll,pitch,yaw,gripper] -> 0..5 pose, 6 gripper
                   # (LOTUS_setting meta/info.json features.action.names 로 확인)


def sample_codebook_hard(cb: dict, n: int, gen=None):
    """(1) 경성 배정 + 1차 항만. 커널 가중 w·대역폭 h·잡음 항 σ_o 를 쓰지 않는다.

        k* ~ Cat(π)
        s̃  = μ_{s,k*} + σ_{s,k*} ⊙ ε          (기존과 동일)
        z̃  = (s̃ − s̄)/std(s)
        õ  = m_{k*} + A_{k*}(z̃ − z̄_{k*})      (강등 셀은 A=0 이라 õ = m_{k*})

    기존 soft_kernel 은 Σ_j w_j(m_j + A_jδ_j) + sqrt(Σ_j w_j σ_o,j²)⊙ε′ 이다.
    코드북 생성은 건드리지 않아 파일 호환이 유지된다.
    """
    pi, mu_s, sig_s = cb["pi"], cb["mu_s"], cb["sig_s"]
    k = torch.multinomial(pi, n, replacement=True, generator=gen)      # (n,)
    eps = torch.randn(n, mu_s.shape[1], device=mu_s.device, generator=gen)
    s = mu_s[k] + sig_s[k] * eps                                      # (n,16)
    z = (s - cb["mean_s"]) / cb["std_s"]                              # (n,16)
    d = (z - cb["zbar"][k]).unsqueeze(2)                               # (n,16,1)
    A = cb["A"][k].float()                                             # (n,3072,16)
    o = cb["m"][k] + torch.bmm(A, d).squeeze(2)                        # (n,3072)
    return s, o


def speed_weights(v_ref: torch.Tensor, eps_noise: torch.Tensor, c: float,
                  w_clip: float, pose_dims: int | None = None,
                  mode: str = "abs") -> torch.Tensor:
    """(2)(3) 공통 — 기준 속도의 역제곱 가중 w̃. (B,H,A) 를 받아 (B,H) 를 돌려준다.

        ṽ_ref = v_ref + ε                     de-noised 프레임
        ρ[b,s] = ‖ṽ_ref[b,s,0:6]‖             pose 성분의 스텝 속도
        w      = 1 / (ρ² + c)
        w      = min(w, median(w)·w_clip)      위로만 자른다
        w̃      = w / mean(w)                   배치 평균 1 -> λ 스케일 불변

    θ 와 무관한 상수라 detach 한다. 느린 스텝(ρ 작음)에 큰 가중이 가서
    "정지 구간의 미세한 속도 차이"가 손실에 제대로 반영되게 하는 것이 목적이다.
    """
    P = POSE_DIMS if pose_dims is None else pose_dims
    with torch.no_grad():
        a = (v_ref + eps_noise)[..., :P]                               # (B,H,P)
        if mode == "diff":
            # ★ 액션이 **절대 위치**일 때(관절 위치 제어) ‖a‖ 는 속도가 아니라
            #   정규화 원점(= 가동범위 중앙)으로부터의 거리다. 스텝 차분을 써야
            #   비로소 변위 = 속도가 된다. LIBERO(EEF delta)에서는 mode="abs" 가
            #   곧 속도라서 기본값을 바꾸지 않는다.
            d = a.diff(dim=1)                                          # (B,H-1,P)
            d = torch.cat([d[:, :1], d], dim=1)                        # (B,H,P) 첫 스텝 패딩
            rho = d.norm(dim=-1)                                       # (B,H)
        else:
            rho = a.norm(dim=-1)                                       # (B,H)
        w = 1.0 / (rho.pow(2) + c)
        w = torch.minimum(w, w.median() * w_clip)
        return (w / w.mean().clamp_min(1e-12)).detach()                # (B,H)


def weighted_sq(resid: torch.Tensor, w: torch.Tensor,
                pose_dims: int | None = None) -> torch.Tensor:
    """가중 제곱오차. pose 6차원에만 w 를 곱하고 gripper 는 그대로 둔다.

        L = (1/(H·A)) · mean_b Σ_s [ w[b,s]·Σ_{d<6} r² + r[...,6]² ]

    mean(w)=1 이므로 평균 스케일이 기존 MSE 와 같아 λ_lvl 을 다시 훑지 않아도 된다.
    """
    P = POSE_DIMS if pose_dims is None else pose_dims
    H, A = resid.shape[1], resid.shape[2]
    pose = (resid[..., :P].pow(2).sum(-1) * w).sum(1)                  # (B,)
    grip = resid[..., P:].pow(2).sum(-1).sum(1)                        # (B,)
    return ((pose + grip) / (H * A)).mean()


def action_speed_med2(actions: torch.Tensor, pose_dims: int | None = None,
                      mode: str = "abs") -> float:
    """c_j 의 재료 — median_{데모·스텝} ‖a_s[0:P]‖². actions: (N, A) 또는 (N,H,A).

    mode="diff" 는 speed_weights 와 **같은 양**을 재야 c 의 스케일이 맞으므로
    스텝 차분의 제곱 중앙값을 돌려준다. (N,H,A) 가 필요하다.
    """
    P = POSE_DIMS if pose_dims is None else pose_dims
    if mode == "diff":
        if actions.dim() < 3:
            raise ValueError("mode='diff' 는 (N,H,A) 액션이 필요하다")
        d = actions[..., :P].diff(dim=1)
        d = torch.cat([d[:, :1], d], dim=1)
        return float(d.pow(2).sum(-1).reshape(-1).median())
    a = actions.reshape(-1, actions.shape[-1])
    return float(a[:, :P].pow(2).sum(-1).median())

# ===========================================================================
#  D. LOTUS 드라이버 — 스플릿 · 코드북 · 앵커 · 학습 · 롤아웃 · 리포트
# ===========================================================================

# 경로는 환경변수로 덮어쓴다 — 다른 서버로 그대로 옮길 수 있어야 한다.
LOTUS_ROOT = Path(os.environ.get("LOTUS_ROOT",
                                 "/home/sa090180/Datasets/lerobot/LOTUS_setting"))
MODEL_ROOT = Path(os.environ.get("CLARE_MODEL_ROOT", str(Path.home() / "Models")))
SETTING_JSON = LOTUS_ROOT / "lotus_setting.json"
LAMBDA_ANCHOR = 1.0       # L = L_FM + λ_a · L_anchor  (l2_codebook_id 와 같다)

SUITE_DIR = {"goal": "LIBERO-GOAL", "object": "LIBERO-OBJECT", "50": "LIBERO-50"}

# gym_libero 핸들 접두사. lotus_setting.json 의 entry["source"] 로 고른다.
_SRC_PREFIX = {"libero_goal": "Libero_Goal_Task_",
               "libero_object": "Libero_Object_Task_",
               "libero_spatial": "Libero_Spatial_Task_",
               "libero_90": "Libero_90_Task_",
               "libero_10": "Libero_10_Task_"}


def _env_handle(e: dict) -> str:
    """lotus_setting.json 의 태스크 항목 -> gym_libero 핸들.

    goal/object 는 번호가 스위트 안에서 닫혀 있어 suite_index 를 그대로 쓴다.
    LIBERO-50 은 LIBERO-100 인덱스라 90 미만이면 libero_90 의 그 번호,
    90 이상이면 libero_10 의 (index-90) 번이다 (lotus_setting.json 의 verified 항목).
    한 스위트 안에서 핸들 접두사가 **섞인다** — goal/object 와 결정적으로 다른 점이다.
    """
    pre = _SRC_PREFIX[e["source"]]
    if "suite_index" in e:                      # goal / object
        return f"{pre}{e['suite_index']}"
    i = e["libero100_index"]                    # LIBERO-50
    return f"{pre}{i - 90 if e['source'] == 'libero_10' else i}"


def _benchmark_of(handle: str) -> str:
    """핸들 -> benchmark 이름. LiberoEnv.resolved_benchmark 와 같은 표를 쓴다."""
    from lerobot.envs.configs import LIBERO_TASK_PREFIX_TO_BENCHMARK
    return LIBERO_TASK_PREFIX_TO_BENCHMARK.get(handle.rsplit("_Task_", 1)[0], "libero_90")


def suite_spec(suite: str) -> dict:
    """suite -> 실험 구조 전부. lotus_setting.json 이 유일한 출처다.

    스위트 태스크 id 는 base 를 0..n_base-1 로 두고 그 뒤에 stage1, stage2, ...
    순서로 이어 붙인 번호다. goal/object 에서는 기존 번호(base 0..5, stage k -> 5+k)
    와 정확히 같고, LIBERO-50 에서는 base 0..24, stage1 이 25..29, stage5 가 45..49 다.

    ★ LIBERO-50 이 goal/object 와 다른 점 세 가지
      1. base 가 25 태스크 (goal/object 는 6).
      2. stage 가 5 개이고 **스테이지마다 새 태스크가 5 개**다 (goal/object 는 1 개).
         lotus_setting.json 의 assumptions 에 적힌 LOTUS 분할 그대로다.
      3. 평가 핸들이 Libero_90_* 와 Libero_10_* 로 섞인다.

    반환 키
      dir/root      LOTUS_setting 하위 디렉터리
      n_base        base phase 가 joint 로 배운 태스크 수
      stage_tasks   [stage1 이 새로 배우는 태스크 id 리스트, stage2 .., ..]
      n_task        전체 태스크 수
      env_task      태스크 id -> gym 핸들
      instr         태스크 id -> 명령어
      base_demos    base 스플릿의 **태스크당** demo 수
      stage_demos   stage 스플릿의 태스크당 demo 수
      learned_at    태스크 id -> 그 태스크를 배운 스테이지 (base 는 전부 0)
    """
    if suite not in SUITE_DIR:
        raise SystemExit(f"suite 는 {'|'.join(SUITE_DIR)} 중 하나 (받은 값: {suite})")
    d = SUITE_DIR[suite]
    exp = json.loads(SETTING_JSON.read_text())["experiments"][d]

    env_task: dict[int, str] = {}
    instr: dict[int, str] = {}
    learned: dict[int, int] = {}
    for t, e in enumerate(exp["base"]):
        env_task[t], instr[t], learned[t] = _env_handle(e), e["instruction"], 0
    stage_tasks: list[list[int]] = []
    nxt = len(exp["base"])
    for s in exp["lifelong_stages"]:
        ids = []
        for e in s["tasks"]:
            env_task[nxt] = _env_handle(e)
            instr[nxt] = e["instruction"]
            learned[nxt] = s["stage"]
            ids.append(nxt)
            nxt += 1
        stage_tasks.append(ids)
    if len(set(env_task.values())) != nxt:
        raise SystemExit("핸들이 중복이다 — lotus_setting.json 의 인덱스를 확인하라")
    return {"dir": d, "root": LOTUS_ROOT / d,
            "n_base": len(exp["base"]), "stage_tasks": stage_tasks, "n_task": nxt,
            "env_task": env_task, "instr": instr,
            "base_demos": exp["base_n_demos"], "stage_demos": exp["lifelong_n_demos"],
            "learned_at": [learned[t] for t in range(nxt)]}


def load_split(root: Path, policy_cfg):
    """LOTUS_setting 의 한 스플릿을 연다.

    ★ delta_timestamps 를 반드시 넣어야 한다. 없으면 observation.state 가 (B,8) 로,
      action 이 (B,7) 로 나와 (B,2,8) / (B,16,7) 을 기대하는 정책과 어긋난다.
      make_dataset(cfg) 이 내부에서 하는 일과 같다(datasets/factory.py:78-127).
    """
    meta = LeRobotDatasetMetadata("local/lotus", root=str(root))
    dt = resolve_delta_timestamps(policy_cfg, meta)
    return LeRobotDataset("local/lotus", root=str(root), delta_timestamps=dt)


def instructions_of(root: Path) -> dict[int, str]:
    """스플릿의 task_index -> 명령어."""
    return {json.loads(l)["task_index"]: json.loads(l)["task"]
            for l in (root / "meta" / "tasks.jsonl").read_text().splitlines() if l.strip()}


# ═════════════════════════════════════════════════════════════════════════════
#  (s, o) 수집 — 저수지 없이 전 프레임
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def collect_pairs(policy, dataset, device, batch_size: int, num_workers: int,
                  n_demos: int):
    """데이터셋을 한 바퀴 돌며 (s 16, o 3072) 쌍을 모은다.

    그래디언트가 없고 DINOv2 는 동결이라 forward 만 돈다. 저수지를 쓰지 않으므로
    **모든 프레임이 정확히 한 번씩** 들어간다.
    반환은 (태스크별 (s,o), 태스크별 demo 수) 두 개다 — demo 수로 K 를 정한다.

    ★ 태스크 구분은 task_index 가 아니라 **에피소드 블록**(episode_index // n_demos)
      으로 한다. LOTUS 스플릿은 태스크마다 n_demos 개 에피소드가 연속으로 들어 있고
      그 순서가 lotus_setting.json 의 태스크 순서와 같다(전 스플릿 검증함).
      task_index 를 쓰면 LIBERO-50/stage3 처럼 **명령어가 같은 서로 다른 두 태스크**가
      한 인덱스로 합쳐져(10+10=20 ep) 코드북 하나로 뭉개진다. 앵커가 없애려던
      "장면과 명령어가 어긋나는 조합"이 그 안에서 되살아난다.
      goal/object 에서는 두 방식의 결과가 완전히 같다(episode_index//50 == task_index).
    """
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False,
        pin_memory=device.type == "cuda", drop_last=False,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    buf: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    act: dict[int, list[torch.Tensor]] = {}   # 태스크별 액션 -> c_j 의 speed_med2
    eps: dict[int, set] = {}          # 태스크별 에피소드 집합 -> demo 수
    policy.eval()
    for raw in loader:
        batch = prep_batch(policy, to_device(raw, device))
        cls = rgb_cls(policy, batch)
        n = batch["observation.state"].shape[0]
        s = batch["observation.state"].detach().flatten(1).float().cpu()   # (B,16)
        o = cls.detach().reshape(n, -1).float().cpu()                      # (B,3072)
        a = batch["action"].detach().float().cpu()                         # (B,H,A)
        ei = raw["episode_index"].cpu()
        ti = ei // n_demos                      # 에피소드 블록 = 스플릿 내 태스크 위치
        for t in ti.unique().tolist():
            m = ti == t
            buf.setdefault(int(t), []).append((s[m], o[m]))
            act.setdefault(int(t), []).append(a[m])
            eps.setdefault(int(t), set()).update(ei[m].tolist())
    policy.train()
    return ({t: (torch.cat([x for x, _ in v]), torch.cat([y for _, y in v]))
             for t, v in buf.items()},
            {t: len(v) for t, v in eps.items()},
            {t: torch.cat(v) for t, v in act.items()})


def make_books(policy, root: Path, device, args, task_ids: list[int],
               n_demos: int) -> dict[int, dict]:
    """스플릿 하나에서 코드북을 만든다. task_ids 는 스위트 기준 태스크 번호로,
    스플릿 안의 에피소드 블록 순서(0,1,2,..)와 같은 순서여야 한다.

    ★ 셀 수 K 는 **demo 수에 비례**해서 정한다 (K = k_per_demo x demo).
      LOTUS 셋팅은 base 가 태스크당 50 demo, lifelong 이 10 demo 로 5배 차이가 난다.
      둘에 같은 K 를 쓰면 lifelong 쪽 셀당 표본이 10개 안팎으로 떨어지고,
      grad_min_frames=24 에 전부 걸려 **A_k 가 모든 셀에서 0 으로 강등된다**
      (실측: 997 프레임 / K=96 -> 88셀 전부 강등). 그러면 이름만 v3 이고 실제로는
      v1(0차) 앵커가 된다.

      k_per_demo=2 면  50 demo -> K=100,  10 demo -> K=20.
      실측 강등 비율(K=20): GOAL 3~5/20, OBJECT 0~1/20 로 떨어진다.
    """
    ds = load_split(root, policy.config)
    pairs, ndemo, acts = collect_pairs(policy, ds, device, args.batch_size,
                                       args.num_workers, n_demos)
    if len(pairs) != len(task_ids):
        raise SystemExit(f"{root.name}: 에피소드 블록 {len(pairs)}개인데 "
                         f"태스크 {len(task_ids)}개다 — n_demos={n_demos} 를 확인하라")
    books = {}
    for local_idx, (s, o) in sorted(pairs.items()):
        suite_task = task_ids[local_idx]
        K = max(args.k_min, args.k_per_demo * ndemo[local_idx])
        cb = build_codebook(s.to(device), o.to(device), K, seed=args.seed,
                            h_scale=args.h_scale, ridge_rho=args.ridge_rho,
                            grad_min_frames=args.grad_min_frames)
        # c_j 의 재료. 액션은 MIN_MAX 로 [-1,1] 이라 스케일이 태스크마다 다르지 않다.
        cb["speed_med2"] = action_speed_med2(acts[local_idx])
        nk = cb["n_k"]
        logging.info(
            f"[LOTUS] 코드북 task{suite_task}  demo={ndemo[local_idx]} -> K={K}  "
            f"N={cb['N']} (전 프레임)  "
            f"K_eff={cb['K_eff']} (병합 {cb['n_merged']})  "
            f"n_k min/med/max={int(nk.min())}/{int(nk.median())}/{int(nk.max())}  "
            f"h={cb['h']:.4f}  강등 {cb['n_demoted']}셀  "
            f"speed_med2={cb['speed_med2']:.4f}")
        books[suite_task] = to_device_cb(cb, device)
    del ds
    return books


# ═════════════════════════════════════════════════════════════════════════════
#  v4 — 세 가지 변경 (전부 플래그. 기본값이면 아래 함수를 아예 타지 않는다)
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
#  앵커 손실 — l2_codebook_id.L2CodebookAnchor.loss 와 같은 계산
# ═════════════════════════════════════════════════════════════════════════════
def anchor_loss(policy, teacher, books, instructions, x_t, t, st_shape, cls_dim,
                lam_lvl, device, args=None, diag=None):
    """과거 태스크마다 코드북에서 (s̃, õ) 를 뽑아 teacher 와 속도장을 맞춘다.

    청크 backward 고정 — j 마다 즉시 backward 하고 그래프를 버린다. 그래서 앵커
    그래디언트는 과거에 대한 **합**이 된다(평균이 아니다). 반환값은 로그용이다.

    args 가 없거나 기본값이면 v3 와 **비트 단위로 같은 경로**를 탄다.
      --sampler_mode hard_linear  -> 샘플러를 경성 배정 + 1차 항으로
      --anchor_weight speedinv    -> L_j 를 teacher 속도 역제곱으로 가중
    FM 손실은 항상 기존 MSE 다 — 속도 정규화도 EMA 도 쓰지 않는다.
    """
    sampler = getattr(args, "sampler_mode", "soft_kernel") if args else "soft_kernel"
    aw = getattr(args, "anchor_weight", "mse") if args else "mse"
    n = x_t.shape[0]
    tcol = t[:, None, None]
    lvl = []
    for j in sorted(books):
        cb = books[j]
        s_t, o_t = (sample_codebook_hard(cb, n) if sampler == "hard_linear"
                    else sample_codebook(cb, n))
        s_t, o_t = s_t.detach(), o_t.detach()
        bb = {"observation.state": s_t.view(st_shape).to(x_t.dtype)}
        cls_j = o_t.reshape(-1, cls_dim).to(x_t.dtype)
        past = [instructions[j]] * n

        tl_S = cond_tail(policy, bb, cls_j)
        with torch.no_grad():
            tl_T = cond_tail(teacher, bb, cls_j)
            cond_T = make_cond(encode_lang(teacher, past), tl_T)
        cond_S = make_cond(encode_lang(policy, past), tl_S)

        def vel(pol, xx, tt, cond):
            return pol.dit_flow.velocity_net(noisy_actions=xx, time=tt, global_cond=cond)

        # teacher 부트스트랩 x_t — teacher 가 실제로 지나가는 지점에서 잰다
        with torch.no_grad():
            eps0 = torch.randn_like(x_t)
            a_hat = (eps0 + vel(teacher, eps0, torch.zeros_like(t), cond_T)).detach()
        eps1 = torch.randn_like(x_t)
        x_j = ((1 - tcol) * eps1 + tcol * a_hat).detach()

        with torch.no_grad():
            vt0 = vel(teacher, x_j, t, cond_T)
        vs0 = vel(policy, x_j, t, cond_S)
        resid = vs0 - vt0.to(vs0.dtype)                       # (B,H,A)
        if aw == "speedinv":
            # ṽ_T = v_T + ε₁ 프레임에서 잰 teacher 속도로 가중.
            # 잔차는 프레임을 옮겨도 같다: ṽ_S − ṽ_T = v_S − v_T.
            pd = getattr(args, "pose_dims", None)
            w = speed_weights(vt0, eps1, cb["speed_med2"] * args.kappa, args.w_clip,
                              pose_dims=pd,
                              mode=getattr(args, "speed_mode", "abs"))
            L_j = weighted_sq(resid, w, pose_dims=pd)
            if diag is not None:
                diag.setdefault("w_mean", []).append(float(w.mean()))
                diag.setdefault("w_max", []).append(float(w.max()))
        else:
            L_j = resid.flatten(1).pow(2).mean(1).mean()
        (LAMBDA_ANCHOR * lam_lvl * L_j).backward()
        lvl.append(L_j.detach())
        if diag is not None and diag.get("sanity_pending"):
            # ṽ_T ≈ â 여야 한다. 0.2 를 넘으면 잡음 규약이 어긋난 것이다.
            with torch.no_grad():
                r = float((vt0 + eps1 - a_hat).norm(dim=-1).mean()
                          / a_hat.norm(dim=-1).mean().clamp_min(1e-9))
            diag["sanity_ratio"] = r
            diag["sanity_pending"] = False
    if not lvl:
        return torch.zeros((), device=device)
    return (lam_lvl * (sum(lvl) / len(lvl))).detach()


# ═════════════════════════════════════════════════════════════════════════════
#  평가
# ═════════════════════════════════════════════════════════════════════════════
def build_eval_cfg(args, policy_path: str, env_task: str) -> TrainPipelineConfig:
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device
    policy_cfg.push_to_hub = False
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="local/lotus"),
        # LIBERO-50 은 핸들이 Libero_90_*/Libero_10_* 로 섞여 "libero_50" 같은
        # 벤치마크 이름이 없다. 핸들에서 되찾는다(LiberoEnv.resolved_benchmark 와 동일).
        env=LiberoEnv(benchmark=_benchmark_of(env_task), task=env_task),
        policy=policy_cfg, output_dir=None, job_name="lotus_eval",
        seed=args.eval_seed, num_workers=args.num_workers,
        batch_size=args.batch_size, steps=args.steps, eval_freq=0,
        eval=EvalConfig(n_episodes=args.eval_episodes,
                        batch_size=args.eval_batch_size, max_episodes_rendered=0),
    )
    cfg.optimizer = policy_cfg.get_optimizer_preset()
    cfg.scheduler = policy_cfg.get_scheduler_preset()
    cfg.checkpoint_path = None
    return cfg


def rollout_sr(policy, cfg, env_task: str, args) -> float | None:
    """시뮬레이터 롤아웃 성공률(%). 시작 시드는 --eval_seed."""
    import copy
    from lerobot.envs.factory import make_env
    from lerobot.scripts.eval import eval_policy
    env = None
    try:
        env_cfg = copy.deepcopy(cfg.env)
        env_cfg.task = env_task
        env = make_env(env_cfg, n_envs=args.eval_batch_size, use_async_envs=False)
        policy.eval()
        with torch.no_grad():
            info = eval_policy(env, policy, args.eval_episodes, start_seed=args.eval_seed)
        return float(info["aggregated"]["pc_success"])
    except Exception as e:
        logging.warning(f"[LOTUS] SR 실패 ({env_task}): {type(e).__name__}: {e}")
        return None
    finally:
        if env is not None:
            env.close()
        policy.train()


# ═════════════════════════════════════════════════════════════════════════════
#  지표 — tools/plot_sr_matrix.py · l2_family_report.py 와 같은 LIBERO 정의
# ═════════════════════════════════════════════════════════════════════════════
def metrics(R, learned_at, n_task):
    """FWT / NBT / AUC / AvgSR.

        FWT_t = R[learned_at(t)][t]              배운 직후
        NBT_t = mean_{k>learned} (FWT_t − R[k][t])   이후 잃은 양 (+면 망각)
        AUC_t = mean_{k>=learned} R[k][t]        시퀀스 내내 낸 평균
    """
    fwt, nbt, auc = [], [], []
    for t in range(n_task):
        col = [R[k][t] for k in range(learned_at[t], len(R)) if R[k][t] is not None]
        if not col:
            continue
        fwt.append(col[0])
        auc.append(sum(col) / len(col))
        if len(col) > 1:
            nbt.append(col[0] - sum(col[1:]) / len(col[1:]))
    last = [R[-1][t] for t in range(n_task) if R[-1][t] is not None]
    mean = lambda v: (sum(v) / len(v)) if v else None                  # noqa: E731
    return {"AvgSR": mean(last), "FWT": mean(fwt), "NBT": mean(nbt), "AUC": mean(auc)}


def write_report(out: Path, R, learned_at, instr, args, n_task, spec):
    m = metrics(R, learned_at, n_task)
    n_stage = len(spec["stage_tasks"])
    per_stage = len(spec["stage_tasks"][0])
    bd, sd = spec["base_demos"], spec["stage_demos"]
    W = max(86, 8 + 6 * n_task)                 # 태스크가 50개면 표가 그만큼 넓어진다
    L = ["=" * W,
         f"LOTUS 셋팅 lifelong — libero_{args.suite}  (L2_codebook v3 앵커)",
         "=" * W, "",
         f"base phase   {spec['n_base']} task joint, 200,000 step (체크포인트 재사용)",
         f"lifelong     {n_stage} stage x {per_stage} task x {sd} demo, "
         f"stage 당 {args.steps:,} step",
         f"평가         칸당 {args.eval_episodes} rollout, eval seed {args.eval_seed}",
         f"코드북       K = {args.k_per_demo} x demo  (base {bd} demo -> {args.k_per_demo*bd}, "
         f"lifelong {sd} demo -> {args.k_per_demo*sd}), 전 프레임 수집(저수지 없음), "
         f"ridge_rho={args.ridge_rho}, λ_lvl={args.lambda_level}", "",
         "지표 (LIBERO 벤치마크 Liu et al. 2023 — L2_SR_matrix.txt 와 같은 정의)",
         "  FWT = mean_t R[배운 스테이지][t]                높을수록 좋음",
         "  NBT = mean_t mean_{k>배운} (FWT_t − R[k][t])    **낮을수록** 좋음",
         "  AUC = mean_t mean_{k>=배운} R[k][t]             높을수록 좋음",
         "  AvgSR = 마지막 행 평균", "",
         "-" * W,
         "SR matrix   행 = 스테이지(0 = base 직후), 열 = 스위트 태스크",
         "-" * W,
         f"{'stage':<8}" + "".join(f"{'t'+str(t):>6}" for t in range(n_task))]
    for k in range(len(R)):
        L.append(f"{k:<8}" + "".join(
            (f"{R[k][t]:6.0f}" if R[k][t] is not None else f"{'':>6}")
            for t in range(n_task)))
    f1 = lambda v: "-" if v is None else f"{v:.1f}"                    # noqa: E731
    fs = lambda v: "-" if v is None else f"{v:+.1f}"                   # noqa: E731
    L += ["-" * W,
          f"AvgSR {f1(m['AvgSR'])}   FWT↑ {f1(m['FWT'])}   "
          f"NBT↓ {fs(m['NBT'])}   AUC↑ {f1(m['AUC'])}", "",
          "태스크별 (배운 스테이지 / 명령어)"]
    for t in range(n_task):
        L.append(f"  t{t:<3} stage{learned_at[t]}  {spec['env_task'][t]:<20} "
                 f"{instr.get(t, '')[:60]}")
    L += ["", "주의",
          f"  · 칸당 {args.eval_episodes} rollout 이라 이항 표준오차가 개별 칸 최대 "
          f"±{50/ (args.eval_episodes ** 0.5):.0f}%p 다. 행/열 평균으로 읽어야 한다.",
          f"  · base {spec['n_base']} 태스크는 joint 로 배웠으므로 stage 0 이 그 "
          f"{spec['n_base']}개의 FWT 다.",
          "  · 앵커는 청크 backward 라 과거 태스크 수에 비례해 실효 가중이 커진다.", ""]
    if per_stage > 1:
        L += [f"  · 이 스위트는 스테이지마다 {per_stage} 태스크를 **joint 로** 배운다. "
              f"따라서 한 행에 새 태스크 {per_stage}개의 FWT 가 함께 들어간다.", ""]
    out.write_text("\n".join(L) + "\n")
    return m


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["goal", "object", "50"], required=True)
    ap.add_argument("--steps", type=int, default=10000, help="lifelong 스테이지당 스텝")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--k_per_demo", type=int, default=2,
                    help="코드북 셀 수를 demo 수에 비례해 정한다: K = k_per_demo x demo. "
                         "base 50 demo -> K=100, lifelong 10 demo -> K=20.")
    ap.add_argument("--k_min", type=int, default=8, help="셀 수 하한")
    # ── v4 세 가지 변경. 기본값이면 v3 와 완전히 같은 경로를 탄다 ──────────
    ap.add_argument("--sampler_mode", choices=["soft_kernel", "hard_linear"],
                    default="soft_kernel",
                    help="hard_linear: 경성 배정 + 1차 항만 (커널 가중·잡음 항 없음)")
    ap.add_argument("--anchor_weight", choices=["mse", "speedinv"], default="mse",
                    help="speedinv: teacher 예측 속도의 역제곱으로 앵커 손실 가중")
    ap.add_argument("--kappa", type=float, default=1.0, help="c = kappa x speed_med2")
    ap.add_argument("--w_clip", type=float, default=20.0, help="w 를 median(w)x이 값에서 자름")
    ap.add_argument("--n_stages", type=int, default=0,
                    help="lifelong 스테이지 수를 앞에서 이만큼만 쓴다. 0 = 전부. "
                         "스모크에서 1 로 줄이는 용도이고 본 run 에서는 건드리지 않는다.")
    ap.add_argument("--h_scale", type=float, default=1.0)
    ap.add_argument("--ridge_rho", type=float, default=0.05)
    ap.add_argument("--grad_min_frames", type=int, default=24)
    ap.add_argument("--lambda_level", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42, help="학습/k-means 시드")
    ap.add_argument("--eval_seed", type=int, default=7, help="롤아웃 시작 시드")
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=20)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # GPU 지정은 셸 래퍼(Lotus_l2cb_cl.sh)가 torch import 전에 export 한다.
    # 여기서 세우면 CUDA 가 이미 초기화된 뒤라 늦을 수 있다.
    init_logging()
    spec = suite_spec(args.suite)
    if args.n_stages and args.n_stages < len(spec["stage_tasks"]):
        # 스모크용 축소. 잘린 뒤쪽 태스크는 평가·명령어 목록에서도 함께 빼야
        # SR 행렬의 열과 learned_at 이 어긋나지 않는다.
        spec["stage_tasks"] = spec["stage_tasks"][:args.n_stages]
        keep = spec["n_base"] + sum(len(x) for x in spec["stage_tasks"])
        spec["n_task"] = keep
        spec["env_task"] = {t: v for t, v in spec["env_task"].items() if t < keep}
        spec["instr"] = {t: v for t, v in spec["instr"].items() if t < keep}
        spec["learned_at"] = spec["learned_at"][:keep]
        logging.info(f"[LOTUS] ★ 스모크 축소: stage {args.n_stages}개, task {keep}개만 쓴다")
    root = spec["root"]
    n_base, n_task = spec["n_base"], spec["n_task"]
    stage_tasks = spec["stage_tasks"]
    n_stages = len(stage_tasks)
    env_task, instr = spec["env_task"], spec["instr"]
    base_ckpt = MODEL_ROOT / f"Lotus_clare_libero_{args.suite}_pretrain"
    if not (base_ckpt / "model.safetensors").exists():
        raise SystemExit(f"base 체크포인트가 없다: {base_ckpt}")
    out_dir = Path(args.out) if args.out else REPO / "results" / f"lotus_l2cb_{args.suite}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), (out_dir / "config.json").open("w"), indent=2, ensure_ascii=False)

    device = get_safe_torch_device(args.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    # 태스크 t 를 배운 스테이지. base 태스크는 전부 스테이지 0.
    learned_at = spec["learned_at"]
    R: list[list[float | None]] = [[None] * n_task for _ in range(n_stages + 1)]

    # 명령어는 lotus_setting.json 에서 온다. 앵커가 이 문자열로 조건을 만들기 때문에
    # 데이터셋이 내놓는 batch["task"] 와 어긋나면 조용히 틀린 앵커가 된다 — 대조한다.
    for split, ids in [("base", list(range(n_base)))] + [
            (f"stage{i+1}", ids) for i, ids in enumerate(stage_tasks)]:
        have = set(instructions_of(root / split).values())
        miss = [instr[t] for t in ids if instr[t] not in have]
        if miss:
            raise SystemExit(f"{split}: lotus_setting.json 의 명령어가 데이터셋에 없다 "
                             f"({len(miss)}개, 예: {miss[0]!r})")
    logging.info(f"[LOTUS] suite={args.suite}  base {n_base} task  "
                 f"{n_stages} stage x {len(stage_tasks[0])} task  총 {n_task} task")

    # ── stage 0: base 체크포인트 로드 ────────────────────────────────────────
    cfg = build_eval_cfg(args, str(base_ckpt), env_task[0])
    base_meta = LeRobotDatasetMetadata("local/lotus", root=str(root / "base"))
    policy = make_policy(cfg=cfg.policy, ds_meta=base_meta)
    policy.to(device)
    logging.info(f"[LOTUS] base 체크포인트 로드 -> {base_ckpt}")

    t0 = time.perf_counter()
    books = make_books(policy, root / "base", device, args,
                       task_ids=list(range(n_base)), n_demos=spec["base_demos"])
    logging.info(f"[LOTUS] base 코드북 {len(books)}개 ({time.perf_counter()-t0:.0f}s)")
    teacher = snapshot(policy)

    for t in range(n_base):
        R[0][t] = rollout_sr(policy, cfg, env_task[t], args)
        logging.info(f"[LOTUS] SR stage0 task{t} ({env_task[t]}) = {R[0][t]}")
        write_report(out_dir / "report.txt", R, learned_at, instr, args, n_task, spec)

    # ── stage 1..N ───────────────────────────────────────────────────────────
    for stage in range(1, n_stages + 1):
        new_tasks = stage_tasks[stage - 1]       # 이 스테이지가 새로 배우는 태스크들
        n_seen = new_tasks[-1] + 1               # 여기까지 본 태스크 수 (평가 범위)
        sroot = root / f"stage{stage}"
        ds = load_split(sroot, policy.config)
        loader = torch.utils.data.DataLoader(
            ds, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True,
            pin_memory=device.type == "cuda", drop_last=False,
            multiprocessing_context="spawn" if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0)
        dl = cycle(loader)

        cfg.steps = args.steps
        optimizer, sched = make_optimizer_and_scheduler(cfg, policy)
        policy.train()
        logging.info(f"══ [LOTUS] stage {stage}  새 태스크 "
                     f"t{new_tasks[0]}..t{new_tasks[-1]} ({len(new_tasks)}개)  "
                     f"{ds.num_frames} frame  보유 코드북 {len(books)}개")

        diag = {"sanity_pending": args.anchor_weight == "speedinv"}
        for it in range(args.steps):
            batch = prep_batch(policy, to_device(next(dl), device))
            cls = rgb_cls(policy, batch)
            tail = cond_tail(policy, batch, cls)
            x_t, t_, target = sample_fm(policy, batch)   # target = a − ε
            cond = make_cond(encode_lang(policy, list(batch["task"])), tail)
            pred = policy.dit_flow.velocity_net(noisy_actions=x_t, time=t_, global_cond=cond)

            loss_fm = fm_loss(pred, target)
            loss_fm.backward()                       # 앵커도 j 마다 즉시 backward 한다
            la = anchor_loss(policy, teacher, books, instr, x_t, t_,
                             batch["observation.state"].shape, cls.shape[-1],
                             args.lambda_level, device, args=args, diag=diag)
            torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                           cfg.optimizer.grad_clip_norm,
                                           error_if_nonfinite=False)
            optimizer.step()
            optimizer.zero_grad()
            if sched is not None:
                sched.step()
            if it % args.log_every == 0 or it == args.steps - 1:
                extra = ""
                if "w_mean" in diag:
                    extra += (f" anc_w={sum(diag['w_mean'])/len(diag['w_mean']):.2f}"
                              f"/{max(diag['w_max']):.1f}")
                    diag["w_mean"], diag["w_max"] = [], []
                if "sanity_ratio" in diag:
                    extra += f" sanity(ṽ_T≈â)={diag.pop('sanity_ratio'):.3f}"
                logging.info(f"[LOTUS] stage{stage} step={it:6d} "
                             f"fm={float(loss_fm.detach()):.4f} anc={float(la):.4f} "
                             f"lr={optimizer.param_groups[0]['lr']:.2e}{extra}")
                with (out_dir / "diag.csv").open("a") as f:
                    f.write(f"{stage},{it},{float(loss_fm.detach()):.6f},"
                            f"{float(la):.6f}\n")

        # 체크포인트 저장 -> 코드북 추가 -> teacher 갱신
        sdir = out_dir / "checkpoints" / f"stage{stage}"
        ck = get_step_checkpoint_dir(sdir, args.steps, args.steps)
        save_checkpoint(ck, args.steps, cfg, policy, optimizer, sched)
        update_last_checkpoint(ck)
        del loader, dl, ds
        torch.cuda.empty_cache()

        books.update(make_books(policy, sroot, device, args,
                                task_ids=new_tasks, n_demos=spec["stage_demos"]))
        del teacher
        teacher = snapshot(policy)
        torch.cuda.empty_cache()

        for t in range(n_seen):
            R[stage][t] = rollout_sr(policy, cfg, env_task[t], args)
            logging.info(f"[LOTUS] SR stage{stage} task{t} ({env_task[t]}) = {R[stage][t]}")
            write_report(out_dir / "report.txt", R, learned_at, instr, args, n_task, spec)

    m = write_report(out_dir / "report.txt", R, learned_at, instr, args, n_task, spec)
    json.dump({"suite": args.suite, "R": R, "learned_at": learned_at,
               "stage_tasks": stage_tasks,
               "env_task": {str(k): v for k, v in env_task.items()},
               "metrics": m, "instructions": {str(k): v for k, v in instr.items()}},
              (out_dir / "metrics.json").open("w"), indent=2, ensure_ascii=False)
    print((out_dir / "report.txt").read_text())
    logging.info(f"[LOTUS] 완료 -> {out_dir}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
