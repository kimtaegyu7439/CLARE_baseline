#!/usr/bin/env python
"""Lotus_L3_V7dea — LOTUS 셋팅 lifelong, **L3 코드북 + V7 dea 앵커**. 단일 파일.

    python Lotus_L3_V7dea.py --suite 50     --beta 3.0
    python Lotus_L3_V7dea.py --suite goal   --beta 3.0
    python Lotus_L3_V7dea.py --suite object --beta 3.0
    python Lotus_L3_V7dea.py --selftest            # 모델 없이 수식 검증

★ 이 파일은 **자체 완결**이다. 저장소의 다른 모듈(l2_codebook_id, L2_V6,
  Lotus_l2cb_cl)을 import 하지 않는다. 필요한 함수는 전부 안에 들어 있어서
  이 파일 하나만 있으면 돌아간다. 외부 의존은 lerobot / torch 뿐이다.

════════════════════════════════════════════════════════════════════════════
L3 가 L2 와 다른 점 — 코드북 셀 수
════════════════════════════════════════════════════════════════════════════
L2 까지는 모든 태스크에 같은 규칙을 썼다.

    K = max(k_min, k_per_demo x demo)

LOTUS 셋팅은 base 가 태스크당 50 demo, lifelong 이 10 demo 라서 이 규칙이
    base      50 demo -> K=100
    lifelong  10 demo -> K=20
를 낸다. L3 는 **lifelong 만 태스크당 K 를 고정**한다 (--lifelong_k, 기본 25).
base 는 기존 규칙(K=100)을 그대로 둔다.

    L3:  base K=100 (규칙),  lifelong K=25 (고정, 태스크마다 각각)

★ "스테이지당 25" 가 아니라 **태스크당 25** 다. LIBERO-50 은 한 스테이지가
  5 태스크를 한꺼번에 배우지만 코드북은 books[suite_task] 로 태스크마다 하나씩
  만들어지므로(make_books), 스테이지 하나에 K=25 코드북이 5 개 생긴다.

왜 25 인가: lifelong 태스크는 10 demo = 1,400~1,900 프레임이다. K=20 이면
셀당 ~90 프레임, K=25 면 ~70 프레임이다. grad_min_frames=24 보다 충분히 커서
A_k 가 강등되지 않는다(실측 강등 0~2 셀). K 를 더 올리면 셀당 프레임이 24 에
가까워지고, 그 지점을 넘으면 모든 셀이 강등되어 이름만 1 차 앵커가 된다.

════════════════════════════════════════════════════════════════════════════
V7 dea 앵커
════════════════════════════════════════════════════════════════════════════
앵커 손실을 **액션 공간**에서 크기 항과 방향 항으로 쪼개고, 방향 항의 계수를
과거 태스크별로 적응 배분한다.

    ‖a_s − ā‖²  =  (r_s − r̄)²  +  2·r_s·r̄·(1 − cos)
                 =      m       +  2·r_s·r̄·  e

    팔        m_j 계수   e_j 계수
    sqerr     1/N       2·r_s·r̄ / N      <- 분해 항등식의 원래 계수 (= MSE 앵커)
    unifmd    1/N       β / N
    dea       1/N       β·w_j            <- Σ_j w_j = 1 이라 /N 이 없다

    w_j = (e_j + ε_w) / (Σ_m e_m + N_valid·ε_w)     태스크 축에 걸쳐 합이 1

dea 는 2-pass 다 — w_j 가 Σ_m e_m 에 의존하므로 모든 과거 태스크의 e 를 먼저
알아야 한다. teacher forward 는 pass 1 결과를 재사용하므로 1 배이고 student
forward 만 2 배가 된다. sqerr/unifmd 는 1-pass 로 끝난다.

★ 비용: LOTUS 는 base 가 25 태스크라 stage 1 부터 과거가 25 개다. 실측
  2.1 s/step (N=25, A6000) 이고 N 에 거의 비례한다 — a + b·N 로 맞추면
  고정비 0.32 s + 태스크당 0.073 s. LIBERO-50 한 시드가 약 40 시간이다.

════════════════════════════════════════════════════════════════════════════
학습/롤아웃 분리
════════════════════════════════════════════════════════════════════════════
--skip_eval 로 롤아웃을 전부 건너뛰면 스테이지 체크포인트만 남는다. LIBERO-50
은 스테이지마다 최대 50 칸을 돌아 롤아웃이 학습을 오래 막으므로, 학습을 GPU
하나에서 돌리고 롤아웃을 다른 GPU 에서 따로 돌릴 때 쓴다.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.configs.default import DatasetConfig, EvalConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle
from lerobot.envs.configs import LiberoEnv
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.scripts.E0 import to_device
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint,
)
from lerobot.utils.utils import get_safe_torch_device, init_logging

REPO = Path(__file__).resolve().parent

# 경로는 환경변수를 먼저 보고, 없으면 서버1 의 기존 값을 쓴다.
LOTUS_ROOT = Path(os.environ.get("LOTUS_DATA_ROOT")
                  or Path(os.environ.get("HF_LEROBOT_HOME",
                                         "/home/sa090180/Datasets/lerobot")) / "LOTUS_setting")
MODEL_ROOT = Path(os.environ.get("CLARE_MODEL_ROOT", "/home/sa090180/Models"))
SETTING_JSON = LOTUS_ROOT / "lotus_setting.json"
SUITE_DIR = {"goal": "LIBERO-GOAL", "object": "LIBERO-OBJECT", "50": "LIBERO-50"}

# lotus_setting.json 의 source 필드 -> gym-libero 환경 핸들 접두사
_SRC_PREFIX = {
    "libero_10": "Libero_10_Task_",
    "libero_90": "Libero_90_Task_",
    "libero_goal": "Libero_Goal_Task_",
    "libero_object": "Libero_Object_Task_",
    "libero_spatial": "Libero_Spatial_Task_",
}

POSE_DIMS = 6          # action = [x,y,z,roll,pitch,yaw,gripper] -> 0..5 pose, 6 gripper
LAMBDA_ANCHOR = 1.0    # L = L_FM + λ_a · L_anchor


# ============================================================================
#  코드북 / 조건 인코딩 (l2_codebook_id 에서 그대로)
# ============================================================================

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


def to_device_cb(cb: dict, device) -> dict:
    return {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in cb.items()}


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


def encode_lang(policy, texts: list[str]) -> torch.Tensor:
    """명령어 -> 조건 벡터의 앞 512차원. LanguageEncoder 는 문자열 캐시가 있어 싸다."""
    dit = policy.dit_flow                           # texts: 길이 B 의 문자열 리스트
    with torch.no_grad():
        emb = dit.language_encoder(texts)           # (B,512) CLIP 텍스트 임베딩. 동결.
    # language_embedding_projection 은 학습 대상이므로 no_grad 밖에 둔다.
    return dit.language_embedding_projection(emb)   # (B,512) -> (B,512)  cond 의 앞부분


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


# ============================================================================
#  크기/방향 분해 (L2_V6 에서 그대로)
# ============================================================================

def to_action(v: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """속도 -> 그 속도가 가리키는 끝점(액션). 선형 경로 v = (a − x_t)/(1−t) 의 역.

        a = x_t + (1 − t) · v

    왜 쓰는가: v 는 공통 노이즈 ε 를 품어(‖v‖≈‖ε‖) student·teacher 가 거의 같은
    방향을 가리킨다. 각도가 t 에 따라 6 배까지 압축된다(t=0.1 에서 6.4x).
    a 로 바꾸면 그 공통 성분이 빠져 각도가 t 와 무관해진다.
    ★ m 과 e 를 **같은 공간**에서 뽑아야 한다. 섞으면
      ‖·‖² = m + 2·r·r·e 항등식이 깨져 sqerr 이 MSE 를 복원하지 못한다.

    ── shape: 바뀌지 않는다 ────────────────────────────────────────────────
        v, x_t : (B,H,A)   t : (B,)
        반환    : (B,H,A)   ← 공간만 옮기고 모양은 그대로. 그래서 split_mag_dir
                             에 v 대신 a 를 넣어도 코드가 그대로 돈다.
    t.view(-1,1,1) 이 (B,) -> (B,1,1) 로 축을 끼워, 샘플마다 자기 t 가 그 샘플의
    H x A 원소 전부에 걸리게 한다. (B,) 그대로는 뒤 축(A=7)과 맞으려 해 에러난다.

    ── 숫자 감각 (실측) ────────────────────────────────────────────────────
        ‖ε‖ ≈ √6 = 2.45   ‖a‖ ≈ 0.85   ‖v‖ ≈ 2.49
    속도의 대부분이 공통 노이즈다. 그래서 액션이 직각(90°)으로 틀어져 있어도
    속도 각도는 27° 로 보인다. 압축률이 t 에 의존하므로(t=0.1 에서 6.4x,
    t=0.9 에서 1.0x) 같은 정책 불일치가 t 에 따라 6 배 다른 e 로 들어온다.
    sqerr 은 계수 2·r_s·r̄ 가 t 와 함께 커져 자동 보정되지만, β 를 쓰는
    unifmd/dea 에는 그 보정이 없다 — dir_space=act (V7) 가 그 왜곡을 없앤다.
    """
    # a = x_t + (1−t)·v.  (B,H,A) + (B,1,1)*(B,H,A) -> (B,H,A)
    return x_t + (1.0 - t).view(-1, 1, 1) * v


def split_mag_dir(v_s: torch.Tensor, v_t: torch.Tensor, pose_dims: int | None,
                  eps: float = 1e-8):
    """★★ 여기가 앵커 손실이 **크기 항과 방향 항으로 쪼개지는 유일한 지점**이다.

    (B,H,A) 두 개 -> (m, e, r_s, r_t, valid) 각 (B,H).

    ── 무엇을 하는가 ───────────────────────────────────────────────────────
    student 와 teacher 의 속도 벡터를 각각 "크기 x 단위방향" 으로 분해한다.

        v_θ = r_θ · u_θ ,   v̄ = r̄ · ū        (r = ‖v‖₂,  ‖u‖₂ = 1)

    그리고 둘의 불일치를 **두 개의 스칼라**로 나눈다.

        m = (r_θ − r̄)²        크기 오차.  [0, ∞)
        e = 1 − u_θ·ū         방향 오차.  [0, 2]   (0 같은 방향 / 1 직각 / 2 정반대)

    ── 왜 쪼개는가: 이 항등식이 근거다 (수치 검증 완료) ────────────────────
        ‖v_θ − v̄‖²  =  (r_θ − r̄)²  +  2·r_θ·r̄·(1 − cos)
                     =      m       +  2·r_θ·r̄·  e

    즉 기존 MSE 앵커는 이미 "크기 오차 + 방향 오차" 의 합인데, 방향 항의 계수가
    2·r_θ·r̄ 로 **속도가 빠른 구간일수록 크다.** 느린 구간(r 이 작은 구간)의 방향은
    사실상 감독을 못 받는다 — 이것이 V4 의 speedinv 가중이 손대려던 문제다.
    V6 는 계수를 손으로 정해(unifmd: β, dea: β·w_j) 그 배분을 직접 통제한다.
    그래서 anchor_form=sqerr 로 두면 계수가 2·r_θ·r̄ 로 돌아가 **MSE 와 항등**이 된다
    — 세 팔을 같은 코드 경로로 비교할 수 있는 이유다 (loss() 의 표 참고).

    ── shape 이 어떻게 바뀌는가 ────────────────────────────────────────────
        입력   v_s, v_t : (B,H,A)  = (32,16,7)   B=배치, H=청크(horizon), A=액션축
        ①슬라이스       : (B,H,P)  = (32,16,6)   뒤 축만 pose P=6 으로 잘라냄
        ②norm(dim=-1)   : (B,H)    = (32,16)     ← A 축이 **사라진다**. 벡터->크기
        ③m, e           : (B,H)    = (32,16)     (t,h) 칸마다 스칼라 2개
    즉 **마지막 축(액션 성분)이 접히고, (배치 x 청크) 격자만 남는다.** 이후 loss()
    에서 이 격자에 계수를 곱하고 .mean() 으로 스칼라까지 접는다.

    ── pose_dims 로 gripper 를 빼는 이유 ──────────────────────────────────
    pose_dims 를 주면 그 앞쪽 축만 쓴다(gripper 제외). 노트의 v 는 "같은 정규화
    액션 좌표"의 벡터라 pose/gripper 를 섞어도 되지만, 이 코드베이스의 앵커는
    줄곧 pose 6 축만 가중해 왔으므로 기본을 그에 맞춘다.
    ★ 그리퍼는 정답이 사실상 ±1 이진값이라 "방향"이라는 개념이 맞지 않는다.
      6 축 pose 와 한 벡터로 묶어 각도를 재면 그리퍼 전환이 각도를 지배한다.
    """
    # P: 실제로 쓸 축 수. None 이면 전체(A), 보통 POSE_DIMS=6.
    P = v_s.shape[-1] if pose_dims is None else pose_dims
    a, b = v_s[..., :P], v_t[..., :P]                 # 각 (B,H,A)->(B,H,P)

    # ── ① 크기: 마지막 축을 L2 노름으로 접는다. (B,H,P) -> (B,H) ───────────
    r_s = a.norm(dim=-1)                              # (B,H)  student 속도 크기 r_θ
    r_t = b.norm(dim=-1)                              # (B,H)  teacher 속도 크기 r̄
    m = (r_s - r_t).pow(2)                            # (B,H)  크기 오차 m = (r_θ−r̄)²

    # ── ② 방향: 단위벡터 내적 = cos. (B,H,P) -> (B,H) ─────────────────────
    # (a*b).sum(-1) 이 내적이고, 각 크기로 나누면 단위벡터 내적이 된다.
    # 즉 u_θ·ū 를 u 를 따로 만들지 않고 한 번에 구한다(0 나눗셈만 막으면 된다).
    # 방향은 단위벡터 내적. r 이 0 이면 정의되지 않으므로 eps 로 막고 valid 로 뺀다.
    cos = (a * b).sum(-1) / (r_s.clamp_min(eps) * r_t.clamp_min(eps))
    # clamp(-1,1): 부동소수 오차로 |cos| 가 1 을 살짝 넘으면 e 가 음수가 된다.
    e = 1.0 - cos.clamp(-1.0, 1.0)                    # (B,H) ∈ [0,2]
    # valid: 어느 한쪽 크기가 ~0 이면 각도가 정의되지 않는다. 그 칸은 방향 항에서
    # 빼고(loss 의 torch.where), dea 가중의 정규화 분모에서도 뺀다.
    valid = (r_t > eps) & (r_s > eps)                 # (B,H) bool
    return m, e, r_s, r_t, valid                      # 각 (B,H) — A 축은 접혔다


def dea_weights(e_stack: torch.Tensor, valid_stack: torch.Tensor,
                eps_w: float) -> torch.Tensor:
    """(N,B,H) 방향오차 -> (N,B,H) 가중. **태스크 축(N)에 걸쳐** 합이 1.

        w_j = (e_j + ε_w) / (Σ_m e_m + N_valid·ε_w)

    노트 (2). ε_w 는 e 가 전부 0 일 때의 0/0 을 막고, 작을수록 적응이 세진다.
    무효(근0 teacher) 태스크는 분자·분모에서 모두 빼고 유효 태스크로만 정규화한다.

    ── 무엇을 위한 가중인가 ────────────────────────────────────────────────
    방향 "예산" 을 과거 태스크들 사이에 나눠 주는 것이다. Σ_j w_j = 1 이므로
    **총량은 고정**이고, 방향이 많이 틀어진 태스크가 더 큰 몫을 받는다.
    즉 균일 배분(unifmd, w_j = 1/N)의 적응형 버전이다.

    ── shape 이 어떻게 바뀌는가 ────────────────────────────────────────────
        입력  e_stack, valid_stack : (N,B,H)   N=과거 태스크 수. split_mag_dir 의
                                              (B,H) 결과를 torch.stack 으로 쌓은 것
        nv  = valid.sum(0)         : (B,H)     ← 축 0(태스크)이 접힌다
        den = e.sum(0) + nv·ε_w    : (B,H)     ← 같은 (t,h) 칸의 모든 태스크 합
        num                        : (N,B,H)
        반환 num/den (브로드캐스트) : (N,B,H)   (B,H) 가 N 축으로 자동 확장
    ★ 정규화 축이 **0(태스크)** 이라는 점이 핵심이다. (B,H) 칸마다 독립적으로
      "이 칸에서 태스크들이 예산을 어떻게 나누나" 를 계산한다. 배치나 청크 축으로
      합이 1 이 되는 게 아니다.

    ── 왜 .detach() 인가 ───────────────────────────────────────────────────
    w_j 는 **계수**이지 학습 대상이 아니다. detach 를 빼면 "e_j 를 줄이는" 대신
    "w_j 를 줄이는"(= 다른 태스크의 e 를 키우는) 경로로도 그래디언트가 흘러,
    과거 태스크를 일부러 더 망가뜨리는 쪽이 손실을 낮추는 퇴화가 생긴다.
    노트의 sg[·] (stop-gradient) 가 이것이다.

    ── 왜 2-pass 가 강제되는가 ────────────────────────────────────────────
    w_j 가 Σ_m e_m 에 의존하므로, 태스크 j 의 손실을 backward 하기 전에
    **모든 태스크의 e 를 먼저** 알아야 한다. 그래서 dea 만 pass 1(no_grad)로
    전체 e 를 모은 뒤 pass 2 에서 다시 계산한다 (loss() 참고).
    sqerr/unifmd 는 계수가 태스크에 의존하지 않아 1-pass 로 끝난다.
    """
    # 무효 칸의 e 를 0 으로 눌러 둔다 — 아래 e.sum(0) 분모에 섞이지 않게.
    e = torch.where(valid_stack, e_stack, torch.zeros_like(e_stack))
    nv = valid_stack.sum(0).clamp_min(1)                      # (B,H) 유효 태스크 수
    # 분자: 유효 태스크만 (e + ε_w), 무효는 0 -> 가중 0 을 받는다.
    num = torch.where(valid_stack, e + eps_w, torch.zeros_like(e))   # (N,B,H)
    # 분모: 같은 (t,h) 칸의 유효 태스크 e 합 + 유효 개수 x ε_w.
    # ★ N 이 아니라 nv 를 곱한다 — 무효 태스크까지 세면 Σ_j w_j < 1 이 된다.
    den = e.sum(0) + nv * eps_w                               # (B,H)
    # (N,B,H) / (B,H) -> 브로드캐스트로 (N,B,H). detach 는 위 설명 참고.
    return (num / den.clamp_min(1e-12)).detach()              # (N,B,H)


# ============================================================================
#  LOTUS 셋팅 정의 / 데이터 / 평가 (Lotus_l2cb_cl 에서 그대로)
# ============================================================================

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


def action_speed_med2(actions: torch.Tensor) -> float:
    """c_j 의 재료 — median_{데모·스텝} ‖a_s[0:6]‖². actions: (N, A) 또는 (N,H,A)."""
    a = actions.reshape(-1, actions.shape[-1])
    return float(a[:, :POSE_DIMS].pow(2).sum(-1).median())


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


# ============================================================================
#  코드북 생성 — L3 (lifelong 만 K 고정)
# ============================================================================

def make_books(policy, root: Path, device, args, task_ids: list[int],
               n_demos: int, k_fixed: int | None = None) -> dict[int, dict]:
    """스플릿 하나에서 코드북을 만든다. task_ids 는 스위트 기준 태스크 번호로,
    스플릿 안의 에피소드 블록 순서(0,1,2,..)와 같은 순서여야 한다.

    ★ 셀 수 K 는 **demo 수에 비례**해서 정한다 (K = k_per_demo x demo).
      LOTUS 셋팅은 base 가 태스크당 50 demo, lifelong 이 10 demo 로 5배 차이가 난다.
      둘에 같은 K 를 쓰면 lifelong 쪽 셀당 표본이 10개 안팎으로 떨어지고,
      grad_min_frames=24 에 전부 걸려 **A_k 가 모든 셀에서 0 으로 강등된다**
      (실측: 997 프레임 / K=96 -> 88셀 전부 강등). 그러면 이름만 v3 이고 실제로는
      v1(0차) 앵커가 된다.

      k_per_demo=2 면  50 demo -> K=100,  10 demo -> K=20.

    ★ L3 는 lifelong 만 k_fixed 로 고정한다(기본 25). base 는 규칙대로 K=100.
      "스테이지당 25" 가 아니라 **태스크당 25** 다 — 코드북이 books[suite_task]
      로 태스크마다 하나씩 만들어지므로, 5 태스크 스테이지면 K=25 코드북 5 개다.
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
        # ★ L3: k_fixed 가 주어지면 **태스크마다 그 값으로 고정**한다.
        #   main 이 lifelong 호출에만 k_fixed=args.lifelong_k 를 넘긴다.
        #   base 는 k_fixed=None 이라 기존 규칙(K = k_per_demo x demo)을 쓴다.
        #   우선순위: codebook_k(모든 태스크 강제) > k_fixed(lifelong) > 비례 규칙
        if getattr(args, "codebook_k", 0) > 0:
            K = args.codebook_k
        elif k_fixed:
            K = max(args.k_min, k_fixed)
        else:
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


# ============================================================================
#  V7 dea 앵커
# ============================================================================

def anchor_loss(policy, teacher, books, instructions, x_t, t, st_shape, cls_dim,
                lam_lvl, device, args=None, diag=None):
    """V7 dea 앵커. 과거 태스크마다 코드북에서 (s̃, õ) 를 뽑아 teacher 의 속도장을
    **액션 공간에서** 따라가게 한다.

    청크 backward 고정 — j 마다 즉시 backward 하고 그래프를 버린다. N 개를 다
    쌓으면 활성값이 N 배라 메모리가 터진다. 그래디언트는 파라미터에 누적되므로
    수학적으로는 한꺼번에 backward 한 것과 같다. 반환값은 로그용이다.

    ── 계수 배분 (파일 머리의 표) ──────────────────────────────────────────
        sqerr     m:1/N   e:2·r_s·r̄/N    분해 항등식의 원래 계수 (= MSE 앵커)
        unifmd    m:1/N   e:β/N
        dea       m:1/N   e:β·w_j        Σ_j w_j = 1 이라 /N 이 없다

    ── dea 는 2-pass ──────────────────────────────────────────────────────
    w_j 가 Σ_m e_m 으로 정규화되므로 **모든 과거 태스크의 e 를 먼저** 알아야
    한다. pass 1 은 가중치 계산용이라 no_grad 다. teacher forward 는 pass 1 에서
    만든 (x_j, v̄) 를 재사용하므로 2-pass 라도 teacher 비용은 1 배이고 student
    forward 만 2 배가 된다.

    ★ 과거 태스크 정의: books 는 태스크 번호마다 코드북이 하나씩이다. base 가
      25 태스크인 LIBERO-50 도 25 개의 개별 태스크로 들어온다 — 한 덩어리로
      뭉쳐 있지 않으므로 dea 가중이 태스크 단위로 제대로 나뉜다.
    """
    form = getattr(args, "anchor_form", "dea") if args else "dea"
    beta = float(getattr(args, "beta", 2.0) if args else 2.0)
    eps_w = float(getattr(args, "eps_w", 1e-3) if args else 1e-3)
    space = getattr(args, "dir_space", "act") if args else "act"
    reduce = getattr(args, "anchor_reduce", "mean") if args else "mean"
    sampler = getattr(args, "sampler_mode", "hard_linear") if args else "hard_linear"
    pd = getattr(args, "pose_dims", 0) if args else 0
    P = pd if pd else POSE_DIMS

    js = sorted(books)       # ★ sorted 필수 — W[i] 와 js[i] 가 같은 태스크여야 한다
    if not js:
        return torch.zeros((), device=device)
    N = len(js)
    n = x_t.shape[0]
    tcol = t[:, None, None]
    inv_N = (1.0 / N) if reduce == "mean" else 1.0

    def vel(pol, xx, tt, cond):
        return pol.dit_flow.velocity_net(noisy_actions=xx, time=tt, global_cond=cond)

    # ── 질의 재료: 태스크마다 한 번. teacher forward 는 여기서 끝난다 ────────
    qs = []
    for j in js:
        cb = books[j]
        s_t, o_t = (sample_codebook_hard(cb, n) if sampler == "hard_linear"
                    else sample_codebook(cb, n))
        s_t, o_t = s_t.detach(), o_t.detach()
        bb = {"observation.state": s_t.view(st_shape).to(x_t.dtype)}
        cls_j = o_t.reshape(-1, cls_dim).to(x_t.dtype)
        past = [instructions[j]] * n
        with torch.no_grad():
            tl_T = cond_tail(teacher, bb, cls_j)
            cond_T = make_cond(encode_lang(teacher, past), tl_T)
            # teacher 부트스트랩: 과거 j 의 정답을 모르니 teacher 에게 묻는다.
            # t=0 (순수 노이즈) 에서 v 를 받아 끝점 a ≈ ε + v 를 추정한다.
            eps0 = torch.randn_like(x_t)
            a_hat = (eps0 + vel(teacher, eps0, torch.zeros_like(t), cond_T)).detach()
            eps1 = torch.randn_like(x_t)
            x_j = ((1 - tcol) * eps1 + tcol * a_hat).detach()
            v_t = vel(teacher, x_j, t, cond_T).detach()
        qs.append(dict(bb=bb, cls_j=cls_j, past=past, x_j=x_j, v_t=v_t,
                       eps1=eps1, a_hat=a_hat))

    def student_v(q):
        tl_S = cond_tail(policy, q["bb"], q["cls_j"])
        cond_S = make_cond(encode_lang(policy, q["past"]), tl_S)
        return vel(policy, q["x_j"], t, cond_S)

    def pair(v_s, q):
        """두 벡터를 **같은** 공간으로. 한쪽만 옮기면 분해 항등식이 깨진다."""
        if space == "act":
            return to_action(v_s, q["x_j"], t), to_action(q["v_t"], q["x_j"], t)
        return v_s, q["v_t"]

    # ── pass 1: dea 면 전 태스크 e 를 모아 태스크 축으로 정규화 ──────────────
    W = None
    if form == "dea":
        es, vs = [], []
        with torch.no_grad():
            for q in qs:
                p_s, p_t = pair(student_v(q), q)
                _, e, _, _, va = split_mag_dir(p_s, p_t, P)
                es.append(e)
                vs.append(va)
        W = dea_weights(torch.stack(es), torch.stack(vs), eps_w)      # (N,B,H)

    # ── pass 2: 태스크마다 손실 -> 즉시 backward ────────────────────────────
    lvl = []
    for i, q in enumerate(qs):
        p_s, p_t = pair(student_v(q), q)
        m, e, r_s, r_t, valid = split_mag_dir(p_s, p_t, P)             # 각 (B,H)
        if form == "sqerr":
            # .detach(): 계수는 학습 대상이 아니다. 빼면 "r 을 줄여 계수를
            # 낮추는" 경로로 그래디언트가 흘러 속도를 0 으로 눌러 버린다.
            coef = (2.0 * r_s * r_t).detach() * inv_N
        elif form == "unifmd":
            coef = torch.full_like(e, beta * inv_N)
        elif form == "dea":
            coef = beta * W[i]
        else:
            raise ValueError(f"알 수 없는 anchor_form: {form}")
        L_m = inv_N * m.mean()
        # 무효 칸(teacher 속도가 ~0 이라 각도가 정의되지 않는 칸)은 0 으로 둔다.
        # e[valid].mean() 이 아니라 where+mean 을 쓰는 이유는 분모를 B x H 로
        # 고정해 유효 칸 수가 배치마다 달라도 손실 스케일이 안 흔들리게 하려는 것.
        L_e = torch.where(valid, coef * e, torch.zeros_like(e)).mean()
        L_j = L_m + L_e
        (LAMBDA_ANCHOR * lam_lvl * L_j).backward()
        lvl.append(L_j.detach())
        if diag is not None:
            # LC 식 로그는 w_mean/w_max 쌍을 함께 읽고 함께 비운다.
            diag.setdefault("w_mean", []).append(float(coef.mean()))
            diag.setdefault("w_max", []).append(float(coef.max()))
            for key, val in (("dea_m", float(m.mean())),
                             ("dea_e", float(e[valid].mean()) if valid.any() else 0.0)):
                lst = diag.setdefault(key, [])
                lst.append(val)
                del lst[:-512]          # 10k step x N 태스크라 직접 잘라 둔다
    return (lam_lvl * (sum(lvl) / N)).detach()


def selftest() -> int:
    """모델 없이 수식을 검증한다. 분해 항등식 / sqerr==MSE / Σw=1 / to_action."""
    torch.manual_seed(0)
    B, H, A = 32, 16, 7
    v_s = torch.randn(B, H, A, dtype=torch.float64)
    v_t = torch.randn(B, H, A, dtype=torch.float64)
    bad = 0
    print("Lotus_L3_V7dea 자체 검증")
    for P in (6, A):
        m, e, r_s, r_t, valid = split_mag_dir(v_s, v_t, P)
        lhs = (v_s[..., :P] - v_t[..., :P]).pow(2).sum(-1)
        err = (lhs - (m + 2.0 * r_s * r_t * e)).abs().max().item()
        ok = err < 1e-9
        bad += not ok
        print(f"  ① 분해 항등식 P={P}: 최대오차 {err:.2e}  {'OK' if ok else '✗'}")
    P = A
    m, e, r_s, r_t, valid = split_mag_dir(v_s, v_t, P)
    mine = m.mean() + torch.where(valid, (2.0 * r_s * r_t) * e,
                                  torch.zeros_like(e)).mean()
    orig = (v_s - v_t).flatten(1).pow(2).mean(1).mean()      # 원본 MSE 앵커
    rel = abs(float(mine) - float(orig) * A) / float(orig * A)
    ok = rel < 1e-9
    bad += not ok
    print(f"  ② sqerr == MSE×A : {float(mine):.6f} vs {float(orig)*A:.6f}  "
          f"상대오차 {rel:.2e}  {'OK' if ok else '✗'}")
    N = 25
    es = torch.rand(N, B, H, dtype=torch.float64) * 2.0
    vs = torch.ones(N, B, H, dtype=torch.bool)
    err = (dea_weights(es, vs, 1e-3).sum(0) - 1.0).abs().max().item()
    # 허용오차 1e-9: dea_weights 의 nv * eps_w 가 int64 x float -> float32 로
    # 승격되어 ε_w 항만 float32 정밀도가 된다. 실제 학습은 e 가 float32 라 무관.
    ok = err < 1e-9
    bad += not ok
    print(f"  ③ Σ_j w_j == 1 (N={N}): 최대오차 {err:.2e}  {'OK' if ok else '✗'}")
    vs[0] = False
    W = dea_weights(es, vs, 1e-3)
    ok = (W[0].abs().max().item() < 1e-12
          and (W.sum(0) - 1.0).abs().max().item() < 1e-9)
    bad += not ok
    print(f"  ④ 무효 태스크 제외 후에도 합 1 : {'OK' if ok else '✗'}")
    tt = torch.rand(B, dtype=torch.float64) * 0.98
    x = torch.randn(B, H, A, dtype=torch.float64)
    a = to_action(v_s, x, tt)
    back = (a - x) / (1 - tt[:, None, None])
    ok = a.shape == v_s.shape and (back - v_s).abs().max().item() < 1e-9
    bad += not ok
    print(f"  ⑤ to_action 가역·모양 보존 : {'OK' if ok else '✗'}")
    print()
    print("전부 통과" if bad == 0 else f"✗ {bad} 항목 불일치")
    return 1 if bad else 0


# ============================================================================
#  메인
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # required=True 로 두면 --selftest 만 줘도 파서가 먼저 막는다. 뒤에서 검사한다.
    ap.add_argument("--suite", choices=["goal", "object", "50"], default=None)
    ap.add_argument("--steps", type=int, default=10000, help="lifelong 스테이지당 스텝")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--k_per_demo", type=int, default=2,
                    help="코드북 셀 수를 demo 수에 비례해 정한다: K = k_per_demo x demo. "
                         "base 50 demo -> K=100, lifelong 10 demo -> K=20.")
    ap.add_argument("--k_min", type=int, default=8, help="셀 수 하한")
    ap.add_argument("--codebook_k", type=int, default=0,
                    help="0 이면 K = k_per_demo x demo (기본 규칙). >0 이면 **모든 태스크**의 "
                         "셀 수를 이 값으로 고정한다(base 도 포함). demo 수에 비례시키지 않고 "
                         "셀 수를 직접 통제하고 싶을 때 쓴다.")
    ap.add_argument("--skip_base_eval", action="store_true",
                    help="base 태스크의 롤아웃을 건너뛴다. lifelong 구간만 재면 될 때 쓴다 — "
                         "libero_50 은 base 가 25 태스크라 stage 마다 25 칸을 더 돌게 되고, "
                         "그게 전체 시간의 대부분이다. ★ base 열은 비워지므로 리포트의 "
                         "지표도 lifelong 태스크만으로 계산된다(base 는 애초에 안 잰다).")
    # ── L3: lifelong 코드북 셀 수 ─────────────────────────────────────────
    ap.add_argument("--lifelong_k", type=int, default=25,
                    help="lifelong 태스크의 코드북 셀 수를 **태스크마다** 이 값으로 "
                         "고정한다 (L3 의 정의). base 는 기존 규칙 K=k_per_demo x demo "
                         "(50 demo -> 100) 를 그대로 쓴다. 0 이면 lifelong 도 기존 규칙 "
                         "(10 demo -> 20). ★ 스테이지당이 아니라 태스크당이다 — 5 태스크 "
                         "스테이지면 K=25 코드북이 5 개 생긴다.")
    # ── V7 dea 앵커 ───────────────────────────────────────────────────────
    ap.add_argument("--sampler_mode", choices=["soft_kernel", "hard_linear"],
                    default="hard_linear",
                    help="hard_linear: 경성 배정 + 1차 항만 (커널 가중·잡음 항 없음)")
    ap.add_argument("--anchor_form", choices=["dea", "sqerr", "unifmd"], default="dea",
                    help="방향 항의 계수를 정하는 방식. sqerr 은 MSE 앵커와 항등이라 "
                         "기준선으로 쓴다")
    ap.add_argument("--beta", type=float, default=3.0, help="방향 항 세기")
    ap.add_argument("--eps_w", type=float, default=1e-3,
                    help="dea 가중의 0/0 방지. 작을수록 적응이 세다")
    ap.add_argument("--dir_space", choices=["vel", "act"], default="act",
                    help="act = 끝점 액션 공간에서 분해 (V7). vel = 속도 공간 (V6)")
    ap.add_argument("--anchor_reduce", choices=["mean", "sum"], default="mean",
                    help="크기 항에 1/N 을 걸지(mean) 그대로 둘지(sum). ★ LOTUS 원본 "
                         "앵커는 과거에 대한 **합**이라 sum 이 그쪽과 같은 스케일이다")
    ap.add_argument("--pose_dims", type=int, default=0,
                    help="0 이면 POSE_DIMS(6, gripper 제외). 7 로 두면 전 축")
    ap.add_argument("--skip_eval", action="store_true",
                    help="롤아웃을 전부 건너뛰고 학습만 한다. 스테이지 체크포인트는 "
                         "남으므로 다른 GPU 에서 따로 SR 을 채울 수 있다")
    ap.add_argument("--selftest", action="store_true",
                    help="모델 없이 수식만 검증하고 끝낸다")
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
    if args.selftest:
        raise SystemExit(selftest())
    if args.suite is None:
        raise SystemExit("--suite 가 필요하다 (goal|object|50). 수식만 보려면 --selftest.")

    # GPU 지정은 셸 래퍼가 torch import 전에 export 한다.
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
    logging.info(f"[L3] lifelong K={args.lifelong_k or '규칙'} (태스크당)  "
                 f"base K=k_per_demo x demo  |  앵커 {args.anchor_form} β={args.beta} "
                 f"space={args.dir_space} reduce={args.anchor_reduce} "
                 f"P={args.pose_dims or POSE_DIMS}"
                 + ("  |  롤아웃 건너뜀(--skip_eval)" if args.skip_eval else ""))

    # ── stage 0: base 체크포인트 로드 ────────────────────────────────────────
    cfg = build_eval_cfg(args, str(base_ckpt), env_task[0])
    base_meta = LeRobotDatasetMetadata("local/lotus", root=str(root / "base"))
    policy = make_policy(cfg=cfg.policy, ds_meta=base_meta)
    policy.to(device)
    logging.info(f"[LOTUS] base 체크포인트 로드 -> {base_ckpt}")

    t0 = time.perf_counter()
    # base 는 기존 규칙(K = k_per_demo x demo = 100). k_fixed 를 넘기지 않는다.
    books = make_books(policy, root / "base", device, args,
                       task_ids=list(range(n_base)), n_demos=spec["base_demos"],
                       k_fixed=None)
    logging.info(f"[LOTUS] base 코드북 {len(books)}개 ({time.perf_counter()-t0:.0f}s)")
    teacher = snapshot(policy)

    if args.skip_eval or getattr(args, "skip_base_eval", False):
        logging.info(f"[LOTUS] base 롤아웃 건너뜀 (--skip_base_eval) — base {n_base}개 열은 빈 칸")
    else:
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

        diag = {}
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

        # ★ L3: lifelong 은 태스크마다 K=args.lifelong_k (기본 25) 로 고정한다.
        books.update(make_books(policy, sroot, device, args,
                                task_ids=new_tasks, n_demos=spec["stage_demos"],
                                k_fixed=args.lifelong_k))
        del teacher
        teacher = snapshot(policy)
        torch.cuda.empty_cache()

        for t in range(n_seen):
            if args.skip_eval:
                break            # 체크포인트만 남긴다 — 롤아웃은 다른 GPU 에서
            if getattr(args, "skip_base_eval", False) and learned_at[t] == 0:
                continue          # base 태스크는 아예 안 잰다 — 시간의 대부분이 여기에 쓰인다
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
