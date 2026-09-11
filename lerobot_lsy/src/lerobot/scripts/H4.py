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

"""H4 — 태스크 간 Fisher 충돌(cross-task Fisher conflict)을 측정한다.

가설(H4)
    LIBERO-spatial의 태스크들은 같은 물체·같은 동작 통계를 공유하므로 "중요한
    파라미터"의 집합이 태스크 간에 심하게 겹친다. 겹치면 EWC의 λ가 만드는
    stability–plasticity 파레토가 퇴화한다: 지키면 새 걸 못 배우고, 배우면 지킬 수
    없는, "좋은 λ가 존재하지 않는" 구조.

이 스크립트가 재는 것 (E0가 학습을 돌려서 SR로 보는 현상을, 학습 전에 파라미터
공간에서 직접 재는 쪽이다)

    [A] 공통 앵커 θ*(사전학습 체크포인트)에서 태스크 k=0..N-1 각각의
        대각 Fisher F_k 와 평균 그래디언트 g_k 를 잰다.
        ★ 반드시 같은 θ*에서 재야 한다. 순차 학습 중에 재면 파라미터가 이동한
          효과와 태스크 차이가 섞여 "겹침"이 무엇의 겹침인지 알 수 없게 된다.

    [B] 세 종류의 수치를 낸다.
        1) 중요 파라미터 부공간이 얼마나 겹치는가
             cosine(F_i, F_j),  Bhattacharyya overlap Σ√(p_i q_i),
             상위 p% 마스크 교집합 / chance(=p)  ← "lift". 1이면 우연 수준.
        2) 이전 태스크의 큰 Fisher가 다음 태스크의 update 성분을 얼마나 덮는가
             blocked_gain(p) = 새 태스크가 얻을 수 있는 손실 감소량 중
                               "이전 태스크 상위 p% 좌표"에 들어 있는 비율.
             chance는 p이므로 0.01에서 0.5가 나오면 50배 겹친 것이다.
        3) λ의 파레토가 퇴화했는가  ← H4의 본체
             EWC가 스스로 가정하는 대각 2차 모델 안에서 닫힌 형태로 계산한다.
               새 태스크:  L_new(θ*+δ) ≈ L_new(θ*) + gᵀδ + ½ δᵀdiag(F_new)δ
               EWC 페널티: ½ λ δᵀdiag(F_old)δ
             ⇒ 좌표별 최적 스텝  δ_i(λ) = −g_i / (F_new,i + λ F_old,i)
                                       = s_i · δ_i(0),   s_i = F_new,i/(F_new,i+λF_old,i)
             plasticity(λ) = 새 태스크 손실 감소 / λ=0일 때의 감소 = Σw_i(2s_i−s_i²)/Σw_i
             forgetting(λ) = 옛 태스크 손상 / λ=0일 때의 손상   = Σ F_old,i δ_i(λ)² / Σ F_old,i δ_i(0)²
             둘 다 (1,1)에서 (0,0)으로 간다. 좋은 λ가 있으려면 곡선이 좌상단
             (forgetting≈0, plasticity≈1)으로 불룩해야 한다.

             ★ 퇴화의 기준선: 곡선의 모양을 정하는 것은 좌표별 비율
               r_i = F_old,i / F_new,i 하나뿐이다 (s_i = 1/(1+λ r_i)).
               r이 좌표마다 같으면 — 즉 두 태스크의 중요 파라미터가 같은 모양이면 —
               s가 상수가 되어 곡선이 plasticity = 2√f − f 로 **정확히** 고정되고
               그 위에서 max(plasticity − forgetting) = 0.5 이다.
               r이 퍼져 있어야 λ가 좌표를 골라내 (0,1) 코너 쪽으로 갈 수 있다.
               그래서 pareto_gain G = max_λ(plasticity − forgetting) 이
                   G ≈ 0.5  → H4 성립(좋은 λ 없음)
                   G → 1    → λ로 분리 가능
               로 읽히는 한 개짜리 요약이 된다 (degeneracy = 1−2(G−0.5)로도 낸다).

               주의 두 가지.
                 · G는 r의 **퍼짐**만 본다. r 전체가 작아지면(=옛 태스크가 애초에
                   손상될 게 별로 없으면) 모양은 그대로이고 λ*만 옮겨간다. 그래서
                   λ*와 절대 수준(gain_free / damage_free)을 같이 남긴다.
                 · λ*가 격자 끝이면 G는 "격자 안에서의 최선"이다.
                   pareto_gain_at_grid_edge=True로 표시하니 그때는 --lambdas를 넓힌다.
                 · 격자에 아예 의존하지 않는 짝은 ratio_log_std (gain-가중 log r의
                   표준편차)다. 0에 가까울수록 퇴화. G와 같이 읽으면 된다.

    [C] (선택, --measure_steps>0) 위 예측을 실제로 확인한다. θ*에서 λ를 바꿔 가며
        짧게 학습하고 Δθ = θ − θ* 를 λ=0의 Δθ와 비교한다. EWC가 "중요 좌표만"
        골라 막는지(선택적) 아니면 그냥 전체를 줄이는지(퇴화)를 본다.

    [D] JSONL로 쌓고 --plot_only 로 그림을 그린다.

train.py 뼈대([1]~[11] 순서)와 E0.py의 관례를 그대로 따른다. 학습을 하지 않는
것이 기본이므로 gym_libero도 시뮬레이터도 필요 없다.

사용 예
    python H4.py --dataset.repo_id=... --policy.path=... --output_dir=... --num_tasks=4
    python H4.py --plot_only --results=... --out=...
"""

import json
import logging
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import torch
import torch.multiprocessing as mp
from termcolor import colored
from torch.amp import GradScaler

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import format_big_number, get_safe_torch_device, init_logging

# 분석은 파라미터 텐서를 조각내서 float64로 누적한다. 194M 파라미터 × 3벡터를
# 한꺼번에 float64로 올리면 GPU가 터지므로 조각 크기를 고정한다.
CHUNK = 4_000_000


@dataclass
class H4Config(TrainPipelineConfig):
    """train.py 인자 전부 + Fisher 충돌 측정용 인자."""

    # ── 어떤 태스크들을 볼 것인가 ────────────────────────────────────────────
    num_tasks: int = 4
    dataset_prefix: str = "continuallearning/libero_spatial_image_task_"
    holdout_episodes: int = 5      # E0와 같은 분할. Fisher는 학습 에피소드에서만 잰다.

    # ── [A] Fisher / 그래디언트 추정 ─────────────────────────────────────────
    fisher_batches: int = 100      # E0.build_ewc_state와 같은 기본값
    fisher_batch_size: int = 8
    stats_dir: str = ""            # 비면 output_dir/stats. 재실행 시 캐시로 재사용.
    recompute: bool = False        # 캐시가 있어도 다시 잰다

    # ── [B] 분석 ─────────────────────────────────────────────────────────────
    # λ*가 어디에 앉을지 미리 모르므로 넉넉히 깐다. 격자 끝에 걸리면 결과에
    # pareto_gain_at_grid_edge=True 가 찍히니 그때 더 넓히면 된다.
    lambdas: str = ("0,1e-4,1e-3,1e-2,0.03,0.1,0.3,1,3,10,30,100,300,1000,3000,"
                    "1e4,1e5,1e6,1e7,1e8,inf")
    top_p: str = "0.0001,0.001,0.01,0.05,0.1,0.25"   # 상위 p 비율 마스크
    curv_damping: float = 1e-3     # 곡률 바닥값 (평균 대비 비율). F_new≈0 좌표에서
                                   # g²/F_new 가 폭발하는 것을 막는 trust-region 항.
    subsample: int = 1_000_000     # 상위 p% 임계값(분위수)을 추정할 표본 크기
    layer_report: int = 12         # 그림에 올릴 레이어 그룹 개수

    # ── [C] 실측(선택) ───────────────────────────────────────────────────────
    measure_steps: int = 0         # 0이면 [C] 통째로 생략
    measure_lambdas: str = "0,10,100,1000"
    measure_top_p: float = 0.01
    measure_stages: str = ""       # 비면 k=1..num_tasks-1 전부

    # ── 출력 ─────────────────────────────────────────────────────────────────
    run_tag: str = ""
    results_path: str = "outputs/H4/h4_results.jsonl"

    def validate(self):
        """부모 validate()는 output_dir가 이미 있으면 FileExistsError를 낸다.

        H4는 Fisher 캐시를 재사용하려고 같은 output_dir로 여러 번 들어오는 것이
        정상 동작이라 그 검사만 우회한다. (학습 산출물을 덮어쓸 일이 없다 —
        아래 [1]에서 save_checkpoint를 끈다.)
        """
        out = self.output_dir
        if isinstance(out, Path) and out.is_dir():
            self.output_dir = None
            super().validate()
            self.output_dir = out
        else:
            super().validate()


# ═════════════════════════════════════════════════════════════════════════════
#  데이터 (E0.py와 동일한 분할/샘플러 규약)
# ═════════════════════════════════════════════════════════════════════════════
def split_episodes(repo_id: str, root: str | None, holdout: int) -> tuple[list[int], list[int]]:
    total = LeRobotDatasetMetadata(repo_id, root=root).total_episodes
    if not 0 < holdout < total:
        raise ValueError(f"holdout_episodes={holdout} invalid for {repo_id} (total={total})")
    return list(range(total - holdout)), list(range(total - holdout, total))


def episode_sampler(cfg: H4Config, dataset, episodes: list[int], shuffle: bool = True):
    """에피소드 부분집합만 뽑는 샘플러.

    ★ LeRobotDataset(episodes=[...])를 쓰면 안 된다 (E0.py의 같은 함수 주석 참조).
      데이터셋은 통째로 열고 샘플러에서 가른다.
    """
    return EpisodeAwareSampler(
        dataset.episode_data_index,
        episode_indices_to_use=episodes,
        drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
        shuffle=shuffle,
    )


def to_device(batch: dict, device) -> dict:
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device, non_blocking=device.type == "cuda")
    return batch


def open_task_dataset(cfg: H4Config, task: int) -> LeRobotDataset:
    """태스크 k의 데이터셋을 학습 때와 같은 delta_timestamps로 연다."""
    repo_id = f"{cfg.dataset_prefix}{task}"
    return LeRobotDataset(
        repo_id,
        delta_timestamps=resolve_delta_timestamps(cfg.policy, LeRobotDatasetMetadata(repo_id)),
        video_backend=cfg.dataset.video_backend,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  [A] 공통 앵커에서 태스크별 Fisher + 평균 그래디언트
# ═════════════════════════════════════════════════════════════════════════════
def estimate_task_stats(cfg: H4Config, policy: PreTrainedPolicy, task: int, device) -> dict:
    """태스크 k의 (F_k, ĝ_k²)를 앵커 θ*에서 잰다.

    F_i  = (1/M) Σ_b (∂L_b/∂θ_i)²     ← E0.build_ewc_state와 정확히 같은 추정량
    ĝ_i² = ((1/M) Σ_b ∂L_b/∂θ_i)²

    두 값의 역할이 다르다는 점이 중요하다. 2차 모델에서 좌표 i의 손실 감소는
    ΔL_i = ½ g_i²/F_i 이므로,
        분자 ĝ²  = 그 좌표가 손실을 줄이라고 얼마나 세게 미는가 (1차, 방향)
        분모 F   = 그 좌표가 얼마나 뻣뻣한가                    (2차, 곡률)
    분자에 F를 다시 넣으면 gain이 F/(F+damp) ≈ 1 로 상수가 되어
    blocked_gain(p)가 데이터와 무관하게 항상 p(=우연 수준)가 된다. 섞으면 안 된다.

    ĝ²의 편향에 대하여: ĝ²는 미니배치 잡음 때문에 E[ĝ²] = E[g]² + Var/M 만큼
    위로 편향돼 있다. 이전 버전은 이를 ĝ² − (F − ĝ²)/(M−1) 로 보정했는데,
    M=100에서 실측해보니(43.9M 파라미터, libero_spatial task0→task1)
        보정으로 0이 되는 좌표 22.6%, 그러나 총 질량의 4.3%뿐
        blocked_gain@1%   0.0191 → 0.0191 (변화 없음)
        pareto_gain       0.6334 → 0.6321 (0.2%)
    라 결론에 영향이 없어 제거했다. 편향은 Var/M 이므로 M에 반비례한다.
    fisher_batches를 20 이하로 줄일 일이 생기면 보정을 되살려야 한다.
    """
    trainable = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    fisher = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in trainable.items()}
    gsum = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in trainable.items()}

    dataset = open_task_dataset(cfg, task)
    train_eps, _ = split_episodes(f"{cfg.dataset_prefix}{task}", None, cfg.holdout_episodes)
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=cfg.fisher_batch_size,
        sampler=episode_sampler(cfg, dataset, train_eps),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    it = cycle(loader)

    M = cfg.fisher_batches
    logging.info(f"[H4] task {task}: estimating Fisher/grad over {M} batches (bs={cfg.fisher_batch_size})")
    policy.eval()
    t0 = time.perf_counter()
    for b in range(M):
        policy.zero_grad(set_to_none=True)
        policy.forward(to_device(next(it), device))[0].backward()
        for n, p in trainable.items():
            if p.grad is not None:
                g = p.grad.detach().float()
                fisher[n] += g.pow(2)
                gsum[n] += g
        if (b + 1) % 20 == 0:
            logging.info(f"[H4]   task {task}: {b + 1}/{M} ({time.perf_counter() - t0:.0f}s)")
    policy.zero_grad(set_to_none=True)
    policy.train()

    out_f, out_g2 = {}, {}
    for n in fisher:
        out_f[n] = (fisher[n] / M).flatten().cpu()
        out_g2[n] = (gsum[n] / M).pow(2).flatten().cpu()
    return {"fisher": out_f, "grad2": out_g2, "batches": M, "task": task}


def task_stats_path(cfg: H4Config, task: int) -> Path:
    root = Path(cfg.stats_dir) if cfg.stats_dir else Path(cfg.output_dir) / "stats"
    return root / f"h4_stats_task_{task}.pt"


def load_stats(path: Path) -> dict:
    """mmap으로 연다. 태스크 4개 × (F,g²) = 수 GB라 통째로 올리면 아깝다."""
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


# ═════════════════════════════════════════════════════════════════════════════
#  [B] 스트리밍 누적기
# ═════════════════════════════════════════════════════════════════════════════
def parse_floats(s: str) -> list[float]:
    return [float("inf") if x.strip() == "inf" else float(x) for x in s.split(",") if x.strip()]


def group_of(name: str) -> str:
    """'model.blocks.3.attn.qkv.weight' -> 'model.blocks.*.attn.qkv' (레이어 묶음)."""
    parts = ["*" if p.isdigit() else p for p in name.split(".")]
    return ".".join(parts[:-1]) if len(parts) > 1 else name


def shrink(fo: torch.Tensor, h: torch.Tensor, lam: float) -> torch.Tensor:
    """s = h / (h + λ·F_old).  λ=0과 λ=inf는 0/0·inf가 나므로 따로 처리한다."""
    if lam == 0.0:
        return torch.ones_like(h)
    if math.isinf(lam):
        # F_old = 0 인 좌표는 페널티가 없으니 그대로 움직인다. 나머지는 완전 동결.
        return (fo <= 0).to(h.dtype)
    return h / (h + lam * fo)


class PairAcc:
    """(F_old, F_new, g_new²) 스트림에 대한 스칼라 누적기.

    파라미터 텐서를 하나씩(다시 CHUNK 단위로) 밀어 넣으면 전체 벡터를 메모리에
    올리지 않고도 모든 지표가 나온다. 레이어별 분해도 같은 객체를 그룹마다 하나씩
    두는 것으로 끝난다.
    """

    def __init__(self, lambdas: list[float], ps: list[float]):
        self.lambdas, self.ps = lambdas, ps
        self.numel = 0
        self.dot = self.no2 = self.nn2 = self.bc = 0.0     # 부공간 겹침용
        self.sum_gain = self.sum_g2 = self.fden = 0.0      # 분모들
        self.plast = [0.0] * len(lambdas)
        self.forget = [0.0] * len(lambdas)
        self.m_old = [0.0] * len(ps)                       # |상위 p% of F_old|
        self.m_both = [0.0] * len(ps)                      # |상위 p% 교집합|
        self.b_gain = [0.0] * len(ps)                      # 그 안에 들어간 손실감소 몫
        self.b_g2 = [0.0] * len(ps)                        # 그 안에 들어간 그래디언트 에너지
        self.wL = self.wL2 = 0.0                           # 비율 r=F_old/F_new의 가중 로그 분포

    def iadd(self, o: "PairAcc"):
        """누적기는 전부 합이라 그냥 더하면 된다. 레이어별로 한 번만 돌고 전체는 합쳐서 얻는다."""
        self.numel += o.numel
        for k in ("dot", "no2", "nn2", "bc", "sum_gain", "sum_g2", "fden", "wL", "wL2"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        for k in ("plast", "forget", "m_old", "m_both", "b_gain", "b_g2"):
            a, bl = getattr(self, k), getattr(o, k)
            for i in range(len(a)):
                a[i] += bl[i]

    def add(self, fo_t, fn_t, g2_t, taus_o, taus_n, damp: float):
        for i in range(0, fo_t.numel(), CHUNK):
            fo = fo_t[i : i + CHUNK].double()
            fn = fn_t[i : i + CHUNK].double()
            g2 = g2_t[i : i + CHUNK].double()

            self.numel += fo.numel()
            self.dot += float((fo * fn).sum())
            self.no2 += float((fo * fo).sum())
            self.nn2 += float((fn * fn).sum())
            self.bc += float((fo * fn).clamp_min(0).sqrt().sum())

            # 곡률에 바닥을 깐다. damp는 F_new의 평균(=1) 대비 비율이다.
            h = fn + damp
            gain = g2 / h              # ∝ λ=0에서 얻는 손실 감소 (×2)
            step2 = g2 / (h * h)       # δ_free²
            self.sum_gain += float(gain.sum())
            self.sum_g2 += float(g2.sum())
            self.fden += float((fo * step2).sum())

            # 파레토 모양을 결정하는 것은 좌표별 비율 r_i = F_old,i / F_new,i 하나뿐이다.
            # (s_i = 1/(1+λ r_i) 이므로) r이 좌표마다 같으면 s가 상수가 되어 곡선이
            # 2√f−f 로 퇴화하고, r이 퍼져 있으면 λ가 좌표를 골라낼 수 있다.
            # 그래서 "새 태스크가 배울 몫(gain)으로 가중한 log r의 표준편차"가
            # λ 격자에 의존하지 않는 퇴화 지표가 된다. 0에 가까울수록 퇴화.
            lr = (fo / h).clamp_min(1e-12).log()
            self.wL += float((gain * lr).sum())
            self.wL2 += float((gain * lr * lr).sum())

            for j, lam in enumerate(self.lambdas):
                s = shrink(fo, h, lam)
                self.plast[j] += float((gain * (2 * s - s * s)).sum())
                self.forget[j] += float((fo * step2 * s * s).sum())

            for j, (to_, tn_) in enumerate(zip(taus_o, taus_n, strict=True)):
                mo = fo >= to_
                self.m_old[j] += float(mo.sum())
                self.m_both[j] += float((mo & (fn >= tn_)).sum())
                self.b_gain[j] += float(gain[mo].sum())
                self.b_g2[j] += float(g2[mo].sum())

    def result(self) -> dict:
        eps = 1e-30
        plast = [p / (self.sum_gain + eps) for p in self.plast]
        forget = [f / (self.fden + eps) for f in self.forget]
        gains = [p - f for p, f in zip(plast, forget, strict=True)]
        best = max(range(len(gains)), key=lambda i: gains[i])
        lam_best = self.lambdas[best]
        # λ*가 격자 끝에 걸리면 pareto_gain은 "격자 안에서의 최선"일 뿐 진짜 최대가 아니다.
        finite = [i for i, x in enumerate(self.lambdas) if 0 < x < float("inf")]
        edge = bool(finite) and best in (finite[0], finite[-1])
        mL = self.wL / (self.sum_gain + eps)
        var = max(self.wL2 / (self.sum_gain + eps) - mL * mL, 0.0)
        return {
            "numel": self.numel,
            # ── 1) 중요 파라미터 부공간이 얼마나 겹치는가 ──
            "cosine": self.dot / (math.sqrt(self.no2 * self.nn2) + eps),
            "bhattacharyya": self.bc / (self.numel + eps),   # F는 둘 다 평균 1로 정규화됨
            "top_p": self.ps,
            "mask_overlap": [b / (m + eps) for b, m in zip(self.m_both, self.m_old, strict=True)],
            "mask_lift": [
                (b / (m + eps)) / p for b, m, p in zip(self.m_both, self.m_old, self.ps, strict=True)
            ],
            # ── 2) 이전 태스크 상위 좌표가 새 태스크 update를 얼마나 덮는가 ──
            "blocked_gain": [b / (self.sum_gain + eps) for b in self.b_gain],
            "blocked_grad": [b / (self.sum_g2 + eps) for b in self.b_g2],
            # ── 3) λ 파레토 ──
            "lambdas": ["inf" if math.isinf(x) else x for x in self.lambdas],
            "plasticity": plast,
            "forgetting": forget,
            "pareto_gain": gains[best],
            "pareto_gain_lambda": "inf" if math.isinf(lam_best) else lam_best,
            "pareto_gain_at_grid_edge": edge,   # True면 λ 격자를 넓혀야 한다
            "plasticity_at_best": plast[best],
            "forgetting_at_best": forget[best],
            # 완전 겹침(F_old ∝ F_new)일 때의 이론값. 관측값이 이것과 붙으면 H4 성립.
            "pareto_gain_degenerate": 0.5,
            # 0.5(완전 퇴화) ~ 1.0(분리 가능)을 1~0으로 뒤집은 읽기 편한 요약.
            "degeneracy": max(0.0, min(1.0, 1.0 - 2.0 * (gains[best] - 0.5))),
            # ── λ 격자와 무관한 퇴화 지표: log r 의 gain-가중 분포 ──
            "ratio_log_std": math.sqrt(var),          # 0에 가까울수록 퇴화
            "ratio_gmean": math.exp(mL),              # 1/이 값이 대략 λ*
            # ── 절대 수준: "애초에 얼마나 손상될 게 있었나" (파레토 모양과는 별개) ──
            "gain_free": self.sum_gain,               # λ=0에서 새 태스크가 얻는 몫
            "damage_free": self.fden,                 # λ=0에서 옛 태스크가 입는 몫
        }


def global_thresholds(vals: torch.Tensor, ps: list[float]) -> list[float]:
    """정규화된 Fisher 표본에서 상위 p% 경계값을 구한다."""
    v = vals[torch.isfinite(vals)]
    if v.numel() == 0:
        return [float("inf")] * len(ps)
    q = torch.tensor([1.0 - p for p in ps], dtype=torch.float64)
    return [float(x) for x in torch.quantile(v.double(), q)]


def analyze_pair(
    cfg: H4Config,
    old_files: list[dict],
    new_file: dict,
    names: list[str],
    lambdas: list[float],
    ps: list[float],
    device,
) -> tuple[dict, dict]:
    """F_old = Σ_{j∈old} F_j (평균 1로 정규화), F_new = F_new (평균 1로 정규화).

    E0가 EWC state를 만들 때 하는 정규화(누적 후 mean=1)를 그대로 따라간다.
    그래야 여기서 나온 λ 눈금이 E0의 λ 스윕과 같은 축 위에 놓인다.

    반환: (전체 지표, {레이어그룹: 지표})
    """
    # ── pass 1: 평균과 임계값 추정용 표본 ─────────────────────────────────────
    tot = sum(new_file["fisher"][n].numel() for n in names)
    rate = min(1.0, cfg.subsample / max(tot, 1))
    gen = torch.Generator().manual_seed(cfg.seed if cfg.seed is not None else 0)
    sum_o = sum_n = 0.0
    samp_o, samp_n = [], []
    for n in names:
        fo = old_files[0]["fisher"][n].clone()
        for f in old_files[1:]:
            fo += f["fisher"][n]
        fn = new_file["fisher"][n]
        sum_o += float(fo.double().sum())
        sum_n += float(fn.double().sum())
        k = max(1, int(round(fo.numel() * rate)))
        idx = torch.randint(0, fo.numel(), (k,), generator=gen)
        samp_o.append(fo[idx])
        samp_n.append(fn[idx])
    mean_o = max(sum_o / tot, 1e-30)
    mean_n = max(sum_n / tot, 1e-30)
    taus_o = global_thresholds(torch.cat(samp_o) / mean_o, ps)
    taus_n = global_thresholds(torch.cat(samp_n) / mean_n, ps)
    del samp_o, samp_n

    # g²도 스케일을 맞춘다. 모든 지표가 g²의 **비율**만 쓰므로 상수는 무해하지만,
    # damp가 F_new의 평균 대비 비율이라는 해석을 유지하려면 F_new와 같이 나눠야 한다.
    gsum = sum(float(new_file["grad2"][n].double().sum()) for n in names)
    gmean = max(gsum / tot, 1e-30)

    # ── pass 2: 본 계산 ───────────────────────────────────────────────────────
    # ★ 원본은 mmap 백업 텐서다. 절대 in-place로 건드리면 안 되므로 copy=True로 뜬다.
    per_layer: dict[str, PairAcc] = {}
    for n in names:
        fo = old_files[0]["fisher"][n].to(device=device, dtype=torch.float32, copy=True)
        for f in old_files[1:]:
            fo += f["fisher"][n].to(device, torch.float32)
        fo /= mean_o
        fn = new_file["fisher"][n].to(device, torch.float32) / mean_n
        g2 = new_file["grad2"][n].to(device, torch.float32) / gmean

        per_layer.setdefault(group_of(n), PairAcc(lambdas, ps)).add(
            fo, fn, g2, taus_o, taus_n, cfg.curv_damping
        )
        del fo, fn, g2

    total = PairAcc(lambdas, ps)
    for acc in per_layer.values():
        total.iadd(acc)
    return total.result(), {gname: a.result() for gname, a in per_layer.items()}


# ═════════════════════════════════════════════════════════════════════════════
#  [C] 실측: EWC가 실제로 update를 얼마나 막는가
# ═════════════════════════════════════════════════════════════════════════════
def ewc_penalty(policy: PreTrainedPolicy, state: dict) -> torch.Tensor:
    """0.5·Σ_i F_i (θ_i − θ*_i)²  (E0.py와 동일)."""
    total = None
    for n, p in policy.named_parameters():
        if p.requires_grad and n in state["fisher"]:
            t = (state["fisher"][n] * (p - state["anchor"][n]).pow(2)).sum()
            total = t if total is None else total + t
    if total is None:
        raise RuntimeError("EWC penalty가 아무 파라미터도 못 잡았다 (이름 불일치)")
    return 0.5 * total


def short_run_delta(
    cfg: H4Config, policy: PreTrainedPolicy, anchor: dict, dataset, train_eps,
    ewc_state: dict | None, lam: float, device,
) -> dict:
    """θ*에서 lam으로 measure_steps만큼 학습하고 Δθ = θ − θ* 를 CPU에 돌려준다.

    ★ λ마다 시드를 같은 값으로 되돌린다. 배치 순서와 flow-matching의 시각/노이즈
      샘플까지 λ 사이에 맞춰야 Δθ의 차이를 "EWC가 막은 몫"으로 읽을 수 있다.
      안 맞추면 표본 잡음이 λ 효과와 섞인다.
    """
    if cfg.seed is not None:
        set_seed(cfg.seed)
    with torch.no_grad():
        for n, p in policy.named_parameters():
            if n in anchor:
                p.copy_(anchor[n])
    policy.zero_grad(set_to_none=True)

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        sampler=episode_sampler(cfg, dataset, train_eps),
        pin_memory=device.type == "cuda",
        drop_last=False,
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )
    dl_iter = cycle(loader)

    policy.train()
    mse_sum = pen_sum = 0.0
    for step in range(cfg.measure_steps):
        batch = to_device(next(dl_iter), device)
        with torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
            mse, _ = policy.forward(batch)
            # λ=inf·0 = nan을 피하려고 곱셈을 조건 안에 둔다 (E0.update_policy와 같은 이유).
            if ewc_state is not None and 0 < lam < float("inf"):
                penalty = ewc_penalty(policy, ewc_state)
                loss = mse + lam * penalty
            else:
                penalty = torch.zeros((), device=device)
                loss = mse
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optimizer.grad_clip_norm,
                                       error_if_nonfinite=False)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        mse_sum += float(mse.detach())
        pen_sum += float(penalty.detach())
        if cfg.log_freq > 0 and (step + 1) % cfg.log_freq == 0:
            logging.info(f"[H4]   λ={lam:g} step {step + 1}/{cfg.measure_steps} "
                         f"mse={mse_sum / (step + 1):.4f} pen={pen_sum / (step + 1):.3e}")

    delta = {}
    with torch.no_grad():
        for n, p in policy.named_parameters():
            if n in anchor:
                delta[n] = (p.detach() - anchor[n]).flatten().cpu()
    del loader, dl_iter, optimizer, lr_scheduler, grad_scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()      # 다음 λ가 Adam 상태(파라미터 2배)를 새로 잡는다
    return {"delta": delta, "mse": mse_sum / max(cfg.measure_steps, 1),
            "penalty": pen_sum / max(cfg.measure_steps, 1)}


def compare_delta(base: dict, cur: dict, fo_norm: dict, tau: float, device) -> dict:
    """Δθ_λ 를 Δθ_0 와 비교한다.

    EWC가 의도대로 동작한다면 "이전 태스크 상위 p% 좌표"에서만 크게 줄어들고
    나머지는 거의 그대로여야 한다(선택적 차단). 부공간이 겹치면 두 곳이 똑같이
    줄어든다 — 보호는 못 하면서 가소성만 잃는, 딱 H4의 그림이다.
    """
    acc = {k: 0.0 for k in ("n2b_top", "n2c_top", "n2b_rest", "n2c_rest", "dot", "b2", "c2")}
    for n, b in base.items():
        fo = fo_norm[n].to(device, torch.float32)
        bb = b.to(device, torch.float32)
        cc = cur[n].to(device, torch.float32)
        m = fo >= tau
        acc["n2b_top"] += float(bb[m].double().pow(2).sum())
        acc["n2c_top"] += float(cc[m].double().pow(2).sum())
        acc["n2b_rest"] += float(bb[~m].double().pow(2).sum())
        acc["n2c_rest"] += float(cc[~m].double().pow(2).sum())
        acc["dot"] += float((bb.double() * cc.double()).sum())
        acc["b2"] += float(bb.double().pow(2).sum())
        acc["c2"] += float(cc.double().pow(2).sum())
    eps = 1e-30
    top = math.sqrt(acc["n2c_top"] / (acc["n2b_top"] + eps))
    rest = math.sqrt(acc["n2c_rest"] / (acc["n2b_rest"] + eps))
    return {
        "shrink_all": math.sqrt(acc["c2"] / (acc["b2"] + eps)),
        "shrink_top": top,          # 이전 태스크 상위 좌표에서 살아남은 update 크기 비율
        "shrink_rest": rest,        # 나머지 좌표에서 살아남은 비율
        "selectivity": rest / (top + eps),   # 1이면 무차별 축소 = 퇴화
        "cosine_with_free": acc["dot"] / (math.sqrt(acc["b2"] * acc["c2"]) + eps),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  메인 (train.py와 같은 [1]~ 순서)
# ═════════════════════════════════════════════════════════════════════════════
@parser.wrap()
def main(cfg: H4Config):
    # ── [1] 설정 ─────────────────────────────────────────────────────────────
    cfg.validate()
    cfg.save_checkpoint = False          # H4는 정책을 저장하지 않는다
    if cfg.measure_steps > 0:
        # 프리셋 스케줄러의 총 길이를 실측 구간에 맞춘다 (make_optimizer_and_scheduler가
        # cfg.steps를 호출 시점에 읽으므로 여기서 바꾸면 된다).
        cfg.steps = cfg.measure_steps
    logging.info(pformat(cfg.to_dict()))
    logging.info(colored(f"[H4] tasks 0..{cfg.num_tasks - 1}  anchor={cfg.policy.pretrained_path}",
                         "green", attrs=["bold"]))

    # ── [2] 로거: H4는 스칼라 표만 내므로 wandb를 쓰지 않는다 ─────────────────

    # ── [3] 재현성 ───────────────────────────────────────────────────────────
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # ── [4] 디바이스 ─────────────────────────────────────────────────────────
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── [5] 데이터셋 (정책 생성용 메타는 --dataset.repo_id 하나면 된다) ───────
    logging.info("Creating dataset")
    dataset0 = make_dataset(cfg)

    # ── [6] 평가 환경 없음 (SR은 E0가 잰다. H4는 시뮬레이터가 필요 없다) ──────

    # ── [7] 정책 = 공통 앵커 θ* ──────────────────────────────────────────────
    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset0.meta)
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f"[H4] trainable params: {n_train} ({format_big_number(n_train)})")

    # ── [8] 앵커 사본 (실측에서 매번 θ*로 되돌리는 데 쓴다) ───────────────────
    anchor = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}

    results = Path(cfg.results_path)
    results.parent.mkdir(parents=True, exist_ok=True)
    tag = cfg.run_tag or "h4"

    def emit(row: dict):
        row = {"run_tag": tag, "seed": cfg.seed, **row}
        with results.open("a") as f:
            f.write(json.dumps(row) + "\n")

    # ── [A] 태스크별 Fisher + 그래디언트 ─────────────────────────────────────
    logging.info(colored("[H4][A] per-task Fisher at the common anchor", "cyan", attrs=["bold"]))
    for k in range(cfg.num_tasks):
        path = task_stats_path(cfg, k)
        if path.exists() and not cfg.recompute:
            logging.info(f"[H4] task {k}: cached -> {path}")
            continue
        stats = estimate_task_stats(cfg, policy, k, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, path)
        logging.info(colored(f"[H4] task {k}: saved -> {path}", "green"))
        del stats

    files = [load_stats(task_stats_path(cfg, k)) for k in range(cfg.num_tasks)]
    names = sorted(files[0]["fisher"].keys())
    lambdas = parse_floats(cfg.lambdas)
    ps = parse_floats(cfg.top_p)

    # ── [B1] 태스크 쌍마다: 중요 부공간이 얼마나 겹치는가 ────────────────────
    logging.info(colored("[H4][B1] pairwise Fisher overlap", "cyan", attrs=["bold"]))
    for i in range(cfg.num_tasks):
        for j in range(cfg.num_tasks):
            if i == j:
                continue
            res, _ = analyze_pair(cfg, [files[i]], files[j], names, lambdas, ps, device)
            emit({"kind": "pair", "old": i, "new": j, **res})
            pi = ps.index(0.01) if 0.01 in ps else 0
            logging.info(
                f"[H4] pair {i}->{j}  cos={res['cosine']:.3f}  BC={res['bhattacharyya']:.3f}  "
                f"lift@{ps[pi]:g}={res['mask_lift'][pi]:.1f}x  "
                f"pareto_gain={res['pareto_gain']:.3f}"
            )

    # ── [B2] 순차 학습 그대로: F_old = Σ_{j<k} F_j 대 F_k ────────────────────
    logging.info(colored("[H4][B2] sequential stages (accumulated old Fisher vs new task)",
                         "cyan", attrs=["bold"]))
    for k in range(1, cfg.num_tasks):
        res, layers = analyze_pair(cfg, files[:k], files[k], names, lambdas, ps, device)
        emit({"kind": "stage", "stage": k, "old": list(range(k)), "new": k, **res})
        big = sorted(layers.items(), key=lambda kv: -kv[1]["numel"])[: cfg.layer_report]
        for g, r in big:
            emit({"kind": "layer", "stage": k, "group": g, **r})
        pi = ps.index(0.01) if 0.01 in ps else 0
        logging.info(
            f"[H4] stage k={k}  cos={res['cosine']:.3f}  BC={res['bhattacharyya']:.3f}  "
            f"lift@{ps[pi]:g}={res['mask_lift'][pi]:.1f}x  "
            f"blocked_gain@{ps[pi]:g}={res['blocked_gain'][pi]:.3f}  "
            f"pareto_gain={res['pareto_gain']:.3f} (degenerate=0.5) at λ={res['pareto_gain_lambda']}  "
            f"degeneracy={res['degeneracy']:.2f}  log_r_std={res['ratio_log_std']:.2f}"
        )
        if res["pareto_gain_at_grid_edge"]:
            logging.warning(colored(
                f"[H4] stage k={k}: λ*={res['pareto_gain_lambda']} 가 격자 끝이다. "
                f"--lambdas 를 넓혀야 pareto_gain이 진짜 최대가 된다 "
                f"(참고: 1/ratio_gmean ≈ {1.0 / max(res['ratio_gmean'], 1e-30):.3g}).", "yellow"))

    # ── [C] 실측 (선택) ──────────────────────────────────────────────────────
    if cfg.measure_steps > 0:
        logging.info(colored("[H4][C] measured: how much does EWC actually block the update?",
                             "cyan", attrs=["bold"]))
        stages = ([int(x) for x in cfg.measure_stages.split(",") if x.strip()]
                  or list(range(1, cfg.num_tasks)))
        mlams = parse_floats(cfg.measure_lambdas)
        if 0.0 not in mlams:                 # 기준선 Δθ_0가 없으면 비교가 안 된다
            mlams = [0.0] + mlams
        mlams = sorted(mlams)

        for k in stages:
            # E0가 넘겨주는 것과 같은 EWC state를 메모리에서 만든다: 누적 Fisher, mean=1, 앵커 θ*.
            fo = {}
            tot = s = 0.0
            for n in names:
                v = files[0]["fisher"][n].clone()
                for f in files[1:k]:
                    v += f["fisher"][n]
                fo[n] = v
                s += float(v.double().sum())
                tot += v.numel()
            scale = max(s / tot, 1e-30)
            fo = {n: (v / scale) for n, v in fo.items()}
            gen = torch.Generator().manual_seed(cfg.seed if cfg.seed is not None else 0)
            rate = min(1.0, cfg.subsample / max(tot, 1))
            tau = global_thresholds(
                torch.cat([
                    v[torch.randint(0, v.numel(), (max(1, int(round(v.numel() * rate))),), generator=gen)]
                    for v in fo.values()
                ]),
                [cfg.measure_top_p],
            )[0]
            # 저장된 Fisher는 평탄화돼 있다(분석이 그쪽이 편해서). 페널티는 파라미터
            # 원래 shape이어야 하므로 여기서만 되돌린다 — 안 그러면 브로드캐스트가
            # 조용히 어긋난 채로 곱해진다.
            shapes = {n: p.shape for n, p in policy.named_parameters() if p.requires_grad}
            ewc_state = {"fisher": {n: v.view(shapes[n]).to(device) for n, v in fo.items()},
                         "anchor": {n: anchor[n] for n in fo}}

            dsk = open_task_dataset(cfg, k)
            train_eps, _ = split_episodes(f"{cfg.dataset_prefix}{k}", None, cfg.holdout_episodes)

            base = None
            for lam in mlams:
                logging.info(f"[H4] measuring k={k} λ={lam:g} ({cfg.measure_steps} steps)")
                run = short_run_delta(cfg, policy, anchor, dsk, train_eps, ewc_state, lam, device)
                if lam == 0.0:
                    base = run["delta"]
                    emit({"kind": "measured", "stage": k, "lambda": 0.0, "steps": cfg.measure_steps,
                          "top_p": cfg.measure_top_p, "mse": run["mse"], "penalty": run["penalty"],
                          "shrink_all": 1.0, "shrink_top": 1.0, "shrink_rest": 1.0,
                          "selectivity": 1.0, "cosine_with_free": 1.0})
                    continue
                cmp = compare_delta(base, run["delta"], fo, tau, device)
                emit({"kind": "measured", "stage": k,
                      "lambda": "inf" if math.isinf(lam) else lam, "steps": cfg.measure_steps,
                      "top_p": cfg.measure_top_p, "mse": run["mse"], "penalty": run["penalty"], **cmp})
                logging.info(f"[H4]   λ={lam:g}  shrink top={cmp['shrink_top']:.3f} "
                             f"rest={cmp['shrink_rest']:.3f}  selectivity={cmp['selectivity']:.2f} "
                             f"cos={cmp['cosine_with_free']:.3f}")
                del run
            del ewc_state, fo, base, dsk

        # 실측이 파라미터를 움직였으니 앵커로 되돌려 둔다.
        with torch.no_grad():
            for n, p in policy.named_parameters():
                if n in anchor:
                    p.copy_(anchor[n])

    logging.info(colored(f"[H4] done -> {results}", "green", attrs=["bold"]))


# ═════════════════════════════════════════════════════════════════════════════
#  그림
# ═════════════════════════════════════════════════════════════════════════════
def plot_h4(results_path: str, out_path: str) -> None:
    rows = [json.loads(x) for x in Path(results_path).read_text().splitlines() if x.strip()]
    if not rows:
        raise SystemExit(f"no rows in {results_path}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    stages = [r for r in rows if r["kind"] == "stage"]
    pairs = [r for r in rows if r["kind"] == "pair"]
    layers = [r for r in rows if r["kind"] == "layer"]
    measured = [r for r in rows if r["kind"] == "measured"]

    # ── 요약 CSV: 한 줄에 스테이지 하나 ──────────────────────────────────────
    csv_path = str(Path(out_path).with_suffix(".csv"))
    keys = ["stage", "cosine", "bhattacharyya", "pareto_gain", "degeneracy", "pareto_gain_lambda",
            "pareto_gain_at_grid_edge", "ratio_log_std", "ratio_gmean",
            "plasticity_at_best", "forgetting_at_best", "gain_free", "damage_free"]
    with open(csv_path, "w") as f:
        f.write(",".join(keys + ["blocked_gain@1%", "mask_lift@1%"]) + "\n")
        for r in sorted(stages, key=lambda r: r["stage"]):
            i = r["top_p"].index(0.01) if 0.01 in r["top_p"] else 0
            f.write(",".join(str(r[k]) for k in keys)
                    + f",{r['blocked_gain'][i]},{r['mask_lift'][i]}\n")
    print(f"saved table  -> {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        print("matplotlib 없음 -> 그림 생략 (pip install matplotlib 후 --plot_only 다시)")
        return

    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    (a, b, c), (d, e, g) = axes

    # (a) 태스크 쌍별 Fisher 겹침 행렬 ────────────────────────────────────────
    tasks = sorted({r["old"] for r in pairs} | {r["new"] for r in pairs})
    if tasks:
        M = np.full((len(tasks), len(tasks)), np.nan)
        for r in pairs:
            M[tasks.index(r["old"]), tasks.index(r["new"])] = r["bhattacharyya"]
        for i in range(len(tasks)):
            M[i, i] = 1.0
        im = a.imshow(M, cmap="magma", vmin=0, vmax=1)
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                       color="w" if M[i, j] < 0.6 else "k", fontsize=10)
        fig.colorbar(im, ax=a, fraction=0.046)
        a.set(xticks=range(len(tasks)), yticks=range(len(tasks)),
              xticklabels=tasks, yticklabels=tasks, xlabel="task j", ylabel="task i",
              title="(a) Fisher overlap  $\\sum_i \\sqrt{p_i q_i}$\n1.0 = identical important-parameter subspace")

    # (b) 상위 p% 마스크 교집합 / chance ──────────────────────────────────────
    # 로그축이라 lift=0(교집합이 비어 있음)은 찍을 수 없다. 그런 점은 빼고 그린다.
    for r in sorted(stages, key=lambda r: r["stage"]):
        pts = [(p, v) for p, v in zip(r["top_p"], r["mask_lift"], strict=True) if v > 0]
        if pts:
            b.plot(*zip(*pts), "-o", ms=4, label=f"k={r['stage']}")
    b.axhline(1.0, color="gray", ls="--", lw=1)
    b.text(0.02, 0.06, "chance (independent subspaces)", transform=b.transAxes,
           color="gray", fontsize=9)
    b.set(xscale="log", yscale="log", xlabel="top-p fraction of old-task Fisher",
          ylabel="overlap / chance  (lift)",
          title="(b) do the top-Fisher coordinates coincide?")
    b.grid(alpha=0.3, which="both")
    if b.get_legend_handles_labels()[0]:
        b.legend(fontsize=9, title="stage")

    # (c) ★ λ 파레토 ─────────────────────────────────────────────────────────
    f_ref = np.linspace(0, 1, 200)
    c.plot(f_ref, 2 * np.sqrt(f_ref) - f_ref, "k--", lw=1.5,
           label="degenerate ($F_{old}\\propto F_{new}$)")
    c.plot([0, 0, 1], [0, 1, 1], color="gray", ls=":", lw=1.5, label="ideal (disjoint)")
    ordered = sorted(stages, key=lambda r: r["stage"])
    for r in ordered:
        c.plot(r["forgetting"], r["plasticity"], "-o", ms=5, label=f"k={r['stage']}")
    # λ 눈금은 마지막 스테이지 곡선 하나에만 단다. 전부 달면 (0,0) 근처에서 겹쳐 못 읽는다.
    if ordered:
        r = ordered[-1]
        for lam, x, y in zip(r["lambdas"], r["forgetting"], r["plasticity"], strict=True):
            if lam in (0.1, 1, 10, 100, 1000) and y > 0.08:
                c.annotate(f"λ={lam:g}", (x, y), fontsize=7.5, xytext=(5, -10),
                           textcoords="offset points", color="0.25")
    c.set(xlabel="forgetting  (old-task damage, rel. to λ=0)",
          ylabel="plasticity  (new-task gain, rel. to λ=0)", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03),
          title="(c) stability-plasticity Pareto of $\\lambda$\n"
                "on the dashed line = degenerate; toward the top-left corner = separable")
    c.grid(alpha=0.3)
    c.legend(fontsize=8, loc="lower right")

    # (d) 새 태스크 update 중 옛 태스크 상위 좌표에 갇힌 몫 ────────────────────
    for r in sorted(stages, key=lambda r: r["stage"]):
        d.plot(r["top_p"], r["blocked_gain"], "-o", ms=4, label=f"k={r['stage']}")
    d.plot(f_ref, f_ref, color="gray", ls="--", lw=1)
    d.text(0.35, 0.22, "chance", transform=d.transAxes, color="gray", fontsize=9, rotation=32)
    d.set(xscale="log", xlabel="top-p fraction of old-task Fisher", ylim=(0, 1.02),
          ylabel="fraction of new-task learnable gain inside",
          title="(d) how much of the new task lives on protected weights")
    d.grid(alpha=0.3, which="both")
    if d.get_legend_handles_labels()[0]:
        d.legend(fontsize=9, title="stage")

    # λ은 0과 inf를 포함하므로 로그축에 올리려면 양 끝을 유한한 자리로 옮겨 찍는다.
    def lam_x(v):
        if v == "inf" or (isinstance(v, float) and math.isinf(v)):
            return 1e5
        return 1e-3 if float(v) == 0.0 else float(v)

    # (e) λ 하나를 따라가며 본 plasticity / forgetting ─────────────────────────
    for r in sorted(stages, key=lambda r: r["stage"]):
        xs = [lam_x(x) for x in r["lambdas"]]
        (ln,) = e.plot(xs, r["plasticity"], "-o", ms=4, label=f"k={r['stage']} plasticity")
        e.plot(xs, r["forgetting"], "--s", ms=4, color=ln.get_color(),
               label=f"k={r['stage']} forgetting")
    e.set(xscale="log", xlabel="EWC lambda  (1e-3 = 0, 1e5 = inf)", ylabel="relative",
          ylim=(-0.03, 1.03),
          title="(e) plasticity and forgetting vs $\\lambda$\n"
                "is there a $\\lambda$ where solid stays high while dashed is already low?")
    e.grid(alpha=0.3, which="both")
    e.legend(fontsize=7, ncol=2)

    # (f) 레이어별 파레토 여유 (또는 실측이 있으면 실측) ──────────────────────
    if measured:
        for st in sorted({r["stage"] for r in measured}):
            sub = sorted([r for r in measured if r["stage"] == st], key=lambda r: lam_x(r["lambda"]))
            xs = [lam_x(r["lambda"]) for r in sub]
            (ln,) = g.plot(xs, [r["shrink_top"] for r in sub], "-o", ms=4,
                           label=f"k={st} top-{sub[0]['top_p']:.0%} Fisher")
            g.plot(xs, [r["shrink_rest"] for r in sub], "--s", ms=4, color=ln.get_color(),
                   label=f"k={st} rest")
        g.set(xscale="log", xlabel="EWC lambda", ylabel="$\\|\\Delta\\theta_\\lambda\\|/\\|\\Delta\\theta_0\\|$",
              ylim=(-0.03, 1.10),
              title="(f) measured: how much of the free update survives\n"
                    "solid = protected coords, dashed = rest; equal = indiscriminate")
        g.grid(alpha=0.3, which="both")
        g.legend(fontsize=7, ncol=2)
    elif layers:
        st = max(r["stage"] for r in layers)
        sub = sorted([r for r in layers if r["stage"] == st], key=lambda r: r["pareto_gain"])
        names = [r["group"].replace("model.", "") for r in sub]
        g.barh(range(len(sub)), [r["pareto_gain"] for r in sub], color="#4c72b0")
        g.axvline(0.5, color="crimson", ls="--", lw=1.5)
        g.text(0.5, len(sub) - 0.4, " degenerate", color="crimson", fontsize=9, va="top")
        g.set(yticks=range(len(sub)),
              xlabel="pareto gain  $\\max_\\lambda$(plasticity - forgetting)",
              xlim=(0, 1), title=f"(f) per-layer, stage k={st}\n0.5 = fully conflicted, 1.0 = separable")
        g.set_yticklabels(names, fontsize=7)
        g.grid(alpha=0.3, axis="x")

    gains = [r["pareto_gain"] for r in stages]
    if gains:
        edge = " [lambda grid edge hit — widen --lambdas]" if any(
            r.get("pareto_gain_at_grid_edge") for r in stages) else ""
        head = (f"H4 cross-task Fisher conflict — pareto gain {min(gains):.2f}..{max(gains):.2f}"
                f"  (0.50 = degenerate, 1.00 = separable){edge}")
    else:
        head = "H4 cross-task Fisher conflict"
    fig.suptitle(head, fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out_path}")


if __name__ == "__main__":
    if "--plot_only" in sys.argv:
        kv = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
        init_logging()
        plot_h4(kv.get("results", "outputs/H4/h4_results.jsonl"),
                kv.get("out", "outputs/H4/H4_fisher_conflict.png"))
    else:
        mp.set_start_method("spawn", force=True)
        init_logging()
        main()
