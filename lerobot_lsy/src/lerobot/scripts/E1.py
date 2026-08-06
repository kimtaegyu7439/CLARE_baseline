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

"""E1 — 저장된 Fisher가 실제 손실 증가를 예측하는가 (EWC 페널티의 보정 검사).

가설
    EWC가 task 0,1,2를 지나며 들고 온 누적 Fisher F = F₀+F₁+F₂ 는 각 태스크를 끝낸
    "그 시점의" 파라미터에서 측정된 값이다. 그런데 앵커는 계속 이동했으므로
    F₀, F₁이 재던 loss landscape은 이미 그 자리에 없다. 그렇다면 EWC가 "이만큼
    나빠질 것"이라고 말하는 값은 실제 손상과 어긋나 있을 것이다.

방법 — 학습을 전혀 하지 않는다. 파라미터를 손으로 흔들어 보고 두 숫자를 비교한다.
    1) 단위 노름 방향 u 를 N개 뽑는다 (기본 100개, 가우시안 정규화).
    2) 상대 거리 r ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.3}.
    3) θ = θ* + r‖θ*‖u   ← 텐서 덧셈이 전부다. 옵티마이저도 데이터도 안 쓴다.
    4) 두 값을 잰다.
         예측  Ω  = ½ Σ_i F_i (θ_i − θ*_i)²        EWC가 말하는 손상
         실제  ΔL = L_old(θ) − L_old(θ*)           held-out에서 실제로 나빠진 양
    5) (Ω, ΔL)을 각 축의 최댓값으로 스케일해 산점도를 그린다.
         y=x 위에 몰림   → landscape이 바뀌어도 Fisher가 예측을 해낸다
         구름처럼 퍼짐   → 저장된 Fisher 정보가 불확실하다 (가설 지지)

Ω를 계산하는 요령
    Ω(u,r) = ½ r²‖θ*‖² Σ_i F_i u_i²  이므로, 방향마다 q = Σ F_i u_i² 를 한 번만 구하면
    모든 r에 대한 Ω가 공짜로 나온다. 실제로 파라미터를 흔들 필요는 ΔL 쪽에만 있다.

★ ΔL은 반드시 짝지은(paired) 차이여야 한다
    flow matching 손실은 매 호출 노이즈 ε와 시각 t를 새로 뽑는다. 그냥 재면 r=0.01에서의
    ΔL이 미니배치 잡음에 통째로 묻힌다. 그래서
      · held-out 배치를 미리 고정해 GPU에 올려 두고 (매번 같은 배치)
      · 평가 직전마다 torch.manual_seed로 ε와 t까지 고정한다
    그러면 ΔL이 θ만의 함수가 되어 작은 r에서도 의미가 있다.

축 스케일에 대하여
    E0.build_ewc_state가 F를 mean(F)=1로 정규화해 저장하므로 Ω의 절대 크기에는 의미가
    없다. 그래서 "각 축의 max로 스케일"하는 것이 맞고, 이 그림이 묻는 것은 절대 보정이
    아니라 **순위/모양이 보존되는가**이다.

전제
    E0가 만든 체크포인트와 ewc_state.pt가 있어야 한다 (--cl_root, --stage).
    시뮬레이터(gym_libero)는 필요 없다.

사용 예
    python E1.py --cl_root=outputs/E0/.../lam100 --stage=2 --policy.path=<pretrain> ...
    python E1.py --plot_only --results=... --out=...
"""

import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import torch
import torch.multiprocessing as mp
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import format_big_number, get_safe_torch_device, init_logging

# 데이터 분할/샘플러 규약은 E0·H4·H5와 동일해야 한다. 복사하지 않고 가져온다.
from lerobot.scripts.H4 import episode_sampler, open_task_dataset, split_episodes, to_device


@dataclass
class E1Config(TrainPipelineConfig):
    """train.py 인자 전부 + 섭동 실험용 인자."""

    # ── 어디서 θ*와 저장된 Fisher를 가져오는가 ───────────────────────────────
    cl_root: str = ""              # 예: outputs/E0/libero_spatial/seed_42/lam100
    stage: int = 2                 # task_{stage} 를 끝낸 시점. F = F₀+..+F_stage
    dataset_prefix: str = "continuallearning/libero_spatial_image_task_"
    holdout_episodes: int = 5      # E0와 같은 분할

    # ── 섭동 ─────────────────────────────────────────────────────────────────
    n_directions: int = 100
    radii: str = "0.01,0.02,0.05,0.1,0.2,0.3"
    direction_seed: int = 777

    # ── ΔL 측정 (고정 배치 + 고정 노이즈) ────────────────────────────────────
    eval_batches: int = 6          # 태스크당 캐시할 배치 수
    eval_batch_size: int = 16
    eval_seed: int = 12345         # ε와 t를 고정하는 시드

    # ── 출력 ─────────────────────────────────────────────────────────────────
    run_tag: str = ""
    results_path: str = "outputs/E1/e1_results.jsonl"

    def validate(self):
        """E1은 학습을 하지 않고 기존 산출물만 읽으므로 output_dir 존재 검사를 우회한다."""
        out = self.output_dir
        if isinstance(out, Path) and out.is_dir():
            self.output_dir = None
            super().validate()
            self.output_dir = out
        else:
            super().validate()


# ═════════════════════════════════════════════════════════════════════════════
#  θ*와 저장된 Fisher
# ═════════════════════════════════════════════════════════════════════════════
def stage_ckpt(cfg: E1Config) -> Path:
    return Path(cfg.cl_root) / f"task_{cfg.stage}" / "checkpoints" / "last" / "pretrained_model"


def stage_ewc_state(cfg: E1Config) -> Path:
    return Path(cfg.cl_root) / f"task_{cfg.stage}" / "ewc_state.pt"


def load_policy_at(cfg: E1Config, ckpt: Path, ds_meta):
    if not ckpt.exists():
        raise FileNotFoundError(
            f"체크포인트가 없다: {ckpt}\n  --cl_root / --stage 를 확인해라 "
            f"(예: --cl_root=outputs/E0/libero_spatial/seed_42/lam100 --stage=2)."
        )
    pcfg = PreTrainedConfig.from_pretrained(ckpt)
    pcfg.pretrained_path = ckpt
    pcfg.device = cfg.policy.device
    return make_policy(cfg=pcfg, ds_meta=ds_meta)


# ═════════════════════════════════════════════════════════════════════════════
#  고정 배치 캐시 + 짝지은 손실 측정
# ═════════════════════════════════════════════════════════════════════════════
def cache_batches(cfg: E1Config, tasks: list[int], device) -> dict[int, list[dict]]:
    """held-out 배치를 태스크마다 eval_batches개씩 GPU에 올려 둔다.

    600번(=방향×거리)의 평가마다 데이터를 다시 읽으면 그쪽이 병목이 된다.
    또 매번 **같은 배치**여야 ΔL이 짝지은 차이가 된다.
    """
    cached = {}
    for t in tasks:
        ds = open_task_dataset(cfg, t)
        _, holdout = split_episodes(f"{cfg.dataset_prefix}{t}", None, cfg.holdout_episodes)
        torch.manual_seed(cfg.eval_seed)          # 어느 배치를 고를지도 고정
        loader = torch.utils.data.DataLoader(
            ds,
            num_workers=0,
            batch_size=cfg.eval_batch_size,
            sampler=episode_sampler(cfg, ds, holdout),
            pin_memory=False,
            drop_last=True,
        )
        it = iter(loader)
        batches = []
        for _ in range(cfg.eval_batches):
            batches.append(to_device(next(it), device))
        cached[t] = batches
        logging.info(f"[E1] task {t}: cached {len(batches)} held-out batches on {device}")
        del ds, loader, it
    return cached


@torch.no_grad()
def eval_loss(policy, cached: dict[int, list[dict]], seed: int) -> dict[int, float]:
    """태스크별 held-out 손실. 배치도 ε/t도 고정이라 θ만의 함수다."""
    policy.eval()
    out = {}
    for t, batches in cached.items():
        torch.manual_seed(seed)                   # ★ ε와 t를 θ 사이에 맞춘다
        tot = 0.0
        for b in batches:
            tot += float(policy.forward(b)[0])
        out[t] = tot / len(batches)
    policy.train()
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  방향 뽑기 / 섭동
# ═════════════════════════════════════════════════════════════════════════════
def sample_direction(shapes: dict, gen: torch.Generator, device) -> dict:
    """단위 노름 가우시안 방향 u (학습 가능한 파라미터 전체에 대해 ‖u‖=1)."""
    u = {n: torch.randn(s, generator=gen, device=device) for n, s in shapes.items()}
    nrm = math.sqrt(sum(float(v.double().pow(2).sum()) for v in u.values()))
    for n in u:
        u[n] /= nrm
    return u


def apply_perturbation(policy, anchor: dict, u: dict, scale: float) -> None:
    """θ ← θ* + scale·u  (제자리)."""
    with torch.no_grad():
        for n, p in policy.named_parameters():
            if n in anchor:
                p.copy_(anchor[n] + scale * u[n])


def restore(policy, anchor: dict) -> None:
    with torch.no_grad():
        for n, p in policy.named_parameters():
            if n in anchor:
                p.copy_(anchor[n])


# ═════════════════════════════════════════════════════════════════════════════
#  메인 (train.py / E0 / H4 / H5 와 같은 [1]~ 순서)
# ═════════════════════════════════════════════════════════════════════════════
@parser.wrap()
def main(cfg: E1Config):
    # [1] 설정
    cfg.validate()
    cfg.save_checkpoint = False
    if not cfg.cl_root:
        raise SystemExit(
            "--cl_root 가 필요하다 (예: outputs/E0/libero_spatial/seed_42/lam100). "
            "E1은 E0가 만든 θ*와 ewc_state.pt를 읽는 실험이다."
        )
    logging.info(pformat(cfg.to_dict()))
    radii = [float(x) for x in cfg.radii.split(",") if x.strip()]
    tasks = list(range(cfg.stage + 1))
    logging.info(colored(
        f"[E1] anchor=task_{cfg.stage} of {cfg.cl_root}  tasks={tasks}  "
        f"{cfg.n_directions} directions x {len(radii)} radii", "green", attrs=["bold"]))

    # [3] 재현성
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # [4] 디바이스
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # [5] 데이터셋 (정책 생성용 메타)
    logging.info("Creating dataset")
    dataset0 = make_dataset(cfg)

    # [7] 정책 = θ*
    logging.info("Creating policy at the CL anchor")
    policy = load_policy_at(cfg, stage_ckpt(cfg), dataset0.meta)
    anchor = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
    shapes = {n: tuple(v.shape) for n, v in anchor.items()}
    theta_norm = math.sqrt(sum(float(v.double().pow(2).sum()) for v in anchor.values()))
    n_train = sum(v.numel() for v in anchor.values())
    logging.info(f"[E1] trainable={format_big_number(n_train)}  ‖θ*‖={theta_norm:.4f}")

    # [7b] 저장된 Fisher
    sp = stage_ewc_state(cfg)
    if not sp.exists():
        raise SystemExit(
            f"ewc_state.pt 가 없다: {sp}\n"
            f"  λ=0 팔은 EWC를 안 써서 Fisher를 저장하지 않는다. λ>0 팔을 지정해라."
        )
    fisher = torch.load(sp, map_location="cpu", weights_only=False)["fisher"]
    stored_anchor = torch.load(sp, map_location="cpu", weights_only=False).get("anchor", {})
    fisher = {n: v.to(device) for n, v in fisher.items()}
    missing = [n for n in anchor if n not in fisher]
    if missing:
        raise SystemExit(f"저장된 Fisher에 없는 파라미터 {len(missing)}개 (이름 불일치)")
    # 저장된 앵커와 체크포인트가 같은 지점인지 확인. 다르면 Ω의 기준점이 어긋난 것이다.
    if stored_anchor:
        md = max(float((anchor[n] - stored_anchor[n].to(device)).abs().max())
                 for n in anchor if n in stored_anchor)
        logging.info(f"[E1] stored anchor vs checkpoint: max|diff|={md:.3e}"
                     + ("  (일치)" if md < 1e-6 else "  ★ 어긋남 — Ω의 기준점을 확인해라"))

    # [8] held-out 배치 캐시 + 기준 손실 L_old(θ*)
    cached = cache_batches(cfg, tasks, device)
    base = eval_loss(policy, cached, cfg.eval_seed)
    base_sum = sum(base.values())
    logging.info(f"[E1] baseline L(θ*): " + "  ".join(f"task{t}={v:.5f}" for t, v in base.items())
                 + f"  | sum={base_sum:.5f}")

    results = Path(cfg.results_path)
    results.parent.mkdir(parents=True, exist_ok=True)
    tag = cfg.run_tag or f"stage{cfg.stage}"
    fh = results.open("a")

    # [9] 방향 × 거리
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.direction_seed)
    t0 = time.perf_counter()
    for d in range(cfg.n_directions):
        u = sample_direction(shapes, gen, device)
        # Ω(u,r) = ½ r²‖θ*‖² Σ F_i u_i²  — 방향마다 q를 한 번만 구하면 모든 r이 나온다.
        q = sum(float((fisher[n].double() * u[n].double().pow(2)).sum()) for n in u)

        for r in radii:
            scale = r * theta_norm
            omega = 0.5 * (scale ** 2) * q
            apply_perturbation(policy, anchor, u, scale)
            cur = eval_loss(policy, cached, cfg.eval_seed)
            row = {
                "run_tag": tag, "seed": cfg.seed, "cl_root": str(cfg.cl_root),
                "stage": cfg.stage, "direction": d, "radius": r,
                "omega": omega,                                   # 예측 (½ΣF δ²)
                "delta_L_sum": sum(cur.values()) - base_sum,      # 실제 (합계)
                "delta_L": {str(t): cur[t] - base[t] for t in cur},   # 실제 (태스크별)
                "loss": {str(t): cur[t] for t in cur},
                "base_loss": {str(t): base[t] for t in base},
                "theta_norm": theta_norm, "fisher_quad": q,
            }
            fh.write(json.dumps(row) + "\n")
        fh.flush()
        restore(policy, anchor)
        del u
        if (d + 1) % 10 == 0:
            el = time.perf_counter() - t0
            logging.info(f"[E1] {d + 1}/{cfg.n_directions} directions  "
                         f"({el:.0f}s, ETA {el / (d + 1) * (cfg.n_directions - d - 1):.0f}s)")

    fh.close()
    logging.info(colored(f"[E1] done -> {results}", "green", attrs=["bold"]))


# ═════════════════════════════════════════════════════════════════════════════
#  그림
# ═════════════════════════════════════════════════════════════════════════════
def _spearman(x, y):
    """순위 상관. scipy 없이 numpy만으로."""
    import numpy as np

    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float)
        r[order] = np.arange(len(a), dtype=float)
        return r

    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def plot_e1(results_path: str, out_path: str) -> None:
    rows = [json.loads(x) for x in Path(results_path).read_text().splitlines() if x.strip()]
    if not rows:
        raise SystemExit(f"no rows in {results_path}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    radii = sorted({r["radius"] for r in rows})
    tasks = sorted({int(t) for r in rows for t in r["delta_L"]})

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = str(Path(out_path).with_suffix(".csv"))
    with open(csv_path, "w") as f:
        cols = ["direction", "radius", "omega", "delta_L_sum"] + [f"delta_L_task{t}" for t in tasks]
        f.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda r: (r["radius"], r["direction"])):
            f.write(",".join([str(r["direction"]), str(r["radius"]), str(r["omega"]),
                              str(r["delta_L_sum"])]
                             + [str(r["delta_L"][str(t)]) for t in tasks]) + "\n")
    print(f"saved table  -> {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        print("matplotlib 없음 -> 그림 생략 (pip install matplotlib 후 --plot_only 다시)")
        return

    cmap = plt.get_cmap("viridis")
    col = {r: cmap(i / max(len(radii) - 1, 1)) for i, r in enumerate(radii)}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 12))
    (a, b), (c, d) = axes

    # (a) ★ 본체: 예측 vs 실제 (각 축 max로 스케일) ───────────────────────────
    ox = np.array([r["omega"] for r in rows], float)
    oy = np.array([r["delta_L_sum"] for r in rows], float)
    mx, my = ox.max(), oy.max()
    for rad in radii:
        m = np.array([r["radius"] == rad for r in rows])
        a.scatter(ox[m] / mx, oy[m] / my, s=16, alpha=0.65, color=col[rad], label=f"r={rad:g}")
    a.plot([0, 1], [0, 1], "k--", lw=1.5, label="y = x")
    # ★ y축을 0에서 자르면 안 된다. Ω는 정의상 ≥0인데 ΔL은 음수가 나올 수 있고
    #   (θ*가 옛 태스크 손실의 극소점이 아니면 1차항 gᵀδ가 살아 있어 손실이 줄기도 한다),
    #   그 음수야말로 "2차 근사가 성립하지 않는다"는 직접 증거다.
    lo = min(-0.03, float((oy / my).min()) * 1.05)
    a.axhline(0, color="0.5", lw=0.8)
    a.set(xlabel="predicted  $\\Omega=\\frac{1}{2}\\sum_i F_i(\\theta_i-\\theta_i^*)^2$   (/max)",
          ylabel="actual  $\\Delta L_{old}$   (/max)",
          xlim=(-0.03, 1.03), ylim=(lo, 1.03),
          title="(a) does the stored Fisher predict the real damage?\n"
                "on y=x -> yes;  cloud -> stored Fisher is unreliable")
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc="upper left")
    rho_all = _spearman(ox, oy)
    a.text(0.97, 0.03, f"Spearman (all) = {rho_all:.3f}", transform=a.transAxes,
           ha="right", fontsize=10, bbox=dict(fc="w", alpha=0.8, ec="0.7"))

    # (b) 거리마다 따로 — r이 커질수록 2차 근사가 깨지는지 ────────────────────
    # 반경 안에서만 normalize한다. 반경끼리 크기가 100배 넘게 차이나 섞으면 안 보인다.
    blo = -0.03
    for rad in radii:
        sub = [r for r in rows if r["radius"] == rad]
        x = np.array([r["omega"] for r in sub], float)
        y = np.array([r["delta_L_sum"] for r in sub], float)
        yn = y / max(abs(y).max(), 1e-30)
        blo = min(blo, float(yn.min()) * 1.05)
        b.scatter(x / x.max(), yn, s=14, alpha=0.6, color=col[rad], label=f"r={rad:g}")
    b.plot([0, 1], [0, 1], "k--", lw=1.5)
    b.axhline(0, color="0.5", lw=0.8)
    b.set(xlabel="predicted (/max within radius)", ylabel="actual (/|max| within radius)",
          xlim=(-0.03, 1.03), ylim=(blo, 1.03),
          title="(b) same, normalized inside each radius\n"
                "isolates 'does F rank directions correctly' from 'is the scale right'")
    b.grid(alpha=0.3)
    b.legend(fontsize=8, loc="upper left")

    # (c) 거리별 순위상관 ─────────────────────────────────────────────────────
    rhos = []
    for rad in radii:
        sub = [r for r in rows if r["radius"] == rad]
        rhos.append(_spearman([r["omega"] for r in sub], [r["delta_L_sum"] for r in sub]))
    c.plot(radii, rhos, "-o", ms=7, lw=2, label="sum over old tasks")
    for t in tasks:
        rt = [_spearman([r["omega"] for r in rows if r["radius"] == rad],
                        [r["delta_L"][str(t)] for r in rows if r["radius"] == rad])
              for rad in radii]
        c.plot(radii, rt, "--s", ms=4, alpha=0.75, label=f"task {t}")
    c.axhline(1.0, color="seagreen", ls="--", lw=1)
    c.axhline(0.0, color="crimson", ls="--", lw=1)
    c.set(xscale="log", xlabel="relative perturbation radius r", ylabel="Spearman(Ω, ΔL)",
          ylim=(-0.6, 1.05), xticks=radii, xticklabels=[f"{r:g}" for r in radii],
          title="(c) rank correlation vs distance\n"
                "1 = Fisher orders directions perfectly, 0 = no information")
    c.grid(alpha=0.3, which="both")
    c.legend(fontsize=8)

    # (d) 태스크별 산점도 — 오래된 태스크가 더 어긋나는가 ─────────────────────
    dlo = -0.03
    for t in tasks:
        y = np.array([r["delta_L"][str(t)] for r in rows], float)
        yn = y / max(abs(y).max(), 1e-30)
        dlo = min(dlo, float(yn.min()) * 1.05)
        d.scatter(ox / mx, yn, s=14, alpha=0.5, label=f"task {t}")
    d.plot([0, 1], [0, 1], "k--", lw=1.5)
    d.axhline(0, color="0.5", lw=0.8)
    d.set(xlabel="predicted $\\Omega$ (/max)", ylabel="actual $\\Delta L_{task}$ (/|max|)",
          xlim=(-0.03, 1.03), ylim=(dlo, 1.03),
          title="(d) per task — is the oldest task the worst predicted?\n"
                "(F is a single sum, so all share one prediction)")
    d.grid(alpha=0.3)
    d.legend(fontsize=9)

    n_dir = len({r["direction"] for r in rows})
    # ★ 제목에는 반경 안에서의 상관을 쓴다. 전체(rho_all)는 반경이 커지면 Ω도 ΔL도
    #   같이 커진다는 사실만 반영해서 항상 1에 가깝게 나온다 — F의 품질과 무관하다.
    med = float(np.median([x for x in rhos if not math.isnan(x)])) if rhos else float("nan")
    fig.suptitle(
        f"E1: is EWC's stored Fisher still calibrated after the anchor moved?  "
        f"{n_dir} directions x {len(radii)} radii\n"
        f"within-radius Spearman(Ω, ΔL) median = {med:.3f}   "
        f"(pooled across radii = {rho_all:.3f}, trivially high: larger r -> both grow)",
        fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out_path}")


if __name__ == "__main__":
    if "--plot_only" in sys.argv:
        kv = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
        init_logging()
        plot_e1(kv.get("results", "outputs/E1/e1_results.jsonl"),
                kv.get("out", "outputs/E1/E1_fisher_calibration.png"))
    else:
        mp.set_start_method("spawn", force=True)
        init_logging()
        main()
