#!/usr/bin/env python
"""클러스터 state 를 가우시안으로 뽑아도 되는가 — 실제 state 와 샘플을 **겹쳐** 보인다.

    python L2_state_gauss_overlay.py \
        --npz results/L2_state_gauss_calib_libero_spatial_frame_k70/whitened.npz,... \
        --out results/L2_state_gauss_overlay

════════════════════════════════════════════════════════════════════════════
왜 클러스터를 고르지 않아도 되는가
════════════════════════════════════════════════════════════════════════════
클러스터마다 중심 μ_k 와 폭 σ_k 가 다르니 원좌표에서 겹쳐 그리면 의미가 없다.
그래서 각 클러스터를 **자기 가우시안으로 표준화**해서 한 공간으로 옮긴다.

    z = (x − μ̂_k) / σ̂_k          (μ̂, σ̂ 는 그 클러스터 프레임으로 추정)

가우시안 가정이 맞다면 **모든 클러스터의 z 가 N(0, I) 하나로 모인다.** 그래서
클러스터를 고를 필요가 없고, 스위트의 셀 전부(500개 안팎)가 한 그림에 들어간다.
고르지 않았으므로 고르기에서 오는 편향도 없다.

★ 이 좌표계에서 **우리 샘플러가 뽑는 것은 정확히 N(0, I)** 다. 근사가 아니다:
  sample_codebook_hard 가 s̃ = μ_{s,k} + σ_{s,k}⊙ε 이므로 표준화하면 그대로 ε 이다.
  따라서 "가우시안 샘플" 쪽 점들은 실제로 우리가 앵커로 쓰는 그 분포다.

════════════════════════════════════════════════════════════════════════════
두 패널
════════════════════════════════════════════════════════════════════════════
(a) 산점도 — 16 차원 중 두 축. 눈으로 보는 증거다.
    ★ PCA 축은 쓰지 않는다. 실제 데이터의 최대분산 방향을 고르면 실제 쪽이
      구조적으로 더 넓게 나와 비교가 불공정해진다. 고정 시드의 **무작위 직교
      투영**을 쓴다 — N(0,I) 는 회전 불변이라 어느 방향으로 봐도 같아야 한다.

(b) 마할라노비스 거리 — 16 차원 **전부**를 쓰는 요약이다. 산점도는 두 축만
    보여주므로 나머지 14 축이 어긋나도 안 보인다. 이 패널이 그걸 막는다.
    기준 분포는 이론식이 아니라 **모수 부트스트랩**이다. 셀마다 같은 크기로
    N(0,I) 에서 뽑아 **같은 절차로** μ̂·σ̂ 를 추정하고 표준화한 것이다
    (L2_state_gauss_calibration.py 의 --dump_whitened 가 함께 저장한다).
    이렇게 하면 추정 오차, 대각 가정, 셀 크기, 적합=평가(전 프레임 모드)까지
    전부 자동으로 반영된다. χ² 나 F 를 쓰면 이 중 일부가 빠져 "안 맞는다"가
    표본 사정만으로 나올 수 있다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

SUITE_COL = {"libero_spatial": "#2ca089", "libero_goal": "#7a5bb5",
             "libero_object": "#d9722c", "libero_10": "#2f7fb5"}
C_REAL = "0.62"


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True, help="whitened.npz 경로들, 쉼표로 구분")
    p.add_argument("--out", default="results/L2_state_gauss_overlay")
    p.add_argument("--seed", type=int, default=0, help="투영·샘플 시드")
    p.add_argument("--max_points", type=int, default=6000,
                   help="산점도에 찍을 점 수(양쪽 같게). 전부 찍으면 뭉개진다")
    p.add_argument("--font_scale", type=float, default=1.0)
    return p.parse_args()


def load(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {"z": d["z"].astype(np.float64), "n": d["n_train"].astype(int),
            "zr": d["z_ref"].astype(np.float64),
            "suite": str(d["suite"]), "dim": int(d["dim"]),
            "n_cells": int(d["n_cells"]), "split": str(d.get("split", "?"))}


def main():
    a = parse()
    rng = np.random.default_rng(a.seed)
    data = [load(p.strip()) for p in a.npz.split(",") if p.strip()]
    dim = data[0]["dim"]
    fs = a.font_scale
    plt.rcParams.update({"font.size": 10 * fs, "axes.labelsize": 10.5 * fs,
                         "xtick.labelsize": 9.5 * fs, "ytick.labelsize": 9.5 * fs,
                         "axes.titlesize": 11 * fs})

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.5),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # ── (a) 산점도: 전 스위트·전 클러스터를 모아 한 번에 ─────────────────────
    Z = np.concatenate([d["z"] for d in data])
    # 고정 시드 무작위 직교 투영. PCA 를 쓰지 않는 이유는 파일 상단 설명 참고.
    Q, _ = np.linalg.qr(np.random.default_rng(a.seed).normal(size=(dim, 2)))
    m = min(a.max_points, len(Z))
    real = Z[rng.choice(len(Z), m, replace=False)] @ Q
    ZR = np.concatenate([d["zr"] for d in data])
    samp = ZR[rng.choice(len(ZR), m, replace=False)] @ Q

    axL.scatter(real[:, 0], real[:, 1], s=4, c=C_REAL, alpha=.30, lw=0,
                label=f"real states ({len(Z):,} frames)", rasterized=True)
    axL.scatter(samp[:, 0], samp[:, 1], s=4, c="#c0392b", alpha=.30, lw=0,
                label="drawn from the cluster's Gaussian", rasterized=True)
    axL.set_aspect("equal")
    lim = float(np.percentile(np.abs(np.concatenate([real, samp])), 99.7))
    axL.set_xlim(-lim, lim); axL.set_ylim(-lim, lim)
    axL.set_xlabel("standardized state (random 2-D view)")
    axL.set_ylabel("standardized state")
    axL.set_title("Every cluster, standardized and pooled", pad=8)
    leg = axL.legend(loc="upper left", frameon=False, fontsize=8.6 * fs,
                     handletextpad=.3, borderpad=.1, labelspacing=.3)
    for h in leg.legend_handles:
        h.set_alpha(.9); h.set_sizes([22])

    # ── (b) 마할라노비스 거리: 16 차원 전부 ─────────────────────────────────
    hi = 0.0
    for d in data:
        dd = (d["z"] ** 2).sum(1)
        hi = max(hi, float(np.percentile(dd, 99)))
    bins = np.linspace(0, hi, 60)
    for d in data:
        dd = (d["z"] ** 2).sum(1)
        col = SUITE_COL.get(d["suite"], "0.4")
        axR.hist(dd, bins=bins, density=True, histtype="step", lw=1.5, color=col,
                 label=f"{d['suite'].replace('_', ' ')}  ({d['n_cells']} clusters)")
    ref = (ZR ** 2).sum(1)
    axR.hist(ref, bins=bins, density=True, histtype="stepfilled", lw=0,
             color="0.5", alpha=.22, zorder=0)
    axR.hist(ref, bins=bins, density=True, histtype="step", lw=1.6, color="0.25",
             ls=(0, (4, 2.5)), label="if the cluster were Gaussian", zorder=6)
    axR.set_xlim(0, hi)
    axR.set_xlabel("distance from the cluster centre (Mahalanobis, all 16 dims)")
    axR.set_ylabel("density")
    axR.set_title("Same in all 16 dimensions", pad=8)
    axR.legend(frameon=False, fontsize=8.6 * fs, handletextpad=.6,
               labelspacing=.35, loc="upper right")
    for ax in (axL, axR):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("0.7"); ax.spines["bottom"].set_color("0.7")
        ax.tick_params(colors="0.35", length=3)

    fig.suptitle("Cluster states are Gaussian enough to sample from", y=.985,
                 fontsize=12.5 * fs)
    fig.tight_layout(rect=(0, 0, 1, .95))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"saved -> {out}.{ext}")

    print(f"\n표준화 점 {len(Z):,}개, 셀 "
          f"{sum(d['n_cells'] for d in data)}개, {dim}차원")
    for d in data:
        dd = (d["z"] ** 2).sum(1)
        r = (d["zr"] ** 2).sum(1)
        print(f"  {d['suite']:<16} 셀 {d['n_cells']:>4}  점 {len(dd):>6,}  "
              f"거리 중앙값 실제 {np.median(dd):6.2f} / 기준 {np.median(r):6.2f}  "
              f"(각 차원 분산 평균 {float(d['z'].var(0).mean()):.3f}, 1.0 이면 정확)")


if __name__ == "__main__":
    main()
