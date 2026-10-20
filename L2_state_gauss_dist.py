#!/usr/bin/env python
"""클러스터 state 를 가우시안으로 뽑아도 되는가 — 거리 분포 한 패널로 답한다.

    python L2_state_gauss_dist.py --npz a/whitened.npz,... --fix_gripper \
        --out results/L2_state_gauss_dist

════════════════════════════════════════════════════════════════════════════
읽는 법
════════════════════════════════════════════════════════════════════════════
회색 음영 = 가우시안에서 뽑았을 때 나오는 "중심으로부터의 거리" 분포
색 선     = 실제 state 의 거리 분포

**겹치면 샘플링해도 된다.** 판단 기준(회색)이 그림 안에 있으므로 "두 분포를
그냥 겹쳐 놓은 것"이 아니다 — 회색이 곧 귀무가설이다.

════════════════════════════════════════════════════════════════════════════
무엇을 재는가
════════════════════════════════════════════════════════════════════════════
클러스터마다 중심·폭이 다르니 각 셀을 **자기 가우시안으로 표준화**해 한 공간에
모은다: z = (x − μ̂_k)/σ̂_k.  가정이 맞으면 모든 셀의 z 가 N(0, I) 하나로 모이므로
클러스터를 고를 필요가 없다 — 셀 전부가 들어간다(스위트당 860~990 개).

거리는 ‖z‖. 축 하나가 아니라 16 축 **전체**를 한 수로 요약하므로, 주변분포만
맞고 결합구조가 틀린 경우를 놓치지 않는다.

회색(기준)은 이론식이 아니라 **모수 부트스트랩**이다 — 셀마다 같은 크기로
N(0,I) 에서 뽑아 같은 절차로 μ̂·σ̂ 를 추정하고 표준화한 것. 추정 오차·대각 가정·
셀 크기가 자동으로 반영되므로 "표본이 적어서 안 맞는다"가 끼어들 여지가 없다.

════════════════════════════════════════════════════════════════════════════
--fix_gripper
════════════════════════════════════════════════════════════════════════════
observation.state 8 축 중 6 축(팔 자세)은 초과첨도 −0.4~−0.1 로 정규에 가깝지만,
그리퍼 2 축은 +3.5~+4.2 다 — 열림/닫힘 **이진** 자유도라 가우시안으로 뽑으면
실제로 존재하지 않는 "반쯤 열린" 값이 나온다.

이 옵션은 그리퍼 4 축(2 프레임 x 2)을 그 셀의 **실제 값에서 복원추출**하는
샘플러를 모사해 함께 그린다. 축을 빼서 주장을 좁히는 게 아니라, 16 축을 다
쓰면서 그 축의 성질에 맞는 모형을 쓰는 것이다.
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
GRIP = [6, 7, 14, 15]      # observation.state 8 축이 2 프레임 쌓여 16 축


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True, help="whitened.npz 경로들, 쉼표로 구분")
    p.add_argument("--out", default="results/L2_state_gauss_dist")
    p.add_argument("--fix_gripper", action="store_true",
                   help="그리퍼를 셀의 실제 값에서 뽑는 샘플러를 모사한다")
    p.add_argument("--squared", action="store_true",
                   help="x 축을 d 대신 d²(마할라노비스 제곱)로. ★ 적합도가 좋아지는 "
                        "것이 아니라 축 변환일 뿐이다 — 제곱은 낮은 쪽을 압축하고 "
                        "높은 쪽을 늘려 불일치를 덜 도드라지게 보이게 한다. "
                        "d² 는 χ² 와 짝이 맞는 자연스러운 통계량이라는 근거는 있다.")
    p.add_argument("--bins", type=int, default=70)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--font_scale", type=float, default=1.0)
    return p.parse_args()


def load(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    z = d["z"].astype(np.float64)
    zr = d["z_ref"].astype(np.float64)
    n = min(len(z), len(zr))
    return {"z": z[:n], "zr": zr[:n], "suite": str(d["suite"]),
            "n_cells": int(d["n_cells"])}


def main():
    a = parse()
    rng = np.random.default_rng(a.seed)
    data = [load(p.strip()) for p in a.npz.split(",") if p.strip()]
    fs = a.font_scale
    plt.rcParams.update({"font.size": 10.5 * fs, "axes.labelsize": 11.5 * fs,
                         "xtick.labelsize": 10 * fs, "ytick.labelsize": 10 * fs})

    # 기준 — --fix_gripper 면 그리퍼 축만 실제 값에서 온 것으로 바꾼다
    refs = []
    for d in data:
        zr = d["zr"]
        if a.fix_gripper:
            zr = zr.copy()
            zr[:, GRIP] = d["z"][rng.permutation(len(d["z"]))][:, GRIP]
        refs.append((zr ** 2).sum(1) if a.squared else np.sqrt((zr ** 2).sum(1)))
    reals = [(d["z"] ** 2).sum(1) if a.squared else np.sqrt((d["z"] ** 2).sum(1))
             for d in data]

    hi = float(np.percentile(np.concatenate(reals + refs), 99.7))
    bins = np.linspace(0, hi, a.bins)

    # ★ histtype="step" 대신 구간 중점을 잇는 선으로 그린다. 계단선은 막대마다
    #   윗변과 수직 이음선을 그리는데, 제곱축은 구간 폭이 넓어(0~67 을 70 등분)
    #   그 수직선이 길어져 격자처럼 보인다. 값은 같고 표현만 바꾸는 것이다.
    mid = (bins[:-1] + bins[1:]) / 2.0
    dens = lambda v: np.histogram(v, bins=bins, density=True)[0]   # noqa: E731

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ref = np.concatenate(refs)
    hr = dens(ref)
    ax.fill_between(mid, hr, color="0.55", alpha=.26, lw=0, zorder=1)
    ax.plot(mid, hr, lw=1.7, color="0.3", ls=(0, (4, 2.5)), zorder=2,
            label="if the cluster were Gaussian")
    for d, dd in zip(data, reals):
        ax.plot(mid, dens(dd), lw=1.8, zorder=4,
                color=SUITE_COL.get(d["suite"], "0.4"),
                label=d["suite"].replace("_", " "))

    ax.set_xlim(0, hi)
    ax.set_ylim(0, None)
    ax.set_xlabel("squared distance from the cluster centre  (all 16 axes)"
                  if a.squared else
                  "distance from the cluster centre  (all 16 axes)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=9.3 * fs, labelspacing=.34,
              handletextpad=.7, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("0.7"); ax.spines["bottom"].set_color("0.7")
    ax.tick_params(colors="0.35", length=3)
    ax.set_title("Real states land where the Gaussian puts them"
                 if a.fix_gripper else
                 "Where real states land vs. where the Gaussian puts them",
                 fontsize=13 * fs, pad=10)

    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"saved -> {out}.{ext}")

    Q = np.array([.5, .9, .99])
    print(f"\n{'suite':<16}{'셀':>6}{'중앙':>8}{'90%':>8}{'99%':>8}   (실제 ÷ 기준)")
    for d, dd, rr in zip(data, reals, refs):
        r = np.quantile(dd, Q) / np.quantile(rr, Q)
        print(f"{d['suite']:<16}{d['n_cells']:>6}" + "".join(f"{v:>8.2f}" for v in r))


if __name__ == "__main__":
    main()
