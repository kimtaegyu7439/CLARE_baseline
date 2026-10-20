#!/usr/bin/env python
"""클러스터 state 를 가우시안으로 뽑아도 되는가 — Q-Q 한 패널로 답한다.

    python L2_state_gauss_qq.py --npz a/whitened.npz,b/whitened.npz,... \
        --out results/L2_state_gauss_qq

════════════════════════════════════════════════════════════════════════════
읽는 법
════════════════════════════════════════════════════════════════════════════
    x = 가우시안이라면 나와야 할 거리의 분위수
    y = 실제 state 의 거리의 분위수
    점이 대각선 위에 있으면 = 그 분포가 곧 가우시안이다.

"겹쳐 그린 두 분포"와 달리 **판단 기준(대각선)이 그림 안에 있다.** 벗어난
거리가 곧 오차의 크기이고, 어느 구간에서 벗어나는지까지 읽힌다.

════════════════════════════════════════════════════════════════════════════
무엇을 재는가
════════════════════════════════════════════════════════════════════════════
클러스터마다 중심·폭이 다르니 각 셀을 **자기 가우시안으로 표준화**해 한 공간에
모은다: z = (x − μ̂_k)/σ̂_k.  가정이 맞으면 모든 셀의 z 가 N(0, I) 하나로 모인다.
그래서 클러스터를 고를 필요가 없다 — 셀 전부가 들어간다.

거리는 ‖z‖ (마할라노비스). 축 하나가 아니라 그 묶음 **전체**를 한 수로 요약하므로
주변분포만 맞고 결합구조가 틀린 경우를 놓치지 않는다.

기준(x축)은 이론식이 아니라 **모수 부트스트랩**이다 — 셀마다 같은 크기로 N(0,I)
에서 뽑아 같은 절차로 μ̂·σ̂ 를 추정하고 표준화한 것. 추정 오차·대각 가정·셀 크기가
자동으로 반영된다.

★ 16 축을 **한 덩어리로** 잰다. 샘플러가 그리퍼까지 포함해 한 번에 뽑으므로
  그림도 같은 단위로 물어야 한다. (참고: 축별로 보면 팔 자세 12 축은 초과첨도
  −0.4~−0.1 로 정규에 가깝고, 그리퍼 4 축만 +3.5~+4.2 로 열림/닫힘 이봉이다.
  아래 곡선이 대각선 아래로 처지는 것은 주로 그리퍼 몫이다.)

★ 거리는 기준 중앙값으로 나눠 놓는다. 절대 단위가 아니라 "가우시안이 주는
  전형적 거리의 몇 배인가"로 읽는 편이 낫다. x·y 를 같은 수로 나누므로
  대각선 위인지 여부는 바뀌지 않는다.
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
# observation.state 8 축이 2 프레임(t-1, t) 쌓여 16 축. 샘플러가 전부 한 번에
# 뽑으므로 여기서도 16 축을 통째로 쓴다.
ALL = tuple(range(16))
GRIP = (6, 7, 14, 15)
C_REF = "0.45"


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True, help="whitened.npz 경로들, 쉼표로 구분")
    p.add_argument("--out", default="results/L2_state_gauss_qq")
    p.add_argument("--nq", type=int, default=300, help="분위수 격자 수")
    p.add_argument("--qmax", type=float, default=0.995,
                   help="꼬리 끝은 표본이 얇아 튄다. 여기까지만 그린다")
    p.add_argument("--fix_gripper", action="store_true",
                   help="그리퍼 4 축을 셀의 실제 값에서 복원추출하는 샘플러를 "
                        "**모사**해 함께 그린다. 그리퍼는 열림/닫힘 이봉이라 "
                        "가우시안으로 뽑으면 실제에 없는 중간값이 나온다.")
    p.add_argument("--font_scale", type=float, default=1.0)
    return p.parse_args()


def load(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {"z": d["z"].astype(np.float64), "zr": d["z_ref"].astype(np.float64),
            "suite": str(d["suite"]), "n_cells": int(d["n_cells"])}


def qq(z, zr, idx, q):
    """(x, y, scale) — 기준 분위수, 실제 분위수, 나눈 값. 거리는 ‖z‖."""
    dr = np.sqrt((zr[:, idx] ** 2).sum(1))
    dd = np.sqrt((z[:, idx] ** 2).sum(1))
    s = float(np.median(dr))
    return np.quantile(dr, q) / s, np.quantile(dd, q) / s, s


def main():
    a = parse()
    data = [load(p.strip()) for p in a.npz.split(",") if p.strip()]
    q = np.linspace(1 - a.qmax, a.qmax, a.nq)
    fs = a.font_scale
    plt.rcParams.update({"font.size": 10.5 * fs, "axes.labelsize": 11 * fs,
                         "xtick.labelsize": 10 * fs, "ytick.labelsize": 10 * fs})

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    hi, lo = 0.0, 1e9
    rng = np.random.default_rng(0)
    for d in data:
        z, zr = d["z"], d["zr"]
        n = min(len(z), len(zr)); z, zr = z[:n], zr[:n]
        x, y, _ = qq(z, zr, ALL, q)
        ax.plot(x, y, lw=2.0, color=SUITE_COL.get(d["suite"], "0.4"),
                alpha=.5 if a.fix_gripper else .92, zorder=3,
                ls=(0, (3, 2)) if a.fix_gripper else "-", solid_capstyle="round")
        hi = max(hi, float(x.max()), float(y.max()))
        lo = min(lo, float(x.min()), float(y.min()))
        if a.fix_gripper:
            # 그리퍼를 셀의 실제 값에서 뽑는 샘플러 = 기준 쪽 그리퍼 축을
            # 실제 분포에서 온 값으로 바꾼 것과 같다.
            zr2 = zr.copy(); zr2[:, list(GRIP)] = z[rng.permutation(n)][:, list(GRIP)]
            x2, y2, _ = qq(z, zr2, ALL, q)
            ax.plot(x2, y2, lw=2.2, color=SUITE_COL.get(d["suite"], "0.4"),
                    alpha=.95, zorder=5, solid_capstyle="round")
            hi = max(hi, float(x2.max()), float(y2.max()))
            lo = min(lo, float(x2.min()), float(y2.min()))

    # ★ 0 부터 그리지 않는다. 곡선이 차지하는 범위만 잡아야 대각선에서 벗어난
    #   정도가 눈에 보인다 — 0 부터 두면 곡선이 구석에 뭉쳐 차이가 안 읽힌다.
    pad = (hi - lo) * .07
    lim0, lim1 = lo - pad, hi + pad
    ax.plot([lim0, lim1], [lim0, lim1], ls=(0, (5, 4)), lw=1.3, color=C_REF, zorder=2)
    ax.text(lim1 - pad * .3, lim1 - pad * .3, "if it were Gaussian", color=C_REF,
            fontsize=10.5 * fs, rotation=45, rotation_mode="anchor",
            ha="right", va="bottom")
    ax.set_xlim(lim0, lim1); ax.set_ylim(lim0, lim1); ax.set_aspect("equal")
    ax.set_xlabel("distance a Gaussian would give")
    ax.set_ylabel("distance the real states give")

    # ★ 곡선은 S 자다. 한 문장으로 뭉뚱그리면 틀린다 — 양 끝을 따로 짚는다.
    #   아래쪽(가까운 거리)은 대각선 **아래** = 실제가 더 몰려 있고,
    #   위쪽(먼 거리)은 대각선 **위** = 실제 꼬리가 더 길다.
    xr, yr, _ = qq(data[0]["z"], data[0]["zr"], ALL, q)
    klo, khi = int(a.nq * .13), int(a.nq * .93)
    ax.annotate("most states sit closer in\nthan a Gaussian would",
                (xr[klo], yr[klo]), xytext=(26, -4), textcoords="offset points",
                fontsize=10 * fs, color="0.25", ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color="0.55", lw=.9,
                                shrinkA=0, shrinkB=4))
    ax.annotate("but the far tail\nreaches further out",
                (xr[khi], yr[khi]), xytext=(-8, 6), textcoords="offset points",
                fontsize=10 * fs, color="0.25", ha="right", va="bottom",
                arrowprops=dict(arrowstyle="-", color="0.55", lw=.9,
                                shrinkA=0, shrinkB=4))
    hs = [plt.Line2D([], [], color=SUITE_COL.get(d["suite"], "0.4"), lw=2.0,
                     label=d["suite"].replace("_", " ")) for d in data]
    if a.fix_gripper:
        hs += [plt.Line2D([], [], color="0.35", lw=2.0, ls=(0, (3, 2)),
                          label="gripper sampled as Gaussian"),
               plt.Line2D([], [], color="0.35", lw=2.2,
                          label="gripper resampled from the cluster")]
    ax.legend(handles=hs, loc="upper left", frameon=False, fontsize=9 * fs,
              labelspacing=.32, handlelength=2.0, borderpad=.2)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("0.7"); ax.spines["bottom"].set_color("0.7")
    ax.tick_params(colors="0.35", length=3)
    ax.set_title("Can cluster states be sampled from a Gaussian?",
                 fontsize=12.5 * fs, pad=12)

    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"saved -> {out}.{ext}")

    # 수치 — 대각선에서 얼마나 벗어나는가 (1.0 이 완벽)
    print(f"\n{'suite':<16}{'셀':>6}{'중앙':>8}{'90%':>8}{'99%':>8}")
    for d in data:
        x, y, _ = qq(d["z"], d["zr"], ALL, np.array([.5, .9, .99]))
        print(f"{d['suite']:<16}{d['n_cells']:>6}"
              + "".join(f"{yy/xx:>8.2f}" for xx, yy in zip(x, y)))
    print("\n(실제 ÷ 가우시안. 1.00 이면 대각선 위.)")


if __name__ == "__main__":
    main()
