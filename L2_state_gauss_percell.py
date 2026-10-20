#!/usr/bin/env python
"""클러스터 **하나하나**가 가우시안인가 — 셀별로 재고 셀들 사이 분포를 본다.

    python L2_state_gauss_percell.py --npz a/whitened.npz,... \
        --out results/L2_state_gauss_percell

════════════════════════════════════════════════════════════════════════════
왜 합산한 그림으로는 부족한가
════════════════════════════════════════════════════════════════════════════
셀을 전부 표준화해 한 덩어리로 모아 거리 분포를 보면, 그건 **"평균적인 셀"** 에
대한 진술이다. 셀 절반이 잘 맞고 절반이 엉망이어도 섞으면 그럭저럭 맞아 보이고,
반대로 셀마다 제각각이면 개별 셀은 괜찮은데도 합산이 어긋나 보인다.

우리가 실제로 하는 일은 **특정 셀 k 의 가우시안에서 뽑는 것**이다. 그러니
"셀 k 에서 뽑아도 되는가"를 셀마다 따로 물어야 하고, 그 답이 셀들에 걸쳐
얼마나 고른지를 봐야 한다. 이 파일이 그걸 그린다.

════════════════════════════════════════════════════════════════════════════
통계량
════════════════════════════════════════════════════════════════════════════
셀 k 마다:
    1. 그 셀의 가우시안에서 "확률 q 짜리 영역"의 반지름 r_q 를 정한다
    2. 그 셀의 **실제** state 중 몇 %가 r_q 안에 들어오는지 센다
가우시안이 맞으면 그 값이 q 여야 한다. 셀 하나 = 점 하나.

★ r_q 는 이론식이 아니라 **모수 부트스트랩**에서 얻는다 — 셀과 같은 크기로
  N(0,I) 에서 뽑아 같은 절차로 표준화한 점들의 분위수다. μ̂·σ̂ 를 같은 프레임에서
  추정한 탓에 생기는 자유도 손실이 자동으로 반영된다.

★ 임계값은 **같은 n 을 가진 셀들의 기준점을 모두 모아** 계산한다. 셀 하나의
  부트스트랩 점 n 개(50~80)로 90% 분위수를 잡으면 임계값 자체가 흔들려,
  셀 간 산포에 "셀이 다른 탓"과 "임계값이 흔들린 탓"이 섞인다. 기준 분포는
  (n, 차원)에만 의존하므로 같은 n 끼리 모으는 것이 정당하다.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

SUITE_COL = {"libero_spatial": "#2ca089", "libero_goal": "#7a5bb5",
             "libero_object": "#d9722c", "libero_10": "#2f7fb5"}


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True, help="whitened.npz 경로들, 쉼표로 구분")
    p.add_argument("--out", default="results/L2_state_gauss_percell")
    p.add_argument("--q", type=float, default=0.90, help="--hist 모드의 명목 확률")
    p.add_argument("--strip", action="store_true",
                   help="셀마다 점 하나. 기본 x 는 포함률 — 그 클러스터의 명목 q "
                        "영역이 실제로 담는 state 비율(%%). 기준선은 q 자체다.")
    p.add_argument("--x", choices=["coverage", "reach"], default="coverage",
                   help="coverage: 명목 q 영역이 담는 실제 비율(%%), 기준선 q. "
                        "reach: 샘플 도달거리 ÷ 실제 도달거리, 기준선 1.0")
    p.add_argument("--dots", action="store_true",
                   help="명목 수준 몇 곳에서 셀별 포함률을 점으로. 대각선이 점 구름을 "
                        "가로지르므로 '약속한 만큼 담았나'가 간격으로 바로 읽힌다.")
    p.add_argument("--levels", default="20,40,60,80,90,95",
                   help="--dots 에서 볼 명목 수준(%%)")
    p.add_argument("--reach", action="store_true",
                   help="셀마다 점 하나: x = 실제 state 가 도달한 거리, "
                        "y = 가우시안 샘플이 도달한 거리. 대각선 아래 = 샘플이 "
                        "그 클러스터 범위를 벗어나지 않는다.")
    p.add_argument("--hist", action="store_true",
                   help="명목 한 지점에서의 셀별 분포(히스토그램). 기본은 누적 곡선")
    p.add_argument("--grid", default="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95",
                   help="곡선의 명목 격자(%%)")
    p.add_argument("--font_scale", type=float, default=1.0)
    return p.parse_args()


def load(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    z, zr = d["z"].astype(np.float64), d["z_ref"].astype(np.float64)
    c, n = d["cell"].astype(int), d["n_train"].astype(int)
    m = min(len(z), len(zr), len(c), len(n))
    return {"z": z[:m], "zr": zr[:m], "cell": c[:m], "n": n[:m],
            "suite": str(d["suite"])}


def per_cell(d: dict, q: float) -> np.ndarray:
    """셀마다 '가우시안 q 영역 안에 들어온 실제 state 비율(%)'."""
    dd = np.sqrt((d["z"] ** 2).sum(1))
    rr = np.sqrt((d["zr"] ** 2).sum(1))
    # 같은 n 끼리 기준점을 모아 임계 반지름을 구한다 (윗 설명 참고)
    thr_of_n = {nn: float(np.quantile(rr[d["n"] == nn], q))
                for nn in np.unique(d["n"])}
    by = defaultdict(list)
    for i, c in enumerate(d["cell"]):
        by[c].append(i)
    out = []
    for c, idx in by.items():
        idx = np.asarray(idx)
        thr = thr_of_n[d["n"][idx[0]]]
        out.append(100.0 * float(np.mean(dd[idx] <= thr)))
    return np.asarray(out)


def curve(d: dict, grid: np.ndarray) -> np.ndarray:
    """(셀 수, 격자) — 셀마다 명목 q 영역 안에 들어온 실제 state 비율(%).

    임계 반지름은 같은 n 을 가진 셀들의 부트스트랩 기준점을 모아 구한다.
    셀 하나의 기준점 n 개로 분위수를 잡으면 임계값이 흔들려 셀 간 산포에
    "셀이 다른 탓"과 "임계값이 흔들린 탓"이 섞인다.
    """
    dd = np.sqrt((d["z"] ** 2).sum(1))
    rr = np.sqrt((d["zr"] ** 2).sum(1))
    thr_of_n = {nn: np.quantile(rr[d["n"] == nn], grid / 100.0)
                for nn in np.unique(d["n"])}
    by = defaultdict(list)
    for i, c in enumerate(d["cell"]):
        by[c].append(i)
    out = np.empty((len(by), len(grid)))
    for r, (c, idx) in enumerate(by.items()):
        idx = np.asarray(idx)
        v = np.sort(dd[idx])
        thr = thr_of_n[d["n"][idx[0]]]
        out[r] = 100.0 * np.searchsorted(v, thr, side="right") / len(v)
    return out


def main():
    a = parse()
    data = [load(p.strip()) for p in a.npz.split(",") if p.strip()]
    qs = a.q * 100
    fs = a.font_scale
    plt.rcParams.update({"font.size": 10.5 * fs, "axes.labelsize": 11.5 * fs,
                         "xtick.labelsize": 10 * fs, "ytick.labelsize": 10 * fs})

    if a.dots:
        lv = np.array([float(x) for x in a.levels.split(",")])
        fig, ax = plt.subplots(figsize=(7.8, 5.6))
        rng = np.random.default_rng(0)
        # ★ 스위트를 겹쳐 뿌리면 마지막에 그린 것만 보인다. 명목 지점마다
        #   가로를 4 등분해 스위트별 소열을 만든다.
        span, ns = 5.2, len(data)
        ax.plot([0, 100], [0, 100], lw=1.6, color="0.25", ls=(0, (5, 4)), zorder=6)
        for i, d in enumerate(data):
            C = curve(d, lv)            # (셀, 명목)
            col = SUITE_COL.get(d["suite"], "0.4")
            off = (i - (ns - 1) / 2) * (span / ns)
            for j, g in enumerate(lv):
                x = g + off + rng.uniform(-span / ns / 2.3, span / ns / 2.3,
                                          C.shape[0])
                ax.scatter(x, C[:, j], s=3.0, lw=0, alpha=.20, color=col,
                           zorder=3, rasterized=True)
            ax.plot(lv + off, np.median(C, 0), lw=0, marker="o", ms=5.5,
                    mec="white", mew=1.0, color=col, zorder=7,
                    label=d["suite"].replace("_", " "))
        ax.set_xlim(lv.min() - 6, 100); ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        # 두 축을 같은 키워드 꼴로 — "coverage"를 공유하고 promised/actual 만 다르다
        ax.set_xlabel("promised coverage  (%)")
        ax.set_ylabel("actual coverage  (%)")
        # 대각선 설명. 60-80 구름 사이 빈 골짜기에 두고 대각선을 가리킨다.
        # 아래는 범례, 왼쪽은 구름이라 남는 자리가 여기뿐이다.
        ax.annotate("promised = actual", xy=(70, 70), xytext=(81, 47),
                    fontsize=10.5 * fs, color="0.25", ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=.9,
                                    shrinkA=6, shrinkB=2))
        # 범례는 오른쪽 아래. 왼쪽 위에 두면 x=20/40 구름의 윗꼬리를 덮는다.
        leg = ax.legend(frameon=False, fontsize=9.5 * fs, labelspacing=.32,
                        loc="lower right", markerscale=1.0,
                        bbox_to_anchor=(1.0, 0.0))
        # 범례 항목 **바로 아래**에 붙인다. 항목 수 x 상수로 잡으면 --font_scale
        # 을 올렸을 때 범례만 키가 커져 마지막 항목과 겹친다 — 그려 놓고 실제
        # 높이를 재서 그 밑에 놓는다.
        fig.canvas.draw()
        bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
        ax.text(bb.x1, bb.y1 + .014, "one dot = one cluster",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9.5 * fs, color="0.45")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("0.7"); ax.spines["bottom"].set_color("0.7")
        ax.tick_params(colors="0.35", length=3)
        ax.set_title("Per-cluster Gaussian coverage", fontsize=11.5 * fs, pad=10)
        fig.tight_layout()
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None,
                        bbox_inches="tight")
            print(f"saved -> {out}.{ext}")
        print(f"\n{'suite':<16}" + "".join(f"{f'{g:.0f}%':>16}" for g in lv))
        for d in data:
            C = curve(d, lv)
            row = f"{d['suite']:<16}"
            for j in range(len(lv)):
                q1, q3 = np.percentile(C[:, j], [25, 75])
                row += f"{f'{np.median(C[:, j]):.0f} ({q1:.0f}-{q3:.0f})':>16}"
            print(row)
        return

    if a.strip:
        cov = a.x == "coverage"
        rows = []
        for d in data:
            if cov:
                rows.append((d["suite"], per_cell(d, a.q)))
            else:
                dd = np.sqrt((d["z"] ** 2).sum(1)); rr = np.sqrt((d["zr"] ** 2).sum(1))
                by = defaultdict(list)
                for i, c in enumerate(d["cell"]):
                    by[c].append(i)
                rows.append((d["suite"], np.array(
                    [rr[np.asarray(v)].max() / dd[np.asarray(v)].max()
                     for v in by.values()])))
        ref = qs if cov else 1.0

        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        rng = np.random.default_rng(0)
        ax.axvline(ref, color="0.25", lw=1.6, ls=(0, (4, 2.5)), zorder=5)
        for i, (nm, r) in enumerate(rows):
            y = len(rows) - 1 - i + rng.uniform(-.28, .28, len(r))
            ax.scatter(r, y, s=6, lw=0, alpha=.35, rasterized=True,
                       color=SUITE_COL.get(nm, "0.4"), zorder=4)
            ax.text(-.012, len(rows) - 1 - i, nm.replace("_", " "),
                    transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=10.5 * fs, color=SUITE_COL.get(nm, "0.4"))
            # 그 줄의 중앙값을 굵은 눈금으로 + 수치
            med = float(np.median(r))
            ax.plot([med, med], [len(rows) - 1 - i - .34, len(rows) - 1 - i + .34],
                    color=SUITE_COL.get(nm, "0.4"), lw=2.4, zorder=6, solid_capstyle="butt")
            ax.text(med, len(rows) - 1 - i - .42,
                    (f"{med:.1f}%" if cov else f"{med:.2f}"),
                    ha="center", va="top", fontsize=9.5 * fs, color="0.3")
        ax.set_yticks([]); ax.set_ylim(-.85, len(rows) - .3)
        lo = min(float(np.percentile(r, .3)) for _n, r in rows)
        hi = max(float(np.percentile(r, 99.7)) for _n, r in rows)
        ax.set_xlim(lo - (hi - lo) * .04, max(ref, hi) + (hi - lo) * .06)
        ax.set_xlabel(f"share of each cluster's real states inside its own "
                      f"{qs:.0f}% Gaussian region" if cov else
                      "how far the Gaussian's samples reach  ÷  how far the "
                      "cluster's real states reach")
        ax.text(ref, len(rows) - .34,
                (f"  what a Gaussian promises: {qs:.0f}%" if cov
                 else "  equal reach"),
                ha="left", va="center", fontsize=10 * fs, color="0.25")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("0.7")
        ax.tick_params(colors="0.35", length=3)
        ax.set_title("One dot = one cluster", fontsize=13 * fs, pad=10)
        fig.tight_layout()
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None,
                        bbox_inches="tight")
            print(f"saved -> {out}.{ext}")
        print()
        for nm, r in rows:
            q1, q3 = np.percentile(r, [25, 75])
            print(f"  {nm:<16} 셀 {len(r):>4}  중앙 {np.median(r):.2f} "
                  f"(IQR {q1:.1f}-{q3:.1f})")
        return

    if a.reach:
        fig, ax = plt.subplots(figsize=(6.5, 6.2))
        hi = 0.0
        stat = []
        for d in data:
            dd = np.sqrt((d["z"] ** 2).sum(1)); rr = np.sqrt((d["zr"] ** 2).sum(1))
            by = defaultdict(list)
            for i, c in enumerate(d["cell"]):
                by[c].append(i)
            X = np.array([dd[np.asarray(v)].max() for v in by.values()])
            Y = np.array([rr[np.asarray(v)].max() for v in by.values()])
            ax.scatter(X, Y, s=7, lw=0, alpha=.45, zorder=4,
                       color=SUITE_COL.get(d["suite"], "0.4"),
                       label=d["suite"].replace("_", " "), rasterized=True)
            hi = max(hi, X.max(), Y.max())
            stat.append((d["suite"], len(X), 100 * np.mean(Y <= X)))
        ax.plot([0, hi * 1.05], [0, hi * 1.05], ls=(0, (5, 4)), lw=1.3,
                color="0.4", zorder=5)
        ax.set_xlim(0, hi * 1.05); ax.set_ylim(0, hi * 1.05); ax.set_aspect("equal")
        ax.set_xlabel("how far the cluster's real states reach")
        ax.set_ylabel("how far the Gaussian's samples reach")
        ax.text(hi * 1.02, hi * 1.02, "equal reach", color="0.4",
                fontsize=10.5 * fs, rotation=45, rotation_mode="anchor",
                ha="right", va="bottom")
        ax.text(.97, .06, "below the line:\nsamples stay inside\nthe cluster's own range",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10.5 * fs, color="0.25")
        leg = ax.legend(frameon=False, fontsize=9.5 * fs, labelspacing=.32,
                        loc="upper left", markerscale=2.6)
        for h in leg.legend_handles:
            h.set_alpha(.95)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("0.7"); ax.spines["bottom"].set_color("0.7")
        ax.tick_params(colors="0.35", length=3)
        ax.set_title("One dot = one cluster", fontsize=13 * fs, pad=10)
        fig.tight_layout()
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None,
                        bbox_inches="tight")
            print(f"saved -> {out}.{ext}")
        print()
        for nm, n_, pc in stat:
            print(f"  {nm:<16} 셀 {n_:>4}   샘플이 실제 범위 안에 머문 셀 {pc:.1f}%")
        return

    if not a.hist:
        grid = np.array([float(x) for x in a.grid.split(",")])
        fig, ax = plt.subplots(figsize=(6.6, 6.2))
        ax.plot([0, 100], [0, 100], ls=(0, (5, 4)), lw=1.3, color="0.45", zorder=2)
        ax.text(97, 97, "if it were Gaussian", color="0.45", fontsize=10.5 * fs,
                rotation=45, rotation_mode="anchor", ha="right", va="bottom")
        for d in data:
            C = curve(d, grid)
            col = SUITE_COL.get(d["suite"], "0.4")
            lo, hi = np.percentile(C, [25, 75], axis=0)
            ax.fill_between(grid, lo, hi, color=col, alpha=.16, lw=0, zorder=3)
            ax.plot(grid, np.median(C, 0), lw=1.9, color=col, zorder=4,
                    label=d["suite"].replace("_", " "))
        ax.set_xlim(grid.min(), 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
        ax.set_xlabel("size of the Gaussian region  (nominal %)")
        ax.set_ylabel("real states it actually contains  (%)")
        ax.legend(frameon=False, fontsize=9.5 * fs, labelspacing=.32,
                  loc="upper left")
        ax.axvline(90, color="0.6", lw=.8, alpha=.6, zorder=1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("0.7"); ax.spines["bottom"].set_color("0.7")
        ax.tick_params(colors="0.35", length=3)
        ax.set_title("What a cluster's Gaussian region really contains",
                     fontsize=13 * fs, pad=10)
        fig.tight_layout()
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None,
                        bbox_inches="tight")
            print(f"saved -> {out}.{ext}")
        print(f"\n명목 -> 실제 (셀별 중앙값, 괄호는 IQR).  셀 하나가 곡선 하나")
        show = [50.0, 80.0, 90.0, 95.0]
        print(f"{'suite':<16}{'셀':>6}" + "".join(f"{f'{g:.0f}%':>18}" for g in show))
        for d in data:
            C = curve(d, grid)
            row = f"{d['suite']:<16}{C.shape[0]:>6}"
            for g in show:
                j = int(np.argmin(abs(grid - g)))
                q1, q3 = np.percentile(C[:, j], [25, 75])
                row += f"{f'{np.median(C[:, j]):.1f} ({q1:.0f}-{q3:.0f})':>18}"
            print(row)
        return

    vals = [per_cell(d, a.q) for d in data]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bins = np.arange(0, 102.5, 2.5)
    for d, v in zip(data, vals):
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8,
                color=SUITE_COL.get(d["suite"], "0.4"), zorder=4,
                label=d["suite"].replace("_", " "))
    ax.axvline(qs, color="0.3", lw=1.5, ls=(0, (4, 2.5)), zorder=5)
    ax.annotate(f"what a Gaussian promises: {qs:.0f}%", (qs, ax.get_ylim()[1]),
                xytext=(-9, -6), textcoords="offset points", rotation=90,
                fontsize=10 * fs, color="0.3", ha="right", va="top")
    lo = min(float(np.percentile(v, 0.5)) for v in vals)
    ax.set_xlim(max(0, lo - 4), 101)
    ax.set_xlabel(f"share of the cluster's real states inside its own "
                  f"{qs:.0f}% Gaussian region")
    ax.set_ylabel("clusters (density)")
    ax.legend(frameon=False, fontsize=9.5 * fs, labelspacing=.34, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("0.7"); ax.spines["bottom"].set_color("0.7")
    ax.tick_params(colors="0.35", length=3)
    ax.set_title("Each cluster, measured on its own", fontsize=13 * fs, pad=10)

    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"saved -> {out}.{ext}")

    print(f"\n명목 {qs:.0f}% 영역에 들어온 실제 state 비율 — 셀 하나가 값 하나")
    print(f"{'suite':<16}{'셀':>6}{'중앙':>8}{'IQR':>16}"
          f"{'80%p 이상':>11}{'70%p 이상':>11}")
    for d, v in zip(data, vals):
        q1, q3 = np.percentile(v, [25, 75])
        print(f"{d['suite']:<16}{len(v):>6}{np.median(v):>8.1f}"
              f"{f'{q1:.1f}~{q3:.1f}':>16}"
              f"{100*np.mean(v >= 80):>10.0f}%{100*np.mean(v >= 70):>10.0f}%")


if __name__ == "__main__":
    main()
