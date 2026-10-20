#!/usr/bin/env python
"""코드북 셀의 가우시안 근사 — 최종 2패널 그림.

묻는 것
    "셀마다 둔 가우시안이 실제 state 의 퍼짐을 얼마나 담아내는가?"

    ① 예상 비율 vs 실제 비율   명목 포함 비율을 쓸어가며 실제로 들어온 비율을 잰다.
                               대각선에 가까울수록 근사가 퍼짐을 잘 설명한다.
    ② 클러스터별 편차          명목 90% 하나에 대해 셀마다 실제 비율을 찍는다.

★ 핵심은 held-out 을 **두 가지로** 뗀 것이다. 이 둘을 같이 봐야 답이 갈린다.
    같은 에피소드, 새 프레임   셀 안의 퍼짐만 본다. 가우시안 가정 자체의 시험이다.
    새 에피소드               물체 배치가 다른 실행. 에피소드 간 변동이 더해진다.
  둘이 갈라지면 "셀 안이 가우시안이 아니다"가 아니라 "에피소드 간 변동을 셀별
  가우시안이 담당하지 않는다"가 맞는 서술이다. 하나만 그리면 그 구분을 못 한다.

모델
    그림은 **대각 가우시안** — sample_codebook 이 실제로 쓰는 모델 — 만 그린다.
    완전공분산(--full_cov_s)도 같이 계산돼 JSON 에 있지만, ridge 가 Σ 를 부풀려
    포함 비율이 낙관적으로 나오므로 같은 축에 겹치면 오해를 부른다.

포함 비율의 기준
    모수를 표본에서 추정했으므로 χ² 가 아니라 F 를 쓴다(계산은 L2_state_gauss_calibration.py).
        T·n(n-d)/(d(n²-1)) ~ F(d, n-d)

    python L2_state_gauss_calib_plot.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

C_SAME, C_NEW, C_REF = "#009E73", "#D55E00", "0.45"
# 스위트별 색 — 여러 스위트를 한 그림에 겹칠 때 쓴다.
SUITE_COL = {"libero_spatial": "#009E73", "libero_10": "#0072B2",
             "libero_goal": "#6A3D9A", "libero_object": "#D55E00"}


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frame_json",
                   default="results/L2_state_gauss_calib_frame_k100/calibration_summary.json",
                   help="쉼표로 여러 개를 주면 스위트마다 곡선 하나씩 그린다. 이름과 색은 "
                        "각 JSON 의 suite 필드에서 온다.")
    p.add_argument("--episode_json", default="results/L2_state_gauss_calib/calibration_summary.json")
    p.add_argument("--model", choices=["diag", "full"], default="diag")
    p.add_argument("--radius", action="store_true",
                   help="y 를 '필요 반지름 비'로. 완벽 보정이 수평선 1.0 이 된다. 16 차원에서 "
                        "질량 축이 차이를 과장하므로 같은 불일치를 거리 단위로 보는 것이다.")
    p.add_argument("--delta", action="store_true",
                   help="y 를 (실측 − 명목) 으로. 대각선이 수평선 0 이 되어 '얼마나 떴는가'가 "
                        "작게 읽힌다. 같은 데이터이고 축만 바꾼 것이다.")
    p.add_argument("--wide", action="store_true",
                   help="--panels one 을 가로로 넓게(7.1x5.0). 다른 그림과 나란히 놓을 때. "
                        "등비 제약을 풀어야 실제로 넓어지므로 대각선이 45도가 아니게 된다.")
    p.add_argument("--panels", choices=["two", "one"], default="two",
                   help="one = ① 보정 곡선만. 주장이 '명목 대비 실측 포함률'이면 ② 는 "
                        "보조라 빼도 된다 — 다만 사분위 띠와 셀 수는 남겨야 한다.")
    p.add_argument("--only", choices=["both", "frame", "episode"], default="both",
                   help="frame = 같은 에피소드(새 프레임) 곡선만 그린다.")
    p.add_argument("--out", default="results/L2_state_gauss_calib/state_gauss_calibration")
    return p.parse_args()


def radius_ratio(d: dict, model: str) -> np.ndarray:
    """셀 x 명목격자 의 "필요 반지름 비" — 1.0 이면 완벽 보정.

    16 차원에서는 질량이 껍질에 몰려 있어 반지름이 조금만 어긋나도 담기는 질량이
    크게 달라진다(명목 50% 에서 반지름 12% 차이가 질량으로는 24%p 로 보인다).
    같은 불일치를 **거리 단위**로 다시 적는 것이다.

        실측 포함률 c 를 담는 이론 임계값이 thr(c), 우리가 쓴 임계값이 thr(q) 이니
        sqrt(thr(q)/thr(c)) = "정확히 맞추려면 타원체를 몇 배로 줄여야 하는가".

    임계값은 측정과 같은 Hotelling T2(F(d, n-d)). 셀마다 n 이 달라 셀별로 계산한다.
    """
    from scipy.stats import f as f_dist
    rows = d["per_cell"]
    dim = int(d["dim"])
    grid = np.array(d["nominal_grid"])
    def thr(q, n):
        return dim * (n ** 2 - 1) / (n * (n - dim)) * float(f_dist.ppf(q, dim, n - dim))
    out = np.full((len(rows), len(grid)), np.nan)
    for i, r in enumerate(rows):
        n = int(r["n_train"])
        for j, q in enumerate(grid):
            c = float(r[f"cov_{model}"][j])
            # ★ 포화된 셀(포함률 0% 또는 100%)은 비가 **정의되지 않는다**. 경계값으로
            #   눌러 넣으면 높은 명목에서 값이 튀어 띠가 무너진다. NaN 으로 두고
            #   분위 계산에서 뺀다(nanpercentile/nanmedian).
            if 0.0 < c < 1.0:
                out[i, j] = (thr(float(q), n) / thr(c, n)) ** 0.5
    return out


def load(path: str, model: str):
    d = json.loads(Path(path).read_text())
    rows = d["per_cell"]
    grid = np.array(d["nominal_grid"]) * 100
    cov = np.array([r[f"cov_{model}"] for r in rows]) * 100          # (셀, 명목격자)
    at90 = np.array([r[f"at_{model}"][-1] for r in rows]) * 100      # 명목 90% 에서
    task = np.array([r["task"] for r in rows])
    return dict(grid=grid, cov=cov, at90=at90, task=task, meta=d,
                n_test=np.array([r["n_test"] for r in rows]),
                n_train=np.array([r["n_train"] for r in rows]))


def main():
    a = parse()
    paths = [p.strip() for p in a.frame_json.split(",") if p.strip()]
    multi = len(paths) > 1               # 스위트마다 곡선 하나
    S = load(paths[0], a.model)          # 같은 에피소드, 새 프레임
    if multi:
        N = None
        # 밑줄은 빼고 보여 준다 — 폴더명이 아니라 사람이 읽는 이름이다.
        series = [(d, SUITE_COL.get(d["meta"]["suite"], "0.45"),
                   d["meta"]["suite"].replace("_", " "))
                  for d in (load(p, a.model) for p in paths)]
    else:
        N = load(a.episode_json, a.model) if a.only != "frame" else None  # 새 에피소드
        # --only frame 이면 한 곡선만 — 주장이 "코드북이 보는 분포 안에서의 포함률"이면
        # 새 에피소드 계열은 그 주장에 쓰이지 않는다.
        series = ([(S, C_SAME, "same episodes, new frames")]
                  + ([(N, C_NEW, "new episodes")] if N is not None else []))
    tasks = sorted(set(S["task"].tolist())
                   | (set(N["task"].tolist()) if N is not None else set()))

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10.5,
                                "axes.edgecolor": "0.72", "axes.linewidth": .8,
                                "xtick.color": "0.4", "ytick.color": "0.4"})
    import matplotlib.pyplot as plt

    one = a.panels == "one"
    if one:
        # A 안: 그림 안에는 데이터만 둔다. 제목·범례·캡션은 논문 캡션이 맡고,
        # 구조적으로 비는 대각선 아래 삼각형을 수치와 히스토그램으로 채운다.
        fig, axes = plt.subplots(1, 1, figsize=(7.1, 5.0) if a.wide else (5.6, 5.4))
        axes = [axes]
        fig.subplots_adjust(left=.105 if a.wide else .135, right=.985,
                            top=.915 if a.wide else .925, bottom=.135)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.9))
        fig.subplots_adjust(left=.075, right=.985, top=.845, bottom=.255, wspace=.23)

    # ① 예상 vs 실제 ---------------------------------------------------------
    ax = axes[0]
    radius = one and a.radius          # y = 필요 반지름 비 (1.0 이 완벽)
    delta = one and a.delta and not radius   # y = 실측 − 명목
    if radius:
        ax.axhline(1.0, ls=(0, (4, 3)), lw=1.1, color=C_REF, zorder=2)
        # 오른쪽 끝은 곡선이 1.0 을 가로지르며 90% 마커가 모이는 자리다. 왼쪽 위는
        # 범례가 쓰므로, 기준선 **바로 아래 왼쪽** — 곡선이 0.8 근처에 있어 비어 있는
        # 유일한 자리에 붙인다.
        ax.text(.012, 1.0, " perfect calibration", color=C_REF, fontsize=9.5,
                transform=ax.get_yaxis_transform(), ha="left", va="top", zorder=6,
                bbox=dict(fc="white", ec="none", alpha=.75, pad=1.5))
    elif delta:
        ax.axhline(0, ls=(0, (4, 3)), lw=1.1, color=C_REF, zorder=2)
        ax.text(.985, .04, "perfect calibration", color=C_REF, fontsize=9.5,
                transform=ax.transAxes, ha="right", va="bottom")
    else:
        ax.plot([0, 100], [0, 100], ls=(0, (4, 3)), lw=1.1, color=C_REF, zorder=2)
        ax.text(70, 64, "perfect", color=C_REF, fontsize=9.5, rotation=45,
                rotation_mode="anchor", ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=1.5))
    for d, col, lbl in series:
        Y = radius_ratio(d["meta"], a.model) if radius else d["cov"]
        g = d["grid"]
        if radius:
            # 명목이 높아질수록 포화된 셀(radius_ratio 가 NaN 으로 둔 것)이 늘어
            # 표본이 얇아진다. 남은 셀이 절반 미만인 지점은 중앙값·IQR 이 더 이상
            # "그 스위트의 셀들"을 대표하지 못하므로 그리지 않는다.
            ok = np.isfinite(Y).sum(0)
            keep = (g <= 95) & (ok >= max(5, int(.5 * Y.shape[0])))
            Y, g = Y[:, keep], g[keep]
            d["_grid_r"] = g
        lo, hi = np.nanpercentile(Y, [25, 75], axis=0)
        med = np.nanmedian(Y, 0)
        if delta:                       # 축만 바꾼다 — 데이터는 그대로
            lo, hi, med = lo - g, hi - g, med - g
        if radius:
            d["_rr"] = Y                # 90% 마커·표에서 다시 쓴다
        if one and multi:
            # 스위트별 IQR 을 곡선 색의 반투명 띠로. 넷이 겹치므로 alpha 를 낮게 두고
            # 선을 얇고 진하게 얹어 신호가 띠에 묻히지 않게 한다.
            ax.fill_between(g, lo, hi, color=col, alpha=.16, lw=0, zorder=3)
        elif one:
            # 곡선과 대각선 사이를 칠한다 — "위에 있다 = 명목보다 더 담는다"가
            # 한눈에 읽힌다. 띠는 이 그림의 핵심 증거라 더 진하게.
            ax.fill_between(g, g, med, color=col, alpha=.055, lw=0, zorder=2)
            ax.fill_between(g, lo, hi, color=col, alpha=.28, lw=0, zorder=3)
        else:
            ax.fill_between(g, lo, hi, color=col, alpha=.15, lw=0, zorder=3)
        ax.plot(g, med, color=col, zorder=5, label=lbl,
                lw=1.4 if (one and multi) else (2.6 if one else 2.3))
    # 명목 90% 지점을 눈금으로 찍어 준다 — ② 와 같은 지점이다
    for d, col, _lbl in series:
        i90 = int(np.argmin(abs((d["_grid_r"] if radius else d["grid"]) - 90)))
        v = (float(np.nanmedian(d["_rr"][:, i90])) if radius
             else float(np.median(d["at90"])))
        ax.plot([90], [v - 90 if delta else v], marker="o", ms=6, color=col, zorder=6,
                mec="white", mew=.9)
        if not one:     # one 은 수치를 오른쪽 아래 빈 공간에 크게 적는다
            ax.annotate(f"{v:.0f}%", (90, v), textcoords="offset points",
                        xytext=(9, -3), color=col, fontsize=10.5, fontweight="medium")
    ax.axvline(90, color=C_REF, lw=.7, alpha=.5, zorder=1)
    if one and multi:
        # ── 빈 삼각형: 스위트별 수치 표 ─────────────────────────────────────
        # IQR 은 띠가, 셀 수는 캡션이 맡는다 — 표에는 중앙값만.
        # 축 비율 좌표로 둔다 — 데이터 좌표면 --delta 로 y 범위가 바뀔 때 표가 날아간다.
        X0, X1, Y0 = .50, .84, .34
        ax.text(X0, Y0, "At nominal 90%", fontsize=10, color="0.3",
                transform=ax.transAxes, ha="left", va="bottom")
        for i, (d, col, nm) in enumerate(series):
            i90 = int(np.argmin(abs((d["_grid_r"] if radius else d["grid"]) - 90)))
            txt = (f"{np.nanmedian(d['_rr'][:, i90]):.2f}" if radius
                   else f"{np.median(d['at90']):.1f}%")
            y = Y0 - .075 * (i + 1)
            ax.text(X0, y, nm, fontsize=9, color=col, transform=ax.transAxes,
                    ha="left", va="bottom", fontweight="semibold")
            ax.text(X1, y, txt, fontsize=9, color="0.35",
                    transform=ax.transAxes, ha="right", va="bottom")
    elif one:
        # ── 빈 삼각형 ①: 핵심 수치 ──────────────────────────────────────────
        d, col, _ = series[0]
        v = float(np.median(d["at90"]))
        q1, q3 = np.percentile(d["at90"], [25, 75])
        ax.text(52, 41, "At nominal 90%", fontsize=10, color="0.35",
                ha="left", va="bottom")
        ax.text(52, 31, f"{v:.1f}%", fontsize=19, color=col,
                ha="left", va="bottom", fontweight="semibold")
        ax.text(52, 24, f"IQR {q1:.1f}–{q3:.1f}   ({len(d['at90'])} clusters)",
                fontsize=9, color="0.45", ha="left", va="bottom")
        # ── 빈 삼각형 ②: 셀별 포함률 히스토그램 (뺀 ② 패널의 정보) ──────────
        # 수치 텍스트 아래로 내리고, 메인 x 축 라벨과 겹치지 않게 띄운다.
        ins = ax.inset_axes([.525, .085, .44, .115])
        ins.hist(d["at90"], bins=np.arange(0, 105, 5), color=col, alpha=.75, lw=0)
        ins.axvline(90, color=C_REF, lw=.9, ls=(0, (3, 2)))
        ins.set_xlim(0, 100); ins.set_yticks([])
        ins.tick_params(labelsize=7.5, length=2, colors="0.45")
        ins.set_xlabel("per-cluster coverage (%)", fontsize=7.5,
                       color="0.45", labelpad=1.5)
        for s in ("top", "right", "left"):
            ins.spines[s].set_visible(False)
        ins.spines["bottom"].set_color("0.75")
    if one:
        # 측정은 명목 5% 부터다. 축을 0 부터 두면 5 에서 띠가 수직으로 끊겨 보인다.
        # (0,0) 을 앞에 붙이는 방법도 있지만 5% 에서 이미 25% 라 그 구간이 직선
        # 보간이 되어 재지 않은 값을 그리게 된다 — 축을 자르는 쪽이 정직하다.
        # 오른쪽 끝은 실제로 그려진 지점까지만 — 포화로 잘린 구간에 빈 여백이 남지 않게.
        right = (max(float(d["_grid_r"].max()) for d, _c, _l in series) if radius else 100)
        ax.set_xlim(float(series[0][0]["grid"].min()), right)
    else:
        ax.set_xlim(0, 100)
    if radius:
        lo_all = min(float(np.nanpercentile(d["_rr"], 25, axis=0).min()) for d, _c, _l in series)
        hi_all = max(float(np.nanpercentile(d["_rr"], 75, axis=0).max()) for d, _c, _l in series)
        ax.set_ylim(min(lo_all, .95) - .04, max(hi_all, 1.05) + .04)
    elif delta:
        hi_all = max(float((np.percentile(d["cov"], 75, axis=0) - d["grid"]).max())
                     for d, _c, _l in series)
        ax.set_ylim(-4, hi_all + 4)
    else:
        ax.set_ylim(0, 103)
    if not (one and a.wide):    # 넓게 쓰려면 등비를 풀어야 한다 — 아니면 좌우가 흰 여백
        ax.set_aspect("equal")
    ax.set_xlabel("Nominal coverage of the Gaussian region (%)")
    ax.set_ylabel("Radius scale for exact calibration" if radius else
                  "Observed − nominal (%p)" if delta else
                  "Real held-out states inside (%)")
    ax.set_title("Per-cluster Gaussian coverage" if one else
                 "① Does the Gaussian explain the spread?", fontsize=12.5, pad=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if not one:     # 곡선이 하나뿐이면 범례는 자리만 차지한다
        ax.legend(frameon=False, fontsize=10, loc="upper left")
    elif multi:     # 스위트가 여럿이면 어느 색이 어느 스위트인지 필요하다
        ax.legend(frameon=False, fontsize=9.5, loc="upper left",
                  handlelength=1.6, borderaxespad=.8)

    # ② 클러스터별 편차 (명목 90%) -------------------------------------------
    # --panels one 이면 ① 만 남긴다. 주장이 "명목 대비 실측 포함률"이면 ② 는 보조다.
    ax = axes[0] if one else axes[1]
    if not one:
        ax.axhline(90, ls=(0, (4, 3)), lw=1.1, color=C_REF, zorder=2)
    if not one:
        ax.text(len(tasks) - .45, 91.5, "nominal 90%", color=C_REF, fontsize=9.5,
                ha="right", va="bottom")
    rng = np.random.default_rng(0)
    for j, (d, col, lbl) in ([] if one else list(enumerate(series))):
        for i, t in enumerate(tasks):
            v = d["at90"][d["task"] == t]
            if not len(v):
                continue
            x = i + (j - .5) * .40 + rng.uniform(-.085, .085, len(v))
            ax.scatter(x, v, s=15, color=col, alpha=.55, edgecolors="none", zorder=4,
                       label=lbl if i == 0 else None)
            ax.plot([i + (j - .5) * .40 - .13, i + (j - .5) * .40 + .13],
                    [np.median(v)] * 2, color=col, lw=2.4, zorder=5)
    if not one:
        ax.set_xticks(range(len(tasks))); ax.set_xticklabels([str(t) for t in tasks])
        ax.set_xlim(-.6, len(tasks) - .4); ax.set_ylim(-3, 103)
        ax.set_xlabel("Task"); ax.set_ylabel("Inside the nominal 90% region (%)")
        ax.set_title("② How far off is each cluster?", fontsize=12, pad=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=10, loc="lower left")

    m = (N if N is not None else S)["meta"]
    if not one:     # 단일 패널은 축 제목이 같은 말을 하므로 전체 제목을 두지 않는다
        fig.suptitle("How much of the real state spread does the per-cluster Gaussian capture?",
                     fontsize=13.5, y=.955)
    if one:
        # A 안: 그림 안에 설명 텍스트를 두지 않는다. 아래 한 줄은 논문 캡션용으로
        # stdout 에 찍어 준다.
        who = (", ".join(f"{d['meta']['suite']} ({len(d['at90'])})" for d, _c, _n in series)
               if multi else f"{m['suite']}, {len(tasks)} tasks")
        print("  [caption] "
              f"{who}, K={m['K']}, 10 tasks each, "
              f"clusters with ≥21 train / ≥15 held-out frames. Coverage is measured in "
              f"the original {m['dim']}-D state space with the finite-sample threshold "
              f"F(d, n−d) in place of χ². "
              + ("Each line is the median over that suite's clusters; the table gives "
                 "the per-cluster IQR at the nominal 90% level."
                 if multi else "Line = median over clusters, band = 25–75%."))
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None)
            print("saved ->", f"{out}.{ext}")
        for d, _col, nm in series:
            print(f"  {nm:26s} 명목 90% -> 실제 중앙값 {np.median(d['at90']):.1f}% "
                  f"(IQR {np.percentile(d['at90'],25):.1f}~{np.percentile(d['at90'],75):.1f}), "
                  f"셀 {len(d['at90'])}개")
        return
    fig.text(.5, .155,
             f"{m['suite']}, {len(tasks)} tasks, K={m['K']}, diagonal Gaussian "
             f"(the model sample_codebook actually draws from). Coverage is measured in the "
             f"original {m['dim']}-D state space,\nwith the finite-sample threshold "
             f"T·n(n−d)/(d(n²−1)) ~ F(d, n−d) instead of χ², so that "
             f"estimating (μ, Σ) from n frames is already accounted for.\n"
             + (f"Clusters: {len(S['at90'])}, each with ≥21 train and ≥15 held-out "
                f"frames. Line = median over clusters, band = 25–75%."
                if N is None else
                f"Clusters: {len(S['at90'])} (same-episode) / {len(N['at90'])} "
                f"(new-episode), each with ≥50 train and ≥15 held-out frames. "
                f"① line = median over clusters, band = 25–75%.  "
                f"② one dot = one cluster, bar = median."),
             ha="center", va="top", fontsize=9, color="0.3", linespacing=1.6)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None)
        print("saved ->", f"{out}.{ext}")

    for d, _col, nm in series:
        print(f"  {nm:26s} 명목 90% -> 실제 중앙값 {np.median(d['at90']):.1f}% "
              f"(IQR {np.percentile(d['at90'],25):.1f}~{np.percentile(d['at90'],75):.1f}), "
              f"셀 {len(d['at90'])}개")


if __name__ == "__main__":
    main()
