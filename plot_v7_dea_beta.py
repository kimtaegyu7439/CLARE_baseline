#!/usr/bin/env python
"""V7 dea (액션 공간) β 애블레이션 그림 — results/Action_dea_ablation.{png,pdf}

    python plot_v7_dea_beta.py

왼쪽 축  AvgSR / FWT↑ / AUC↑   — 전부 성공률(%) 이라 같은 눈금을 쓴다
오른쪽 축 NBT↓                  — 망각량(%p) 이라 범위가 ±6 으로 훨씬 작다

★ 이중 축은 일반적으로 피해야 한다. 두 눈금의 정렬이 임의라서 없는 상관을
  있어 보이게 만들기 때문이다. 여기서는 요청대로 쓰되, 오해를 줄이려고
  NBT 는 **점선 + 다른 마커 + 축·눈금·레이블을 같은 색**으로 묶고 범례에
  "(right axis)" 를 적었다. 그래도 두 축의 높낮이를 서로 비교하면 안 된다.
  위아래 두 패널로 나눈 판본이 더 정확하다 — --panels 로 그릴 수 있다.

데이터 출처
  β0.1~3.5  이 서버의 results/clare_v7_dea_*_goal_seed7/sr_matrix.csv 에서 직접 읽는다
  β4.0~6.0  다른 4090 서버에서 돈 값이라 여기 없다. EXTRA 에 적어 둔다.
  ★ β2.0 은 폴더명에 접미사가 없다(clare_v7_dea_goal_seed7). 기본 실행이라
    그렇다. 빼면 최고점이 사라져 곡선이 왜곡되므로 반드시 포함한다.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clare_v4_report import metrics, read_matrix

REPO = Path(__file__).resolve().parent
RES = REPO / "results"

# 다른 서버에서 돈 값 (AvgSR, FWT, NBT, AUC)
EXTRA = {
    4.0: (87.6, 91.6, +2.4, 89.7),
    4.5: (88.8, 91.0, +1.9, 89.3),
    5.5: (90.2, 91.8, +2.3, 90.2),
    6.0: (88.8, 92.0, +2.0, 90.8),
}

# 검증된 팔레트 (validate_palette.js 통과: CVD ΔE 9.2, normal ΔE 24.0)
C_AVG, C_FWT, C_AUC, C_NBT = "#2a78d6", "#1baf7a", "#eb6834", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def collect() -> list[tuple[float, float, float, float, float]]:
    rows: dict[float, tuple] = {}
    for d in sorted(RES.glob("clare_v7_dea_*goal_seed7")):
        n = d.name
        if "bs" in n:                      # batch 변형(_b24bs)은 다른 조건이다
            continue
        m = re.search(r"_b([0-9.]+)_goal", n)
        if m:
            beta = float(m.group(1))
        elif n == "clare_v7_dea_goal_seed7":
            beta = 2.0                     # 접미사 없는 기본 실행
        else:
            continue
        R, nt = read_matrix(d)
        if not R:
            continue
        filled = sum(1 for r in R for v in r if v is not None)
        if filled < nt * (nt + 1) // 2:    # 미완주는 넣지 않는다
            continue
        mm = metrics(R, nt)
        rows[beta] = (mm["AvgSR"], mm["FWT"], mm["NBT"], mm["AUC"])
    for b, v in EXTRA.items():
        rows.setdefault(b, v)
    return [(b, *rows[b]) for b in sorted(rows)]


def end_labels(fig, items, min_px=13.0):
    """선 끝 직접 레이블을 겹치지 않게 배치한다.

    색만으로 계열을 구분하지 않게 하고, 대비가 낮은 색(aqua)의 접근성
    경고도 이걸로 해소한다. items = [(ax, x, y, text, color), ...].

    ★ 두 축을 쓰면 서로 다른 눈금의 끝점이 화면상 같은 높이에 올 수 있다
      (AvgSR 88.8 과 NBT 2.0 이 실제로 겹쳤다). 그래서 데이터 좌표가 아니라
      **픽셀 좌표**로 모아서 min_px 간격이 되도록 위아래로 민다.
    """
    fig.canvas.draw()                       # 변환을 쓰려면 한 번 그려야 한다
    pts = []
    for ax, x, y, text, color in items:
        _, py = ax.transData.transform((x, y))
        pts.append([py, ax, x, y, text, color])
    pts.sort(key=lambda r: r[0])
    for i in range(1, len(pts)):
        gap = pts[i][0] - pts[i - 1][0]
        if gap < min_px:
            pts[i][0] = pts[i - 1][0] + min_px
    for py, ax, x, y, text, color in pts:
        _, y0 = ax.transData.transform((x, y))
        ax.annotate(text, xy=(x, y), xytext=(7, py - y0),
                    textcoords="offset points", color=color, fontsize=9,
                    fontweight="medium", va="center", ha="left", clip_on=False)


def draw(rows, panels: bool, out_stem: Path):
    beta = [r[0] for r in rows]
    avg, fwt, nbt, auc = ([r[i] for r in rows] for i in (1, 2, 3, 4))
    mk = dict(marker="o", markersize=5.5, linewidth=2.0,
              markeredgecolor="white", markeredgewidth=1.0)

    if panels:
        fig, (ax, bx) = plt.subplots(
            2, 1, figsize=(7.6, 5.6), sharex=True,
            gridspec_kw=dict(height_ratios=[2.1, 1.0], hspace=0.12))
    else:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        bx = ax.twinx()

    tips = []
    for ys, c, lab in ((avg, C_AVG, "AvgSR"), (fwt, C_FWT, "FWT↑"),
                       (auc, C_AUC, "AUC↑")):
        ax.plot(beta, ys, color=c, label=lab, **mk)
        tips.append((ax, beta[-1], ys[-1], lab, c))

    nbt_lab = "NBT↓" if panels else "NBT↓ (right axis)"
    bx.plot(beta, nbt, color=C_NBT, linestyle="--", label=nbt_lab,
            marker="s", markersize=5.5, linewidth=2.0,
            markeredgecolor="white", markeredgewidth=1.0)
    tips.append((bx, beta[-1], nbt[-1], "NBT↓", C_NBT))
    bx.axhline(0, color=C_NBT, linewidth=0.8, alpha=0.35, zorder=0)

    ax.set_ylabel("Success rate (%)", color=INK, fontsize=10)
    bx.set_ylabel("NBT (%p)", color=C_NBT, fontsize=10)
    bx.tick_params(axis="y", colors=C_NBT, labelsize=9)
    for s in ("top",) + (() if panels else ("left",)):
        bx.spines[s].set_visible(False)
    if not panels:
        bx.spines["right"].set_color(C_NBT)

    (bx if panels else ax).set_xlabel("β  (direction-term strength)",
                                      color=INK, fontsize=10)
    for a in (ax, bx):
        a.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
        a.set_axisbelow(True)
        a.tick_params(colors=INK2, labelsize=9)
        for s in ("top", "right") if (panels or a is ax) else ():
            if not (a is ax and s == "right" and not panels):
                a.spines[s].set_visible(False)
        a.spines["left"].set_color(GRID)
        a.spines["bottom"].set_color(GRID)
    ax.set_xticks(beta)
    ax.set_xticklabels([f"{b:g}" for b in beta])
    ax.set_xlim(min(beta) - 0.25, max(beta) + 0.75)

    end_labels(fig, tips)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = bx.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", frameon=False,
              fontsize=9, ncol=2, labelcolor=INK2)

    fig.suptitle("Action dea ablation", x=0.055, y=0.975, ha="left",
                 fontsize=13, fontweight="semibold", color=INK)
    ax.set_title("V7 dea (action space) · libero_goal · seed 7 · "
                 "10,000 step/task · 50 rollout/cell",
                 loc="left", fontsize=9, color=INK2, pad=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    for ext in ("png", "pdf"):
        p = out_stem.with_suffix("." + ext)
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"  -> {p.relative_to(REPO)}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", action="store_true",
                    help="이중 축 대신 위아래 두 패널로 그린다(더 정확한 판본)")
    a = ap.parse_args()
    rows = collect()
    print(f"{'beta':>6}{'AvgSR':>8}{'FWT':>8}{'NBT':>8}{'AUC':>8}")
    for r in rows:
        print(f"{r[0]:6g}{r[1]:8.1f}{r[2]:8.1f}{r[3]:+8.1f}{r[4]:8.1f}")
    print(f"총 {len(rows)} 점")
    stem = RES / ("Action_dea_ablation_panels" if a.panels else "Action_dea_ablation")
    draw(rows, a.panels, stem)


if __name__ == "__main__":
    main()
