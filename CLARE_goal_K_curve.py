#!/usr/bin/env python
"""코드북 셀 수 K 에 따른 FWT / NBT / AUC — 가로 3 패널.

    python CLARE_goal_K_curve.py
    python CLARE_goal_K_curve.py --ks 70,96,100,120,140 --seeds 42

읽는 것  results/clare_v4_goal_k<K>_seed<seed>/sr_matrix.csv
         results/clare_v4_goal_seed<seed>/sr_matrix.csv        K=96 (스윕 이전 런이라 이름에 k 가 없다)

행렬·지표 정의는 clare_v4_report 와 공유한다 — 같은 셋팅의 표와 그림이 서로 다른
계산을 쓰는 일이 없게 하려는 것이다.

★ 완주한 런만 그린다. 반쯤 찬 행렬의 지표는 다른 K 와 **같은 것을 재고 있지 않다**
  (AvgSR 은 마지막 완성 행 기준, NBT 는 이후 스테이지 수가 적어 작게 나온다).
  빠진 K 는 그리지 않고 표준출력에 적는다.

★ NBT 는 낮을수록 좋다. 축 방향은 뒤집지 않되 제목에 ↓ 를 달아 혼동을 막는다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from clare_v4_report import metrics, read_matrix   # noqa: E402

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
# (지표 키, 패널 제목, 높을수록 좋은가)
# (지표 키, 패널 제목, 높을수록 좋은가). 제목은 지표 이름 + 방향 화살표만 둔다.
PANELS = (("FWT", "FWT ↑", True),
          ("NBT", "NBT ↓", False),
          ("AUC", "AUC ↑", True))
SEED_COL = {7: "#2ca089", 42: "#7a5bb5", 1000: "#d9722c"}


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ks", default="70,96,100,120,140")
    p.add_argument("--seeds", default="42")
    p.add_argument("--base_k", type=int, default=96,
                   help="폴더 이름에 k 가 없는 기존 런의 K")
    p.add_argument("--out", default="results/Clare_setting_libero_goal2_K")
    p.add_argument("--font_scale", type=float, default=1.0)
    return p.parse_args()


def rundir(k: int, seed: int, base_k: int) -> Path:
    return RES / (f"clare_v4_goal_seed{seed}" if k == base_k
                  else f"clare_v4_goal_k{k}_seed{seed}")


def main():
    a = parse()
    ks = [int(x) for x in a.ks.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    fs = a.font_scale
    plt.rcParams.update({"font.size": 10.5 * fs, "axes.labelsize": 11 * fs,
                         "xtick.labelsize": 10 * fs, "ytick.labelsize": 10 * fs,
                         "axes.titlesize": 11.5 * fs})

    # data[seed][key] = {K: 값}.  완주하지 않은 런은 버린다(위 설명 참고).
    data: dict = {s: {kk: {} for kk, _t, _g in PANELS} for s in seeds}
    skipped: list[str] = []
    for s in seeds:
        for k in ks:
            d = rundir(k, s, a.base_k)
            R, n = read_matrix(d)
            if not R:
                skipped.append(f"K={k} seed{s}: 결과 없음")
                continue
            filled = sum(1 for r in R for c in r if c is not None)
            if filled < n * (n + 1) // 2:
                skipped.append(f"K={k} seed{s}: 진행 중 {filled}/{n*(n+1)//2}칸")
                continue
            m = metrics(R, n)
            for kk, _t, _g in PANELS:
                if isinstance(m.get(kk), (int, float)):
                    data[s][kk][k] = m[kk]

    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.0))
    for ax, (key, title, higher) in zip(axs, PANELS):
        for s in seeds:
            xy = sorted(data[s][key].items())
            if not xy:
                continue
            x = [p[0] for p in xy]
            y = [p[1] for p in xy]
            col = SEED_COL.get(s, "0.4")
            ax.plot(x, y, "-o", lw=1.9, ms=6, color=col, mec="white", mew=1.1,
                    zorder=4, label=f"seed {s}")
            # 최적 K 를 표시 — NBT 만 최소가 좋다
            bi = int(np.argmax(y) if higher else np.argmin(y))
            ax.scatter([x[bi]], [y[bi]], s=150, facecolors="none",
                       edgecolors=col, lw=1.6, zorder=5)
        ax.set_xticks(ks)
        ax.set_xlabel("TSM K")
        ax.set_title(title, pad=8)
        ax.set_ylabel(f"{key} (%)")
        ax.grid(alpha=.18)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("0.7"); ax.spines["bottom"].set_color("0.7")
        ax.tick_params(colors="0.35", length=3)
    if len(seeds) > 1:
        axs[0].legend(frameon=False, fontsize=9.5 * fs, loc="best")

    # 전체 제목은 두지 않는다 — 패널 제목(FWT↑/NBT↓/AUC↑)과 축 이름으로 충분하고,
    # 캡션이 그 역할을 맡는다. rect 도 원래대로 되돌려 위 여백을 없앤다.
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"saved -> {out}.{ext}")

    print()
    for s in seeds:
        print(f"  seed {s}")
        print("    " + "K".ljust(8) + "".join(f"{kk:>10}" for kk, _t, _g in PANELS))
        for k in ks:
            row = f"    {k:<8}"
            for kk, _t, _g in PANELS:
                v = data[s][kk].get(k)
                row += f"{'-':>10}" if v is None else f"{v:>10.1f}"
            print(row)
    for line in skipped:
        print("  (제외) " + line)


if __name__ == "__main__":
    main()
