#!/usr/bin/env python
"""B_fill_probe 결과 시각화 — 속도장이 현재 태스크로 채워지는 과정."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent
D = json.load(open(REPO / "results/B_fill/probe.json"))
rows = D["rows"]; maps = {int(k): np.array(v) for k, v in D["maps"].items()}
P = np.array(D["P"]); Q = np.array(D["Q"])
steps = [r["step"] for r in rows]

SHOW = [s for s in (1000, 5000, 20000) if s in maps] or sorted(maps)[:: max(1, len(maps) // 3)][:3]

fig = plt.figure(figsize=(15.0, 9.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.40, wspace=0.28,
                      left=0.062, right=0.985, top=0.875, bottom=0.075)

# ── 위: 평면 지도 (기준 + 변화량) ───────────────────────────────────────────
base = SHOW[0]
M0 = maps[base]
ax = fig.add_subplot(gs[0, 0])
v0 = abs(M0).max()
im0 = ax.pcolormesh(Q, P, M0, cmap="RdBu_r", vmin=-v0, vmax=v0, shading="auto")
ax.contour(Q, P, M0, levels=[0.0], colors="k", linewidths=1.6)
ax.plot([0, 0], [0, 1], color="#0b3d91", lw=3.0, solid_capstyle="round")
ax.plot([0, 1], [0, 0], color="#8b1a1a", lw=3.0, solid_capstyle="round")
ax.scatter([0], [0], c="k", s=45, zorder=5)
ax.text(0.03, 1.05, "task 1 path", color="#0b3d91", fontsize=9)
ax.text(1.02, 0.05, "task 0 path", color="#8b1a1a", fontsize=9, ha="right")
ax.text(-0.50, 0.03, "noise $\\varepsilon$", fontsize=9)
ax.set_xlabel("component toward $a_0$   (units of $\\|g_1\\|$)")
ax.set_ylabel("component toward $a_1$")
ax.set_title(f"baseline: after {base:,} steps", fontsize=11)
fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.02,
             label="$\\cos(v,g_0)-\\cos(v,g_1)$")

DIFF = [s for s in SHOW[1:]]
dmax = max(abs(maps[s] - M0).max() for s in DIFF) if DIFF else 1.0
for i, s_ in enumerate(DIFF):
    ax = fig.add_subplot(gs[0, i + 1])
    imd = ax.pcolormesh(Q, P, maps[s_] - M0, cmap="RdBu_r", vmin=-dmax, vmax=dmax,
                        shading="auto")
    ax.contour(Q, P, maps[s_] - M0, levels=[0.0], colors="k", linewidths=1.0, alpha=.6)
    ax.plot([0, 0], [0, 1], color="#0b3d91", lw=3.0, solid_capstyle="round")
    ax.plot([0, 1], [0, 0], color="#8b1a1a", lw=3.0, solid_capstyle="round")
    ax.scatter([0], [0], c="k", s=45, zorder=5)
    ax.set_xlabel("component toward $a_0$")
    ax.set_title(f"change from {base:,} to {s_:,} steps", fontsize=11)
    if i == len(DIFF) - 1:
        fig.colorbar(imd, ax=ax, fraction=0.046, pad=0.02,
                     label="red = swung toward task 0")

# ── 아래 왼쪽: 점유율 ────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
y = [r["claimed"] * 100 for r in rows]
ax.plot(steps, y, "o-", lw=2.3, ms=6, color="#8b1a1a")
ax.fill_between(steps, y, 100, color="#4a7fb5", alpha=0.16)
ax.fill_between(steps, 0, y, color="#8b1a1a", alpha=0.16)
ax.text(steps[len(steps)//2], (100 + y[len(steps)//2]) / 2, "free space\nfor the next task",
        ha="center", va="center", fontsize=9.5, color="#2b5580")
ax.text(steps[len(steps)//2], y[len(steps)//2] / 2, "already claimed\nby task 0",
        ha="center", va="center", fontsize=9.5, color="#7a1414")
ax.set_xlabel("training steps on task 0"); ax.set_ylabel("% of points on task 1's path")
ax.set_ylim(0, 100)
ax.set_title("(1) The current task fills the space", fontsize=11)
ax.annotate(f"{y[0]:.0f}%", xy=(steps[0], y[0]), xytext=(steps[0], y[0]+9), fontsize=10, color="#7a1414")
ax.annotate(f"{y[-1]:.0f}%", xy=(steps[-1], y[-1]), xytext=(steps[-1]-2600, y[-1]+9), fontsize=10, color="#7a1414")

# ── 아래 가운데: 학습 비용 ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.plot(steps, [r["cost"] for r in rows], "o-", lw=2.3, ms=6, color="#2f6db5")
ax.set_xlabel("training steps on task 0")
ax.set_ylabel("$\\|v-g_1\\|/\\|g_1\\|$  on task 1's path")
ax.set_title("(2) How far the field must move to fit task 1", fontsize=11)

# ── 아래 오른쪽: 페널티의 저항 ───────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.plot(steps, [r["conflict"] for r in rows], "o-", lw=2.3, ms=6, color="#7a4fa3")
ax.set_xlabel("training steps on task 0")
ax.set_ylabel("$\\hat G=\\|v(o_1,\\ell_0)-g_1\\|/\\|g_1\\|$")
ax.set_title("(3) What the anchor would hold in place", fontsize=11)
ax.legend(handles=[Line2D([], [], color="none",
                          label="seq-FT: free to overwrite\nB (anchor): pays $\\lambda\\hat G^2$ to move")],
          fontsize=9, frameon=False, loc="lower right", handlelength=0)

fig.suptitle("Flow matching fills a continuous field — longer training leaves less room for the next task\n"
             "(task 0 only, libero_spatial, cosine schedule over 20,000 steps)",
             fontsize=12.5, y=0.965)
p = REPO / "results/B_fill/fill.png"
fig.savefig(p, dpi=155)
print("saved ->", p)
print(f"{'step':>7}{'claimed%':>11}{'cost':>9}{'conflict':>10}")
for r in rows:
    print(f"{r['step']:>7}{r['claimed']*100:>11.1f}{r['cost']:>9.3f}{r['conflict']:>10.3f}")
