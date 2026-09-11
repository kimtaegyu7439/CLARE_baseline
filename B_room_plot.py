#!/usr/bin/env python
"""빈 자리 가설 ④ — 페널티가 있을 때만 '채워짐'이 손해가 되는가."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
R = [json.loads(l) for l in open(REPO / "results/B_room/results.jsonl")]
F = json.load(open(REPO / "results/B_fill/probe.json"))["rows"]
S = sorted({r["start_steps"] for r in R})
C = {0.0: "#b0413e", 3.0: "#2f6db5"}
LB = {0.0: "$\\lambda=0$  (no penalty)", 3.0: "$\\lambda=3$  (anchor)"}


def g(lam, key):
    return [next(r["sr_after"][key] for r in R
                 if r["start_steps"] == s and r["lambda_anchor"] == lam) for s in S]


fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
fig.subplots_adjust(left=0.058, right=0.985, top=0.80, bottom=0.145, wspace=0.30)

ax = axes[0]
for lam in (0.0, 3.0):
    ax.plot(S, g(lam, "task1"), "o-", lw=2.4, ms=8, color=C[lam], label=LB[lam])
ax.set_xscale("log"); ax.set_xticks(S); ax.set_xticklabels([f"{s:,}" for s in S])
ax.set_xlabel("steps of task-0 training before task 1\n(= how filled the field is)")
ax.set_ylabel("task 1 SR after learning it")
ax.set_title("Acquisition — the penalty flips the sign", fontsize=11.5)
ax.set_ylim(70, 100); ax.legend(fontsize=9.5, frameon=False, loc="lower left")
ax.annotate("", xy=(S[-1], 95), xytext=(S[0], 80),
            arrowprops=dict(arrowstyle="->", color=C[0.0], lw=1.1, alpha=.45))
ax.annotate("", xy=(S[-1], 85), xytext=(S[0], 95),
            arrowprops=dict(arrowstyle="->", color=C[3.0], lw=1.1, alpha=.45))
ax.text(3400, 78, "free to overwrite:\nmore filling even helps", fontsize=8.8, color=C[0.0])
ax.text(3400, 96.4, "must pay $\\lambda\\hat G^2$ to move:\nmore filling hurts",
        fontsize=8.8, color=C[3.0])

ax = axes[1]
for lam in (0.0, 3.0):
    ax.plot(S, g(lam, "task0"), "o-", lw=2.4, ms=8, color=C[lam], label=LB[lam])
ax.set_xscale("log"); ax.set_xticks(S); ax.set_xticklabels([f"{s:,}" for s in S])
ax.set_xlabel("steps of task-0 training before task 1")
ax.set_ylabel("task 0 SR retained")
ax.set_title("Retention — the mirror image", fontsize=11.5)
ax.set_ylim(0, 105); ax.legend(fontsize=9.5, frameon=False, loc="center right")

ax = axes[2]
fs = [r["step"] for r in F]; gg = [r["conflict"] for r in F]
ax.plot(fs, [x ** 2 for x in gg], "o-", lw=2.2, ms=5, color="#7a4fa3")
ax.set_xlabel("steps of task-0 training")
ax.set_ylabel("$\\hat G^{2}$   (what $\\lambda$ multiplies)")
ax.set_title("The price of moving, measured", fontsize=11.5)
for s in S:
    if s in fs:
        v = gg[fs.index(s)] ** 2
        ax.axvline(s, color="k", lw=.6, ls=":", alpha=.5)
        ax.annotate(f"{v:.2f}", xy=(s, v), xytext=(s * 1.04, v - 0.045), fontsize=9)
r0, r1 = gg[fs.index(S[0])] ** 2, gg[fs.index(S[-1])] ** 2
ax.text(0.42, 0.12, f"×{r1/r0:.1f} from {S[0]:,} to {S[-1]:,} steps",
        transform=ax.transAxes, fontsize=10, color="#7a4fa3")

fig.suptitle("Does a filled velocity field block the next task?   "
             "Only when a penalty forbids overwriting it.\n"
             "task 0 → task 1, libero_spatial, 5,000 steps on task 1, 20 rollouts/cell",
             fontsize=12.3, y=0.965)
p = REPO / "results/B_room/room.png"
fig.savefig(p, dpi=155)
print("saved ->", p)
print(f"\n{'출발':>7}{'λ':>4}{'task1 습득':>11}{'task0 보존':>11}{'평균':>8}")
for s in S:
    for lam in (0.0, 3.0):
        t1 = next(r["sr_after"]["task1"] for r in R if r["start_steps"] == s and r["lambda_anchor"] == lam)
        t0 = next(r["sr_after"]["task0"] for r in R if r["start_steps"] == s and r["lambda_anchor"] == lam)
        print(f"{s:>7}{lam:>4.0f}{t1:>11.0f}{t0:>11.0f}{(t0+t1)/2:>8.1f}")
