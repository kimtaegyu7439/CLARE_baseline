#!/usr/bin/env python
"""B_default 결과를 그림으로. 무조건부 필드를 누가 차지하는가."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
rows = json.load(open(REPO / "results/B_default/rows.json"))
K = 4
ARMS = ["seq-FT", "B2λ3", "ER"]
LBL = {"seq-FT": "seq-FT  (no dropout, no anchor)",
       "B2λ3": "B2 $\\lambda$=3  (p_drop=0.1 + anchor)",
       "ER": "ER  (replay)"}
C = {"seq-FT": "#b0413e", "B2λ3": "#2f6db5", "ER": "#3f8f4f"}


def sel(**kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def m(arm, k, j, o=None):
    rs = sel(arm=arm, stage=k, instr=j) if o is None else sel(arm=arm, stage=k, instr=j, obs=o)
    return float(np.mean([r["mean"] for r in rs])) if rs else np.nan


fig = plt.figure(figsize=(14.5, 9.2))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                      left=0.065, right=0.985, top=0.885, bottom=0.075)

# ── A: 팔마다 stage x instruction 히트맵 (관측 출처 전부 평균) ────────────────
for i, arm in enumerate(ARMS):
    ax = fig.add_subplot(gs[0, i])
    M = np.array([[m(arm, k, j) for j in range(K)] for k in range(K)])
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=0.55)
    for k in range(K):
        for j in range(K):
            ax.text(j, k, f"{M[k,j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if M[k, j] < 0.33 else "black")
    ax.set_xticks(range(K)); ax.set_xticklabels([f"$\\ell_{j}$" for j in range(K)])
    ax.set_yticks(range(K)); ax.set_yticklabels([f"stage {k}" for k in range(K)])
    ax.set_title(LBL[arm], fontsize=11, pad=8)
    ax.set_xlabel("instruction fed to the model")
    if i == 0:
        ax.set_ylabel("checkpoint after learning task k")
    for k in range(K):                       # 대각선 = 방금 배운 태스크
        ax.add_patch(plt.Rectangle((k - .5, k - .5), 1, 1, fill=False, ec="red", lw=2.2))
fig.colorbar(im, ax=fig.axes[:3], fraction=0.020, pad=0.012,
             label="$d_j=\\|v(o,\\ell_j)-v(o,\\varnothing)\\|\\,/\\,\\|v^*\\|$")

# ── B: 방금 배운 태스크의 d (빨간 대각선) 를 스테이지별로 ─────────────────────
ax = fig.add_subplot(gs[1, 0])
for arm in ARMS:
    ax.plot(range(K), [m(arm, k, k) for k in range(K)], "o-", lw=2.2, ms=7,
            color=C[arm], label=arm)
ax.axhline(0, color="k", lw=.7, ls=":")
ax.set_xticks(range(K)); ax.set_xlabel("stage k"); ax.set_ylim(-0.02, 0.30)
ax.set_ylabel("$d_k$  (current task vs no-instruction)")
ax.set_title("Does the newest task BECOME the default field?", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.annotate("$d_k\\approx0$: instruction is redundant,\nthe default field IS task $k$",
            xy=(1.5, 0.01), xytext=(0.55, 0.115), fontsize=9, color=C["B2λ3"],
            arrowprops=dict(arrowstyle="->", color=C["B2λ3"], lw=1.3))

# ── C: 관측 출처별 분해 (stage 3) — "거의 모든 지점"인가 국소인가 ─────────────
ax = fig.add_subplot(gs[1, 1])
w = 0.26
for i, arm in enumerate(ARMS):
    ax.bar(np.arange(K) + (i - 1) * w, [m(arm, 3, 3, o) for o in range(K)],
           w, color=C[arm], label=arm)
ax.set_xticks(range(K))
ax.set_xticklabels([f"obs from\ntask {o}" for o in range(K)])
ax.set_ylabel("$d_3$  at stage 3")
ax.set_title("Is it global? $d_3$ by where the observation came from", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.set_ylim(0, 0.30)

# ── D: 지점 단위 분포 — 정말 "거의 모든" 지점인가 ────────────────────────────
ax = fig.add_subplot(gs[1, 2])
for arm in ARMS:
    v = np.concatenate([np.array(r["vals"]) for r in sel(arm=arm, stage=3, instr=3)])
    xs = np.sort(v)
    ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), lw=2.2, color=C[arm], label=arm)
ax.axvline(0.05, color="k", lw=.8, ls="--")
ax.text(0.056, 0.35, "$d_3<0.05$", fontsize=9)
ax.set_xlabel("$d_3$ at an individual point"); ax.set_ylabel("cumulative fraction of points")
ax.set_xlim(0, 0.8); ax.set_ylim(0, 1.02)
ax.set_title("Per-point CDF at stage 3 (all 256 points pooled)", fontsize=11)
ax.legend(fontsize=9, frameon=False, loc="lower right")

fig.suptitle("Who owns the unconditional velocity field?   "
             "$d_j=\\|v(o,\\ell_j)-v(o,\\varnothing)\\|/\\|v^*\\|$   "
             "(libero_spatial, 4 tasks, 5000 steps/task)",
             fontsize=12.5, y=0.962)
p = REPO / "results/B_default/default_field.png"
fig.savefig(p, dpi=160)
print("saved ->", p)

for arm in ARMS:
    v = np.concatenate([np.array(r["vals"]) for r in sel(arm=arm, stage=3, instr=3)])
    print(f"{arm:>7}  stage3 d_3:  평균 {v.mean():.3f}  중앙 {np.median(v):.3f}  "
          f"p90 {np.quantile(v,.9):.3f}   d<0.05 인 지점 비율 {(v<0.05).mean()*100:.1f}%")
