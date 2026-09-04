#!/usr/bin/env python
"""x_t 의 태스크 의존성 — 실행 구간 vs 미리보기 구간 vs 전체.

캐시된 행동(results/xt_probe/actions.pt)만 쓴다. GPU 수집 불필요.

  action_delta_indices = [-1, 0, 1, ..., 14]   (modeling_dit_flow_mt.py:230)
    index 0     -> t-1        버림
    index 1..8  -> t ~ t+7    **실행** (n_action_steps=8)
    index 9..15 -> t+8~t+14   미리보기, 버림
  단 flow-matching 학습 손실은 16 스텝 전부에 걸린다.
"""
from __future__ import annotations
import numpy as np, torch
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "xt_probe"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEG = {"executed  idx 1:9  (t..t+7)": slice(1, 9),
       "preview   idx 9:16 (t+8..t+14)": slice(9, 16),
       "all       idx 0:16 (FM loss)": slice(0, 16)}
TS = np.linspace(0, 1, 51)
M = 6000


def stats(src, dst, g):
    i = torch.randint(len(src), (M,), device=DEV, generator=g)
    j = torch.randint(len(dst), (M,), device=DEV, generator=g)
    da = (src[i] - dst[j]).flatten(1)
    ai = src[i].flatten(1)
    eps = torch.randn(M, da.shape[1], device=DEV, generator=g)
    ab, rl = [], []
    for t in TS:
        dx = t * da                                  # 같은 ε -> 상쇄
        xt = (1 - t) * eps + t * ai
        ab.append(float(dx.norm(dim=1).mean()))
        rl.append(100 * float((dx.norm(dim=1) / xt.norm(dim=1).clamp_min(1e-8)).mean()))
    return np.array(ab), np.array(rl)


def main():
    A0 = [x.to(DEV) for x in torch.load(OUT / "actions.pt", weights_only=False)]
    K = len(A0)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9))
    cm = plt.cm.viridis(np.linspace(0, .92, K - 1))
    lines = []
    for col, (nm, sl) in enumerate(SEG.items()):
        A = [x[:, sl] for x in A0]
        g = torch.Generator(device=DEV).manual_seed(0)
        wb, wr = stats(A[0], A[0], g)                     # within task0
        for r, (y0, lab) in enumerate(((wb, "abs"), (wr, "rel"))):
            ax[r, col].plot(TS, y0, "k--", lw=2.6, zorder=5,
                            label="within task0 (baseline)")
        for j in range(1, K):
            ab, rl = stats(A[0], A[j], g)
            ax[0, col].plot(TS, ab, color=cm[j - 1], lw=1.6, label=f"task{j}")
            ax[1, col].plot(TS, rl, color=cm[j - 1], lw=1.6)
            if col == 0:
                lines.append((j, ab, rl))
        ax[0, col].set_title(nm, fontsize=11)
        for r in (0, 1):
            ax[r, col].grid(alpha=.3); ax[r, col].set_xlim(0, 1)
            ax[r, col].set_xlabel("flow-matching time  t")
        ax[1, col].set_ylim(0, None)
    ax[0, 0].set_ylabel(r"$\|x_t^{A}-x_t^{B}\|$   absolute")
    ax[1, 0].set_ylabel(r"$\|x_t^{A}-x_t^{B}\|/\|x_t\|$   (%)")
    ax[0, 0].legend(fontsize=7, ncol=2, loc="upper left")
    ax[1, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Task-specificity of the anchor coordinate $x_t=(1-t)\\varepsilon+t\\,a$\n"
                 "libero_spatial, task0 vs task1..9, 1500 chunks/task, shared $\\varepsilon$",
                 fontsize=12)
    plt.tight_layout(); plt.savefig(OUT / "xt_task_gap.png", dpi=135)

    L = ["x_t 태스크 의존성 — libero_spatial, 태스크당 1500 청크, 같은 ε", "",
         "index 0=t-1(버림)  1~8=실행  9~15=미리보기(버림).  FM 손실은 16개 전부.", ""]
    idx = [int(t * 50) for t in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)]
    for nm, sl in SEG.items():
        A = [x[:, sl] for x in A0]
        g = torch.Generator(device=DEV).manual_seed(0)
        wb, wr = stats(A[0], A[0], g)
        bt = np.mean([stats(A[0], A[j], g)[1] for j in range(1, K)], 0)
        ba = np.mean([stats(A[0], A[j], g)[0] for j in range(1, K)], 0)
        L += [f"[{nm}]",
              f"  {'t':<26}" + "".join(f"{t:>9.1f}" for t in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)),
              f"  {'절대 within':<26}" + "".join(f"{wb[i]:9.3f}" for i in idx),
              f"  {'절대 between(평균)':<26}" + "".join(f"{ba[i]:9.3f}" for i in idx),
              f"  {'상대% within':<26}" + "".join(f"{wr[i]:9.1f}" for i in idx),
              f"  {'상대% between(평균)':<26}" + "".join(f"{bt[i]:9.1f}" for i in idx),
              f"  {'between/within 비':<26}" + "".join(f"{bt[i]/max(wr[i],1e-9):9.2f}" for i in idx), ""]
    (OUT / "xt_task_gap.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"saved -> {OUT/'xt_task_gap.png'}")


if __name__ == "__main__":
    main()
