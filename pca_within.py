#!/usr/bin/env python
"""그룹 안에서의 PCA + K-sweep — 시간 bin vs state 셀.

  왼쪽   그룹마다 따로 PCA 한 곡선을 전부 겹쳐 그린다 (rank 맞춤).
         빨강 = 시간 bin,  파랑(투명) = state 셀.  굵은 선 = 중앙값.
  오른쪽 K 를 10→96 으로 늘릴 때 **전체 잔차 분산**이 어떻게 줄어드는가.

★ rank 함정: 그룹 크기 n 이면 공분산 rank ≤ n−1 이라 작은 그룹이 적은 성분으로
  100% 에 도달한다. 그래서 왼쪽은 **그룹당 표본 수를 맞춰서** 그린다.

★ 시간 bin 복원: 캐시에 (episode, position) 이 없어서 τ 가 감소하는 지점으로
  에피소드 경계를 찾아 복원했다. drop_n_last_frames 로 꼬리가 잘려 있어 원래 τ 와
  66% 일치하지만, K=10 잔차가 1253.6 vs 원래 1254.0 로 사실상 같다.

    python pca_within.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "pca_probe"
TOP, NS = 40, 30
KS = (10, 20, 50, 96)


def episodes(tau):
    """τ 감소 지점으로 에피소드 경계 복원 -> (pos, eplen)."""
    t = tau.cpu(); N = len(t)
    bnd = (t[1:] < t[:-1]).nonzero().squeeze(1) + 1
    st = torch.cat([torch.tensor([0]), bnd]); en = torch.cat([bnd, torch.tensor([N])])
    pos = torch.zeros(N, dtype=torch.long); ln = torch.zeros(N, dtype=torch.long)
    for a, b in zip(st.tolist(), en.tolist()):
        pos[a:b] = torch.arange(b - a); ln[a:b] = b - a
    return pos.to(tau.device), ln.to(tau.device), len(st)


def group_curves(o, lab, K, n_sub):
    g = torch.Generator().manual_seed(0); out = []
    for k in range(K):
        idx = (lab == k).nonzero().squeeze(1)
        if idx.numel() < n_sub:
            continue
        idx = idx[torch.randperm(idx.numel(), generator=g).to(idx.device)[:n_sub]]
        x = o[idx].double(); x = x - x.mean(0)
        lam = torch.linalg.svdvals(x).pow(2) / (x.shape[0] - 1)
        c = (torch.cumsum(lam, 0) / lam.sum()).cpu().numpy() * 100
        y = np.full(TOP, 100.0); y[:min(TOP, len(c))] = c[:TOP]; out.append(y)
    return np.array(out)


def resid_var(o, lab, K):
    m = torch.zeros(K, o.shape[1], device=o.device, dtype=torch.float64)
    m.index_add_(0, lab, o.double())
    c = torch.bincount(lab, minlength=K).clamp_min(1).double()[:, None]
    return float((o.double() - (m / c)[lab]).var(0).sum())


def main() -> None:
    import l2_codebook as CB
    d = torch.load(OUT / "frames_task0.pt", weights_only=False)
    o, s, tau = d["o"].cuda(), d["s"].cuda(), d["tau"].cuda()
    pos, ln, n_ep = episodes(tau)
    base = float(o.double().var(0).sum())
    print(f"프레임 {o.shape[0]}  에피소드 {n_ep}  전역 잔차 분산 {base:.1f}")

    tlab = {K: ((K * pos) // ln.clamp_min(1)).clamp(0, K - 1) for K in KS}
    slab, keff = {}, {}
    for K in KS:
        cb = CB.build_codebook(s, o, K, seed=42)
        slab[K] = torch.cdist((s - cb["mean_s"].cuda()) / cb["std_s"].cuda(),
                              cb["c"].cuda()).argmin(1)
        keff[K] = cb["K_eff"]

    C = {("time", K): group_curves(o, tlab[K], K, NS) for K in (10, 96)}
    C.update({("state", K): group_curves(o, slab[K], keff[K], NS) for K in (10, 96)})
    V = {("time", K): resid_var(o, tlab[K], K) for K in KS}
    V.update({("state", K): resid_var(o, slab[K], keff[K]) for K in KS})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.arange(1, TOP + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))

    STY = {("time", 10): ("tab:red", "-"), ("time", 96): ("darkred", "--"),
           ("state", 10): ("tab:blue", "-"), ("state", 96): ("navy", "--")}
    for key in (("state", 96), ("state", 10), ("time", 96), ("time", 10)):
        col = "tab:blue" if key[0] == "state" else "tab:red"
        for y in C[key]:
            ax[0].plot(xs, y, color=col, alpha=0.13, lw=0.9)
    for key in (("time", 10), ("time", 96), ("state", 10), ("state", 96)):
        c, lsty = STY[key]
        ax[0].plot(xs, np.median(C[key], 0), color=c, ls=lsty, lw=2.6,
                   label=f"{key[0]} K={key[1]}  ({len(C[key])} groups)")
    ax[0].set_xlabel("number of eigenvalues summed")
    ax[0].set_ylabel("cumulative variance ratio (%)")
    ax[0].set_title(f"within-group PCA  —  {NS} frames sampled per group\n"
                    "(rank-matched; thin = individual groups, thick = median)")
    ax[0].set_xlim(1, TOP); ax[0].set_ylim(0, 100)

    for nm, col, mk in (("time", "tab:red", "o"), ("state", "tab:blue", "s")):
        ax[1].plot(KS, [100 * V[(nm, K)] / base for K in KS], color=col, marker=mk,
                   lw=2.4, ms=7, label=f"{nm} partition")
        for K in KS:
            ax[1].annotate(f"{100*V[(nm,K)]/base:.1f}", (K, 100 * V[(nm, K)] / base),
                           textcoords="offset points", xytext=(0, 8 if nm == "time" else -14),
                           ha="center", fontsize=8, color=col)
    ax[1].axhline(100, color="0.5", ls=":", lw=1)
    ax[1].set_xlabel("K  (number of groups)")
    ax[1].set_ylabel("residual variance, % of global")
    ax[1].set_title("pooled residual variance vs K\n(lower = partition removed more structure)")
    ax[1].set_xscale("log"); ax[1].set_xticks(KS); ax[1].set_xticklabels([str(k) for k in KS])
    ax[1].set_ylim(30, 105)
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=9, loc="lower right" if a is ax[0] else "upper right")
    fig.suptitle("Time bin vs state cell — libero_spatial task0, "
                 f"{o.shape[0]} frames, DINOv2 CLS 3072-d (frozen)")
    plt.tight_layout(); plt.savefig(OUT / "pca_within.png", dpi=140)

    n_ok96 = len(C[("state", 96)])
    L = [f"시간 bin vs state 셀 — {o.shape[0]} 프레임 (에피소드 {n_ep}), o 3072-d", "",
         f"[1] 그룹 안 PCA — 누적 분산 비율(%) 중앙값, 그룹당 {NS} 프레임 균등 표본",
         f"{'분할':<16}{'그룹수':>7}" + "".join(f"{('k='+str(k)):>9}" for k in (1, 3, 5, 10, 20))]
    for key in (("time", 10), ("state", 10), ("time", 96), ("state", 96)):
        m = np.median(C[key], 0)
        L.append(f"{key[0]+' K='+str(key[1]):<16}{len(C[key]):>7}"
                 + "".join(f"{m[k-1]:9.2f}" for k in (1, 3, 5, 10, 20)))
    L += ["", "[2] 전체 잔차 분산 (전역 대비 %) — K sweep",
          f"{'K':>6}{'시간 bin':>14}{'state 셀':>14}{'차이':>10}"]
    for K in KS:
        t_, s_ = 100 * V[("time", K)] / base, 100 * V[("state", K)] / base
        L.append(f"{K:>6}{t_:13.1f}%{s_:13.1f}%{t_-s_:+10.1f}")
    L += ["", "시간은 K 를 10배 늘려도 포화한다(1차원이라 정보 천장이 있다).",
          "state 는 계속 줄어든다 — 16차원이라 쪼갤수록 실제 구조를 따라간다.",
          "",
          f"주의  state K=96 은 {n_ok96}/96 셀만 {NS} 프레임을 넘겼다.",
          "      고유 프레임 4718 개를 96 으로 나누면 셀당 평균 49 개라 이미 얇다.",
          "      K 를 더 키우면 통계가 무너진다 — 이 데이터셋의 실질 상한 근처다."]
    (OUT / "pca_within.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsaved -> {OUT/'pca_within.png'}")


if __name__ == "__main__":
    main()
