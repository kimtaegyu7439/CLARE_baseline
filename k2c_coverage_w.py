#!/usr/bin/env python
"""K2c — 공유기저 w-공간에서 task1 고유방향이 task6 활성 부분공간에 얼마나 덮이는가.

K1 이 실제로 수송을 수행하는 좌표계는 3072-d raw 가 아니라 공유 basis W_r 로
사영한 r 차원 w-공간이다. 여기서 bin τ 마다

    w   = W_r^T (o_flat − c0)
    v_i = (task1, τ) 중심화 공분산의 고유벡터 (λ_i 내림차순), p_i = λ_i / Σλ
    U   = (task6, τ) 중심화 공분산에서 에너지 90% 를 채우는 상위 m 개
    cov_i(τ) = ‖U(τ)^T v_i‖²  ∈ [0,1]

수치만 낸다. 기준선·판정·대조군 없음.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def eig_desc(Xc: torch.Tensor):
    """중심화 데이터의 공분산 고유쌍. (V (N,d) 내림차순 행, lam (N,))."""
    n, d = Xc.shape
    if d <= n:                                   # w-공간: 공분산을 직접
        Cm = (Xc.T @ Xc).double() / max(1, n - 1)
        ev, U = torch.linalg.eigh(Cm)
        lam = ev.flip(0).clamp_min(0.0)
        V = U.flip(1).T.float()                  # (d, d) 행이 고유벡터
        k = min(d, n - 1)
        return V[:k].contiguous(), lam[:k].float()
    # raw 3072-d: Gram (n x n) 으로 푼다
    G = (Xc @ Xc.T).double()
    ev, U = torch.linalg.eigh(G)
    ev = ev.flip(0).clamp_min(0.0)
    U = U.flip(1)
    k = n - 1
    s = ev[:k].sqrt().clamp_min(1e-8)
    V = (U[:, :k].T.float() @ Xc) / s[:, None].float()
    return V.contiguous(), (ev[:k] / max(1, n - 1)).float()


def active_subspace(V: torch.Tensor, lam: torch.Tensor, frac: float):
    """에너지 frac 을 채우는 상위 m 개. (U (m,d), m)."""
    tot = lam.sum().clamp_min(1e-12)
    m = int((torch.cumsum(lam, 0) / tot < frac).sum()) + 1
    m = min(m, V.shape[0])
    return V[:m], m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--src", type=int, default=6, help="원료 태스크 (수송 source)")
    ap.add_argument("--tgt", type=int, default=1, help="고유방향을 볼 태스크")
    ap.add_argument("--basis", default="results/K0/basis.pt")
    ap.add_argument("--cache", default="results/K0/emb_cache")
    ap.add_argument("--k1_cfg", default="results/K1_spatial_10task/k1_config.json")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--out", default="results/K2c")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.k1_cfg).read_text())
    r = int(cfg["r"])
    b = torch.load(a.basis)
    W = b["W512"].float()[:, :r].contiguous()          # (3072, r)
    c0 = b["c0"].float()
    print(f"[K2c] basis {a.basis}  W512{tuple(b['W512'].shape)} -> W_r{tuple(W.shape)}  "
          f"r={r} (K1 config: {a.k1_cfg})", flush=True)

    emb = {}
    for k in (a.tgt, a.src):
        d = torch.load(Path(a.cache) / f"{a.suite}_task{k}.pt")
        emb[k] = (d["X"].float(), d["T"].long())
        print(f"[K2c] task{k} {tuple(emb[k][0].shape)}", flush=True)

    rec = {"r": r, "suite": a.suite, "src": a.src, "tgt": a.tgt,
           "n_bins": a.n_bins, "bins": {}}
    curves = []
    print(f"\n{'bin':>4}{'n_t'+str(a.tgt):>8}{'n_t'+str(a.src):>8}"
          f"{'m90':>6}{'m80':>6}{'covmin':>9}{'covmed':>9}{'top10 Σp':>10}"
          f"{'|raw m90':>9}{'raw covmed':>12}")
    for t in range(a.n_bins):
        entry = {}
        # ── w-공간 ────────────────────────────────────────────────────────
        ws = {}
        for k in (a.tgt, a.src):
            X, T = emb[k]
            idx = torch.nonzero(T == t, as_tuple=True)[0]
            xw = (X[idx] - c0) @ W
            ws[k] = xw - xw.mean(0, keepdim=True)
        V1, l1 = eig_desc(ws[a.tgt])
        V6, l6 = eig_desc(ws[a.src])
        U90, m90 = active_subspace(V6, l6, 0.90)
        U80, m80 = active_subspace(V6, l6, 0.80)
        p = (l1 / l1.sum().clamp_min(1e-12))
        cov90 = (V1 @ U90.T).pow(2).sum(1).clamp(0, 1)
        cov80 = (V1 @ U80.T).pow(2).sum(1).clamp(0, 1)

        # ── raw 3072 공간 (json 전용) ─────────────────────────────────────
        rw = {}
        for k in (a.tgt, a.src):
            X, T = emb[k]
            idx = torch.nonzero(T == t, as_tuple=True)[0]
            x = X[idx]
            rw[k] = x - x.mean(0, keepdim=True)
        Vr1, lr1 = eig_desc(rw[a.tgt])
        Vr6, lr6 = eig_desc(rw[a.src])
        Ur90, rm90 = active_subspace(Vr6, lr6, 0.90)
        Ur80, rm80 = active_subspace(Vr6, lr6, 0.80)
        pr = (lr1 / lr1.sum().clamp_min(1e-12))
        covr90 = (Vr1 @ Ur90.T).pow(2).sum(1).clamp(0, 1)
        covr80 = (Vr1 @ Ur80.T).pow(2).sum(1).clamp(0, 1)

        entry = {"n_tgt": int(ws[a.tgt].shape[0]), "n_src": int(ws[a.src].shape[0]),
                 "m90": m90, "m80": m80,
                 "p": p.tolist(), "cov90": cov90.tolist(), "cov80": cov80.tolist(),
                 "raw": {"m90": rm90, "m80": rm80, "p": pr.tolist(),
                         "cov90": covr90.tolist(), "cov80": covr80.tolist()}}
        rec["bins"][str(t)] = entry
        curves.append((cov90.numpy(), p.numpy()))
        print(f"{t:>4}{entry['n_tgt']:>8}{entry['n_src']:>8}{m90:>6}{m80:>6}"
              f"{float(cov90.min()):9.3f}{float(cov90.median()):9.3f}"
              f"{float(p[:10].sum()):10.3f}"
              f"{rm90:>9}{float(covr90.median()):12.3f}", flush=True)

    # ── 그림 ─────────────────────────────────────────────────────────────
    N = min(len(c) for c, _ in curves)
    Cv = np.stack([c[:N] for c, _ in curves])          # (bins, N)
    Pv = np.stack([p[:N] for _, p in curves])
    x = np.arange(1, N + 1)
    pmean = Pv.mean(0)
    cmean = Cv.mean(0)
    norm = LogNorm(vmin=max(float(np.min(Pv[Pv > 0])), 1e-8), vmax=float(Pv.max()))

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for bi in range(Cv.shape[0]):
        seg = np.stack([np.column_stack([x[:-1], Cv[bi, :-1]]),
                        np.column_stack([x[1:], Cv[bi, 1:]])], axis=1)
        lc = LineCollection(seg, cmap="viridis", norm=norm, linewidths=0.8, alpha=0.30)
        lc.set_array(Pv[bi, :-1])
        ax.add_collection(lc)
    seg = np.stack([np.column_stack([x[:-1], cmean[:-1]]),
                    np.column_stack([x[1:], cmean[1:]])], axis=1)
    lc = LineCollection(seg, cmap="viridis", norm=norm, linewidths=3.0)
    lc.set_array(pmean[:-1])
    ax.add_collection(lc)
    ax.set_xlim(1, N); ax.set_ylim(0, 1)
    ax.set_xlabel(f"task{a.tgt} eigendirection rank $i$")
    ax.set_ylabel(r"$\mathrm{cov}_i = \|U^{\top}v_i\|^2$")
    ax.set_title(f"Coverage of task{a.tgt} eigendirections (w-space, r={r}) "
                 f"by task{a.src} active subspace (90% energy)", fontsize=11.5)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
    cb.set_label(f"task{a.tgt} variance share (w-space)")
    fig.tight_layout()
    tag = f"task{a.tgt}_src{a.src}"
    fp = out / f"fig_w_coverage_{tag}.png"
    fig.savefig(fp, dpi=300); plt.close(fig)

    jp = out / f"coverage_w_{tag}.json"
    jp.write_text(json.dumps(rec, ensure_ascii=False))
    print(f"\nsaved -> {fp}")
    print(f"saved -> {jp}")


if __name__ == "__main__":
    main()
