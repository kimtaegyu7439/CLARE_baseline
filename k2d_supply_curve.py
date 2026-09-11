#!/usr/bin/env python
"""K2d — 실제 수송 출력 b_1 의 방향별 공급률.

K2c 는 부분공간 사영으로 잰 **선형 프록시**였다. 여기서는 K1 의 실제 파이프라인
(공유기저 사영 -> 분위수 CDF 사상 -> clamp/floor -> 복원)을 그대로 돌려 나온
b_1 이 task1 의 각 고유방향으로 실제 task1 분포 대비 얼마나 퍼지는지 잰다.

    v_i, λ_i   (task1, τ) 중심화 공분산의 고유쌍 (내림차순), p_i = λ_i/Σλ
    supply_i   = Var(v_iᵀ b_1) / λ_i
    shift_i    = (v_iᵀ(mean(b_1) − mean(o_1)))² / λ_i        (json 전용)

★ 공간: k1.K1Anchor.transport 의 반환은 raw 3072-d 다(b.reshape(B,-1,768)).
  따라서 v_i 도 raw 3072-d 에서 잡는다. w-공간이 아니다.

수송 코드는 import 해서 그대로 쓴다. R13 쪽은 R10.loss 안에 두 줄로 박혀 있어
뽑아 쓸 함수가 없어서, R13 이 저장한 mu/sigma 로 같은 식을 쓴다.
"""
from __future__ import annotations

import argparse
import inspect
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
import k1 as K1MOD
import R10 as R10MOD


def eig_desc_raw(Xc: torch.Tensor, kmax: int):
    """raw 공간 공분산 고유쌍. Gram(n x n) 으로 푼다. (V (k,d), lam (k,))."""
    n = Xc.shape[0]
    G = (Xc @ Xc.T).double()
    ev, U = torch.linalg.eigh(G)
    ev = ev.flip(0).clamp_min(0.0); U = U.flip(1)
    k = min(kmax, n - 1)
    s = ev[:k].sqrt().clamp_min(1e-8)
    V = (U[:, :k].T.float() @ Xc) / s[:, None].float()
    return V.contiguous(), (ev[:k] / max(1, n - 1)).float(), float(ev.sum() / max(1, n - 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--stage", type=int, default=6, help="이 stage 의 source 태스크를 쓴다")
    ap.add_argument("--tgt", type=int, default=1, help="수송 목적지 태스크 j")
    ap.add_argument("--k1_dir", default="results/K1_spatial_10task")
    ap.add_argument("--r13_dir", default="results/R13_10task")
    ap.add_argument("--k2c_json", default=None,
                    help="기본은 results/K2c/coverage_w_task{tgt}_src{src}.json")
    ap.add_argument("--cache", default="results/K0/emb_cache")
    ap.add_argument("--cap", type=int, default=500, help="bin 당 source 프레임")
    ap.add_argument("--n_dir", type=int, default=200, help="N = min(n_dir, 유효 rank)")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/K2d")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    k1_dir, r13_dir = Path(a.k1_dir), Path(a.r13_dir)
    cfg = json.loads((k1_dir / "k1_config.json").read_text())
    nb, src, tgt = a.n_bins, a.stage, a.tgt

    print("[K2d] 사용한 코드 경로")
    print(f"  transport : {inspect.getsourcefile(K1MOD.K1Anchor.transport)}"
          f":{inspect.getsourcelines(K1MOD.K1Anchor.transport)[1]}"
          f"  (k1.K1Anchor.transport)")
    print(f"  rotate    : {inspect.getsourcefile(K1MOD.K1Anchor.rotate)}"
          f":{inspect.getsourcelines(K1MOD.K1Anchor.rotate)[1]}")
    print(f"  R13 샘플  : {inspect.getsourcefile(R10MOD.R10Anchor.loss)}"
          f":{inspect.getsourcelines(R10MOD.R10Anchor.loss)[1]}"
          f"  (R10.loss 내 sample_z 분기 — 뽑아 쓸 함수가 없어 같은 식을 쓴다)")
    print(f"  config    : {k1_dir/'k1_config.json'}  (r={cfg['r']}, Q={cfg['Q']}, "
          f"marginal={cfg['marginal']}, basis={cfg['basis']})")
    print(f"  기저      : {k1_dir/'shared_basis.pt'}")
    print(f"  K1 표     : {k1_dir/'stats'}/task{{0..{src}}}.pt   (stage {src} 시점 로드분)")
    print(f"  R13 통계  : {r13_dir/'stats'}/task{tgt}.pt")
    print(f"  임베딩    : {a.cache}/{a.suite}_task{{{tgt},{src}}}.pt")

    # ── 앵커 복원 (stage 6 시점) ────────────────────────────────────────────
    ns = argparse.Namespace(
        out_dir=str(out / "_anchor"), n_bins=nb, lambda_level=cfg["lambda_level"],
        anchor_norm=cfg["anchor_norm"], chunk_backward=False, stats_batches=0,
        stats_workers=0, log_every_anchor=10 ** 9, rho=0.0, warmup_steps=0,
        quantiles=cfg["Q"], rank=cfg["r"], marginal=cfg["marginal"],
        basis=cfg["basis"], iid_sample=cfg["iid_sample"])
    anchor = K1MOD.K1Anchor(ns)
    b = torch.load(k1_dir / "shared_basis.pt")
    anchor.W = None if b["W"] is None else b["W"].float()
    anchor.c0 = b["c0"].float(); anchor.r = b["r"]

    def table(j):
        d = torch.load(k1_dir / "stats" / f"task{j}.pt")
        t = {"qtab": d["qtab"].float(), "m_perp": d["m_perp"].float()}
        t["med"] = K1MOD.quant_at(t["qtab"], 0.5)
        t["s"] = (K1MOD.quant_at(t["qtab"], 0.841) - K1MOD.quant_at(t["qtab"], 0.159)) / 2
        t["s_floor"] = float(0.1 * t["s"].flatten().median())
        return t

    anchor.cur = table(src)                       # stage 6 의 현재 태스크
    anchor.stats = {j: table(j) for j in range(src)}   # 그 시점에 들고 있던 과거 표

    d13 = torch.load(r13_dir / "stats" / f"task{tgt}.pt")
    mu13 = d13["mu"].float().reshape(nb, -1)
    sg13 = d13["sigma"].float().reshape(nb, -1)
    fl13 = float(d13["sigma_floor"])

    emb = {}
    for k in (tgt, src):
        d = torch.load(Path(a.cache) / f"{a.suite}_task{k}.pt")
        emb[k] = (d["X"].float(), d["T"].long())

    k2c_p = Path(a.k2c_json) if a.k2c_json else \
        Path("results/K2c") / f"coverage_w_task{tgt}_src{src}.json"
    k2c = json.loads(k2c_p.read_text()) if k2c_p.exists() else None
    print(f"  K2c 프록시: {k2c_p}  ({'로드됨' if k2c else '없음 — 배경선 생략'})")

    print(f"\n[K2d] 공간 = raw 3072-d  (transport 반환이 (B,4,768) 이라 그 공간에서 잰다)")
    print(f"{'bin':>4}{'n_src':>7}{'n_tgt':>7}{'clamp':>8}"
          f"{'supply med K1':>15}{'supply med R13':>16}{'top10 Σp':>10}")

    rec = {"space": "raw3072", "stage": src, "tgt": tgt, "suite": a.suite,
           "r": cfg["r"], "Q": cfg["Q"], "n_bins": nb, "cap": a.cap, "bins": {}}
    curves = []
    for t in range(nb):
        X1, T1 = emb[tgt]; X6, T6 = emb[src]
        i1 = torch.nonzero(T1 == t, as_tuple=True)[0]
        i6 = torch.nonzero(T6 == t, as_tuple=True)[0]
        if len(i6) > a.cap:
            i6 = i6[torch.from_numpy(rng.choice(len(i6), a.cap, replace=False))]
        o1 = X1[i1]
        o6 = X6[i6]
        n = o6.shape[0]

        # ── 실제 K1 파이프라인 ─────────────────────────────────────────────
        tau = torch.full((n,), t, dtype=torch.long)
        with torch.no_grad():
            w, res = anchor.rotate(o6.reshape(n, -1, 768))
            b1 = anchor.transport(w, res, tau, tgt).reshape(n, -1)
        clamp = anchor.clamp_frac

        # ── R13 방식 (R10.py:249, 281 과 같은 식) ─────────────────────────
        z = torch.randn(n, o1.shape[1]).clamp_(-3.0, 3.0)
        s1 = mu13[t] + sg13[t].clamp_min(fl13) * z

        # ── task1 기준계 ───────────────────────────────────────────────────
        m1 = o1.mean(0, keepdim=True)
        V, lam, _ = eig_desc_raw(o1 - m1, a.n_dir)
        p = lam / lam.sum().clamp_min(1e-12)

        def proj_stats(Y):
            Pj = (Y - m1) @ V.T                       # (n, N)
            return Pj.var(0, unbiased=True), Pj.mean(0)

        vb, mb = proj_stats(b1)
        vs, ms = proj_stats(s1)
        lam_c = lam.clamp_min(1e-12)
        sup1 = (vb / lam_c)
        sup3 = (vs / lam_c)
        sh1 = (mb ** 2) / lam_c
        sh3 = (ms ** 2) / lam_c

        e = {"n_src": int(n), "n_tgt": int(o1.shape[0]), "N": int(V.shape[0]),
             "clamp": float(clamp), "p": p.tolist(),
             "supply": sup1.tolist(), "supply_R13": sup3.tolist(),
             "shift": sh1.tolist(), "shift_R13": sh3.tolist()}
        if k2c:
            e["k2c_cov90_w"] = k2c["bins"][str(t)]["cov90"][:V.shape[0]]
            e["k2c_cov90_raw"] = k2c["bins"][str(t)]["raw"]["cov90"][:V.shape[0]]
        rec["bins"][str(t)] = e
        curves.append((sup1.numpy(), sup3.numpy(), p.numpy(),
                       np.array(e.get("k2c_cov90_raw", []))))
        print(f"{t:>4}{n:>7}{o1.shape[0]:>7}{clamp*100:7.1f}%"
              f"{float(sup1.median()):15.3f}{float(sup3.median()):16.3f}"
              f"{float(p[:10].sum()):10.3f}", flush=True)

    # ── 그림 ────────────────────────────────────────────────────────────────
    N = min(len(c[0]) for c in curves)
    S1 = np.stack([c[0][:N] for c in curves])
    S3 = np.stack([c[1][:N] for c in curves])
    Pv = np.stack([c[2][:N] for c in curves])
    CV = np.stack([c[3][:N] for c in curves]) if len(curves[0][3]) >= N else None
    x = np.arange(1, N + 1)
    norm = LogNorm(vmin=max(float(np.min(Pv[Pv > 0])), 1e-8), vmax=float(Pv.max()))

    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    if CV is not None:
        ax.plot(x, CV.mean(0), "-", color="0.78", lw=1.2, zorder=1,
                label="linear proxy (K2c)")
    for bi in range(S1.shape[0]):
        seg = np.stack([np.column_stack([x[:-1], S1[bi, :-1]]),
                        np.column_stack([x[1:], S1[bi, 1:]])], axis=1)
        lc = LineCollection(seg, cmap="viridis", norm=norm, linewidths=0.8, alpha=0.30)
        lc.set_array(Pv[bi, :-1]); ax.add_collection(lc)
    sm = S1.mean(0)
    seg = np.stack([np.column_stack([x[:-1], sm[:-1]]),
                    np.column_stack([x[1:], sm[1:]])], axis=1)
    lc = LineCollection(seg, cmap="viridis", norm=norm, linewidths=3.0)
    lc.set_array(Pv[:, :-1].mean(0)); ax.add_collection(lc)
    ax.plot(x, S3.mean(0), "--", color="black", lw=1.8, zorder=4, label="R13 (bin mean)")
    ax.axhline(1.0, color="0.7", lw=0.8, zorder=0)
    ax.set_xlim(1, N)
    ax.set_ylim(0, float(max(S1.max(), S3.max())) * 1.05)
    ax.set_xlabel(f"task{tgt} eigendirection rank $i$")
    ax.set_ylabel(r"supply$_i$ = $\mathrm{Var}(v_i^{\top}b_1)\,/\,\lambda_i$")
    ax.set_title(f"Directional supply ratio of transported b{tgt} vs. real task{tgt} "
                 f"(stage-{src} source: task{src})", fontsize=11.5)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
    cb.set_label(f"task{tgt} variance share")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout()
    tag = f"task{tgt}_src{src}"
    fp = out / f"fig_supply_{tag}.png"
    fig.savefig(fp, dpi=300); plt.close(fig)

    jp = out / f"supply_{tag}.json"
    jp.write_text(json.dumps(rec, ensure_ascii=False))
    print(f"\nsaved -> {fp}")
    print(f"saved -> {jp}")


if __name__ == "__main__":
    main()
