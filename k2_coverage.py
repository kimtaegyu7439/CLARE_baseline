#!/usr/bin/env python
"""K2 — task1 붕괴 원인 판정: 원료 커버리지(H1) vs 좌표 무관 누적(H2). 분석 전용.

배경
  10 태스크에서 K1(공유기저 분위수 수송)의 task1 이 stage 6 부터 무너졌다
  (95 -> 30). R13(가우시안 i.i.d. 샘플)은 같은 시점에 덜 무너졌다(-> 70).

  H1  수송의 **원료**인 현재 태스크 편차 z 가 task1 의 변동 방향을 희소하게만
      담고 있어서, b_1 이 task1 영역의 일부만 반복해서 덮는다(재현율 하락).
  H2  희석(1/K)·teacher 표류·태스크 충돌 등 좌표와 무관한 누적 원인.

여기서 재는 것
  1. 원료 커버리지  c_{k->j}(τ) = tr(P_k(τ) Σ_j(τ)) / tr(Σ_j(τ))
     P_k 는 태스크 k 의 (k,τ) 조건부 편차 공분산에서 에너지 90% 를 설명하는
     상위 부분공간의 사영. "현재 태스크의 편차가 과거 태스크의 변동을 얼마나
     담고 있는가" 다. 수송은 이 원료 밖으로는 나갈 수 없다.
  2. 수송점의 정밀도/재현율  실제 태스크 j 프레임과의 최근접 거리로.
     P̂ = median dist(b_j -> 실제 j),  R̂ = median dist(실제 j -> b_j).
     둘 다 실제 j 프레임끼리의 leave-one-out 최근접 거리 중앙값 d_j 로 나눈다.
     P̂ 는 "만든 점이 진짜 같은가", R̂ 는 "진짜 영역을 다 덮는가" 다.
     ★ 거리는 **같은 bin 안에서만** 잰다. 수송이 bin 조건부라 다른 bin 프레임과
       맞는 것은 의미가 없다.
  3. ΔSR 과의 상관  스테이지 하락폭이 커버리지로 설명되는가.
  4. task1 시계열  SR / c / R̂ 를 한 그림에.

수송 코드는 k1.K1Anchor.transport 를 **import 해서 그대로** 쓴다. 저장된
qtab/m_perp/기저를 채워 넣고 호출한다 — 재구현하지 않는다.
R13 쪽은 R10.loss 안에 두 줄로 박혀 있어 뽑아 쓸 함수가 없다. 그래서 R13 이
저장한 mu/sigma 를 읽어 b_j = mu_j[τ] + sigma_j[τ]·z, z~N(0,I) clip(-3,3) 을
그대로 쓴다(R10.py:244-253, 281 과 같은 식).
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
from scipy import stats as sst

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import k1 as K1MOD

D = 3072


# ═════════════════════════════════════════════════════════════════════════════
#  입력
# ═════════════════════════════════════════════════════════════════════════════
def load_sr(p: Path):
    cells = {}
    for line in p.read_text().splitlines():
        if line.startswith("#"):
            continue
        f = line.split(",")
        if not f[0].strip().isdigit():
            continue
        k = int(f[0])
        for t, v in enumerate(f[1:]):
            if v.strip():
                cells[(k, t)] = float(v)
    return cells


def load_emb(cache: Path, suite: str, k: int):
    d = torch.load(cache / f"{suite}_task{k}.pt")
    return d["X"].float(), d["T"].long()


def make_anchor(k1_dir: Path, cfg: dict, out_dir: Path):
    """저장된 기저와 표로 K1Anchor 를 복원한다. 학습은 하지 않는다."""
    ns = argparse.Namespace(
        out_dir=str(out_dir), n_bins=cfg["n_bins"], lambda_level=cfg["lambda_level"],
        anchor_norm=cfg["anchor_norm"], chunk_backward=False, stats_batches=0,
        stats_workers=0, log_every_anchor=10 ** 9, rho=0.0, warmup_steps=0,
        quantiles=cfg["Q"], rank=cfg["r"], marginal=cfg["marginal"],
        basis=cfg["basis"], iid_sample=cfg["iid_sample"])
    a = K1MOD.K1Anchor(ns)
    b = torch.load(k1_dir / "shared_basis.pt")
    a.W = None if b["W"] is None else b["W"].float()
    a.c0 = b["c0"].float()
    a.r = b["r"]
    return a


def load_table(k1_dir: Path, j: int):
    d = torch.load(k1_dir / "stats" / f"task{j}.pt")
    t = {"qtab": d["qtab"].float(), "m_perp": d["m_perp"].float()}
    t["med"] = K1MOD.quant_at(t["qtab"], 0.5)
    t["s"] = (K1MOD.quant_at(t["qtab"], 0.841) - K1MOD.quant_at(t["qtab"], 0.159)) / 2
    t["s_floor"] = float(0.1 * t["s"].flatten().median())
    return t


# ═════════════════════════════════════════════════════════════════════════════
#  1. 원료 커버리지
# ═════════════════════════════════════════════════════════════════════════════
def top_subspace(Xc: torch.Tensor, frac: float = 0.90):
    """중심화 데이터의 에너지 frac 을 설명하는 상위 우특이벡터 V (q, d).

    Gram 행렬(n x n)로 푼다 — n << d 라 3072x3072 공분산을 만들 필요가 없다.
    """
    n = Xc.shape[0]
    G = (Xc @ Xc.T).double()
    ev, U = torch.linalg.eigh(G)                   # 오름차순
    ev = ev.flip(0).clamp_min(0.0)
    U = U.flip(1)
    tot = ev.sum().clamp_min(1e-12)
    q = int((torch.cumsum(ev, 0) / tot < frac).sum()) + 1
    q = min(q, n - 1)
    s = ev[:q].sqrt().clamp_min(1e-8)
    V = (U[:, :q].T.float() @ Xc) / s[:, None].float()   # (q, d) 정규직교
    return V, q


def coverage(Vk: torch.Tensor, Xjc: torch.Tensor) -> float:
    """c = tr(P_k Σ_j)/tr(Σ_j) = ‖X_jc V_k^T‖_F² / ‖X_jc‖_F²."""
    num = (Xjc @ Vk.T).pow(2).sum()
    den = Xjc.pow(2).sum().clamp_min(1e-12)
    return float(num / den)


# ═════════════════════════════════════════════════════════════════════════════
#  2. 정밀도/재현율
# ═════════════════════════════════════════════════════════════════════════════
def nn_dists(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A 의 각 행에서 B 로의 최근접 L2 거리."""
    return torch.cdist(A, B).min(dim=1).values


def loo_scale(X: torch.Tensor) -> float:
    """자기 자신을 뺀 최근접 거리의 중앙값. 매니폴드 고유 간격."""
    Dm = torch.cdist(X, X)
    Dm.fill_diagonal_(float("inf"))
    return float(Dm.min(dim=1).values.median())


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--num_tasks", type=int, default=10)
    ap.add_argument("--k1_dir", default="results/K1_spatial_10task")
    ap.add_argument("--r13_dir", default="results/R13_10task")
    ap.add_argument("--k0_json", default="results/K0_10task/summary.json")
    ap.add_argument("--cache", default="results/K0/emb_cache")
    ap.add_argument("--cov_cap", type=int, default=500, help="커버리지용 bin 당 프레임")
    ap.add_argument("--pr_cap", type=int, default=300, help="정밀/재현용 bin 당 프레임")
    ap.add_argument("--energy", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/K2")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    k1_dir, r13_dir = Path(a.k1_dir), Path(a.r13_dir)
    K = a.num_tasks
    cfg = json.loads((k1_dir / "k1_config.json").read_text())
    nb = cfg["n_bins"]

    sr_k1 = load_sr(k1_dir / "sr_matrix.csv")
    sr_r13 = load_sr(r13_dir / "sr_matrix.csv")
    print(f"[K2] SR  K1 {k1_dir/'sr_matrix.csv'} ({len(sr_k1)} 칸)  "
          f"R13 {r13_dir/'sr_matrix.csv'} ({len(sr_r13)} 칸)", flush=True)
    print(f"[K2] K1 config  basis={cfg['basis']} marginal={cfg['marginal']} "
          f"Q={cfg['Q']} r={cfg['r']} bins={nb}", flush=True)

    # ── 임베딩 (bin 별로 잘라 둔다) ─────────────────────────────────────────
    cache = Path(a.cache)
    binsets, binsets_pr = {}, {}
    for k in range(K):
        X, T = load_emb(cache, a.suite, k)
        cov, pr = [], []
        for t in range(nb):
            idx = torch.nonzero(T == t, as_tuple=True)[0]
            if len(idx) > a.cov_cap:
                idx = idx[torch.from_numpy(rng.choice(len(idx), a.cov_cap, replace=False))]
            cov.append(X[idx])
            sub = idx if len(idx) <= a.pr_cap else \
                idx[torch.from_numpy(rng.choice(len(idx), a.pr_cap, replace=False))]
            pr.append(X[sub])
        binsets[k], binsets_pr[k] = cov, pr
        del X, T
    print(f"[K2] 임베딩 로드 완료  bin 당 커버리지 <= {a.cov_cap}, 정밀/재현 <= {a.pr_cap}",
          flush=True)

    # ── 1. 커버리지 ────────────────────────────────────────────────────────
    Vs, ranks = {}, {}
    for k in range(K):
        Vs[k], ranks[k] = [], []
        for t in range(nb):
            Xc = binsets[k][t] - binsets[k][t].mean(0, keepdim=True)
            V, q = top_subspace(Xc, a.energy)
            Vs[k].append(V); ranks[k].append(q)
        print(f"[K2] task{k} 부분공간 rank(에너지 {a.energy:.0%})  bin별 {ranks[k]}", flush=True)

    Cov = np.full((K, K, nb), np.nan)
    for k in range(K):
        for j in range(K):
            if j == k:
                continue
            for t in range(nb):
                Xjc = binsets[j][t] - binsets[j][t].mean(0, keepdim=True)
                Cov[k, j, t] = coverage(Vs[k][t], Xjc)
    cov_mean = np.nanmean(Cov, axis=2)
    cov_min = np.nanmin(Cov, axis=2)
    del Vs

    # ── 2. 정밀도/재현율 ───────────────────────────────────────────────────
    anchor = make_anchor(k1_dir, cfg, out / "_anchor")
    tables = {j: load_table(k1_dir, j) for j in range(K)}
    r13_ms = {}
    for j in range(K):
        d = torch.load(r13_dir / "stats" / f"task{j}.pt")
        r13_ms[j] = (d["mu"].float().reshape(nb, -1), d["sigma"].float().reshape(nb, -1),
                     float(d["sigma_floor"]))

    dj = {j: [loo_scale(binsets_pr[j][t]) for t in range(nb)] for j in range(K)}

    PR = {}
    for k in range(1, K):
        anchor.cur = tables[k]
        anchor.stats = {j: tables[j] for j in range(k)}
        for j in range(k):
            ph1, rh1, ph3, rh3, cl = [], [], [], [], []
            for t in range(nb):
                src = binsets_pr[k][t]
                tgt = binsets_pr[j][t]
                n = src.shape[0]
                if n < 5 or tgt.shape[0] < 5:
                    continue
                tau = torch.full((n,), t, dtype=torch.long)
                o = src.reshape(n, -1, 768)
                with torch.no_grad():
                    w, res = anchor.rotate(o)
                    b1 = anchor.transport(w, res, tau, j).reshape(n, -1)
                cl.append(anchor.clamp_frac)
                mu, sg, fl = r13_ms[j]
                z = torch.randn(n, D).clamp_(-3.0, 3.0)
                b3 = mu[t] + sg[t].clamp_min(fl) * z
                s = dj[j][t] if dj[j][t] > 1e-6 else 1.0
                ph1.append(float(nn_dists(b1, tgt).median()) / s)
                rh1.append(float(nn_dists(tgt, b1).median()) / s)
                ph3.append(float(nn_dists(b3, tgt).median()) / s)
                rh3.append(float(nn_dists(tgt, b3).median()) / s)
            PR[(k, j)] = {"P_k1": float(np.mean(ph1)), "R_k1": float(np.mean(rh1)),
                          "P_r13": float(np.mean(ph3)), "R_r13": float(np.mean(rh3)),
                          "clamp": float(np.mean(cl)), "cov": float(cov_mean[k, j]),
                          "cov_min": float(cov_min[k, j])}
        print(f"[K2] stage {k}  " + "  ".join(
            f"j{j}: c={PR[(k,j)]['cov']:.2f} R̂={PR[(k,j)]['R_k1']:.2f}/"
            f"{PR[(k,j)]['R_r13']:.2f}" for j in range(k)), flush=True)

    # ── 3. ΔSR 상관 ────────────────────────────────────────────────────────
    k0 = json.loads(Path(a.k0_json).read_text())
    rho0 = {int(t[4:]): v["rho256_mean"] for t, v in k0["verdict"].items()}

    rows = []
    for k in range(2, K):
        for j in range(k):
            d1 = sr_k1.get((k, j)), sr_k1.get((k - 1, j))
            if None in d1:
                continue
            rows.append({"k": k, "j": j, "dSR": d1[0] - d1[1],
                         "cov": PR[(k, j)]["cov"], "R_k1": PR[(k, j)]["R_k1"],
                         "P_k1": PR[(k, j)]["P_k1"], "inv_k": 1.0 / k,
                         "rho0": rho0.get(j, np.nan)})
    dsr = np.array([r["dSR"] for r in rows])
    preds = {"c_coverage": np.array([r["cov"] for r in rows]),
             "R_hat_K1": np.array([r["R_k1"] for r in rows]),
             "inv_k_dilution": np.array([r["inv_k"] for r in rows]),
             "K0_rho_negctrl": np.array([r["rho0"] for r in rows])}
    corr = {}
    for nm, v in preds.items():
        m = ~np.isnan(v)
        rho_, p_ = sst.spearmanr(dsr[m], v[m])
        corr[nm] = {"spearman": float(rho_), "p": float(p_), "n": int(m.sum())}

    # ── 4. 판정 (사전 등록) ────────────────────────────────────────────────
    # (i) stage 6~9 에서 c_{k->1} 이 같은 stage 의 c_{k->j} 중 하위 25% 이내
    qranks = []
    for k in range(6, K):
        vals = np.array([PR[(k, j)]["cov"] for j in range(k)])
        qranks.append(float((vals < PR[(k, 1)]["cov"]).mean()))
    cond_i = bool(np.mean(qranks) <= 0.25)

    # (ii) R̂_{k,1}(K1) 이 stage 에 따라 악화(양의 추세)하고 R13 보다 나쁘다.
    #      반면 P̂ 는 안정(양의 추세가 유의하지 않다).
    ks = [k for k in range(2, K)]
    r1 = np.array([PR[(k, 1)]["R_k1"] for k in ks])
    r3 = np.array([PR[(k, 1)]["R_r13"] for k in ks])
    p1 = np.array([PR[(k, 1)]["P_k1"] for k in ks])
    tr_R, pr_R = sst.spearmanr(ks, r1)
    tr_P, pr_P = sst.spearmanr(ks, p1)
    worse = bool((r1 > r3).mean() > 0.5)
    cond_ii = bool(tr_R > 0 and pr_R < 0.05 and worse and not (tr_P > 0 and pr_P < 0.05))

    # (iii) Spearman(ΔSR, c) >= 0.4, p < 0.05
    cond_iii = bool(corr["c_coverage"]["spearman"] >= 0.4 and corr["c_coverage"]["p"] < 0.05)

    n_ok = sum([cond_i, cond_ii, cond_iii])
    if n_ok == 3:
        verdict = "H1 채택 — 원료 커버리지 가설이 세 조건을 모두 충족한다"
    elif n_ok == 0:
        verdict = ("원료 가설 기각, H2(희석/표류/충돌) 축으로 이동 — "
                   "c_{k->1} 이 다른 태스크와 비슷하고 R̂ 가 평평하다")
    else:
        ok = [n for n, c in [("(i) 커버리지 하위 25%", cond_i),
                             ("(ii) R̂ 악화 + R13 보다 나쁨", cond_ii),
                             ("(iii) Spearman(ΔSR,c)>=0.4", cond_iii)] if c]
        verdict = f"혼합 원인 — 충족: {', '.join(ok)} ({n_ok}/3)"
    negctrl = (f"negative control Spearman(ΔSR, K0 ρ_j) = "
               f"{corr['K0_rho_negctrl']['spearman']:+.3f} "
               f"(p={corr['K0_rho_negctrl']['p']:.3f}) — "
               f"{'유의하지 않음' if corr['K0_rho_negctrl']['p'] >= 0.05 else '★유의함(주의)'}")

    # ── 그림 ───────────────────────────────────────────────────────────────
    C = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.2))
    for ax, M, ttl in ((axes[0], cov_mean, "bin mean"), (axes[1], cov_min, "bin min")):
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                if i == j:
                    ax.text(j, i, "–", ha="center", va="center", fontsize=9, color="0.7")
                else:
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if M[i, j] < 0.6 else "black")
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xlabel("target task $j$"); ax.set_ylabel("source task $k$")
        ax.set_title(f"$c_{{k\\to j}}$  ({ttl})", fontsize=11.5)
        fig.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle("K2 — raw-material coverage: does task $k$'s deviation span task $j$'s "
                 f"variation?  (top {a.energy:.0%} energy subspace)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig1_coverage_heatmap.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0), sharey=True)
    for ax, j in zip(axes, (1, 2, 3)):
        kk = [k for k in range(j + 1, K)]
        ax.plot(kk, [PR[(k, j)]["P_k1"] for k in kk], "-o", color=C[0], lw=2, label="P̂  K1")
        ax.plot(kk, [PR[(k, j)]["R_k1"] for k in kk], "-s", color=C[3], lw=2, label="R̂  K1")
        ax.plot(kk, [PR[(k, j)]["P_r13"] for k in kk], "--o", color=C[0], lw=1.6,
                mfc="none", label="P̂  R13")
        ax.plot(kk, [PR[(k, j)]["R_r13"] for k in kk], "--s", color=C[3], lw=1.6,
                mfc="none", label="R̂  R13")
        ax.axhline(1.0, color="0.5", ls=":", lw=1.4)
        ax.set_xticks(kk); ax.set_xlabel("stage $k$")
        ax.set_title(f"past task $j={j}$", fontsize=11.5)
    axes[0].set_ylabel("distance / LOO-NN scale $d_j$")
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle("K2 — precision / recall of synthesized anchor points (within phase bin). "
                 "1.0 = manifold spacing", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "fig2_pr_by_stage.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    for r in rows:
        j = r["j"]
        ax.scatter(r["cov"], r["dSR"], s=110 if j == 1 else 42,
                   color=C[j], marker="*" if j == 1 else "o",
                   edgecolors="k" if j == 1 else "none", lw=.8, alpha=.9 if j == 1 else .6,
                   zorder=5 if j == 1 else 3)
    for j in range(K):
        if any(r["j"] == j for r in rows):
            ax.scatter([], [], color=C[j], s=40, label=f"task {j}" + ("  ★" if j == 1 else ""))
    ax.axhline(0, color="0.5", lw=1, ls=":")
    cs = corr["c_coverage"]
    ax.set_xlabel(r"raw-material coverage $c_{k\to j}$ (bin mean)")
    ax.set_ylabel(r"$\Delta$SR$_{k,j}$ = SR(after $k$) − SR(after $k{-}1$)")
    ax.set_title(f"K2 — does coverage explain the per-stage SR drop?\n"
                 f"Spearman $\\rho$ = {cs['spearman']:+.3f}  (p = {cs['p']:.3f}, "
                 f"n = {cs['n']})", fontsize=11.5)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig3_dsr_scatter.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    st = list(range(2, K))
    ax.plot(st, [sr_k1.get((k, 1), np.nan) for k in st], "-o", color=C[3], lw=2.4,
            ms=7, label="SR task1 — K1")
    ax.plot(st, [sr_r13.get((k, 1), np.nan) for k in st], "-o", color=C[0], lw=2.4,
            ms=7, label="SR task1 — R13")
    ax.set_xlabel("stage $k$"); ax.set_ylabel("SR (%)"); ax.set_ylim(0, 105)
    ax.set_xticks(st)
    ax2 = ax.twinx()
    ax2.plot(st, [PR[(k, 1)]["cov"] for k in st], "--^", color="seagreen", lw=2,
             label=r"$c_{k\to 1}$ coverage")
    ax2.plot(st, [PR[(k, 1)]["R_k1"] for k in st], "--v", color="darkorange", lw=2,
             label=r"$\hat{R}_{k,1}$ K1 recall")
    ax2.set_ylabel(r"coverage $c$   /   recall $\hat{R}$")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, loc="lower left")
    ax.set_title("K2 — task1 timeline: SR collapse vs coverage and recall", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "fig4_task1_timeline.png", dpi=300); plt.close(fig)

    # ── 저장 + 출력 ────────────────────────────────────────────────────────
    js = {"suite": a.suite, "num_tasks": K, "n_bins": nb, "energy": a.energy,
          "cov_cap": a.cov_cap, "pr_cap": a.pr_cap,
          "k1_dir": str(k1_dir), "r13_dir": str(r13_dir),
          "sr_paths": {"K1": str(k1_dir / "sr_matrix.csv"),
                       "R13": str(r13_dir / "sr_matrix.csv")},
          "subspace_rank": {str(k): ranks[k] for k in ranks},
          "coverage_binmean": cov_mean.tolist(), "coverage_binmin": cov_min.tolist(),
          "pr": {f"{k}->{j}": v for (k, j), v in PR.items()},
          "rows": rows, "correlations": corr,
          "task1_trend": {"R_hat_spearman": float(tr_R), "R_hat_p": float(pr_R),
                          "P_hat_spearman": float(tr_P), "P_hat_p": float(pr_P),
                          "R_worse_than_R13_frac": float((r1 > r3).mean())},
          "conditions": {"i_coverage_bottom25": cond_i,
                         "ii_recall_degrades": cond_ii,
                         "iii_spearman_dsr_cov": cond_iii,
                         "quantile_ranks_stage6_9": qranks},
          "verdict": verdict, "negative_control": negctrl}
    (out / "summary.json").write_text(json.dumps(js, indent=2, ensure_ascii=False))

    print("\n" + "=" * 82)
    print("K2 판정")
    print("=" * 82)
    print(f"  (i)   stage 6~9 에서 c_(k->1) 이 하위 25% 이내      : "
          f"{cond_i}   (분위 순위 {['%.2f' % q for q in qranks]})")
    print(f"  (ii)  R̂_(k,1) 악화 + R13 보다 나쁨, P̂ 는 안정      : {cond_ii}   "
          f"(R̂ 추세 ρ={tr_R:+.2f} p={pr_R:.3f}, P̂ 추세 ρ={tr_P:+.2f} p={pr_P:.3f}, "
          f"R13 보다 나쁜 stage 비율 {(r1>r3).mean():.0%})")
    print(f"  (iii) Spearman(ΔSR, c) >= 0.4 & p<0.05            : {cond_iii}   "
          f"(ρ={corr['c_coverage']['spearman']:+.3f}, p={corr['c_coverage']['p']:.3f}, "
          f"n={corr['c_coverage']['n']})")
    print(f"\n  -> {verdict}")
    print(f"  {negctrl}")
    print("\n  예측변수별 Spearman(ΔSR, ·)")
    for nm, c in corr.items():
        print(f"    {nm:<20} ρ={c['spearman']:+.3f}  p={c['p']:.4f}  n={c['n']}")
    print(f"\n그림  {out/'fig1_coverage_heatmap.png'}\n      {out/'fig2_pr_by_stage.png'}"
          f"\n      {out/'fig3_dsr_scatter.png'}\n      {out/'fig4_task1_timeline.png'}"
          f"\n요약  {out/'summary.json'}")


if __name__ == "__main__":
    main()
