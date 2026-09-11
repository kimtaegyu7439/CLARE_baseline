#!/usr/bin/env python
"""K0 — 공유 기저 전이 검사. 학습 없음, 분석 전용.

K1 은 태스크 0 에서 만든 PCA 기저 W 를 동결하고 이후 모든 태스크의 관측을 그
좌표계에서 수송한다. 그 전제는 "태스크 0 의 주성분이 다른 태스크의 변동도
설명한다" 이다. 성립하지 않으면 수송이 기저 밖 성분을 통째로 날린다.

여기서 재는 것
  rho_k(tau; r) = mean‖W_r^T x‖² / mean‖x‖²        x 는 (k,tau) 자기 평균 중심화
    ★ c0 나 태스크 0 평균으로 중심화하면 안 된다. 평균 이동은 K1 이 m_perp 로
      따로 처리하므로 검사 대상이 아니다. 여기서 볼 것은 **퍼짐의 방향**이다.
  상한  태스크 0 을 에피소드 단위로 A/B 로 갈라 A 로 기저를 만들고 B 에서 잰 rho.
        같은 태스크의 held-out 이므로 표본오차만 남은 값이다.
  하한  r/3072. 무작위 r 차원 부분공간이 잡는 에너지의 기대값.

누출 방향 정렬도  a_i = ‖W_256^T v_i‖², v_i 는 (k,tau) 중심화 데이터의 상위
  고유벡터. a_i 가 낮은 상위 고유벡터가 있으면 그 방향이 기저 밖이라는 뜻이다.

좌표 상관 부활  w = W_256^T x 의 상관행렬 C 에서
  e = ‖C − diag(C)‖_F² / ‖C‖_F². K1 의 좌표별 분위수 사상은 좌표 독립을 가정하지
  않지만(copula 는 순위로 보존된다), e 가 태스크마다 크게 다르면 표 하나로
  기술되는 정도가 달라진다.
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

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
import R10
from B_merge import _ns

from lerobot.datasets.factory import make_dataset                    # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.sampler import EpisodeAwareSampler             # noqa: E402
from lerobot.policies.factory import make_policy                     # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging  # noqa: E402

D = 3072            # (n_obs 2 x n_cam 2) x 768
RANKS = (64, 128, 256, 512)
M_EIG = 10          # 정렬도를 볼 상위 고유벡터 개수


# ═════════════════════════════════════════════════════════════════════════════
#  임베딩 — R13/K1 이 쓰는 추출 경로를 그대로 쓴다
#  (B1.prep_batch -> B1.rgb_cls -> R10.phase_bins. 재구현하지 않는다.)
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract(policy, cfg, dataset, device, n_bins, batch_size, workers):
    """(X (N,3072) float32 cpu, T (N,) bin, E (N,) episode_index)."""
    sampler = EpisodeAwareSampler(
        dataset.episode_data_index,
        drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0), shuffle=False)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=workers, batch_size=batch_size, sampler=sampler,
        drop_last=False, pin_memory=(device.type == "cuda"),
        multiprocessing_context="spawn" if workers > 0 else None)
    ep_len = R10.episode_lengths(dataset)
    X, T, E = [], [], []
    for raw in loader:
        b = B1.prep_batch(policy, B1.to_device(raw, device))
        cls = B1.rgb_cls(policy, b).float()
        n = b["observation.state"].shape[0]
        X.append(cls.reshape(n, -1).cpu())
        T.append(R10.phase_bins(raw, ep_len, n_bins).cpu())
        E.append(raw["episode_index"].long().cpu())
    return torch.cat(X), torch.cat(T), torch.cat(E)


def cap_per_bin(T, n_bins, cap, rng, mask=None):
    """bin 마다 최대 cap 개만 남기는 인덱스. mask 로 후보를 먼저 좁힐 수 있다."""
    keep = []
    for t in range(n_bins):
        idx = torch.nonzero(T == t, as_tuple=True)[0]
        if mask is not None:
            idx = idx[mask[idx]]
        if len(idx) > cap:
            sel = torch.from_numpy(rng.choice(len(idx), cap, replace=False))
            idx = idx[sel]
        keep.append(idx)
    return keep


# ═════════════════════════════════════════════════════════════════════════════
#  분석
# ═════════════════════════════════════════════════════════════════════════════
def fit_basis(X, r_max):
    """전역 평균 c0 로 중심화 후 상위 r_max 주성분. W (3072, r_max) 정규직교."""
    c0 = X.mean(0)
    Xc = (X - c0).double()
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    k = min(r_max, Vh.shape[0])
    W = Vh[:k].T.float().contiguous()
    var = (S ** 2 / max(1, Xc.shape[0] - 1))
    return W, c0, (var / var.sum()).float()


def rho_and_more(x, W, ranks, m_eig, r_corr=256):
    """x (n,3072) 원본. 자기 평균으로 중심화해서 rho / a_i / offdiag 에너지."""
    xc = x - x.mean(0, keepdim=True)
    tot = float(xc.pow(2).sum(1).mean())
    P = xc @ W                                          # (n, r_max)
    cum = P.pow(2).cumsum(1).mean(0)                    # 누적 에너지
    rho = {r: float(cum[r - 1]) / max(tot, 1e-12) for r in ranks if r <= W.shape[1]}

    # 상위 고유벡터 정렬도 — 경제형 SVD 로 우특이벡터를 얻는다
    n = xc.shape[0]
    k = min(m_eig, n - 1, xc.shape[1])
    _, S, Vh = torch.linalg.svd(xc.double(), full_matrices=False)
    V = Vh[:k].float()                                  # (k, 3072)
    Wc = W[:, :r_corr]
    a = (V @ Wc).pow(2).sum(1)                          # ‖W^T v_i‖²  (W 정규직교)
    lam = (S[:k] ** 2).float()
    abar = float((lam * a).sum() / lam.sum().clamp_min(1e-12))

    # 좌표 상관 부활
    w = xc @ Wc                                         # (n, r_corr)
    w = w - w.mean(0, keepdim=True)
    sd = w.std(0, unbiased=False).clamp_min(1e-8)
    C = (w / sd).T @ (w / sd) / n                       # 상관행렬
    off = float((C.pow(2).sum() - C.diag().pow(2).sum()) / C.pow(2).sum().clamp_min(1e-12))
    return rho, a.tolist(), abar, off, float(lam.sum())


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--num_tasks", type=int, default=4)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--cap", type=int, default=500, help="bin 당 최대 프레임")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/K0")
    ap.add_argument("--cache", default=None,
                    help="임베딩 캐시 디렉토리. 기본은 <out>/emb_cache. "
                         "4 태스크 실행과 캐시를 공유하려면 지정한다.")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache) if a.cache else out / "emb_cache"
    cache.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    rng = np.random.default_rng(a.seed)
    ds_prefix, _ = B1.suite_prefixes(a.suite)
    tasks = list(range(a.num_tasks))
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    ck = REPO / "outputs/B2_lam3/libero_spatial_seed42_ours/task_0/checkpoints/005000/pretrained_model"

    # ── 1. 임베딩 (캐시) ────────────────────────────────────────────────────
    policy = None
    raw = {}
    for k in tasks:
        cp = cache / f"{a.suite}_task{k}.pt"
        if cp.exists():
            raw[k] = torch.load(cp)
            print(f"[K0] task{k} 캐시 {tuple(raw[k]['X'].shape)}", flush=True)
            continue
        if policy is None:
            policy = make_policy(cfg=B1.build_cfg(_ns(a), 0, str(ck), Path("/tmp/k0")).policy,
                                 ds_meta=meta)
            policy.eval()
        cfg = B1.build_cfg(_ns(a), k, str(ck), Path("/tmp/k0"))
        ds = make_dataset(cfg)
        X, T, E = extract(policy, cfg, ds, device, a.n_bins, a.batch_size, a.workers)
        raw[k] = {"X": X, "T": T, "E": E}
        torch.save(raw[k], cp)
        print(f"[K0] task{k} 추출 {tuple(X.shape)}  에피소드 {len(E.unique())}개", flush=True)
        del ds
        torch.cuda.empty_cache()
    del policy
    torch.cuda.empty_cache()

    # ── 2. 태스크 0 을 에피소드 단위로 A/B 분할 ─────────────────────────────
    eps0 = raw[0]["E"].unique().sort().values
    half = len(eps0) // 2
    epsA, epsB = set(eps0[:half].tolist()), set(eps0[half:].tolist())
    inA = torch.tensor([int(e) in epsA for e in raw[0]["E"]])
    print(f"[K0] task0 에피소드 {len(eps0)}개 -> A {len(epsA)} / B {len(epsB)}", flush=True)

    # ── 3. bin 별 표본 뽑기 ─────────────────────────────────────────────────
    sets = {}                              # 이름 -> [bin 별 (n,3072)]
    idxA = cap_per_bin(raw[0]["T"], a.n_bins, a.cap, rng, mask=inA)
    idxB = cap_per_bin(raw[0]["T"], a.n_bins, a.cap, rng, mask=~inA)
    sets["task0-A"] = [raw[0]["X"][i] for i in idxA]
    sets["task0-B"] = [raw[0]["X"][i] for i in idxB]
    for k in tasks:
        idx = cap_per_bin(raw[k]["T"], a.n_bins, a.cap, rng)
        sets[f"task{k}"] = [raw[k]["X"][i] for i in idx]

    # ── 4. 기저는 task0-A 전 bin 합쳐서 ─────────────────────────────────────
    XA = torch.cat(sets["task0-A"])
    W, c0, evr = fit_basis(XA, max(RANKS))
    torch.save({"W512": W, "c0": c0, "explained_variance_ratio": evr,
                "suite": a.suite, "n_frames_A": int(XA.shape[0])},
               out / "basis.pt")
    orth = float((W.T @ W - torch.eye(W.shape[1])).norm())
    print(f"[K0] 기저 W{tuple(W.shape)}  task0-A {XA.shape[0]} 프레임  "
          f"‖WᵀW−I‖={orth:.2e}  누적설명분산 r=256 {float(evr[:256].sum())*100:.1f}%  "
          f"r=512 {float(evr[:512].sum())*100:.1f}%", flush=True)

    # ── 5. 격자 계산 ────────────────────────────────────────────────────────
    names = [f"task{k}" for k in tasks] + ["task0-B"]
    G = {}
    for nm in names:
        G[nm] = []
        for t in range(a.n_bins):
            x = sets[nm][t]
            if x.shape[0] < 12:            # 표본이 너무 적은 bin 은 건너뛴다
                G[nm].append(None); continue
            rho, ai, abar, off, lam = rho_and_more(x, W, RANKS, M_EIG)
            G[nm].append({"n": int(x.shape[0]), "rho": rho, "a": ai,
                          "abar": abar, "offdiag": off, "lam_sum": lam})
        ok = [g for g in G[nm] if g]
        print(f"[K0] {nm:>8}  rho(r=256) 평균 {np.mean([g['rho'][256] for g in ok])*100:5.1f}%  "
              f"최소 {np.min([g['rho'][256] for g in ok])*100:5.1f}%  "
              f"ā {np.mean([g['abar'] for g in ok]):.3f}  "
              f"offdiag {np.mean([g['offdiag'] for g in ok]):.3f}", flush=True)

    ref = np.mean([g["rho"][256] for g in G["task0-B"] if g])     # 상한(bin 평균)
    floor = 256 / D

    # ── 6. 판정 ─────────────────────────────────────────────────────────────
    verdict, notes = {}, []
    for k in tasks:
        g = [x for x in G[f"task{k}"] if x]
        rmin = float(np.min([x["rho"][256] for x in g]))
        rmean = float(np.mean([x["rho"][256] for x in g]))
        frac = rmin / ref
        bad_bins = [i for i, x in enumerate(G[f"task{k}"])
                    if x and x["rho"][256] / ref < 0.60]
        if bad_bins or frac < 0.70:
            v = f"기각 — bin {bad_bins} 이 상한의 60% 미만" if bad_bins else \
                f"기각 — 최소 bin 이 상한의 {frac*100:.0f}%"
        elif frac < 0.90:
            v = f"경계 ({frac*100:.0f}%) — 기저를 task0+1 합집합으로 재계산 또는 r=512 권장"
        else:
            v = f"공유 기저 OK ({frac*100:.0f}%)"
        verdict[f"task{k}"] = {"rho256_min": rmin, "rho256_mean": rmean,
                               "frac_of_ceiling": frac, "bad_bins": bad_bins,
                               "verdict": v}
        # 정렬도가 낮은 상위 고유벡터
        A = np.array([x["a"] for x in g])                      # (bins, M)
        LAM = np.array([x["lam_sum"] for x in g])
        for i in range(A.shape[1]):
            if A[:, i].mean() < 0.5:
                notes.append(f"task{k} eigvec#{i+1}: ā_i={A[:,i].mean():.3f} "
                             f"(λ 비중 상위 {M_EIG}개 중 {i+1}번째, "
                             f"bin 최소 {A[:,i].min():.3f})")

    # ── 7. 그림 ─────────────────────────────────────────────────────────────
    C = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for k in tasks:
        g = G[f"task{k}"]
        ys = [x["rho"][256] * 100 for x in g if x]
        ax.scatter([k] * len(ys), ys, s=26, alpha=.30, color=C[k], edgecolors="none")
        ax.plot([k], [np.mean(ys)], "o", ms=11, color=C[k], zorder=5)
        ax.plot([k], [np.min(ys)], "v", ms=9, color=C[k], mfc="none", mew=2, zorder=5)
    ax.plot(tasks, [np.mean([x["rho"][256] * 100 for x in G[f"task{k}"] if x])
                    for k in tasks], "-", color="0.35", lw=1.6, zorder=4)
    ax.axhline(ref * 100, color="crimson", ls="--", lw=1.8,
               label=f"ceiling: task0-B held-out ({ref*100:.1f}%)")
    ax.axhline(floor * 100, color="0.4", ls=":", lw=1.8,
               label=f"floor: random {256}-dim subspace ({floor*100:.1f}%)")
    ax.plot([], [], "o", color="0.3", ms=9, label="bin mean")
    ax.plot([], [], "v", color="0.3", ms=8, mfc="none", mew=2, label="bin min")
    ax.plot([], [], "o", color="0.3", ms=5, alpha=.3, label="individual bins")
    ax.set_xticks(tasks); ax.set_xlabel("task id")
    ax.set_ylabel(r"$\rho$ = retained variance (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"K0 — does the task-0 basis explain other tasks?  ($r=256$ of {D})\n"
                 f"{a.suite}, self-centered within each (task, phase bin)", fontsize=11.5)
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "fig1_rho.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for k in tasks:
        ys = [np.mean([x["rho"][r] * 100 for x in G[f"task{k}"] if x]) for r in RANKS]
        ax.plot(RANKS, ys, "-o", color=C[k], lw=1.9, ms=6, label=f"task {k}")
    ys = [np.mean([x["rho"][r] * 100 for x in G["task0-B"] if x]) for r in RANKS]
    ax.plot(RANKS, ys, "--s", color="crimson", lw=2.2, ms=7, label="task0-B (ceiling)")
    ax.plot(RANKS, [r / D * 100 for r in RANKS], ":", color="0.4", lw=1.8,
            label="random subspace (floor)")
    ax.axvline(256, color="0.75", lw=1, zorder=0)
    ax.set_xscale("log", base=2); ax.set_xticks(RANKS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("basis rank $r$"); ax.set_ylabel(r"$\rho$ bin mean (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"K0 — how large must $r$ be?  ({a.suite})", fontsize=11.5)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig2_rank_sweep.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    ax = axes[0]
    H = np.array([[np.mean([x["a"][i] for x in G[f"task{k}"] if x])
                   for i in range(M_EIG)] for k in tasks])
    im = ax.imshow(H, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H[i,j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if H[i, j] < 0.6 else "black")
    ax.set_xticks(range(M_EIG)); ax.set_xticklabels(range(1, M_EIG + 1))
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels([f"task {k}" for k in tasks])
    ax.set_xlabel("eigenvector rank (within task, bin-averaged)")
    ax.set_title(r"alignment $a_i=\|W_{256}^{\top}v_i\|^2$", fontsize=11.5)
    fig.colorbar(im, ax=ax, fraction=.046)

    ax = axes[1]
    for k in tasks:
        ys = [x["offdiag"] for x in G[f"task{k}"] if x]
        ax.scatter([k] * len(ys), ys, s=26, alpha=.30, color=C[k], edgecolors="none")
        ax.plot([k], [np.mean(ys)], "o", ms=11, color=C[k], zorder=5)
    ax.plot(tasks, [np.mean([x["offdiag"] for x in G[f"task{k}"] if x]) for k in tasks],
            "-", color="0.35", lw=1.6, zorder=4)
    rb = np.mean([x["offdiag"] for x in G["task0-B"] if x])
    ax.axhline(rb, color="crimson", ls="--", lw=1.8, label=f"task0-B ({rb:.3f})")
    ax.set_xticks(tasks); ax.set_xlabel("task id")
    ax.set_ylabel(r"off-diagonal energy of $\mathrm{corr}(W_{256}^{\top}x)$")
    ax.set_title("coordinate correlation after rotation", fontsize=11.5)
    ax.legend(fontsize=9, frameon=False)
    fig.suptitle(f"K0 — leaked directions and residual coordinate correlation ({a.suite})",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig3_leak_and_corr.png", dpi=300); plt.close(fig)

    # ── 8. 저장 + 판정 출력 ─────────────────────────────────────────────────
    js = {"suite": a.suite, "n_bins": a.n_bins, "cap": a.cap, "ranks": list(RANKS),
          "d": D, "ceiling_rho256_binmean": float(ref), "floor_rho256": float(floor),
          "basis": {"orth_err": orth, "n_frames_A": int(XA.shape[0]),
                    "cum_evr": {str(r): float(evr[:r].sum()) for r in RANKS}},
          "grid": {nm: [(g if g else None) for g in G[nm]] for nm in names},
          "verdict": verdict, "low_alignment_notes": notes}
    (out / "summary.json").write_text(json.dumps(js, indent=2, ensure_ascii=False))

    print("\n" + "=" * 78)
    print(f"판정   상한(task0-B, r=256, bin 평균) = {ref*100:.1f}%   "
          f"하한(무작위 256차원) = {floor*100:.1f}%")
    print("=" * 78)
    for k in tasks:
        v = verdict[f"task{k}"]
        print(f"  task{k}  rho256 평균 {v['rho256_mean']*100:5.1f}%  "
              f"최소 {v['rho256_min']*100:5.1f}%  상한의 {v['frac_of_ceiling']*100:5.1f}%"
              f"   -> {v['verdict']}")
    if notes:
        print("\n  정렬도 낮은 상위 고유벡터 (a_i < 0.5):")
        for n_ in notes:
            print("   ", n_)
    else:
        print("\n  상위 10 고유벡터 중 a_i < 0.5 인 것 없음")
    print(f"\n그림   {out/'fig1_rho.png'}\n       {out/'fig2_rank_sweep.png'}"
          f"\n       {out/'fig3_leak_and_corr.png'}\n요약   {out/'summary.json'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
