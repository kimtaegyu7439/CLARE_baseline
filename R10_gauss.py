#!/usr/bin/env python
"""vision encoder 출력이 가우시안인가 — R10 수송의 전제 검증.

R10 은 b_j = mu_j[τ] + sigma_j[τ] · z 로 관측 임베딩을 과거 태스크 분포로 옮긴다.
이 식이 뜻을 가지려면 임베딩 분포가 (적어도 단계별로는) 평균과 표준편차로
기술돼야 한다. 다봉이거나 꼬리가 두꺼우면 b_j 는 어느 태스크의 관측도 아닌
허공의 점이 되고, 앵커가 지키는 영역이 또 어긋난다.

임베딩 = DINOv2 CLS (동결). 샘플당 (n_obs 2 × n_cam 2, 768) = (4, 768).
z = (o − mu[τ]) / max(sigma[τ], floor) 를 만들어 N(0,1) 과 비교한다.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sst

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


@torch.no_grad()
def collect(policy, cfg, dataset, device, n_bins, batch_size):
    sp = EpisodeAwareSampler(dataset.episode_data_index,
                             drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                             shuffle=False)
    dl = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=batch_size,
                                     sampler=sp, drop_last=False)
    ep_len = R10.episode_lengths(dataset)
    X, T = [], []
    for raw in dl:
        b = B1.prep_batch(policy, B1.to_device(raw, device))
        cls = B1.rgb_cls(policy, b).float()
        n = b["observation.state"].shape[0]
        X.append(cls.view(n, -1, cls.shape[-1]).cpu())
        T.append(R10.phase_bins(raw, ep_len, n_bins).cpu())
    return torch.cat(X), torch.cat(T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0,1,2,3")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/R10_gauss")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(a.suite)
    tasks = [int(x) for x in a.tasks.split(",")]
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    ck = REPO / "outputs/B2_lam3/libero_spatial_seed42_ours/task_0/checkpoints/005000/pretrained_model"
    policy = make_policy(cfg=B1.build_cfg(_ns(a), 0, str(ck), Path("/tmp/r10g")).policy,
                         ds_meta=meta)
    policy.eval()

    Z, RAW, info = {}, {}, {}
    for j in tasks:
        cfg = B1.build_cfg(_ns(a), j, str(ck), Path("/tmp/r10g"))
        ds = make_dataset(cfg)
        X, T = collect(policy, cfg, ds, device, a.n_bins, a.batch_size)
        mu = torch.zeros((a.n_bins,) + X.shape[1:]); sg = torch.zeros_like(mu)
        for t in range(a.n_bins):
            m = T == t
            if int(m.sum()) > 1:
                mu[t] = X[m].mean(0); sg[t] = X[m].std(0)
        floor = 0.1 * sg[sg > 0].median()
        Z[j] = ((X - mu[T]) / sg[T].clamp_min(floor)).numpy()
        RAW[j] = X.numpy()
        info[j] = {"n": int(X.shape[0]), "floor": float(floor)}
        print(f"[gauss] task{j}  N={X.shape[0]}  floor={floor:.4f}", flush=True)
        del ds, X, T
        torch.cuda.empty_cache()

    rng = np.random.default_rng(a.seed)
    zc = {j: Z[j].reshape(Z[j].shape[0], -1) for j in tasks}
    D = zc[tasks[0]].shape[1]
    C = plt.cm.tab10.colors
    j0 = tasks[0]

    fig = plt.figure(figsize=(15.5, 9.4))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28,
                          left=0.062, right=0.985, top=0.868, bottom=0.075)

    ax = fig.add_subplot(gs[0, 0])
    for i, d in enumerate(rng.choice(D, 6, replace=False)):
        ax.hist(zc[j0][:, d], bins=60, range=(-4, 4), density=True,
                histtype="step", lw=1.3, color=C[i], alpha=.85)
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=2, label="$N(0,1)$")
    ax.set_title(f"A. Per-dimension histogram of $z$  (task {j0}, 6 random dims)", fontsize=10.5)
    ax.set_xlabel("$z$"); ax.set_ylabel("density"); ax.legend(fontsize=9, frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    for i, j in enumerate(tasks):
        s = rng.choice(zc[j].size, 40000, replace=False)
        v = np.sort(zc[j].ravel()[s])
        q = sst.norm.ppf((np.arange(len(v)) + .5) / len(v))
        ax.plot(q, v, lw=1.6, color=C[i], label=f"task {j}")
    ax.plot([-5, 5], [-5, 5], "k--", lw=1.5)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_title("B. Q-Q plot vs normal", fontsize=10.5)
    ax.set_xlabel("theoretical quantile"); ax.set_ylabel("sample quantile")
    ax.legend(fontsize=9, frameon=False)

    ax = fig.add_subplot(gs[0, 2])
    sk = sst.skew(zc[j0], axis=0); ku = sst.kurtosis(zc[j0], axis=0)
    ax.scatter(sk, ku, s=4, alpha=.25, color=C[0], edgecolors="none")
    ax.axhline(0, color="k", lw=.8, ls=":"); ax.axvline(0, color="k", lw=.8, ls=":")
    ax.scatter([0], [0], marker="*", s=260, color="crimson", zorder=5, label="Gaussian")
    ax.set_title(f"C. Skew vs excess kurtosis per dim (task {j0}, {D} dims)", fontsize=10.5)
    ax.set_xlabel("skewness"); ax.set_ylabel("excess kurtosis")
    ax.legend(fontsize=9, frameon=False)
    ax.text(.03, .96, f"|skew|<0.5 : {100*np.mean(np.abs(sk)<.5):.0f}% of dims\n"
                      f"|kurt|<1   : {100*np.mean(np.abs(ku)<1):.0f}% of dims",
            transform=ax.transAxes, va="top", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 0])
    for i, j in enumerate(tasks):
        ax.hist((zc[j] ** 2).sum(1) / D, bins=80, density=True, histtype="step",
                lw=1.5, color=C[i], label=f"task {j}")
    x = np.linspace(0.01, 3, 400)
    ax.plot(x, sst.chi2.pdf(x * D, D) * D, "k--", lw=2, label=f"$\\chi^2_{{{D}}}/d$")
    ax.set_title("D. $\\|z\\|^2/d$  — multivariate normality", fontsize=10.5)
    ax.set_xlabel("$\\|z\\|^2/d$"); ax.set_ylabel("density")
    ax.set_xlim(0, 3); ax.legend(fontsize=9, frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    slot = ["$t{-}1$ · cam", "$t{-}1$ · wrist", "$t$ · cam", "$t$ · wrist"]
    w = 0.2
    for i, j in enumerate(tasks):
        v = [np.median(np.abs(sst.kurtosis(Z[j][:, s, :], axis=0))) for s in range(Z[j].shape[1])]
        ax.bar(np.arange(len(v)) + (i - 1.5) * w, v, w, color=C[i], label=f"task {j}")
    ax.set_xticks(range(4)); ax.set_xticklabels(slot, fontsize=9)
    ax.set_ylabel("median |excess kurtosis|")
    ax.set_title("E. Non-normality by observation slot", fontsize=10.5)
    ax.legend(fontsize=9, frameon=False)

    ax = fig.add_subplot(gs[1, 2])
    means = {j: RAW[j].reshape(RAW[j].shape[0], -1).mean(0) for j in tasks}
    sds = {j: RAW[j].reshape(RAW[j].shape[0], -1).std(0) for j in tasks}
    M = np.array([[np.mean(np.abs(means[a_] - means[b_]) / np.maximum(sds[b_], 1e-6))
                   for b_ in tasks] for a_ in tasks])
    im = ax.imshow(M, cmap="magma")
    for i in range(len(tasks)):
        for k in range(len(tasks)):
            ax.text(k, i, f"{M[i,k]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if M[i, k] < M.max() * .6 else "black")
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels([f"t{t}" for t in tasks])
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels([f"t{t}" for t in tasks])
    ax.set_title("F. Mean distance between tasks (in $\\sigma$)", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Is the DINOv2 CLS embedding Gaussian?  —  premise of R10 transport "
                 "$b_j=\\mu_j[\\tau]+\\sigma_j[\\tau]\\,z$\n"
                 f"libero_spatial, {a.n_bins} phase bins, (4 slots x 768) = {D} dims per sample",
                 fontsize=12.3, y=0.958)
    p = out / "gaussianity.png"
    fig.savefig(p, dpi=155)
    print("saved ->", p)

    summ = {}
    for j in tasks:
        sk = sst.skew(zc[j], axis=0); ku = sst.kurtosis(zc[j], axis=0)
        n2 = (zc[j] ** 2).sum(1) / D
        summ[f"task{j}"] = {
            "N": info[j]["n"],
            "skew_median": float(np.median(sk)),
            "kurt_median": float(np.median(ku)),
            "frac_skew_lt_0.5": float(np.mean(np.abs(sk) < .5)),
            "frac_kurt_lt_1": float(np.mean(np.abs(ku) < 1)),
            "z2_over_d_mean": float(n2.mean()), "z2_over_d_std": float(n2.std())}
        s = summ[f"task{j}"]
        print(f"[gauss] task{j}  skew med {s['skew_median']:+.3f}  kurt med {s['kurt_median']:+.3f}  "
              f"|skew|<0.5 {100*s['frac_skew_lt_0.5']:.0f}%  |kurt|<1 {100*s['frac_kurt_lt_1']:.0f}%  "
              f"‖z‖²/d {s['z2_over_d_mean']:.3f}±{s['z2_over_d_std']:.3f}")
    json.dump(summ, (out / "summary.json").open("w"), indent=2)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
