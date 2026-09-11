#!/usr/bin/env python
"""vision encoder 출력의 구조 — t-SNE + 가우시안 등고선.

두 질문을 나눠서 답한다. 섞으면 안 된다.

  Q1 (t-SNE 가 답할 수 있는 것)  궤적 단계(phase bin)가 임베딩 공간에서
     구분되고 순서대로 배열되는가? R10 이 단계별 mu/sigma 를 쓰는 근거다.
     bin 이 완전히 겹치면 단계별 통계가 무의미하고, 궤적처럼 늘어서면 정당하다.

  Q2 (t-SNE 로는 답할 수 없는 것)  분포가 가우시안인가?
     t-SNE 는 밀도와 전역 구조를 의도적으로 왜곡한다. 고차원 가우시안이
     2D 에서 균일한 뭉치나 가짜 군집으로 나온다. 가우시안 판정에 쓰면 안 된다.
     -> 대신 2D 사영 + 등고선으로 본다. 가우시안의 2D 사영은 가우시안이고,
        등고선(1σ/2σ/3σ 타원) 안에 39%/86%/99% 가 들어오면 맞는 것이다.

임베딩 = DINOv2 CLS, 샘플당 (n_obs 2 x n_cam 2, 768) = 3072 차원.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1, R10
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
    X, T, E = [], [], []
    for raw in dl:
        b = B1.prep_batch(policy, B1.to_device(raw, device))
        cls = B1.rgb_cls(policy, b).float()
        n = b["observation.state"].shape[0]
        X.append(cls.view(n, -1, cls.shape[-1]).flatten(1).cpu())
        T.append(R10.phase_bins(raw, ep_len, n_bins).cpu())
        E.append(raw["episode_index"].cpu())
    return torch.cat(X).numpy(), torch.cat(T).numpy(), torch.cat(E).numpy()


def ellipse(ax, xy, n_std, **kw):
    """2D 표본의 n_std 타원. 가우시안이면 카이제곱 분위수만큼이 안에 든다."""
    c = np.cov(xy.T)
    v, w = np.linalg.eigh(c)
    ang = np.degrees(np.arctan2(w[1, -1], w[0, -1]))
    ax.add_patch(Ellipse(xy.mean(0), 2 * n_std * np.sqrt(v[-1]), 2 * n_std * np.sqrt(v[0]),
                         angle=ang, fill=False, **kw))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--perplexity", type=float, default=40)
    ap.add_argument("--pca_dim", type=int, default=50, help="t-SNE 전 PCA 축소 차원")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/R10_tsne")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(a.suite)
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    ck = REPO / "outputs/B2_lam3/libero_spatial_seed42_ours/task_0/checkpoints/005000/pretrained_model"
    cfg = B1.build_cfg(_ns(a), a.task, str(ck), Path("/tmp/r10tsne"))
    policy = make_policy(cfg=cfg.policy, ds_meta=meta); policy.eval()
    ds = make_dataset(cfg)
    X, T, E = collect(policy, cfg, ds, device, a.n_bins, a.batch_size)
    print(f"[tsne] task{a.task}  N={X.shape[0]}  dim={X.shape[1]}  "
          f"에피소드 {len(np.unique(E))}개", flush=True)
    del policy
    torch.cuda.empty_cache()

    print("[tsne] PCA -> t-SNE 계산 중...", flush=True)
    Xp = PCA(n_components=a.pca_dim, random_state=a.seed).fit_transform(X)
    emb = TSNE(n_components=2, perplexity=a.perplexity, init="pca",
               random_state=a.seed, max_iter=1000).fit_transform(Xp)
    print("[tsne] 완료", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))
    fig.subplots_adjust(left=0.055, right=0.965, top=0.80, bottom=0.11, wspace=0.24)

    # 1. t-SNE, phase bin 색 --------------------------------------------------
    ax = axes[0]
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=T, cmap=plt.cm.viridis, s=5, alpha=.75,
                    edgecolors="none")
    fig.colorbar(sc, ax=ax, fraction=.046, label="phase bin (0=start, 9=end)")
    ax.set_title("t-SNE — colored by trajectory phase", fontsize=11.5)
    ax.set_xticks([]); ax.set_yticks([])

    # 2. 한 bin 안에서 가우시안인가 (PCA 2D + 등고선) -------------------------
    ax = axes[1]
    tb = a.n_bins // 2
    m = T == tb
    q = PCA(n_components=2, random_state=a.seed).fit_transform(X[m])
    ax.scatter(q[:, 0], q[:, 1], s=8, alpha=.40, color="#2f6db5", edgecolors="none")
    for k_, ls in ((1, "-"), (2, "--"), (3, ":")):
        ellipse(ax, q, k_, ec="crimson", lw=1.8, ls=ls)
    ci = np.linalg.inv(np.cov(q.T)); dq = q - q.mean(0)
    inside = [float(np.mean(np.sum(dq @ ci * dq, 1) <= k_ ** 2)) for k_ in (1, 2, 3)]
    ax.set_title(f"Within phase bin {tb} — PCA 2D + Gaussian ellipses", fontsize=11.5)
    ax.text(.02, .98, "inside (theory)\n"
            + "\n".join(f"{k_}s  {inside[i]*100:4.1f}%  ({p_*100:.0f}%)"
                         for i, (k_, p_) in enumerate(zip((1, 2, 3), (0.393, 0.865, 0.989)))),
            transform=ax.transAxes, va="top", fontsize=9.5, family="monospace")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    fig.suptitle(
        f"Vision encoder (DINOv2 CLS, {X.shape[1]}-dim) — libero_spatial task {a.task}, "
        f"N={X.shape[0]} frames\n"
        "left: t-SNE shows the phase structure   right: ellipse coverage shows Gaussianity",
        fontsize=12, y=0.955)
    fp = out / f"tsne_task{a.task}.png"
    fig.savefig(fp, dpi=155)
    print("saved ->", fp)
    print(f"[tsne] bin {tb} 타원 안 비율 1s/2s/3s = "
          f"{inside[0]*100:.1f}% / {inside[1]*100:.1f}% / {inside[2]*100:.1f}%  "
          f"(이론 39.3 / 86.5 / 98.9)")
    json.dump({"task": a.task, "N": int(X.shape[0]), "dim": int(X.shape[1]),
               "bin": tb, "inside_1_2_3_sigma": inside,
               "theory": [0.393, 0.865, 0.989]},
              (out / f"summary_task{a.task}.json").open("w"), indent=2)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
