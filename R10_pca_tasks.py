#!/usr/bin/env python
"""bin 5 임베딩의 공통 PCA 2D — 태스크별 색.

네 태스크의 phase bin 5 프레임만 모아 **하나의 PCA 공간**에 사영한다.
  겹치면   지배적 변동이 태스크 간 공유 -> 수송(b_j = mu_j + sigma_j·z) 유리
  갈라지면 지배적 변동이 태스크 정체성 -> 수송이 근사에 불과
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
def collect_bin(policy, cfg, dataset, device, n_bins, want, batch_size):
    sp = EpisodeAwareSampler(dataset.episode_data_index,
                             drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                             shuffle=False)
    dl = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=batch_size,
                                     sampler=sp, drop_last=False)
    ep_len = R10.episode_lengths(dataset)
    out = []
    for raw in dl:
        tau = R10.phase_bins(raw, ep_len, n_bins)
        m = (tau == want)
        if not bool(m.any()):
            continue
        b = B1.prep_batch(policy, B1.to_device(raw, device))
        cls = B1.rgb_cls(policy, b).float()
        n = b["observation.state"].shape[0]
        out.append(cls.view(n, -1, cls.shape[-1]).flatten(1)[m.to(device)].cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0,1,2,3")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--bin", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/R10_tsne")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(a.suite)
    tasks = [int(x) for x in a.tasks.split(",")]
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    ck = REPO / "outputs/B2_lam3/libero_spatial_seed42_ours/task_0/checkpoints/005000/pretrained_model"
    policy = make_policy(cfg=B1.build_cfg(_ns(a), 0, str(ck), Path("/tmp/rpca")).policy,
                         ds_meta=meta)
    policy.eval()

    X, lab = [], []
    for j in tasks:
        cfg = B1.build_cfg(_ns(a), j, str(ck), Path("/tmp/rpca"))
        ds = make_dataset(cfg)
        x = collect_bin(policy, cfg, ds, device, a.n_bins, a.bin, a.batch_size)
        X.append(x); lab.append(np.full(len(x), j))
        print(f"[pca] task{j}  bin{a.bin}  N={len(x)}", flush=True)
        del ds
        torch.cuda.empty_cache()
    X = np.concatenate(X); lab = np.concatenate(lab)

    p = PCA(n_components=2, random_state=a.seed).fit(X)     # ★ 네 태스크 공통 공간
    P = p.transform(X)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    fig.subplots_adjust(left=.115, right=.975, top=.87, bottom=.10)
    C = plt.cm.tab10.colors
    for i, j in enumerate(tasks):
        m = lab == j
        ax.scatter(P[m, 0], P[m, 1], s=9, alpha=.45, color=C[i],
                   edgecolors="none", label=f"task {j}  (N={int(m.sum())})")
    ax.set_xlabel(f"PC1 ({p.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({p.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=9.5, frameon=False)
    ax.set_title(f"DINOv2 CLS ({X.shape[1]}-dim), phase bin {a.bin} only\n"
                 "shared PCA space — do tasks overlap or separate?", fontsize=11.5)
    fp = out / f"pca_tasks_bin{a.bin}.png"
    fig.savefig(fp, dpi=155)
    print("saved ->", fp)

    # 겹침 정도를 숫자로도 남긴다
    cen = np.array([P[lab == j].mean(0) for j in tasks])
    within = np.mean([P[lab == j].std(0).mean() for j in tasks])
    between = np.mean([np.linalg.norm(cen[i] - cen[k])
                       for i in range(len(tasks)) for k in range(i + 1, len(tasks))])
    print(f"[pca] 태스크 내 산포 {within:.2f}   태스크 간 중심거리 {between:.2f}   "
          f"비 {between/max(within,1e-9):.2f}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
