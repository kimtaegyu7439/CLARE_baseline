#!/usr/bin/env python
"""분할 방식별 잔차 PCA — 시간 bin vs state 셀.

같은 8000 프레임의 vision embedding(DINOv2 CLS, 3072-d, 동결)을 놓고,
**어떤 기준으로 묶어 평균을 빼면 남는 잔차가 더 저차원인가** 를 비교한다.

    없음      o − 전역평균                     (기준선)
    시간 10   o − 평균[τ]        τ = floor(10·frame_idx/ep_len)
    state 10  o − 평균[k]        k = 표준화 s 공간 k-means(K=10)
    state 96  o − 평균[k]        K=96 (실제 코드북 설정)
    state 96 + grad             o − (m_k + A_k δ)   셀별 접평면까지 뺀 잔차

각 잔차 집합에 PCA 를 걸어 **누적 분산 비율**(고유값 큰 순서로 더한 값 / 전체 합)을
성분 개수에 대해 그린다. 곡선이 왼쪽 위로 붙을수록 그 분할이 남긴 잔차가 저차원이다.

    python pca_probe.py [--n 8000] [--task 0] [--gpu 1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


DEV = torch.device("cuda")


def collect(args):
    """(o, s, tau) 를 n 프레임 모은다. B1 의 셋업을 그대로 쓴다."""
    import B1
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy
    from lerobot.datasets.sampler import EpisodeAwareSampler

    class A:  # build_cfg 가 보는 최소 인자
        pass
    a = A()
    a.suite, a.device, a.seed = args.suite, "cuda", 42
    a.num_workers, a.batch_size, a.steps_per_task = args.workers, 32, 1
    a.log_every, a.eval_episodes, a.eval_batch_size = 100, 2, 2
    a.mode, a.holdout = "ours", 5
    cfg = B1.build_cfg(a, args.task, args.policy, REPO / "results" / "pca_probe" / "tmp")
    dataset = make_dataset(cfg)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta).to(DEV).eval()
    import R10
    ep_len = R10.episode_lengths(dataset)

    sampler = EpisodeAwareSampler(
        dataset.episode_data_index,
        episode_indices_to_use=list(range(dataset.meta.total_episodes)),
        drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0), shuffle=False)
    loader = torch.utils.data.DataLoader(dataset, num_workers=args.workers,
                                         batch_size=32, sampler=sampler, drop_last=False)
    O, S, T = [], [], []
    got = 0
    with torch.no_grad():
        for raw in loader:
            b = B1.prep_batch(policy, B1.to_device(raw, DEV))
            cls = B1.rgb_cls(policy, b)
            n = b["observation.state"].shape[0]
            O.append(cls.reshape(n, -1).float().cpu())
            S.append(b["observation.state"].flatten(1).float().cpu())
            T.append(R10.phase_bins(raw, ep_len, 10).cpu())
            got += n
            if got >= args.n:
                break
    o = torch.cat(O)[:args.n]; s = torch.cat(S)[:args.n]; t = torch.cat(T)[:args.n]
    print(f"[probe] 수집 {o.shape[0]} 프레임   o {tuple(o.shape)}  s {tuple(s.shape)}  "
          f"τ 분포 {torch.bincount(t, minlength=10).tolist()}")
    return o.cuda(), s.cuda(), t.cuda()


def cum_ratio(resid: torch.Tensor, top: int):
    """(누적 분산 비율 % (top,), 총분산, d_eff). 고유값 = 잔차 공분산의 것."""
    x = resid - resid.mean(0)
    lam = torch.linalg.svdvals(x.double()).pow(2) / (x.shape[0] - 1)
    tot = float(lam.sum())
    deff = float(lam.sum() ** 2 / lam.pow(2).sum())
    c = torch.cumsum(lam, 0) / lam.sum()
    return (100 * c[:top]).cpu().numpy(), tot, deff


def group_resid(o, lab, K):
    m = torch.zeros(K, o.shape[1], device=o.device, dtype=torch.float64)
    m.index_add_(0, lab, o.double())
    cnt = torch.bincount(lab, minlength=K).clamp_min(1).double()[:, None]
    return (o.double() - (m / cnt)[lab]).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--policy", default="/home/sa090180/Models/dit_flow_mt_libero_90_pretrain")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--out", default="results/pca_probe")
    ap.add_argument("--recollect", action="store_true")
    args = ap.parse_args()

    out = REPO / args.out; out.mkdir(parents=True, exist_ok=True)
    cache = out / f"frames_task{args.task}.pt"
    if cache.exists() and not args.recollect:
        d = torch.load(cache, weights_only=False)
        o, s, tau = d["o"].cuda(), d["s"].cuda(), d["tau"].cuda()
        print(f"[probe] 캐시 사용 {o.shape[0]} 프레임 ({cache})")
    else:
        o, s, tau = collect(args)
        torch.save({"o": o.cpu(), "s": s.cpu(), "tau": tau.cpu()}, cache)
    import l2_codebook as CB

    curves = {}
    curves["없음 (전역평균)"] = cum_ratio(o - o.mean(0), args.top)
    curves["시간 bin 10"] = cum_ratio(group_resid(o, tau, 10), args.top)
    for K in (10, 96):
        cb = CB.build_codebook(s, o, K, seed=42)
        zs = (s - cb["mean_s"].cuda()) / cb["std_s"].cuda()
        lab = torch.cdist(zs, cb["c"].cuda()).argmin(1)
        curves[f"state 셀 {K}"] = cum_ratio(group_resid(o, lab, cb["K_eff"]), args.top)
        if K == 96:
            cbg = CB.build_codebook(s, o, K, seed=42, grad=True)
            zsg = (s - cbg["mean_s"].cuda()) / cbg["std_s"].cuda()
            lg = torch.cdist(zsg, cbg["c"].cuda()).argmin(1)
            A = cbg["A"].float().cuda(); mo = cbg["m"].cuda()
            zb = cbg["zbar"].cuda()
            d = zsg - zb[lg]                                   # (N,16)
            # pred_i = m_{k(i)} + A_{k(i)} δ_i   — 배치로 (N,3072,16)@(N,16,1)
            pred = mo[lg] + torch.bmm(A[lg], d.unsqueeze(2)).squeeze(2)
            curves["state 셀 96 + 기울기"] = cum_ratio(o - pred, args.top)

    # 저장
    xs = np.arange(1, args.top + 1)
    with (out / "pca_curves.csv").open("w") as f:
        f.write("k," + ",".join(curves) + "\n")
        for i, k in enumerate(xs):
            f.write(f"{k}," + ",".join(f"{curves[c][0][i]:.4f}" for c in curves) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    EN = {"없음 (전역평균)": "none (global mean)",
          "시간 bin 10": "time bin (K=10)",
          "state 셀 10": "state cell (K=10)",
          "state 셀 96": "state cell (K=96)",
          "state 셀 96 + 기울기": "state cell K=96 + gradient"}
    STY = {"없음 (전역평균)": dict(color="0.45", ls="--"),
           "시간 bin 10": dict(color="tab:orange"),
           "state 셀 10": dict(color="tab:blue"),
           "state 셀 96": dict(color="tab:green"),
           "state 셀 96 + 기울기": dict(color="tab:red")}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, (y, tot, de) in curves.items():
        ax[0].plot(xs, y, lw=2, label=f"{EN[name]}  (var {tot:.0f}, d_eff {de:.1f})",
                   **STY[name])
        ax[1].plot(xs, y * tot / 100, lw=2, **STY[name])
    ax[0].set_xlabel("number of eigenvalues summed")
    ax[0].set_ylabel("cumulative variance ratio (%)")
    ax[0].set_title("normalised (each curve / its own total)")
    ax[0].set_ylim(0, 100)
    ax[1].set_xlabel("number of eigenvalues summed")
    ax[1].set_ylabel("cumulative variance (absolute)")
    ax[1].set_title("absolute — lower = partition removed more structure")
    for a in ax:
        a.grid(alpha=.3); a.set_xlim(1, args.top)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Residual PCA by partition — {args.suite} task{args.task}, "
                 f"{o.shape[0]} frames, DINOv2 CLS 3072-d (frozen)")
    plt.tight_layout(); plt.savefig(out / "pca_curves.png", dpi=140)
    # 요약
    base = curves["없음 (전역평균)"][1]
    L = [f"분할별 잔차 PCA — {args.suite} task{args.task}, {o.shape[0]} 프레임, o 3072-d", "",
         "누적 분산 비율 (%)  — 각 곡선을 **자기 총분산으로** 정규화한 값",
         "  낮을수록 스펙트럼이 평평하다(= 지배 방향이 이미 제거됨)", ""]
    L.append(f"{'분할':<24}" + "".join(f"{('k='+str(k)):>9}" for k in (1, 5, 10, 20, 50)))
    for name, (y, tot, de) in curves.items():
        L.append(f"{name:<24}" + "".join(f"{y[k-1]:9.2f}" for k in (1, 5, 10, 20, 50)))
    L += ["", "절대 잔차 분산 — 어느 분할이 구조를 더 많이 제거했는가 (핵심 지표)", "",
          f"{'분할':<24}{'총분산':>12}{'전역 대비':>10}{'d_eff':>9}"]
    for name, (y, tot, de) in curves.items():
        L.append(f"{name:<24}{tot:12.1f}{tot/base:10.3f}{de:9.2f}")
    L += ["", "주의  그룹 수가 많을수록 평균을 더 많이 빼므로 자동으로 유리하다.",
          "      시간 bin 10 과 공정하게 비교할 것은 **state 셀 10** 이다."]
    (out / "pca_summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsaved -> {out/'pca_curves.png'}, {out/'pca_curves.csv'}, {out/'pca_summary.txt'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
