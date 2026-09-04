#!/usr/bin/env python
"""x_t 의 태스크 의존성 — t 가 커질수록 얼마나 갈리는가.

    x_t = (1−t)·ε + t·a          a = 행동 청크 (16,7), 정규화 [-1,1]

같은 ε 을 쓰면  x_t^A − x_t^B = t·(a_A − a_B)  이므로 절대 차이는 t 에 선형이다.
의미가 있는 건 **상대 차이** ‖Δx_t‖ / ‖x_t‖ 다 — t 가 작으면 ε 이 지배해 태스크
정보가 묻히고, t→1 이면 행동이 지배해 완전히 갈린다.

바닥선: 같은 태스크 안의 서로 다른 프레임끼리의 차이. 태스크 간 차이가 이보다
크지 않으면 "x_t 가 태스크를 구분한다"는 전제가 성립하지 않는다.

    python xt_probe.py [--n 1500] [--tasks 10]
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
DEV = torch.device("cuda")
OUT = REPO / "results" / "xt_probe"


def collect_all(args):
    """태스크 0..K-1 의 정규화된 행동 청크. 정책은 **한 번만** 만든다.

    B1 도 태스크 0 에서 policy 를 만들고 끝까지 이어 쓰므로(B1.py:703),
    정규화 통계가 태스크 0 것으로 고정되는 게 실제 파이프라인과 같다.
    태스크마다 make_policy 를 반복하면 DataLoader 워커가 죽는다.
    """
    import B1
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy
    from lerobot.datasets.sampler import EpisodeAwareSampler
    class A: pass
    a = A(); a.suite, a.device, a.seed = args.suite, "cuda", 42
    a.num_workers, a.batch_size, a.steps_per_task = 0, 32, 1
    a.log_every, a.eval_episodes, a.eval_batch_size = 100, 2, 2
    a.mode, a.holdout = "ours", 5
    pol = None
    out = []
    for task in range(args.tasks):
        cfg = B1.build_cfg(a, task, args.policy, OUT / "tmp")
        ds = make_dataset(cfg)
        if pol is None:
            pol = make_policy(cfg.policy, ds_meta=ds.meta).to(DEV).eval()
        sm = EpisodeAwareSampler(ds.episode_data_index,
                                 episode_indices_to_use=list(range(ds.meta.total_episodes)),
                                 drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                                 shuffle=False)
        dl = torch.utils.data.DataLoader(ds, num_workers=0, batch_size=64, sampler=sm)
        acc, got = [], 0
        with torch.no_grad():
            for raw in dl:
                b = B1.prep_batch(pol, B1.to_device(raw, DEV))
                acc.append(b["action"].float().cpu()); got += b["action"].shape[0]
                if got >= args.n: break
        out.append(torch.cat(acc)[:args.n])
        print(f"  task{task}  {out[-1].shape[0]} 청크", flush=True)
        del ds, dl
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--policy", default="/home/sa090180/Models/dit_flow_mt_libero_90_pretrain")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--all_horizon", dest="exec_only", action="store_false", default=True,
                    help="16 스텝 전체로 비교(기본은 실행되는 앞 8 스텝만)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cache = OUT / "actions.pt"
    if cache.exists():
        A = torch.load(cache, weights_only=False)
        print(f"[xt] 캐시 사용 — {len(A)} 태스크")
    else:
        A = collect_all(args)
        torch.save(A, cache)
    A = [x.to(DEV) for x in A]
    print("행동 청크(원본)", tuple(A[0].shape), " 태스크", len(A))
    # ★ 실행 구간은 [1:9] 다. [0:8] 이 아니다.
    #   action_delta_indices = [-1, 0, 1, ..., 14] 이므로 배열 index 0 은 t-1(과거)이고
    #   index 1 이 "지금"이다 (modeling_dit_flow_mt.py:1230-1241).
    #       index 0    -> t-1       버림
    #       index 1~8  -> t ~ t+7   실행 (n_action_steps=8, 0.4초)
    #       index 9~15 -> t+8~t+14  미리보기, 버림
    if args.exec_only:
        A = [x[:, 1:9] for x in A]
        print(f"실행 구간만 사용 → {tuple(A[0].shape)}  (index 1..8 = t ~ t+7)")

    g = torch.Generator(device=DEV).manual_seed(0)
    M = 4000                                   # 비교 쌍 수
    ts = np.linspace(0.0, 1.0, 51)
    cur = A[0]                                 # "현재 태스크" = task0

    def pair_stats(src, dst):
        i = torch.randint(len(src), (M,), device=DEV, generator=g)
        j = torch.randint(len(dst), (M,), device=DEV, generator=g)
        da = (src[i] - dst[j]).flatten(1)                       # (M, 112)
        eps = torch.randn(M, da.shape[1], device=DEV, generator=g)
        a_i = src[i].flatten(1)
        abs_, rel_ = [], []
        for t in ts:
            dx = t * da                                         # 같은 ε -> ε 상쇄
            xt = (1 - t) * eps + t * a_i
            abs_.append(float(dx.norm(dim=1).mean()))
            rel_.append(float((dx.norm(dim=1) / xt.norm(dim=1).clamp_min(1e-8)).mean()) * 100)
        return np.array(abs_), np.array(rel_)

    res = {}
    res["within task0"] = pair_stats(cur, cur)
    for j in range(1, args.tasks):
        res[f"task0 vs task{j}"] = pair_stats(cur, A[j])

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = plt.cm.viridis(np.linspace(0, .92, args.tasks - 1))
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.3))
    for k, (nm, (ab, rl)) in enumerate(res.items()):
        if nm.startswith("within"):
            for a_, y in ((ax[0], ab), (ax[1], rl)):
                a_.plot(ts, y, color="k", ls="--", lw=2.6, label="within task0 (baseline)", zorder=5)
        else:
            c = cm[k - 1]
            ax[0].plot(ts, ab, color=c, lw=1.8, label=nm.replace("task0 vs ", ""))
            ax[1].plot(ts, rl, color=c, lw=1.8)
    ax[0].set_ylabel(r"$\|x_t^{A}-x_t^{B}\|$   (absolute)")
    ax[0].set_title("absolute difference — linear in t by construction")
    ax[1].set_ylabel(r"$\|x_t^{A}-x_t^{B}\| \, / \, \|x_t\|$   (%)")
    ax[1].set_title("relative difference — how much of $x_t$ is task-specific")
    for a_ in ax:
        a_.set_xlabel("flow-matching time  t");  a_.grid(alpha=.3);  a_.set_xlim(0, 1)
    ax[0].legend(fontsize=8, ncol=2, loc="upper left")
    ax[1].legend(fontsize=8, loc="upper left")
    seg = "executed steps only (idx 1:9 = t..t+7)" if args.exec_only else "all 16 steps"
    fig.suptitle("How task-specific is the anchor coordinate $x_t=(1-t)\\varepsilon+t\\,a$ ?\n"
                 f"libero_spatial, {args.n} action chunks/task, shared $\\varepsilon$, {seg}")
    plt.tight_layout(); plt.savefig(OUT / "xt_task_gap.png", dpi=140)

    L = [f"x_t 의 태스크 의존성 — libero_spatial, 태스크당 {args.n} 청크", "",
         f"{'비교':<20}" + "".join(f"{('t='+f'{t:.1f}'):>10}" for t in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)),
         "절대 ‖Δx_t‖"]
    idx = [int(t * 50) for t in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)]
    for nm, (ab, rl) in res.items():
        L.append(f"  {nm:<18}" + "".join(f"{ab[i]:10.3f}" for i in idx))
    L.append("상대 ‖Δx_t‖/‖x_t‖ (%)")
    for nm, (ab, rl) in res.items():
        L.append(f"  {nm:<18}" + "".join(f"{rl[i]:10.1f}" for i in idx))
    (OUT / "xt_task_gap.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\nsaved -> {OUT/'xt_task_gap.png'}")


if __name__ == "__main__":
    main()      # num_workers=0 — 멀티프로세싱 안 씀
