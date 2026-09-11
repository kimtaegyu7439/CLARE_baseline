#!/usr/bin/env python
"""학습된 모델의 출력이 t 를 따라 명령어별로 갈리는가.

x_t 자체는 (1−t)ε+t·a 라 데이터가 정하는 값이고, libero_spatial 에서는
태스크 간 차이가 태스크 내 차이와 같았다(between/within = 0.98).

여기서는 **모델 출력**을 본다. 같은 (x_t, t, 관측)에 명령어만 ℓ_0..ℓ_9 로 바꿔
    v_j = v(x_t, t, o, s, ℓ_j)
를 뽑고, j 들 사이의 흩어짐을 t 의 함수로 잰다. t→1 (행동 생성 직전)에서
갈리는지가 핵심이다.

바닥선: 같은 명령어로 두 번 forward (dropout 은 eval 이라 0 이므로 정확히 0).
대신 **관측을 바꿨을 때의 흩어짐**을 비교 기준으로 같이 낸다.

    python vt_probe.py --ckpt <path>
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
DEV = torch.device("cuda")
OUT = REPO / "results" / "vt_probe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/libero_spatial/er/"
                    "dit_flow_mt_cl_seed_42_libero_spatial_task_9_er/checkpoints/last/pretrained_model")
    ap.add_argument("--task", type=int, default=0, help="관측을 가져올 태스크")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--suite", default="libero_spatial")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import B1
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy
    from lerobot.datasets.sampler import EpisodeAwareSampler
    class A: pass
    a = A(); a.suite, a.device, a.seed = args.suite, "cuda", 42
    a.num_workers, a.batch_size, a.steps_per_task = 0, 32, 1
    a.log_every, a.eval_episodes, a.eval_batch_size = 100, 2, 2
    a.mode, a.holdout = "ours", 5
    cfg = B1.build_cfg(a, args.task, args.ckpt, OUT / "tmp")
    ds = make_dataset(cfg)
    pol = make_policy(cfg.policy, ds_meta=ds.meta).to(DEV).eval()
    print(f"[vt] 체크포인트 {args.ckpt}")

    sm = EpisodeAwareSampler(ds.episode_data_index,
                             episode_indices_to_use=list(range(ds.meta.total_episodes)),
                             drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                             shuffle=True)
    dl = torch.utils.data.DataLoader(ds, num_workers=0, batch_size=args.n, sampler=sm)
    raw = next(iter(dl))
    b = B1.prep_batch(pol, B1.to_device(raw, DEV))
    n = b["observation.state"].shape[0]
    instr = json.loads((REPO / "results/K1_spatial_10task/instructions.json").read_text())
    texts = [instr[f"task{j}"] for j in range(10)]
    print(f"[vt] 관측 {n} 프레임 (task{args.task}), 명령어 10개")

    with torch.no_grad():
        cls = B1.rgb_cls(pol, b)
        tail = B1.cond_tail(pol, b, cls)                       # (n, 2064) 명령어 무관
        conds = torch.stack([B1.make_cond(B1.encode_lang(pol, [t] * n), tail)
                             for t in texts])                  # (10, n, 2576)
        eps = torch.randn(n, 16, 7, device=DEV)
        act = b["action"]                                      # (n,16,7) 정답
        ts = np.linspace(0.02, 1.0, 50)
        spread, rel, vnorm = [], [], []
        for tv in ts:
            tt = torch.full((n,), float(tv), device=DEV)
            x_t = (1 - tt[:, None, None]) * eps + tt[:, None, None] * act
            V = torch.stack([pol.dit_flow.velocity_net(noisy_actions=x_t, time=tt,
                                                       global_cond=conds[j]) for j in range(10)])
            V = V.flatten(2)                                   # (10, n, 112)
            m = V.mean(0, keepdim=True)
            sd = (V - m).norm(dim=2).mean()                    # 명령어 간 흩어짐
            spread.append(float(sd))
            vnorm.append(float(V.norm(dim=2).mean()))
            rel.append(100 * float(sd / V.norm(dim=2).mean()))
    ts, spread, rel, vnorm = ts, np.array(spread), np.array(rel), np.array(vnorm)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    ax[0].plot(ts, spread, "o-", color="tab:red", ms=3, lw=2, label=r"$\|v_j-\bar v\|$ across 10 instructions")
    ax[0].plot(ts, vnorm, "s--", color="0.5", ms=3, lw=1.6, label=r"$\|v\|$ (scale)")
    ax[0].set_ylabel("velocity spread / norm"); ax[0].legend(fontsize=9)
    ax[1].plot(ts, rel, "o-", color="tab:red", ms=3, lw=2)
    ax[1].set_ylabel("relative spread  $\\|v_j-\\bar v\\| / \\|v\\|$  (%)")
    for a_ in ax:
        a_.set_xlabel("flow-matching time  t"); a_.grid(alpha=.3); a_.set_xlim(0, 1)
    fig.suptitle("Does the trained policy's output diverge across instructions as $t\\to1$ ?\n"
                 f"ER final checkpoint (AvgSR 86.0), {n} frames from task{args.task}, same $x_t$ and observation")
    plt.tight_layout(); plt.savefig(OUT / "vt_instruction_spread.png", dpi=140)
    L = [f"명령어 간 velocity 흩어짐 — ER 최종 체크포인트, task{args.task} 관측 {n} 프레임", "",
         f"{'t':>6}{'‖v_j−v̄‖':>12}{'‖v‖':>10}{'상대%':>9}"]
    for i in range(0, 50, 5):
        L.append(f"{ts[i]:6.2f}{spread[i]:12.4f}{vnorm[i]:10.4f}{rel[i]:9.2f}")
    L.append(f"{ts[-1]:6.2f}{spread[-1]:12.4f}{vnorm[-1]:10.4f}{rel[-1]:9.2f}")
    (OUT / "vt_instruction_spread.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\nsaved -> {OUT/'vt_instruction_spread.png'}")


if __name__ == "__main__":
    main()
