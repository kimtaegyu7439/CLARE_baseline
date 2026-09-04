#!/usr/bin/env python
"""denoising 한 스텝이 태스크 차이를 키우는가.

각 태스크 j 에 대해 **자기 것**으로 좌표를 만든다:
    관측 o_j, 상태 s_j, 명령어 ℓ_j, 정답 행동 a_j   (전부 태스크 j 의 프레임)
    x_t^j = (1−t)·ε + t·a_j                        ε 는 태스크끼리 공유

한 오일러 스텝:
    x'^j = x_t^j + dt · v(x_t^j, t, o_j, s_j, ℓ_j)      dt = 1/100 (추론과 동일)

재는 것
    before  태스크 간 ‖x_t^j − x̄_t‖          입력 흩어짐
    after   태스크 간 ‖x'^j − x̄'‖             출력 흩어짐
    ratio   after/before                      >1 이면 denoising 이 태스크를 벌린다
    바닥선  task0 을 두 조각으로 나눈 within  (같은 태스크끼리의 흩어짐)

추가: t 에서 시작해 t=1 까지 완전 적분한 뒤의 최종 행동 흩어짐(몇 개 t 에서만).
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
DEV = torch.device("cuda")
OUT = REPO / "results" / "denoise_probe"


def collect(pol_holder, args):
    import B1
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy
    from lerobot.datasets.sampler import EpisodeAwareSampler
    class A: pass
    a = A(); a.suite, a.device, a.seed = args.suite, "cuda", 42
    a.num_workers, a.batch_size, a.steps_per_task = 0, 32, 1
    a.log_every, a.eval_episodes, a.eval_batch_size = 100, 2, 2
    a.mode, a.holdout = "ours", 5
    packs = []
    for j in range(args.tasks):
        cfg = B1.build_cfg(a, j, args.ckpt, OUT / "tmp")
        ds = make_dataset(cfg)
        if pol_holder[0] is None:
            pol_holder[0] = make_policy(cfg.policy, ds_meta=ds.meta).to(DEV).eval()
        pol = pol_holder[0]
        sm = EpisodeAwareSampler(ds.episode_data_index,
                                 episode_indices_to_use=list(range(ds.meta.total_episodes)),
                                 drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                                 shuffle=True)
        dl = torch.utils.data.DataLoader(ds, num_workers=0, batch_size=args.n, sampler=sm)
        b = B1.prep_batch(pol, B1.to_device(next(iter(dl)), DEV))
        with torch.no_grad():
            cls = B1.rgb_cls(pol, b)
            tail = B1.cond_tail(pol, b, cls)
        packs.append({"tail": tail.cpu(), "act": b["action"].cpu()})
        print(f"  task{j} 수집 {b['action'].shape[0]}", flush=True)
        del ds, dl
    return packs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/libero_spatial/er/"
                    "dit_flow_mt_cl_seed_42_libero_spatial_task_9_er/checkpoints/last/pretrained_model")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--suite", default="libero_spatial")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import B1
    holder = [None]
    packs = collect(holder, args)
    pol = holder[0]
    instr = json.loads((REPO / "results/K1_spatial_10task/instructions.json").read_text())
    K, n = args.tasks, args.n

    tails = torch.stack([p["tail"] for p in packs]).to(DEV)        # (K,n,2064)
    acts = torch.stack([p["act"] for p in packs]).to(DEV)          # (K,n,16,7)
    with torch.no_grad():
        langs = torch.stack([B1.encode_lang(pol, [instr[f"task{j}"]] * n) for j in range(K)])
        conds = torch.cat([langs, tails], dim=-1)                  # (K,n,2576)
    vel = lambda x, t, c: pol.dit_flow.velocity_net(noisy_actions=x, time=t, global_cond=c)

    g = torch.Generator(device=DEV).manual_seed(0)
    eps = torch.randn(n, 16, 7, device=DEV, generator=g)
    dt = 1.0 / 100
    ts = np.linspace(0.02, 0.98, 25)
    bef, aft, wb, wa = [], [], [], []
    with torch.no_grad():
        for tv in ts:
            tt = torch.full((n,), float(tv), device=DEV)
            tc = tt[:, None, None]
            X = (1 - tc)[None] * eps[None] + tc[None] * acts       # (K,n,16,7)
            Xp = torch.stack([X[j] + dt * vel(X[j], tt, conds[j]) for j in range(K)])
            f = lambda Z: float((Z - Z.mean(0, keepdim=True)).flatten(2).norm(dim=2).mean())
            bef.append(f(X)); aft.append(f(Xp))
            # 바닥선: task0 배치를 **between 과 같은 K 조각**으로 나눈다.
            # ‖x−평균‖ 은 그룹 수에 의존한다(2그룹 0.707σ vs 10그룹 0.949σ, 1.34배).
            # 2조각으로 재면 within 이 구조적으로 작게 나와 비교가 불공정하다.
            h = n // K
            X0 = torch.stack([X[0, g*h:(g+1)*h] for g in range(K)])
            Xp0 = torch.stack([Xp[0, g*h:(g+1)*h] for g in range(K)])
            wb.append(f(X0)); wa.append(f(Xp0))
    bef, aft, wb, wa = map(np.array, (bef, aft, wb, wa))

    # 완전 적분: t 에서 시작해 1 까지
    finals = {}
    with torch.no_grad():
        for t0 in (0.1, 0.5, 0.9):
            X = torch.stack([(1 - t0) * eps + t0 * acts[j] for j in range(K)])
            steps = int(round((1 - t0) / dt))
            for s in range(steps):
                tv = t0 + s * dt
                tt = torch.full((n,), tv, device=DEV)
                X = torch.stack([(X[j] + dt * vel(X[j], tt, conds[j])).clamp(-1, 1) for j in range(K)])
            finals[t0] = float((X - X.mean(0, keepdim=True)).flatten(2).norm(dim=2).mean())

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(ts, bef, "o-", color="tab:blue", ms=3, lw=2, label="before step  $x_t$")
    ax[0].plot(ts, aft, "s-", color="tab:red", ms=3, lw=2, label="after 1 Euler step  $x_{t+dt}$")
    ax[0].plot(ts, wb, "--", color="0.5", lw=1.6, label=f"within task0 (baseline, same {10} groups)")
    ax[0].set_ylabel("spread across tasks  $\\|x^j-\\bar x\\|$"); ax[0].legend(fontsize=8)
    ax[0].set_title("absolute spread")
    ax[1].plot(ts, 100 * (aft / bef - 1), "o-", color="tab:red", ms=3, lw=2, label="between tasks")
    ax[1].plot(ts, 100 * (wa / wb - 1), "--", color="0.5", lw=1.8, label="within task0")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_ylabel("change from one denoising step  (%)"); ax[1].legend(fontsize=9)
    ax[1].set_title("does one step amplify (+) or shrink (−) the gap?")
    for a_ in ax:
        a_.set_xlabel("flow-matching time  t"); a_.grid(alpha=.3); a_.set_xlim(0, 1)
    fig.suptitle("One denoising step with each task's own $x_t$, observation and instruction\n"
                 f"ER final checkpoint, {n} frames/task, {K} tasks, dt=1/100 (inference schedule)")
    plt.tight_layout(); plt.savefig(OUT / "denoise_step.png", dpi=140)

    L = [f"denoising 한 스텝의 태스크 분리 효과 — ER 최종, 태스크 {K}, 프레임 {n}, dt=1/100", "",
         f"{'t':>6}{'before':>10}{'after':>10}{'변화%':>9}{'within(b)':>11}{'within 변화%':>13}"]
    for i in range(0, len(ts), 3):
        L.append(f"{ts[i]:6.2f}{bef[i]:10.4f}{aft[i]:10.4f}{100*(aft[i]/bef[i]-1):9.3f}"
                 f"{wb[i]:11.4f}{100*(wa[i]/wb[i]-1):13.3f}")
    L += ["", "t 에서 시작해 t=1 까지 완전 적분한 뒤 최종 행동의 태스크 간 흩어짐"]
    for t0, v in finals.items():
        L.append(f"  t0={t0:.1f} 에서 출발  →  최종 흩어짐 {v:.4f}")
    (OUT / "denoise_step.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\nsaved -> {OUT/'denoise_step.png'}")


if __name__ == "__main__":
    main()
