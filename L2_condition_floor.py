#!/usr/bin/env python
"""조건 붕괴 바닥(floor) 측정 — velocity 가 언어 조건으로 갈라졌는지 재는 진단.

────────────────────────────────────────────────────────────────────────────
왜 이걸 재는가
────────────────────────────────────────────────────────────────────────────
현재 태스크 B 를 배우면서 과거 태스크 A 를 앵커로 붙잡는 손실은

    L = ‖v_θ(x_t,t,o,ℓ_B) − v*_B‖²  +  ‖v_old(x_t,t,o,ℓ_A) − v*_A‖²

인데, 만약 정책이 **명령어를 무시**하고 있다면(condition blindness)

    v_θ(x_t,t,o,ℓ_B) ≈ v_θ(x_t,t,o,ℓ_A) ≈ v_θ(x_t,t,o,∅) ≡ u

라 두 항이 같은 변수 u 를 놓고 다투게 되고, 손실은 하나로 합쳐진다.

    L(u) = μ‖u − v*_B‖² + λ‖u − v*_A‖²

이 볼록함수의 최소는 ∇_u L = 2μ(u−v*_B) + 2λ(u−v*_A) = 0 에서

    u* = (μ v*_B + λ v*_A)/(μ+λ)        (가중평균)
    L(u*) = μλ/(μ+λ) ‖v*_A − v*_B‖²      ← 조건이 안 갈라졌을 때의 **바닥**

즉 조건을 무시하는 모델은 아무리 잘 맞춰도 이 값 밑으로 못 내려간다.
실측 손실이 이 바닥보다 **낮으면** 정책이 명령어에 따라 속도장을 갈랐다는 뜻이다.

────────────────────────────────────────────────────────────────────────────
N개 태스크로의 일반화 (이 스크립트가 재는 것)
────────────────────────────────────────────────────────────────────────────
조건이 N개(ℓ_0..ℓ_{N-1}), 가중이 w_i, W = Σ w_i 일 때

    L(u) = Σ_i w_i ‖u − v*_i‖²
    ∇_u L = 2 Σ_i w_i (u − v*_i) = 0
    u* = (Σ_i w_i v*_i)/W ≡ v̄                        (가중평균, 그대로 일반화)

    floor_N ≡ L(u*) = Σ_i w_i ‖v*_i − v̄‖²             ... (가중분산 형태)
                    = (1/W) Σ_{i<j} w_i w_j ‖v*_i − v*_j‖²   ... (쌍 형태)

쌍 형태가 N=2 에서 정확히 μλ/(μ+λ)‖v*_A−v*_B‖² 로 떨어진다. 확인:
    W=μ+λ, 쌍은 (i,j)=(A,B) 하나뿐 -> (1/(μ+λ))·μλ‖v*_A−v*_B‖². ✓
스크립트는 두 형태를 따로 계산해 일치를 출력한다(항등식 수치 검증).

가중치를 균등(w_i=1/N)으로 두면
    floor_N = (1/N) Σ_i ‖v*_i − v̄‖² = (1/N²) Σ_{i<j} ‖v*_i − v*_j‖²
로, 태스크가 늘수록 쌍이 N(N−1)/2 개로 늘어 바닥이 올라간다. 이게
"태스크가 여럿일 때 바닥이 다르다"의 정확한 내용이다.

────────────────────────────────────────────────────────────────────────────
v*_i 를 어디서 가져오는가 — 이 측정의 유일한 근사
────────────────────────────────────────────────────────────────────────────
v*_i(q) = E[a−ε | q, ℓ_i] 는 질의점 q=(x_t,t,o) 에서 조건 i 의 **참** 속도장이다.
task t 의 데이터에는 i=t 의 표본만 있고 i≠t 의 정답은 없다. 그래서 학습된
정책 자신을 그 조건부기댓값의 플러그인 추정량으로 쓴다.

    v*_i(q) ≈ V_i ≡ v_θ(q, ℓ_i)        (flow matching 이 학습하는 대상이 바로 E[a−ε|q,ℓ_i])

★ 순환논법이 아니다. 정책이 정말 조건맹이면 모든 V_i 가 같아져 floor 가 실측
  손실과 **정확히 같아진다**. floor > actual 이 나온다는 것 자체가 V_i 들이
  서로 다르다는 것, 즉 속도장이 갈라졌다는 것의 직접 증거다.

libero_spatial 이 이 측정에 특히 알맞다. 10개 태스크가 전부
"pick up the black bowl <공간 참조> and place it on the plate" 로, 장면·물체가
같고 명령어의 공간 참조만 다르다. 관측 o 만으로는 태스크를 가릴 수 없으므로
언어가 유일한 판별자다.

────────────────────────────────────────────────────────────────────────────
출력 열
────────────────────────────────────────────────────────────────────────────
  N            섞여 있는 조건 수 = t+1 (task t 까지 배웠으므로 ℓ_0..ℓ_t)
  floor        조건맹 최적 예측기 u*=V̄ 가 task t 데이터에서 내는 실제 손실
                 = E‖V̄ − (a−ε)‖²        ← 표의 floor
  ours actual  실측 FM 손실 = E‖V_t − (a−ε)‖²
  floor(∅)     명령어를 빈 문자열로 준 v_θ(q,∅) 의 손실 (또 다른 조건맹 예측기)
  Σw‖V−V̄‖²    이론식의 바닥 그 자체(가중분산). floor − actual 과 맞아야 한다.
  쌍 형태       (1/N²)Σ_{i<j}‖V_i−V_j‖². 위와 항등이어야 한다(검증용).

손실은 전부 fm_loss 와 같은 정규화(원소별 평균, (B,H,A) 전체 평균)를 쓴다.

    bash L2_condition_floor.sh                       # 기본 = 92.5 실행(libero_spatial)
    python L2_condition_floor.py --tasks 1,2,3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "lerobot_lsy" / "src"))

import torch                                                          # noqa: E402
import torch.nn.functional as F                                       # noqa: E402

import l2_codebook_id as L2                                           # noqa: E402
from l2_codebook_id import (                                          # noqa: E402
    NULL_TEXT, build_cfg, cond_tail, encode_lang, make_cond,
    prep_batch, rgb_cls, sample_fm,
)
from lerobot.datasets.factory import make_dataset                     # noqa: E402
from lerobot.datasets.utils import cycle                              # noqa: E402
from lerobot.policies.factory import make_policy                      # noqa: E402
from lerobot.scripts.E0 import episode_sampler, split_episodes, to_device  # noqa: E402
from lerobot.utils.utils import init_logging                          # noqa: E402


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """fm_loss 와 같은 정규화 — (B,H,A) 원소별 평균."""
    return float(F.mse_loss(a, b, reduction="none").mean())


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_root", default=str(REPO / "outputs" / "L2_codebook"
                                             / "libero_spatial_seed42_ours"),
                   help="task_0..task_9 가 들어 있는 디렉터리")
    p.add_argument("--instructions", default=str(REPO / "results" / "l2_codebook_k96_grad"
                                                 / "instructions.json"))
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--tasks", default="1,2,3,4,5,6,7,8,9",
                   help="측정할 행(스테이지). 'task t' 행은 ℓ_0..ℓ_t 를 섞는다.")
    p.add_argument("--n_batches", type=int, default=40)
    p.add_argument("--n_draws", type=int, default=4,
                   help="배치마다 (ε,t) 를 몇 번 다시 뽑을지. tail/DINOv2 를 재활용해 싸다.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--episodes_per_task", type=int, default=45)
    p.add_argument("--lam", type=float, default=3.0, help="λ_level. 학습 가중 floor 용.")
    p.add_argument("--split", choices=["train", "holdout"], default="train")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(REPO / "results" / "L2_condition_floor.txt"))
    return p.parse_args()


def measure_row(a, t: int, instr: dict, device) -> dict:
    """행 t: task_t 체크포인트를 task t 데이터에서 재고, ℓ_0..ℓ_t 를 섞는다."""
    N = t + 1
    ckpt = Path(a.ckpt_root) / f"task_{t}" / "checkpoints" / "last" / "pretrained_model"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    cfg_args = SimpleNamespace(
        suite=a.suite, device=a.device, seed=a.seed, num_workers=a.num_workers,
        batch_size=a.batch_size, steps_per_task=1, log_every=10,
        eval_episodes=1, eval_batch_size=1,
    )
    scratch = Path(os.environ.get("SCRATCH", "/tmp")) / f"cfloor_t{t}"
    cfg = build_cfg(cfg_args, t, str(ckpt), scratch)
    ds = make_dataset(cfg)

    total = ds.meta.total_episodes
    holdout = total - a.episodes_per_task
    tr_eps, ho_eps = split_episodes(cfg.dataset.repo_id, cfg.dataset.root, holdout)
    eps = tr_eps if a.split == "train" else ho_eps

    policy = make_policy(cfg=cfg.policy, ds_meta=ds.meta)
    policy.eval()

    loader = torch.utils.data.DataLoader(
        ds, num_workers=a.num_workers, batch_size=a.batch_size,
        sampler=episode_sampler(cfg, ds, eps), pin_memory=True, drop_last=True,
        multiprocessing_context="spawn" if a.num_workers > 0 else None,
        persistent_workers=a.num_workers > 0,
    )
    it = cycle(loader)
    net = policy.dit_flow.velocity_net

    acc = {k: 0.0 for k in ("actual", "floor", "floor_null", "floor_lam",
                            "var_form", "pair_form", "tgt_pow")}
    n = 0
    for b in range(a.n_batches):
        batch = prep_batch(policy, to_device(next(it), device))
        B = batch["action"].shape[0]
        with torch.no_grad():
            # DINOv2 CLS 는 동결 -> 배치당 한 번. 언어 512 만 갈아 끼운다.
            tail = cond_tail(policy, batch, rgb_cls(policy, batch))          # (B,2064)
            # 조건별 lang 벡터도 (ε,t) 재추출과 무관하므로 미리 만들어 둔다
            conds = [make_cond(encode_lang(policy, [instr[f"task{j}"]] * B), tail)
                     for j in range(N)]                                      # N x (B,2576)
            cond_null = make_cond(encode_lang(policy, [NULL_TEXT] * B), tail)

            for d in range(a.n_draws):
                # 행마다 같은 (ε,t) 를 쓰도록 시드를 배치·draw 로 고정한다
                torch.manual_seed(1_000_003 * b + 97 * d + 7)
                x_t, tt, target = sample_fm(policy, batch)          # (B,H,A),(B,),(B,H,A)
                V = torch.stack([net(noisy_actions=x_t, time=tt, global_cond=c)
                                 for c in conds])                   # (N,B,H,A)
                V_null = net(noisy_actions=x_t, time=tt, global_cond=cond_null)

                Vt = V[t]                                           # 현재 조건 ℓ_t
                Vbar = V.mean(0)                                    # 균등가중 u* = v̄
                # 학습 가중: 현재 μ=1, 과거 각각 λ  (chunk backward 라 과거는 전부 더해진다)
                w = torch.full((N,), a.lam, device=V.device); w[t] = 1.0
                u_lam = (w[:, None, None, None] * V).sum(0) / w.sum()

                acc["actual"] += mse(Vt, target)
                acc["floor"] += mse(Vbar, target)
                acc["floor_null"] += mse(V_null, target)
                acc["floor_lam"] += mse(u_lam, target)
                acc["tgt_pow"] += float(target.pow(2).mean())
                # 이론식 두 형태 (균등가중, w_i = 1/N)
                acc["var_form"] += float((V - Vbar).pow(2).mean(0).mean())
                pair = sum(float((V[i] - V[j]).pow(2).mean())
                           for i in range(N) for j in range(i + 1, N))
                acc["pair_form"] += pair / (N * N)
                n += 1

    del policy, ds, loader, it
    torch.cuda.empty_cache()
    out = {k: v / n for k, v in acc.items()}
    out.update(N=N, n_samples=n * a.batch_size, ckpt=str(ckpt))
    return out


def main():
    a = parse()
    init_logging()
    device = torch.device(a.device)
    instr = json.loads(Path(a.instructions).read_text())
    rows_want = [int(x) for x in a.tasks.split(",") if x.strip()]

    print(f"ckpt_root  {a.ckpt_root}")
    print(f"suite      {a.suite}   split={a.split}   "
          f"{a.n_batches}배치 x {a.n_draws}draw x B{a.batch_size}")

    rows = []
    for t in rows_want:
        t0 = time.perf_counter()
        r = measure_row(a, t, instr, device)
        r["task"] = t
        rows.append(r)
        print(f"  task {t}  N={r['N']}  floor={r['floor']:.5f}  "
              f"actual={r['actual']:.5f}  gap={r['floor']-r['actual']:+.5f}  "
              f"({time.perf_counter()-t0:.0f}s)")

    L = ["=" * 100,
         "조건 붕괴 바닥 (floor) 대 실측 손실 — velocity 가 언어로 갈라졌는가",
         "=" * 100, "",
         "조건맹(condition blindness) 가정:  v_θ(x_t,t,o,ℓ_i) ≈ u  (모든 i 에 대해 같은 출력)",
         "그때 손실은 하나로 합쳐지고,  L(u) = Σ_i w_i ‖u − v*_i‖²  의 최소가 바닥이다.",
         "",
         "    u* = (Σ_i w_i v*_i)/W = v̄",
         "    floor_N = Σ_i w_i ‖v*_i − v̄‖² = (1/W) Σ_{i<j} w_i w_j ‖v*_i − v*_j‖²",
         "    N=2 에서 μλ/(μ+λ)‖v*_A − v*_B‖² 로 환원된다.",
         "",
         f"실측  {a.suite}, 순차 v3+grad 실행(AvgSR 92.5)의 태스크별 체크포인트.",
         f"      행 t = task_t 체크포인트를 task t 데이터({a.split})에서 재고 ℓ_0..ℓ_t 를 섞는다.",
         f"      표본 {a.n_batches}배치 x {a.n_draws}draw x B{a.batch_size}, (ε,t) 는 행마다 동일 시드.",
         "      v*_i 는 정책의 플러그인 추정 V_i = v_θ(q,ℓ_i). 조건맹이면 floor = actual 이 된다.",
         "      손실 정규화는 fm_loss 와 같다((B,H,A) 원소별 평균, action 은 [-1,1]).", "",
         "-" * 100,
         f"{'row':<8}{'N':>4}{'floor':>11}{'ours actual':>13}{'gap':>11}{'gap/floor':>11}"
         f"{'floor(∅)':>11}{'floor(λ가중)':>14}",
         "-" * 100]
    for r in rows:
        g = r["floor"] - r["actual"]
        L.append(f"{'task '+str(r['task']):<8}{r['N']:>4}{r['floor']:>11.5f}"
                 f"{r['actual']:>13.5f}{g:>+11.5f}{100*g/r['floor']:>10.1f}%"
                 f"{r['floor_null']:>11.5f}{r['floor_lam']:>14.5f}")
    L += ["-" * 100, "",
          "항등식 검증 — 이론식의 바닥을 두 형태로 각각 계산 (균등가중 w_i = 1/N)",
          f"{'row':<8}{'N':>4}{'Σw‖V−V̄‖² (분산형)':>22}{'(1/N²)Σ_{i<j}‖V_i−V_j‖² (쌍형)':>34}"
          f"{'floor−actual':>15}",
          "-" * 100]
    for r in rows:
        L.append(f"{'task '+str(r['task']):<8}{r['N']:>4}{r['var_form']:>22.6f}"
                 f"{r['pair_form']:>34.6f}{r['floor']-r['actual']:>15.6f}")
    L += ["-" * 100, "",
          "읽는 법",
          "  · floor > actual 이면 정책이 명령어에 따라 속도장을 갈랐다는 뜻이다.",
          "    조건을 무시하는 정책은 floor 밑으로 내려갈 수 없기 때문이다.",
          "  · gap/floor 는 '바닥 대비 얼마나 아래로 내려갔나'다. 클수록 분리가 뚜렷하다.",
          "  · floor(∅) 는 명령어를 빈 문자열로 준 또 하나의 조건맹 예측기다. floor 와",
          "    비슷하게 나오면 두 방식이 같은 결론을 준다는 뜻이다.",
          "  · floor(λ가중) 은 실제 학습이 쓰는 가중(현재 μ=1, 과거 각 λ=%.1f, chunk backward)"
          % a.lam,
          "    으로 계산한 바닥이다. 과거 쪽 가중이 커서 균등가중보다 높게 나온다.",
          "  · 분산형과 쌍형이 소수점까지 같으면 위 유도가 수치적으로도 맞는다는 확인이다.",
          f"  · 참고로 목표 ‖a−ε‖² 의 원소별 크기는 평균 {rows[0]['tgt_pow']:.3f} 다"
          if rows else "",
          "    (손실 규모를 가늠하는 기준). action 은 MIN_MAX 로 [-1,1] 에 정규화돼 있다.", ""]
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L[16:]))
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
