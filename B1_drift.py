#!/usr/bin/env python
"""B1 원인 규명 — teacher 드리프트와 손실 항 그래디언트 균형.

두 가설을 구분한다.
  (A) lambda_anchor=1.0 이 작다  -> 앵커 그래디언트가 FM 대비 미미할 것
  (B) rolling teacher 세대 누적 -> 과거 태스크의 속도장이 "유능했던 시점"에서
      스테이지마다 멀어질 것. 앵커는 이미 오염된 teacher 를 목표로 삼으므로
      아무리 세게 당겨도 원본으로 돌아가지 못한다.

측정 1 — drift[k][j]
    태스크 j 관측 위에서, 스테이지 k 모델과 스테이지 j 모델(= j 를 막 배워 유능하던
    시점)의 속도장 상대 거리:
        ‖v_k(x,t,o_j,ℓ_j) − v_j(x,t,o_j,ℓ_j)‖ / ‖v_j(x,t,o_j,ℓ_j)‖
    k−j 가 커질수록 커지면 (B) 가 지지된다. SR 과 대조한다.

측정 2 — 그래디언트 균형
    스테이지 k 를 재현해 L_FM 과 λ·L_anchor 각각의 그래디언트 노름을 따로 잰다.
    비율이 100:1 수준이면 (A) 가 지지된다.

학습하지 않는다. 저장된 체크포인트만 읽는다.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1

from lerobot.datasets.factory import make_dataset                    # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.sampler import EpisodeAwareSampler             # noqa: E402
from lerobot.policies.factory import make_policy                     # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging  # noqa: E402


def _ns(args, task_k):
    return argparse.Namespace(
        suite=args.suite, device=args.device, seed=args.seed,
        num_workers=0, batch_size=args.batch_size, steps_per_task=1,
        log_every=100, eval_episodes=1, eval_batch_size=1,
        mode="drift", p_drop=0.0, lambda_anchor=args.lambda_anchor)


def probe_batches(args, policy, j, device, meta_cache):
    cfg = B1.build_cfg(_ns(args, j), j, args.dummy_ckpt, Path("/tmp/b1_drift_unused"))
    ds = make_dataset(cfg)
    sampler = EpisodeAwareSampler(
        ds.episode_data_index,
        drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0), shuffle=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=True)
    torch.manual_seed(args.seed)
    out = []
    it = iter(loader)
    for _ in range(args.n_batches):
        out.append(B1.to_device(next(it), device))
    del ds, loader, it
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default="outputs/B1_libero_spatial/libero_spatial_seed42_ours")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--num_tasks", type=int, default=10)
    ap.add_argument("--steps_tag", default="020000")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=3)
    ap.add_argument("--lambda_anchor", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/B1_drift")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    init_logging()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(args.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(args.suite)
    N = args.num_tasks

    ckpt = {}
    for k in range(N):
        p = Path(args.ckpt_root) / f"task_{k}" / "checkpoints" / args.steps_tag / "pretrained_model"
        if p.is_dir():
            ckpt[k] = str(p)
    K = sorted(ckpt)
    args.dummy_ckpt = ckpt[K[0]]
    print(f"[drift] 스테이지 {K}")

    instr = [B1.task_instruction(f"{ds_prefix}{j}") for j in range(N)]
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")

    def load(k):
        cfg = B1.build_cfg(_ns(args, 0), 0, ckpt[k], Path("/tmp/b1_drift_unused"))
        pol = make_policy(cfg=cfg.policy, ds_meta=meta)
        pol.eval()
        return pol

    # 태스크별 고정 probe 배치 (모든 모델이 같은 x_t, t 를 보도록 미리 만든다)
    tmp = load(K[0])
    batches, fm_inputs = {}, {}
    for j in K:
        bs = probe_batches(args, tmp, j, device, meta)
        prepped = [B1.prep_batch(tmp, b) for b in bs]
        batches[j] = prepped
        fixed = []
        for b in prepped:
            torch.manual_seed(args.seed + j)
            x_t, t, target = B1.sample_fm(tmp, b)
            fixed.append((x_t, t, target))
        fm_inputs[j] = fixed
    del tmp
    torch.cuda.empty_cache()

    # ── 측정 1: drift[k][j] ──────────────────────────────────────────────────
    ref_v = {}          # ckpt_j 의 태스크 j 속도장 (기준)
    drift = np.full((N, N), np.nan)
    for j in K:
        pol = load(j)
        vs = []
        with torch.no_grad():
            for b, (x_t, t, _) in zip(batches[j], fm_inputs[j]):
                tail = B1.cond_tail(pol, b)
                cond = B1.make_cond(B1.encode_lang(pol, [instr[j]] * x_t.shape[0]), tail)
                vs.append(pol.dit_flow.velocity_net(
                    noisy_actions=x_t, time=t, global_cond=cond).clone())
        ref_v[j] = vs
        del pol; torch.cuda.empty_cache()

    for k in K:
        pol = load(k)
        for j in K:
            if j > k:
                continue
            num, den = 0.0, 0.0
            with torch.no_grad():
                for b, (x_t, t, _), vref in zip(batches[j], fm_inputs[j], ref_v[j]):
                    tail = B1.cond_tail(pol, b)
                    cond = B1.make_cond(B1.encode_lang(pol, [instr[j]] * x_t.shape[0]), tail)
                    v = pol.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond)
                    num += float((v - vref).flatten(1).norm(dim=1).sum())
                    den += float(vref.flatten(1).norm(dim=1).sum())
            drift[k, j] = num / den
            print(f"[drift] stage {k}  task {j}  rel drift = {drift[k, j]:.3f}")
        del pol; torch.cuda.empty_cache()

    # ── 측정 2: 그래디언트 균형 ──────────────────────────────────────────────
    grad_rows = []
    for k in K:
        if k == 0:
            continue
        student = load(k)
        student.train()
        teacher = load(k - 1)
        teacher.eval().requires_grad_(False)
        b = batches[k][0]
        x_t, t, target = fm_inputs[k][0]
        bsz = x_t.shape[0]

        def gnorm(loss):
            student.zero_grad(set_to_none=True)
            loss.backward(retain_graph=False)
            tot = 0.0
            for p in student.parameters():
                if p.grad is not None:
                    tot += float(p.grad.detach().pow(2).sum())
            return tot ** 0.5

        tail = B1.cond_tail(student, b)
        cond = B1.make_cond(B1.encode_lang(student, list(b["task"])), tail)
        g_fm = gnorm(B1.fm_loss(student.dit_flow.velocity_net(
            noisy_actions=x_t, time=t, global_cond=cond), target))

        j = 0
        tail2 = B1.cond_tail(student, b)
        cond_j = B1.make_cond(B1.encode_lang(student, [instr[j]] * bsz), tail2)
        pred_j = student.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond_j)
        with torch.no_grad():
            t_tail = B1.cond_tail(teacher, b)
            t_cond = B1.make_cond(B1.encode_lang(teacher, [instr[j]] * bsz), t_tail)
            v_tgt = teacher.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=t_cond)
        g_an = gnorm(args.lambda_anchor * F.mse_loss(pred_j, v_tgt))

        grad_rows.append((k, g_fm, g_an))
        print(f"[grad] stage {k}  |∇L_FM| {g_fm:.4f}   |∇(λ·L_anchor)| {g_an:.4f}   "
              f"비율 {g_fm/max(1e-12,g_an):.1f}:1")
        del student, teacher; torch.cuda.empty_cache()

    np.save(out / "drift.npy", drift)

    # ── 리포트 ───────────────────────────────────────────────────────────────
    sr = {}
    p = Path("results/B1_libero_spatial/sr_matrix.csv")
    if p.exists():
        for line in p.read_text().splitlines()[1:]:
            f = line.split(",")
            for t_, v in enumerate(f[1:]):
                if v.strip():
                    sr[(int(f[0]), t_)] = float(v)

    L = ["=" * 76, "B1 원인 규명 — teacher 드리프트 vs 그래디언트 균형", "=" * 76, "",
         "측정 1: 상대 드리프트  ‖v_k(o_j,ℓ_j) − v_j(o_j,ℓ_j)‖ / ‖v_j(o_j,ℓ_j)‖",
         "  스테이지 j 에서 태스크 j 를 막 배웠을 때의 속도장 대비, 스테이지 k 에서 얼마나",
         "  멀어졌는지. 앵커의 목표(teacher)가 세대마다 오염되면 이 값이 k−j 에 따라 커진다.",
         "", "-" * 76, "drift[k][j]  (행 = 스테이지 k, 열 = 태스크 j)", "-" * 76,
         "stage " + "".join(f"{j:>9d}" for j in range(N))]
    for k in K:
        L.append(f"{k:>5d} " + "".join(
            f"{drift[k, j]:9.3f}" if not np.isnan(drift[k, j]) else "        ." for j in range(N)))
    L += ["", "-" * 76, "세대 거리(k−j) 별 평균 드리프트 vs 평균 SR", "-" * 76,
          f"{'k-j':>5}{'평균 drift':>13}{'평균 SR':>11}{'n':>5}"]
    for d in range(0, max(K) + 1):
        vals = [(drift[k, k - d], sr.get((k, k - d))) for k in K if k - d >= 0 and not np.isnan(drift[k, k - d])]
        vals = [(a, b) for a, b in vals if b is not None]
        if vals:
            L.append(f"{d:>5}{np.mean([a for a, _ in vals]):>13.3f}"
                     f"{np.mean([b for _, b in vals]):>11.1f}{len(vals):>5}")
    L += ["", "-" * 76, "측정 2: 손실 항별 그래디언트 노름", "-" * 76,
          f"{'stage':>6}{'|∇L_FM|':>12}{'|∇(λ·L_anc)|':>15}{'비율':>10}"]
    for k, a, b in grad_rows:
        L.append(f"{k:>6}{a:>12.4f}{b:>15.4f}{a/max(1e-12,b):>9.1f}:1")
    rep = "\n".join(L)
    (out / "report.txt").write_text(rep)
    print("\n" + rep)
    print(f"\nsaved -> {out/'report.txt'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
