#!/usr/bin/env python
"""Lotus_cov_diag — 코드북 셀의 s 공분산 구조와, 대각/풀 공분산 샘플에서의 교사 액션 일치도.

    python Lotus_cov_diag.py --run results/lotus_v5act_c0.1_k25_scopeLL_goal_seed7 --tasks 6 7 8 9

두 가지를 잰다 (태스크 t 의 교사 = t 를 배운 stage 의 체크포인트, 코드북 = 학습과 같은 K=25).

 1. 셀별 s 공분산(16차원, 전역 표준화 좌표)의 고유값 비율
      f1/f2/f3      상위 1/2/3 고유값이 차지하는 분산 비율 (n_k 가중 평균)
      diag_in3      대각 샘플의 분산 중 상위 3 고유벡터 부분공간 안에 있는 비율
      md2/D         대각 샘플의 풀 공분산 기준 마할라노비스² / 16 (풀 샘플이면 ≈1)
 2. 교사 액션 일치도 — 교사 한 번 부트스트랩(ε + v(ε, t=0)) 을 n_draw 번 평균한 â 와 기준 액션
      real    실제 프레임 (s, o 실제)            기준 = 그 프레임의 데모 액션        → 바닥
      real_õ  실제 s, o 는 A_k 로 재구성          기준 = 같은 프레임의 데모 액션      → õ 재구성 비용
      nn      실제 프레임, 기준 = 가장 가까운 **다른** 실제 프레임의 액션            → NN 기준 자체의 잡음
      diag    s̃ ~ 대각 가우시안, õ = A_k 재구성  기준 = 가장 가까운 실제 프레임의 액션
      full    s̃ ~ 풀 공분산(수축 α), õ 동일      기준 = 동일
    지표는 실행되는 앞 8 스텝 × pose 6 차원: mse, 방향오차 e=1−cos, 교사 속도 r̂, z 공간 NN 거리.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import Lotus_L3_V7dea as LC                      # noqa: E402
from Lotus_We_plot import ckpt, load_args, H_USE  # noqa: E402


def cell_cov_stats(z: torch.Tensor, lab: torch.Tensor, Ke: int, gen) -> dict:
    """z: (N,16) 전역 표준화 s.  셀별 고유값 비율과 대각 샘플의 부분공간 밖 비율."""
    D = z.shape[1]
    rows = []
    for kk in range(Ke):
        zc = z[lab == kk]
        nk = int(zc.shape[0])
        if nk < 5:
            continue
        Sig = torch.cov(zc.T.double())                               # (16,16)
        ev, U = torch.linalg.eigh(Sig)                               # 오름차순
        ev = ev.clamp_min(0).flip(0); U = U.flip(1)
        tot = float(ev.sum().clamp_min(1e-12))
        f = (ev.cumsum(0) / tot)[:3].tolist()
        d = Sig.diagonal()                                           # 대각 분산
        in3 = float((d * U[:, :3].pow(2).sum(1)).sum() / d.sum().clamp_min(1e-12))
        # 대각 샘플의 풀 공분산 기준 마할라노비스² (수축 10% + 작은 릿지)
        Ssh = 0.9 * Sig + 0.1 * torch.diag(d) + 1e-6 * torch.eye(D, dtype=Sig.dtype, device=Sig.device)
        x = d.sqrt()[None] * torch.randn(512, D, generator=gen, device=z.device, dtype=Sig.dtype)
        md2 = (torch.linalg.solve(Ssh, x.T).T * x).sum(1)
        rows.append(dict(k=kk, n=nk, f1=f[0], f2=f[1], f3=f[2], in3=in3,
                         md2_D=float(md2.median() / D), ev=ev.cpu().numpy()))
    n = np.array([r["n"] for r in rows], dtype=float)
    w = n / n.sum()
    agg = {key: float((w * np.array([r[key] for r in rows])).sum())
           for key in ("f1", "f2", "f3", "in3", "md2_D")}
    agg["n_cells"] = len(rows)
    # 셀 평균 고유값 스펙트럼 (비율)
    spec = np.stack([r["ev"] / max(r["ev"].sum(), 1e-12) for r in rows])
    agg["spectrum"] = (w[:, None] * spec).sum(0).tolist()
    return agg, rows


def full_cov_factors(s: torch.Tensor, lab: torch.Tensor, cb: dict, alpha: float) -> torch.Tensor:
    """셀별 raw s 공분산의 촐레스키 인자 (Ke,16,16). 수축 Σ←(1−α)Σ+α·diag(Σ), 하한은 sig_s 와 같다."""
    Ke, D = cb["K_eff"], s.shape[1]
    floor2 = (1e-3 * float(s.std())) ** 2
    L = torch.zeros(Ke, D, D, device=s.device)
    for kk in range(Ke):
        sc = s[lab == kk]
        if sc.shape[0] < 2:
            L[kk] = torch.diag(cb["sig_s"][kk]); continue
        Sig = torch.cov(sc.T.double())
        Sig = (1 - alpha) * Sig + alpha * torch.diag(Sig.diagonal()) + floor2 * torch.eye(D, dtype=Sig.dtype, device=s.device)
        L[kk] = torch.linalg.cholesky(Sig).float()
    return L


def recon_o(cb: dict, s: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """hard_linear 의 õ = m_k + A_k (z̃ − z̄_k)."""
    z = (s - cb["mean_s"]) / cb["std_s"]
    d = (z - cb["zbar"][k]).unsqueeze(2)
    return cb["m"][k] + torch.bmm(cb["A"][k].float(), d).squeeze(2)


@torch.no_grad()
def teacher_ahat(teacher, s: torch.Tensor, o: torch.Tensor, instr: str, st_shape, cls_dim: int,
                 act_shape, n_draw: int) -> torch.Tensor:
    n = s.shape[0]
    bb = {"observation.state": s.view(n, *st_shape[1:]).float()}
    cls = o.reshape(-1, cls_dim).float()
    cond = LC.make_cond(LC.encode_lang(teacher, [instr] * n), LC.cond_tail(teacher, bb, cls))
    acc = torch.zeros(n, *act_shape[1:], device=s.device)
    t0 = torch.zeros(n, device=s.device)
    for _ in range(n_draw):
        eps0 = torch.randn(n, *act_shape[1:], device=s.device)
        acc += eps0 + teacher.dit_flow.velocity_net(noisy_actions=eps0, time=t0, global_cond=cond)
    return acc / n_draw


def agree(a_hat: torch.Tensor, a_ref: torch.Tensor, P: int) -> dict:
    """앞 H_USE 스텝 × pose P 차원. mse / 방향오차 / 교사 속도 / 기준 속도."""
    x, y = a_hat[:, :H_USE, :P], a_ref[:, :H_USE, :P]
    mse = (x - y).pow(2).sum(-1)                                     # (n,H)
    rx, ry = x.norm(dim=-1), y.norm(dim=-1)
    valid = (rx > 1e-6) & (ry > 1e-6)
    cos = (x * y).sum(-1) / (rx * ry).clamp_min(1e-12)
    e = torch.where(valid, 1 - cos, torch.zeros_like(cos))
    return dict(mse=float(mse.mean()), e=float(e[valid].mean()), r_hat=float(rx.mean()),
                r_ref=float(ry.mean()), rel=float((mse.sqrt() / ry.clamp_min(1e-6)).median()))


@torch.no_grad()
def run_task(run: Path, args, t: int, n: int, n_draw: int, alpha: float, device, seed: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy

    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    spec = LC.suite_spec(args.suite)
    root, stage_tasks, instr = spec["root"], spec["stage_tasks"], spec["instr"]
    k = next(i + 1 for i, st in enumerate(stage_tasks) if t in st)
    local_idx = stage_tasks[k - 1].index(t)
    P = args.pose_dims or LC.POSE_DIMS

    cfg = LC.build_eval_cfg(args, str(ckpt(run, k)), spec["env_task"][t])
    meta = LeRobotDatasetMetadata("local/lotus", root=str(root / f"stage{k}"))
    teacher = make_policy(cfg=cfg.policy, ds_meta=meta).to(device).eval()
    teacher.requires_grad_(False)
    ds = LC.load_split(root / f"stage{k}", teacher.config)
    pairs, ndemo, acts = LC.collect_pairs(teacher, ds, device, args.batch_size, args.num_workers,
                                          spec["stage_demos"])
    s, o = pairs[local_idx]; a = acts[local_idx]
    s, o, a = s.to(device), o.to(device), a.to(device)
    N = s.shape[0]
    K = max(args.k_min, args.lifelong_k)
    cb = LC.to_device_cb(LC.build_codebook(s, o, K, seed=args.seed, h_scale=args.h_scale,
                                           ridge_rho=args.ridge_rho, grad_min_frames=args.grad_min_frames), device)
    z = (s - cb["mean_s"]) / cb["std_s"]
    lab = torch.cdist(z, cb["c"]).argmin(1)
    Ke = cb["K_eff"]

    # ── 1. 공분산 구조 ──
    cov_agg, cov_rows = cell_cov_stats(z, lab, Ke, gen)

    # ── 2. 샘플 ──
    L = full_cov_factors(s, lab, cb, alpha)
    kidx = torch.multinomial(cb["pi"], n, replacement=True, generator=gen)
    eps = torch.randn(n, s.shape[1], generator=gen, device=device)
    s_diag = cb["mu_s"][kidx] + cb["sig_s"][kidx] * eps
    s_full = cb["mu_s"][kidx] + torch.bmm(L[kidx], eps.unsqueeze(2)).squeeze(2)
    o_diag, o_full = recon_o(cb, s_diag, kidx), recon_o(cb, s_full, kidx)
    ridx = torch.randperm(N, generator=gen, device=device)[:n]
    s_real, o_real, a_real = s[ridx], o[ridx], a[ridx]
    o_real_rec = recon_o(cb, s_real, lab[ridx])

    def nn_ref(sq: torch.Tensor, exclude: torch.Tensor | None = None):
        d = torch.cdist((sq - cb["mean_s"]) / cb["std_s"], z)          # (n,N)
        if exclude is not None:
            d[torch.arange(sq.shape[0], device=device), exclude] = float("inf")
        j = d.argmin(1)
        return a[j], d.gather(1, j[:, None]).squeeze(1)

    a_nn_d, d_diag = nn_ref(s_diag)
    a_nn_f, d_full = nn_ref(s_full)
    a_nn_r, d_real = nn_ref(s_real, exclude=ridx)

    st_shape, act_shape = (n, 2, s.shape[1] // 2), a.shape
    cls_dim = o.shape[1] // 4
    ah = lambda ss, oo: teacher_ahat(teacher, ss, oo, instr[t], st_shape, cls_dim, act_shape, n_draw)
    a_hat_real = ah(s_real, o_real)
    res = {
        "real":   agree(a_hat_real, a_real, P),
        "real_o": agree(ah(s_real, o_real_rec), a_real, P),
        "nn":     agree(a_hat_real, a_nn_r, P),
        "diag":   agree(ah(s_diag, o_diag), a_nn_d, P),
        "full":   agree(ah(s_full, o_full), a_nn_f, P),
    }
    res["nn"]["d_nn"] = float(d_real.median()); res["diag"]["d_nn"] = float(d_diag.median())
    res["full"]["d_nn"] = float(d_full.median())
    # 실제 데모 액션끼리의 NN 일치도 (교사 없이): 기준 자체의 분산
    res["demo_nn"] = agree(a_real, a_nn_r, P)
    info = dict(task=t, stage=k, N=N, K_eff=Ke, n_demoted=cb["n_demoted"],
                n_k_med=int(cb["n_k"].median()))
    del teacher; torch.cuda.empty_cache()
    return info, cov_agg, cov_rows, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tasks", type=int, nargs="+", default=[6, 7, 8, 9])
    ap.add_argument("--n", type=int, default=640)
    ap.add_argument("--n_draw", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.1, help="풀 공분산 수축 (diag 쪽으로)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run); args = load_args(run, a.device); device = torch.device(a.device)
    LC.init_logging()
    all_out = {}
    print(f"run {run.name}  suite {args.suite}  n={a.n} n_draw={a.n_draw} α={a.alpha}")
    for t in a.tasks:
        info, cov, rows, res = run_task(run, args, t, a.n, a.n_draw, a.alpha, device, a.seed)
        all_out[f"t{t}"] = dict(info=info, cov=cov, cells=[{k: v for k, v in r.items() if k != "ev"} for r in rows], agree=res)
        print(f"\n── t{t} (stage{info['stage']} 교사)  N={info['N']} K_eff={info['K_eff']} n_k med={info['n_k_med']} 강등 {info['n_demoted']}")
        print(f"   공분산  f1={cov['f1']:.3f} f2={cov['f2']:.3f} f3={cov['f3']:.3f}   "
              f"대각샘플 상위3 안 비율={cov['in3']:.3f}   대각샘플 md²/D={cov['md2_D']:.2f}")
        print(f"   {'조건':8s} {'mse':>8s} {'e=1-cos':>8s} {'rel':>6s} {'r_hat':>6s} {'r_ref':>6s} {'d_nn':>6s}")
        for key in ("real", "real_o", "nn", "demo_nn", "diag", "full"):
            r = res[key]
            print(f"   {key:8s} {r['mse']:8.4f} {r['e']:8.4f} {r['rel']:6.2f} {r['r_hat']:6.3f} {r['r_ref']:6.3f} "
                  f"{r.get('d_nn', float('nan')):6.3f}")
    out = Path(a.out) if a.out else run / "cov_diag.json"
    out.write_text(json.dumps(all_out, indent=1, ensure_ascii=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
