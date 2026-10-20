#!/usr/bin/env python
"""앵커의 **방향 항**이 팔마다 얼마나 센가 — 과거 태스크별로 잰다.

    python L2_dirterm_compare.py --run results/clare_v6_sqerr_goal_seed7 \
        --ckpt outputs/CLARE_V6/sqerr_seed7 --stage 9 --cells 3000

무엇을 비교하나
    앵커 손실은 과거 태스크 j 마다 (b,h) 칸에서 스칼라 하나를 만든다. 항등식

        ‖v_s − v_T‖² = (r_s − r_T)²  +  2·r_s·r_T·(1 − cos α)
                       └── 크기 m ──┘   └──── 방향 ────┘

    으로 쪼개면, 각 팔이 **방향 쪽에 얼마를 싣는지**를 같은 축에서 비교할 수 있다.

        sqerr   (= 기존 MSE 앵커)   2·r_s·r_T · e        e = 1 − cos α
        unifmd                      β · e                 β = 2
        dea                         β · w_j · e           Σ_j w_j = 1
        OMP                         λ_DA · L_DA           L_DA = −log((cos α+1)/2)

    작은 각도에서 L_DA ≈ e/2 이므로, λ_DA=1 은 "계수 0.5 짜리 e" 와 같다.
    그러니 기존 MSE 앵커에 L_DA 를 **더하면** 방향 항이 얼마나 세지는지는

        (λ_DA · L_DA) / (2·r_s·r_T · e)

    의 비로 읽으면 된다. 이 스크립트가 그 비를 태스크별로 낸다.

★ 측정은 **속도 손실로 학습한 중립 팔**(V6 sqerr = MSE 앵커)에서 한다.
  L_DA 로 학습한 V8/V9 에서 재면 "L_DA 를 넣으면 어떻게 되나" 가 아니라
  "이미 넣고 수렴한 뒤 얼마인가" 를 재게 된다.
★ 속도 공간(V8)과 액션 공간(V9) 을 **같은 질의점에서 동시에** 낸다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import L2_V4_seq as B1
from L2_V6 import DEAAnchor, dea_weights, split_mag_dir, to_action
from L2_V8 import da_loss

REPO = Path(__file__).resolve().parent


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="results/clare_v6_sqerr_goal_seed7")
    p.add_argument("--ckpt", default="outputs/CLARE_V6/sqerr_seed7")
    p.add_argument("--stage", type=int, default=9)
    p.add_argument("--h_lo", type=int, default=1, help='실행되는 horizon 구간 [lo,hi). dit_flow 는 actions[:, 1:9] 만 로봇에 보낸다 (n_obs_steps-1=1 부터 n_action_steps=8 개). index 0 은 t-1 이라 이미 지난 시점이고, 9~15 는 미리보기라 폐기된다. 16 전체를 재면 **쓰이지도 않는 액션**이 2/3 를 차지한다.')
    p.add_argument("--h_hi", type=int, default=9)
    p.add_argument("--batches", type=int, default=100,
                   help="배치 수. 배치 32 x 100 = 3200 프레임(배치 원소) 기준.")
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--lambda_da", type=float, default=1.0)
    p.add_argument("--eps_dir", type=float, default=1e-6)
    p.add_argument("--eps_w", type=float, default=1e-3)
    p.add_argument("--out", default=None)
    # cfg 조립용
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--eval_seed", type=int, default=7)
    p.add_argument("--eval_episodes", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--steps_per_task", type=int, default=10000)
    p.add_argument("--episodes_per_task", type=int, default=50)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--log_every_anchor", type=int, default=10 ** 9)
    p.add_argument("--codebook_k", type=int, default=96)
    p.add_argument("--n_pairs", type=int, default=8000)
    p.add_argument("--h_scale", type=float, default=1.0)
    p.add_argument("--ridge_rho", type=float, default=0.05)
    p.add_argument("--grad_min_frames", type=int, default=64)
    p.add_argument("--lambda_level", type=float, default=3.0)
    p.add_argument("--sampler_mode", default="hard_linear")
    p.add_argument("--anchor_weight", default="speedinv")
    p.add_argument("--teacher_bf16", action="store_true", default=False)
    p.add_argument("--pose_dims", type=int, default=None)
    p.add_argument("--anchor_form", default="sqerr")
    p.add_argument("--anchor_reduce", default="mean")
    p.add_argument("--dir_space", default="vel")
    a = p.parse_args()
    a.out_dir = a.run
    return a


def main() -> int:
    a = parse()
    B1.init_logging()
    run, ck = Path(a.run), Path(a.ckpt)
    out = Path(a.out) if a.out else REPO / "results" / f"L2_dirterm_stage{a.stage}.txt"
    device = B1.get_safe_torch_device(a.device, log=True)
    ckdir = sorted(ck.glob("*_ours"))[0] if list(ck.glob("*_ours")) else ck
    P = B1.POSE_DIMS
    instr = json.loads((run / "instructions.json").read_text())
    k = a.stage

    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy

    sp = ckdir / f"task_{k}" / "checkpoints" / "last" / "pretrained_model"
    tp = ckdir / f"task_{k-1}" / "checkpoints" / "last" / "pretrained_model"
    cfg = B1.build_cfg(a, k, str(sp), run)
    ds = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=ds.meta); policy.to(device); policy.eval()
    anchor = DEAAnchor(a); anchor.out = run
    for j in range(k):
        f = run / "codebooks" / f"codebook_task{j}.pt"
        if f.exists():
            anchor.books[j] = B1.to_device_cb(
                torch.load(f, map_location="cpu", weights_only=False), device)
    tcfg = B1.build_cfg(a, k - 1, str(tp), run)
    anchor.teacher = make_policy(cfg=tcfg.policy, ds_meta=ds.meta)
    anchor.teacher.to(device); anchor.teacher.eval()
    for q_ in anchor.teacher.parameters():
        q_.requires_grad_(False)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size, num_workers=a.num_workers, shuffle=True,
        multiprocessing_context="spawn" if a.num_workers > 0 else None)
    js = sorted(anchor.books)
    N = len(js)
    n_batch = a.batches
    KEYS = ("m", "e", "sqerr_dir", "unifmd_dir", "dea_dir", "L_DA", "mse")
    # ★ 손실은 학습 루프에서 **배치마다 스칼라 하나**로 나온다((B,H) 평균).
    #   그러니 그 스칼라를 배치별로 쌓아 평균±표준편차를 내는 것이 맞다.
    #   칸을 전부 풀링해 평균만 내면 배치 간 변동을 못 본다.
    acc = {j: {sp_: {kk: [] for kk in KEYS} for sp_ in ("vel", "act")} for j in js}

    with torch.no_grad():
        for bi, raw in enumerate(loader):
            if bi >= n_batch:
                break
            batch = B1.prep_batch(policy, B1.to_device(raw, device))
            anchor.cls = B1.rgb_cls(policy, batch)
            n = batch["observation.state"].shape[0]
            st_shape = batch["observation.state"].shape
            x_t, t, _ = B1.sample_fm(policy, batch)
            tcol = t[:, None, None]

            # dea 가중은 **전 과거 태스크의 e 를 모아야** 계산된다 -> 두 번 돈다
            qs, pairs = [], {"vel": [], "act": []}
            for j in js:
                q = anchor._query(policy, batch, x_t, t, k, instr, j, n,
                                  st_shape, tcol, device)
                v_s = anchor._student_v(policy, q, t)
                qs.append(q)
                pairs["vel"].append((v_s, q["v_t"]))
                pairs["act"].append((to_action(v_s, q["x_j"], t),
                                     to_action(q["v_t"], q["x_j"], t)))
            for sp_ in ("vel", "act"):
                es, vs = [], []
                for p_s, p_t in pairs[sp_]:
                    _, e, _, _, va = split_mag_dir(p_s, p_t, P)
                    es.append(e); vs.append(va)
                W = dea_weights(torch.stack(es), torch.stack(vs), a.eps_w)   # (N,B,H)
                for i, j in enumerate(js):
                    p_s, p_t = pairs[sp_][i]
                    # ★ 로봇에 실제로 나가는 horizon 구간만 (기본 1:9).
                    p_s, p_t = p_s[:, a.h_lo:a.h_hi], p_t[:, a.h_lo:a.h_hi]
                    m, e, r_s, r_t, _ = split_mag_dir(p_s, p_t, P)
                    L_da, _c, _v = da_loss(p_s, p_t, P, a.eps_dir)
                    d = acc[j][sp_]
                    # ★ 노트의 계수표를 **그대로** 쓴다. dea 만 1/N 이 없다 —
                    #   Σ_j w_j = 1 이라 가중 자체가 평균 역할을 하기 때문이다.
                    #     sqerr  2·r_s·r_T/N      unifmd  β/N
                    #     dea    β·w_j            OMP     λ_DA/N
                    #   여기서 1/N 을 빼먹고 재면 dea 만 N 배 작아 보인다.
                    iN = 1.0 / N
                    d["m"].append(float(m.mean()) * iN)
                    d["e"].append(float(e.mean()))          # 원시 각도 오차(계수 없음)
                    d["sqerr_dir"].append(float((2 * r_s * r_t * e).mean()) * iN)
                    d["unifmd_dir"].append(float((a.beta * e).mean()) * iN)
                    d["dea_dir"].append(float((a.beta * W[i][:, a.h_lo:a.h_hi] * e).mean()))
                    d["L_DA"].append(float((a.lambda_da * L_da).mean()) * iN)
                    d["mse"].append(float((p_s[..., :P] - p_t[..., :P])
                                          .pow(2).sum(-1).mean()) * iN)
            del qs, pairs

    cat = lambda j, s_, kk: np.asarray(acc[j][s_][kk])     # noqa: E731  (배치별 스칼라)

    W_ = 96
    L = ["=" * W_,
         f"앵커 **방향 항**의 크기 비교 — 과거 태스크별  (stage {a.stage}, {a.suite})",
         "=" * W_, "",
         f"  측정 런  {run.name}   (MSE 앵커로 학습한 중립 팔)",
         f"  표본     배치 {a.batch_size} x {n_batch} 회 = {a.batch_size*n_batch} 프레임, "
         f"과거 {N} 개",
         f"           horizon 은 **로봇에 나가는 {a.h_lo}~{a.h_hi-1} 만** "
         f"({a.h_hi-a.h_lo}/16). index 0 은 t−1, 9~15 는 폐기된다.",
         f"           (배치마다 (B,H)={a.batch_size}x16 칸을 평균해 스칼라 1 개 -> "
         f"{n_batch} 개를 모아 평균±표준편차)",
         f"  상수     β={a.beta}, λ_DA={a.lambda_da}, ε_dir={a.eps_dir:g}, ε_w={a.eps_w:g}",
         "",
         "  ‖v_s−v_T‖² = m + 2·r_s·r_T·e    (m 크기, e=1−cos 방향)",
         "  각 팔이 **방향 쪽에 싣는 양** (노트의 계수표 그대로, 1/N 포함):",
         "      sqerr(=MSE)  2·r_s·r_T·e / N      unifmd  β·e / N",
         "      dea          β·w_j·e  (/N 없음)   OMP     λ_DA·L_DA / N",
         ""]

    for sp_ in ("vel", "act"):
        tag = "속도 공간 (V6 / V8)" if sp_ == "vel" else "액션 공간 (V7 / V9)"
        L += ["-" * W_, f"{tag}", "-" * W_,
              f"  {'task':<6}{'m':>10}{'e':>10}{'MSE 전체':>11}"
              f"{'sqerr 방향':>12}{'unifmd':>10}{'dea':>10}{'L_DA':>10}"
              f"{'L_DA/sqerr':>12}"]
        tot = {kk: [] for kk in ("m", "e", "mse", "sqerr_dir", "unifmd_dir",
                                 "dea_dir", "L_DA")}
        for j in js:
            v = {kk: cat(j, sp_, kk) for kk in tot}
            for kk in tot:
                tot[kk].append(v[kk])
            rr = v['L_DA'] / np.maximum(v['sqerr_dir'], 1e-12)
            L.append(f"  t{j:<5}{v['m'].mean():>10.5f}{v['e'].mean():>10.5f}"
                     f"{v['mse'].mean():>11.5f}{v['sqerr_dir'].mean():>12.5f}"
                     f"{v['unifmd_dir'].mean():>10.5f}{v['dea_dir'].mean():>10.5f}"
                     f"{v['L_DA'].mean():>10.5f}"
                     f"{rr.mean():>9.3f}±{rr.std():<.3f}")
        g = {kk: np.concatenate(tot[kk]) for kk in tot}
        gr = g['L_DA'] / np.maximum(g['sqerr_dir'], 1e-12)
        L.append(f"  {'전체':<5}{g['m'].mean():>10.5f}{g['e'].mean():>10.5f}"
                 f"{g['mse'].mean():>11.5f}{g['sqerr_dir'].mean():>12.5f}"
                 f"{g['unifmd_dir'].mean():>10.5f}{g['dea_dir'].mean():>10.5f}"
                 f"{g['L_DA'].mean():>10.5f}"
                 f"{gr.mean():>9.3f}±{gr.std():<.3f}")
        L += ["",
              f"  · L_DA / e 의 평균 = {g['L_DA'].mean()/max(g['e'].mean(),1e-12):.4f}"
              f"   (소각 근사 0.5 와 비교)",
              f"  · MSE 중 방향이 차지하는 몫 = "
              f"{100*g['sqerr_dir'].mean()/max(g['mse'].mean(),1e-12):.1f}%",
              f"  · 기존 MSE 앵커에 λ_DA·L_DA 를 **더하면** 방향 항이 "
              f"{100*g['L_DA'].mean()/max(g['sqerr_dir'].mean(),1e-12):.1f}% 세진다",
              f"  · 전체 손실(MSE+L_DA) 대비 L_DA 의 몫 = "
              f"{100*g['L_DA'].mean()/max(g['mse'].mean()+g['L_DA'].mean(),1e-12):.1f}%",
              ""]

    L += ["읽을 때",
          f"  · 숫자는 **실제 손실에 들어가는 값**이다. 노트의 계수표대로 sqerr/unifmd/",
          f"    OMP 에는 1/N (N={N}) 이 곱해져 있고, dea 에는 없다 — Σ_j w_j = 1 이라",
          "    가중이 이미 평균 역할을 하기 때문이다. 열 'e' 만 계수 없는 원시 각도오차다.",
          "  · 그래서 dea 와 unifmd 의 차이는 **w_j 가 균일에서 얼마나 벗어났나** 다.",
          "    w_j 가 e_j 와 양의 상관이면 E[w·e] > E[e]/N 이라 dea 가 더 커진다.",
          "  · sqerr 의 계수 2·r_s·r_T 는 공간마다 다르다 — 속도는 ‖v‖≈2.5 라 ~12,",
          "    액션은 ‖a‖≈0.85 라 ~1.4. 그래서 같은 λ_DA 라도 두 공간에서 상대 세기가",
          "    전혀 다르다. V8(속도)과 V9(액션)의 차이를 읽을 때 이 점이 핵심이다.",
          "  · mag_da 팔은 sqerr 방향 항을 **버리고** L_DA 만 쓴다. 그 경우 방향 세기가",
          "    위 표의 'sqerr 방향' -> 'L_DA' 로 통째로 바뀐다.",
          ""]
    txt = "\n".join(L)
    out.write_text(txt); print(txt); print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
