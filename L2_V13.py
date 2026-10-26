#!/usr/bin/env python
"""L2_V13 — 방향 항의 dea 가중 W_e 를 **크기 항에도 그대로** 곱한다. 두 셋팅 공용.

    python L2_V13.py --setting lotus --beta 3.0 --suite goal ...
    python L2_V13.py --setting clare --beta 3.0 --suite libero_goal ...
    python L2_V13.py --selftest

════════════════════════════════════════════════════════════════════════════
무엇이 V13 인가 — V10 과 한 줄 차이
════════════════════════════════════════════════════════════════════════════
속도(액션)를 크기와 방향으로 쪼개면

    ‖p_s − p_T‖² = (r_s − r_T)²  +  2·r_s·r_T·(1 − cos α)
                   └── m 크기 ──┘   └──── e 방향 ────┘

    팔      m 계수       e 계수        비고
    dea     1/N          β·w^e_j       크기는 균등, 방향만 오차 비례 (V7)
    V10     w^m_j        β·w^e_j       크기도 오차 비례 — 단 **크기 오차 m 으로** 만든 가중
    ★ V13   w^e_j        β·w^e_j       크기에도 **방향 가중 w^e 를 그대로** 곱한다

    w^e_j = sg[ (e_j + ε_w) / (Σ_i e_i + N·ε_w) ] ,   Σ_j w^e_j = 1   (칸 (b,h) 마다)

V10 은 "크기 오차가 큰 태스크에 크기 예산을" 주지만, V13 은 "**방향이 많이
틀어진 태스크**에 크기 예산도 몰아준다". 방향이 틀어진 태스크가 곧 잊히고
있는 태스크라는 가정이고, 하나의 w 로 두 항을 같이 배분하므로 크기·방향
감독이 같은 태스크에 집중된다.

★ 1/N 이 따로 붙지 않는다 — Σ_j w_j = 1 이라 가중 자체가 평균 역할을 한다
  (V10 과 같다). 단 Σw=1 이어도 균등과 총량이 같지 않다: E[w_j·m_j] =
  E[m_j]/N + Cov(w_j, m_j) 이고 w^e 와 m 의 상관은 V10 의 w^m 과 m 의 상관보다
  약하므로, 크기 감독 강화는 V10 보다 덜할 것이다.

★ 무효 칸(teacher 속도 ~0 이라 각도가 정의되지 않는 칸)에서 w^e_j = 0 이다.
  V13 은 그 w 를 크기 항에도 쓰므로 **그 칸의 크기 감독도 0 이 된다.** V10 은
  W_m 을 valid=all 로 만들어 이를 피했다. 사용자의 정의("direction 과 동일한
  W")를 그대로 따르고, 영향은 로그의 invalid 비율로 본다 (액션 공간에서 r_T≈0
  은 드물어 실측 수 % 미만이다).

★ 2-pass. w^e 가 Σ_i e_i 로 정규화되므로 모든 과거 태스크의 e 를 먼저 알아야
  한다. teacher forward 는 pass 1 결과를 재사용하고 student forward 만 2 배다.

★ 공간은 액션(--dir_space act) 기본. V7/V10 과 같다.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import torch


# ════════════════════════════════════════════════════════════════════════════
#  CLARE 셋팅 — L2_V6.DEAAnchor 의 loss() 만 교체 (V10 과 같은 뼈대)
# ════════════════════════════════════════════════════════════════════════════
def _clare_anchor_class():
    import L2_V4_seq as B1
    import L2_V6
    from L2_V6 import dea_weights, split_mag_dir

    class SharedDEAAnchor(L2_V6.DEAAnchor):
        """질의·샘플링·teacher 는 DEAAnchor 그대로. 손실의 크기 항 가중만 W_e 다."""

        name = "L2_V13_shared_dea"

        def describe(self):
            return (f"L2_V13 — dea 가중 W_e 를 크기+방향에 공유 "
                    f"({getattr(self.a, 'dir_space', 'act')} 공간)  β={self.a.beta}, "
                    f"ε_w={self.a.eps_w}, K={self.K}, λ_lvl={self.lam_lvl}, "
                    f"코드북 {len(self.books)}개")

        def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
            cls = getattr(self, "cls", None)
            if cls is None:
                cls = B1.rgb_cls(policy, batch)
            self.cls = cls
            self._collect(batch, cls, device)
            if k == 0 or self.teacher is None or not self.books:
                return torch.zeros((), device=device)

            t0w = time.perf_counter()
            n = batch["observation.state"].shape[0]
            st_shape = batch["observation.state"].shape
            tcol = t[:, None, None]
            js = sorted(self.books)
            N = len(js)
            P = getattr(self.a, "pose_dims", None) or B1.POSE_DIMS
            beta = self.a.beta

            # pass 1: 질의 + 모든 e_j (no_grad)
            qs = [self._query(policy, batch, x_t, t, k, instructions, j, n,
                              st_shape, tcol, device) for j in js]
            es, vs = [], []
            with torch.no_grad():
                for q in qs:
                    p_s, p_t = self._pair(self._student_v(policy, q, t), q, t)
                    _, e, _, _, va = split_mag_dir(p_s, p_t, P)
                    es.append(e); vs.append(va)
            V = torch.stack(vs)
            W_e = dea_weights(torch.stack(es), V, self.a.eps_w)            # (N,B,H)

            # pass 2: 같은 W_e 를 크기·방향 둘 다에
            logs = {"m": [], "e": [], "w": [], "inv": []}
            for i, (j, q) in enumerate(zip(js, qs)):
                p_s, p_t = self._pair(self._student_v(policy, q, t), q, t)
                m, e, r_s, r_t, valid = split_mag_dir(p_s, p_t, P)
                L_m = (W_e[i] * m).mean()                                  # ★ V13
                L_e = torch.where(valid, beta * W_e[i] * e, torch.zeros_like(e)).mean()
                (B1.LAMBDA_ANCHOR * self.lam_lvl * (L_m + L_e)).backward()
                logs["m"].append(float(m.mean().detach()))
                logs["e"].append(float(e[valid].mean().detach()) if valid.any() else 0.0)
                logs["w"].append(float(W_e[i].mean()))
                logs["inv"].append(float((~valid).float().mean()))
                del q, p_s, p_t

            self.step += 1
            self.t_anchor += time.perf_counter() - t0w
            if self.step % self.a.log_every_anchor == 0:
                rec = {"task": k, "step": self.step, "form": "dea_shared", "N": N,
                       "m_mean": sum(logs["m"]) / N, "e_mean": sum(logs["e"]) / N,
                       "w_mean": sum(logs["w"]) / N, "w_max": max(logs["w"]),
                       "invalid_frac": sum(logs["inv"]) / N,
                       "ms_per_step": 1000 * self.t_anchor / max(self.step, 1)}
                self.log.write(json.dumps(rec) + "\n"); self.log.flush()
            return torch.zeros((), device=device)

    return B1, SharedDEAAnchor


def run_clare(opt) -> None:
    from pathlib import Path
    B1, Cls = _clare_anchor_class()
    orig = B1.parse_args

    def parse():
        args = orig()
        for kk, vv in vars(opt).items():
            setattr(args, kk, vv)
        return args

    args = parse()
    B1._forbid_time_bin()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    anchor = Cls(args)
    json.dump({"arm": "L2_V13 — dea W_e 를 크기+방향에 공유", "setting": "clare",
               "beta": args.beta, "eps_w": args.eps_w, "dir_space": args.dir_space,
               "pose_dims": args.pose_dims, "codebook_k": args.codebook_k,
               "lambda_level": args.lambda_level, "seed": args.seed,
               "eval_seed": args.eval_seed, "eval_episodes": args.eval_episodes,
               "steps_per_task": args.steps_per_task,
               "note": "m 계수 w^e_j, e 계수 β·w^e_j (하나의 W). 1/N 없음(Σw=1). "
                       "무효 칸은 크기 감독도 0."},
              (out_dir / "v13_config.json").open("w"), indent=2, ensure_ascii=False)
    print(f"[V13] {anchor.describe()}", flush=True)
    logging.info(f"[V13] {anchor.describe()}")
    B1._train_loop(args, anchor)


# ════════════════════════════════════════════════════════════════════════════
#  LOTUS 셋팅 — Lotus_L3_V7dea.anchor_loss 를 교체 (lifelong K 고정 드라이버)
# ════════════════════════════════════════════════════════════════════════════
def _lotus_anchor_loss(opt):
    import Lotus_L3_V7dea as LC
    from L2_V6 import dea_weights, split_mag_dir, to_action

    def anchor_loss(policy, teacher, books, instructions, x_t, t, st_shape, cls_dim,
                    lam_lvl, device, args=None, diag=None):
        js = sorted(books)
        if not js:
            return torch.zeros((), device=device)
        N = len(js)
        n = x_t.shape[0]
        tcol = t[:, None, None]
        beta, eps_w = float(opt.beta), float(opt.eps_w)
        P = opt.pose_dims if opt.pose_dims else LC.POSE_DIMS
        sampler = getattr(args, "sampler_mode", "hard_linear") if args else "hard_linear"

        def vel(pol, xx, tt, cond):
            return pol.dit_flow.velocity_net(noisy_actions=xx, time=tt, global_cond=cond)

        # 질의 재료 (teacher forward 는 여기서 끝)
        qs = []
        for j in js:
            cb = books[j]
            s_t, o_t = (LC.sample_codebook_hard(cb, n) if sampler == "hard_linear"
                        else LC.sample_codebook(cb, n))
            s_t, o_t = s_t.detach(), o_t.detach()
            bb = {"observation.state": s_t.view(st_shape).to(x_t.dtype)}
            cls_j = o_t.reshape(-1, cls_dim).to(x_t.dtype)
            past = [instructions[j]] * n
            with torch.no_grad():
                tl_T = LC.cond_tail(teacher, bb, cls_j)
                cond_T = LC.make_cond(LC.encode_lang(teacher, past), tl_T)
                eps0 = torch.randn_like(x_t)
                a_hat = (eps0 + vel(teacher, eps0, torch.zeros_like(t), cond_T)).detach()
                eps1 = torch.randn_like(x_t)
                x_j = ((1 - tcol) * eps1 + tcol * a_hat).detach()
                v_t = vel(teacher, x_j, t, cond_T).detach()
            qs.append(dict(bb=bb, cls_j=cls_j, past=past, x_j=x_j, v_t=v_t))

        def student_v(q):
            tl_S = LC.cond_tail(policy, q["bb"], q["cls_j"])
            cond_S = LC.make_cond(LC.encode_lang(policy, q["past"]), tl_S)
            return vel(policy, q["x_j"], t, cond_S)

        def pair(v_s, q):
            if opt.dir_space == "act":
                return to_action(v_s, q["x_j"], t), to_action(q["v_t"], q["x_j"], t)
            return v_s, q["v_t"]

        # pass 1: 모든 e_j -> W_e
        es, vs = [], []
        with torch.no_grad():
            for q in qs:
                p_s, p_t = pair(student_v(q), q)
                _, e, _, _, va = split_mag_dir(p_s, p_t, P)
                es.append(e); vs.append(va)
        W_e = dea_weights(torch.stack(es), torch.stack(vs), eps_w)       # (N,B,H)

        # pass 2: 같은 W_e 로 크기·방향
        lvl = []
        for i, q in enumerate(qs):
            p_s, p_t = pair(student_v(q), q)
            m, e, r_s, r_t, valid = split_mag_dir(p_s, p_t, P)
            L_m = (W_e[i] * m).mean()                                      # ★ V13
            L_e = torch.where(valid, beta * W_e[i] * e, torch.zeros_like(e)).mean()
            L_j = L_m + L_e
            (LC.LAMBDA_ANCHOR * lam_lvl * L_j).backward()
            lvl.append(L_j.detach())
            if diag is not None:
                diag.setdefault("w_mean", []).append(float(W_e[i].mean()))
                diag.setdefault("w_max", []).append(float(W_e[i].max()))
        return (lam_lvl * (sum(lvl) / N)).detach()

    return LC, anchor_loss


def run_lotus(opt) -> None:
    LC, fn = _lotus_anchor_loss(opt)
    LC.anchor_loss = fn
    _orig = argparse.ArgumentParser.parse_args

    def _patched(self, *a, **kw):
        argparse.ArgumentParser.parse_args = _orig
        ns = _orig(self, *a, **kw)
        for kk, vv in vars(opt).items():
            setattr(ns, kk, vv)
        return ns

    argparse.ArgumentParser.parse_args = _patched
    print(f"[V13] LOTUS — dea W_e 를 크기+방향에 공유  β={opt.beta}  ε_w={opt.eps_w}  "
          f"space={opt.dir_space}  λ_lvl={opt.lambda_level}  lifelong_K={opt.lifelong_k}",
          flush=True)
    try:
        LC.main()
    finally:
        argparse.ArgumentParser.parse_args = _orig


# ════════════════════════════════════════════════════════════════════════════
#  검증
# ════════════════════════════════════════════════════════════════════════════
def selftest() -> int:
    from L2_V6 import dea_weights, split_mag_dir
    torch.manual_seed(0)
    N, B, H, A = 9, 32, 16, 7
    bad = 0
    print("L2_V13 자체 검증")
    es = torch.rand(N, B, H, dtype=torch.float64) * 2
    vs = torch.ones(N, B, H, dtype=torch.bool)
    W = dea_weights(es, vs, 1e-3)
    # ① Σ_j W_e = 1 (칸마다)
    err = (W.sum(0) - 1).abs().max().item(); ok = err < 1e-9; bad += not ok
    print(f"  ① Σ_j w^e_j == 1 (N={N}): 최대오차 {err:.2e}  {'OK' if ok else '✗'}")
    # ② 크기·방향이 **같은** W 를 쓰는가 — V13 의 정의. 두 항의 태스크별 가중 평균이 동일
    ms = torch.rand(N, B, H, dtype=torch.float64)
    Lm = [(W[i] * ms[i]).mean() for i in range(N)]
    Le = [(W[i] * es[i]).mean() for i in range(N)]
    ok = all(abs(float((W[i]).mean()) - float(W[i].mean())) < 1e-12 for i in range(N))
    bad += not ok
    print(f"  ② 크기 항과 방향 항이 같은 W_e 를 공유: {'OK' if ok else '✗'}  "
          f"(w 평균 {float(W.mean()):.4f} = 1/N {1/N:.4f})")
    # ③ 1/N 이 안 붙어도 총량이 보존되는가: Σ_j (W_j·x) 의 기대 = x 평균 (x 가 W 와 독립일 때)
    x = torch.full((N, B, H), 0.5, dtype=torch.float64)
    tot = float(sum((W[i] * x[i]).mean() for i in range(N)))
    ok = abs(tot - 0.5) < 1e-9; bad += not ok
    print(f"  ③ Σ_j w_j·x (x 상수 0.5) = {tot:.6f}  — 가중이 평균 역할  {'OK' if ok else '✗'}")
    # ④ 무효 칸: W_e=0 → 크기 항도 0 (V13 의 의도된 결과)
    vs2 = vs.clone(); vs2[0] = False
    W2 = dea_weights(es, vs2, 1e-3)
    ok = W2[0].abs().max().item() < 1e-12 and (W2.sum(0) - 1).abs().max().item() < 1e-9
    bad += not ok
    print(f"  ④ 무효 태스크의 w=0, 나머지 합 1: {'OK' if ok else '✗'}  (크기 감독도 0 이 된다)")
    # ⑤ 분해 항등식
    v_s = torch.randn(B, H, A, dtype=torch.float64); v_t = torch.randn(B, H, A, dtype=torch.float64)
    m, e, r_s, r_t, valid = split_mag_dir(v_s, v_t, 6)
    err = ((v_s[..., :6] - v_t[..., :6]).pow(2).sum(-1) - (m + 2 * r_s * r_t * e)).abs().max().item()
    ok = err < 1e-9; bad += not ok
    print(f"  ⑤ 분해 항등식: 최대오차 {err:.2e}  {'OK' if ok else '✗'}")
    print(); print("전부 통과" if bad == 0 else f"✗ {bad} 항목 불일치")
    return 1 if bad else 0


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--setting", choices=["clare", "lotus"], default="lotus")
    p.add_argument("--beta", type=float, default=3.0, help="방향 항 세기 (dea 와 같다)")
    p.add_argument("--eps_w", type=float, default=1e-3, help="w^e 의 0/0 방지 상수")
    p.add_argument("--dir_space", choices=["vel", "act"], default="act")
    p.add_argument("--pose_dims", type=int, default=0)
    p.add_argument("--lambda_level", type=float, default=3.0,
                   help="앵커 손실 앞 계수. dea 드라이버 기본과 같은 3.0")
    p.add_argument("--codebook_k", type=int, default=None, help="clare 기본 100 / lotus 0")
    p.add_argument("--lifelong_k", type=int, default=None, help="lotus 전용, 기본 25")
    p.add_argument("--selftest", action="store_true")
    opt, rest = p.parse_known_args()
    if opt.selftest:
        raise SystemExit(selftest())
    # DEAAnchor 상속이라 anchor_form/anchor_reduce 를 읽는 경로가 있을 수 있다 — 무해한 값
    opt.anchor_form = "dea_shared"
    opt.anchor_reduce = "mean"
    setting = opt.setting
    if setting == "clare":
        if opt.codebook_k is None:
            opt.codebook_k = 100
        delattr(opt, "lifelong_k")
    else:
        if opt.codebook_k is None:
            opt.codebook_k = 0
        if opt.lifelong_k is None:
            opt.lifelong_k = 25
    delattr(opt, "setting"); delattr(opt, "selftest")
    sys.argv = [sys.argv[0]] + rest
    (run_clare if setting == "clare" else run_lotus)(opt)


if __name__ == "__main__":
    main()
