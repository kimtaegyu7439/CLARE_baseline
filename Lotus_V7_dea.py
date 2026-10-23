#!/usr/bin/env python
"""Lotus_V7_dea — LOTUS 셋팅(LIBERO-50)에 V7 dea 앵커를 끼운다.

    python Lotus_V7_dea.py --selftest                       # 모델 없이 분해 검증
    python Lotus_V7_dea.py --suite 50 --anchor_form sqerr   # 기존 mse 와 항등(기준선)
    python Lotus_V7_dea.py --suite 50 --anchor_form dea --beta 2.0

Lotus_l2cb_cl.py 를 고치지 않고 anchor_loss 만 갈아끼운다(L2_V7/V11/V12 와 같은 방식).
--suite 50 외의 값도 그대로 통한다 — LC 의 인자를 전부 물려받는다.

════════════════════════════════════════════════════════════════════════════
왜 래퍼로 되는가
════════════════════════════════════════════════════════════════════════════
두 드라이버의 앵커는 **손실 몇 줄만** 다르다.

    단계                   Lotus_l2cb_cl        L2_V6/V7
    코드북에서 (s̃,õ)       sample_codebook_hard  동일
    teacher 부트스트랩 x_j  동일                 동일
    student·teacher 속도    동일                 동일
    손실                   resid 제곱(+speedinv)  **액션공간 분해 + dea 가중**
    backward               태스크별 chunk        동일

★ 과거 태스크의 정의는 손댈 게 없었다. LC 도 books[suite_task] 로 **태스크
  번호마다 코드북이 하나씩**이고(make_books), anchor_loss 가 for j in
  sorted(books) 로 돈다. base 25 개도 25 개의 개별 태스크로 들어간다.
  -> "base 가 한 덩어리로 뭉쳐 있어서 dea 가중을 못 나눈다" 는 걱정은 없었다.

════════════════════════════════════════════════════════════════════════════
dea 는 2-pass 여야 한다
════════════════════════════════════════════════════════════════════════════
w_j ∝ e_j / Σ_m e_m 이라 **모든 과거 태스크의 e 를 먼저** 알아야 한다.
    pass 1  no_grad 로 전 태스크 e 수집 -> dea_weights -> W (N,B,H)
    pass 2  태스크마다 coef = β·W[i] 로 손실 -> 즉시 backward
teacher forward 는 pass 1 에서 만든 (x_j, v̄) 를 재사용하므로 1 배다.
**student forward 만 2 배**가 된다. CLARE 쪽 V7 도 같은 비용을 치른다.

★ 비용 경고: LOTUS 는 base 가 25 태스크라 stage 1 에서 이미 과거가 25 개다.
  CLARE 셋팅(과거 최대 9)보다 스텝당 student forward 가 5 배 넘는다.
  sqerr/unifmd 는 1-pass 라 이 배수를 안 낸다. 실제 ms/step 은 짧은 런으로
  재고, 감당이 안 되면 과거 태스크 샘플링(V12 식)을 봐야 한다.

════════════════════════════════════════════════════════════════════════════
계수 배분 — L2_V6 의 표를 그대로 옮긴다
════════════════════════════════════════════════════════════════════════════
    팔        m_j 계수   e_j 계수
    sqerr     1/N       2·r_θ·r̄ / N      <- 분해 항등식의 원래 계수
    unifmd    1/N       β / N
    dea       1/N       β·w_j            <- Σ_j w_j = 1 이라 /N 이 없다

inv_N = 1/N (--anchor_reduce mean, 기본) 또는 1 (sum).
★ **sum 이 LC 원본의 스케일**이다 — 원본 docstring 이 "앵커 그래디언트는 과거에
  대한 합" 이라고 못박고 있다. mean 은 CLARE 쪽 V7 과 맞춘 쪽이다. 과거 수가
  base 25 에서 50 까지 늘어나는 LOTUS 에서 sum 을 쓰면 앵커가 스테이지마다
  세지므로, 여기서는 기본을 mean 으로 둔다.

════════════════════════════════════════════════════════════════════════════
검증
════════════════════════════════════════════════════════════════════════════
--selftest (모델·GPU 불필요, 몇 초)
  ① 분해 항등식  m + 2·r_θ·r̄·e == ‖v_θ−v̄‖²  (pose 축에서)
  ② sqerr 팔이 LC 원본 mse 와 항등인지:
       원본  = Σ resid² / (B·H·A)
       sqerr = Σ ‖resid‖²_P / (B·H)       (reduce=sum, P=A)
     -> sqerr == 원본 × A. P=A(=--pose_dims 0)로 둬야 맞는다. 기본 P=6 은
        gripper 축을 일부러 빼므로 항등이 아니다 — 그건 설계이고 버그가 아니다.
"""
from __future__ import annotations

import argparse
import logging
import sys

import torch

import Lotus_l2cb_cl as LC
from L2_V6 import dea_weights, split_mag_dir, to_action


def make_anchor_loss(opt):
    """opt(anchor_form/beta/eps_w/dir_space/anchor_reduce/pose_dims) 를 묶은
    anchor_loss 를 만든다. 시그니처는 LC.anchor_loss 와 같다."""

    def anchor_loss(policy, teacher, books, instructions, x_t, t, st_shape, cls_dim,
                    lam_lvl, device, args=None, diag=None):
        js = sorted(books)      # ★ sorted 필수 — W[i] 와 js[i] 가 같은 태스크여야 한다
        if not js:
            return torch.zeros((), device=device)
        N = len(js)
        n = x_t.shape[0]
        tcol = t[:, None, None]
        form = opt.anchor_form
        beta = opt.beta
        P = opt.pose_dims if opt.pose_dims else LC.POSE_DIMS
        inv_N = (1.0 / N) if opt.anchor_reduce == "mean" else 1.0
        # 샘플러는 LC 자신의 플래그를 따른다 (여기서 바꾸지 않는다)
        sampler = getattr(args, "sampler_mode", "soft_kernel") if args else "soft_kernel"

        def vel(pol, xx, tt, cond):
            return pol.dit_flow.velocity_net(noisy_actions=xx, time=tt, global_cond=cond)

        # ── 질의 재료: 태스크마다 한 번. teacher forward 는 여기서 끝난다 ────
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
                # teacher 부트스트랩: 과거 j 의 정답을 모르니 teacher 에게 묻는다
                eps0 = torch.randn_like(x_t)
                a_hat = (eps0 + vel(teacher, eps0, torch.zeros_like(t), cond_T)).detach()
                eps1 = torch.randn_like(x_t)
                x_j = ((1 - tcol) * eps1 + tcol * a_hat).detach()
                v_t = vel(teacher, x_j, t, cond_T).detach()
            qs.append(dict(bb=bb, cls_j=cls_j, past=past, x_j=x_j, v_t=v_t,
                           eps1=eps1, a_hat=a_hat))

        def student_v(q):
            tl_S = LC.cond_tail(policy, q["bb"], q["cls_j"])
            cond_S = LC.make_cond(LC.encode_lang(policy, q["past"]), tl_S)
            return vel(policy, q["x_j"], t, cond_S)

        def pair(v_s, q):
            """두 벡터를 **같은** 공간으로. 한쪽만 옮기면 분해 항등식이 깨진다."""
            if opt.dir_space == "act":
                return to_action(v_s, q["x_j"], t), to_action(q["v_t"], q["x_j"], t)
            return v_s, q["v_t"]

        # ── pass 1: dea 면 전 태스크 e 를 모아 태스크 축으로 정규화 ──────────
        W = None
        if form == "dea":
            es, vs = [], []
            with torch.no_grad():
                for q in qs:
                    p_s, p_t = pair(student_v(q), q)
                    _, e, _, _, va = split_mag_dir(p_s, p_t, P)
                    es.append(e); vs.append(va)
            W = dea_weights(torch.stack(es), torch.stack(vs), opt.eps_w)   # (N,B,H)

        # ── pass 2: 태스크마다 손실 -> 즉시 backward (LC 와 같은 chunk) ──────
        lvl = []
        for i, q in enumerate(qs):
            p_s, p_t = pair(student_v(q), q)
            m, e, r_s, r_t, valid = split_mag_dir(p_s, p_t, P)        # 각 (B,H)
            if form == "sqerr":
                coef = (2.0 * r_s * r_t).detach() * inv_N
            elif form == "unifmd":
                coef = torch.full_like(e, beta * inv_N)
            elif form == "dea":
                coef = beta * W[i]              # Σ_i W[i]=1 이라 inv_N 없음
            else:
                raise ValueError(f"알 수 없는 anchor_form: {form}")
            L_m = inv_N * m.mean()
            # 무효 칸(teacher 속도 ~0)은 0 으로. 분모를 B·H 로 고정해 유효 칸
            # 수가 배치마다 달라도 손실 스케일이 흔들리지 않게 한다.
            L_e = torch.where(valid, coef * e, torch.zeros_like(e)).mean()
            L_j = L_m + L_e
            (LC.LAMBDA_ANCHOR * lam_lvl * L_j).backward()
            lvl.append(L_j.detach())

            if diag is not None:
                # LC 의 로그가 w_mean/w_max 쌍을 함께 읽고 함께 비운다.
                # 한쪽만 넣으면 KeyError 가 난다.
                diag.setdefault("w_mean", []).append(float(coef.mean()))
                diag.setdefault("w_max", []).append(float(coef.max()))
                # m/e 는 LC 가 비워 주지 않으므로 직접 잘라 둔다(10k 스텝 x N 태스크).
                for key, val in (("dea_m", float(m.mean())),
                                 ("dea_e", float(e[valid].mean()) if valid.any() else 0.0)):
                    lst = diag.setdefault(key, [])
                    lst.append(val)
                    del lst[:-512]
            if diag is not None and diag.get("sanity_pending"):
                with torch.no_grad():
                    r = float((q["v_t"] + q["eps1"] - q["a_hat"]).norm(dim=-1).mean()
                              / q["a_hat"].norm(dim=-1).mean().clamp_min(1e-9))
                diag["sanity_ratio"] = r
                diag["sanity_pending"] = False

        # 반환은 로그용. 원본과 같이 태스크 평균을 돌려준다(그래디언트는 위에서 흘렀다).
        return (lam_lvl * (sum(lvl) / N)).detach()

    return anchor_loss


# ═════════════════════════════════════════════════════════════════════════════
#  자체 검증 — 모델도 GPU 도 필요 없다
# ═════════════════════════════════════════════════════════════════════════════
def selftest() -> int:
    torch.manual_seed(0)
    B, H, A = 32, 16, 7
    v_s = torch.randn(B, H, A, dtype=torch.float64)
    v_t = torch.randn(B, H, A, dtype=torch.float64)
    bad = 0

    # ① 분해 항등식:  ‖v_s−v_t‖²_P = m + 2·r_s·r_t·e
    for P in (6, A):
        m, e, r_s, r_t, valid = split_mag_dir(v_s, v_t, P)
        lhs = (v_s[..., :P] - v_t[..., :P]).pow(2).sum(-1)
        rhs = m + 2.0 * r_s * r_t * e
        err = (lhs - rhs).abs().max().item()
        ok = err < 1e-9
        bad += not ok
        print(f"  ① 분해 항등식 P={P}: 최대오차 {err:.2e}  {'OK' if ok else '✗ 불일치'}")

    # ② sqerr 팔 == LC 원본 mse × A   (reduce=sum, P=A 일 때)
    P = A
    m, e, r_s, r_t, valid = split_mag_dir(v_s, v_t, P)
    coef = (2.0 * r_s * r_t) * 1.0                       # inv_N=1 -> sum
    mine = 1.0 * m.mean() + torch.where(valid, coef * e, torch.zeros_like(e)).mean()
    orig = (v_s - v_t).flatten(1).pow(2).mean(1).mean()  # LC 원본 그대로
    rel = abs(float(mine) - float(orig) * A) / float(orig * A)
    ok = rel < 1e-9
    bad += not ok
    print(f"  ② sqerr == 원본×A : mine {float(mine):.6f}  원본×{A} "
          f"{float(orig) * A:.6f}  상대오차 {rel:.2e}  {'OK' if ok else '✗ 불일치'}")

    # ③ dea 가중은 태스크 축 합이 1
    N = 25                                               # LOTUS base 와 같은 수
    es = torch.rand(N, B, H, dtype=torch.float64) * 2.0
    vs = torch.ones(N, B, H, dtype=torch.bool)
    W = dea_weights(es, vs, 1e-3)
    err = (W.sum(0) - 1.0).abs().max().item()
    # 허용오차가 1e-9 인 이유: dea_weights 의 `nv * eps_w` 에서 int64 x float 가
    # **float32 로 승격**된다(torch 기본 dtype). 그래서 float64 로 넣어도 ε_w
    # 항만 float32 정밀도가 되고 오차가 ~1e-11 로 앉는다. 실제 학습은 e 가
    # float32 라 차이가 없으므로 L2_V6 을 손대지 않는다 — 기존 V6/V7 런의
    # 수치가 바뀌면 안 된다.
    ok = err < 1e-9
    bad += not ok
    print(f"  ③ Σ_j w_j == 1 (N={N}): 최대오차 {err:.2e}  {'OK' if ok else '✗ 불일치'}")
    # 무효 태스크가 섞여도 유효분만으로 정규화되는지
    vs[0] = False
    W = dea_weights(es, vs, 1e-3)
    ok = (W[0].abs().max().item() < 1e-12
          and (W.sum(0) - 1.0).abs().max().item() < 1e-9)
    bad += not ok
    print(f"  ④ 무효 태스크 제외 후에도 합 1 : {'OK' if ok else '✗ 불일치'}")

    # ⑤ act 공간 변환이 모양을 안 바꾸고 가역인지
    t = torch.rand(B, dtype=torch.float64) * 0.98
    x = torch.randn(B, H, A, dtype=torch.float64)
    a = to_action(v_s, x, t)
    back = (a - x) / (1 - t[:, None, None])
    ok = a.shape == v_s.shape and (back - v_s).abs().max().item() < 1e-9
    bad += not ok
    print(f"  ⑤ to_action 가역·모양 보존 : {'OK' if ok else '✗ 불일치'}")

    print()
    print("전부 통과" if bad == 0 else f"✗ {bad} 항목 불일치")
    return 1 if bad else 0


def main() -> None:
    # V6/V7 전용 인자를 먼저 떼어낸다. LC 의 파서는 이것들을 모른다.
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--anchor_form", choices=["dea", "sqerr", "unifmd"], default="dea")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--eps_w", type=float, default=1e-3)
    p.add_argument("--dir_space", choices=["vel", "act"], default="act")  # V7 = act
    p.add_argument("--anchor_reduce", choices=["mean", "sum"], default="mean")
    p.add_argument("--pose_dims", type=int, default=0,
                   help="0 이면 LC.POSE_DIMS(6, gripper 제외). A 로 두면 전 축.")
    p.add_argument("--skip_eval", action="store_true",
                   help="롤아웃을 전부 건너뛰고 학습만 한다. 스테이지 체크포인트는 "
                        "그대로 남으므로 Lotus_rollout_only.py 로 다른 GPU 에서 "
                        "따로 SR 을 채울 수 있다. LIBERO-50 은 스테이지마다 50 칸을 "
                        "돌아 롤아웃이 학습을 오래 막는데, 그걸 떼어내는 용도다.")
    p.add_argument("--selftest", action="store_true",
                   help="분해·가중 검증만 하고 끝낸다 (모델 불필요)")
    opt, rest = p.parse_known_args()
    if opt.selftest:
        print("Lotus_V7_dea 자체 검증")
        raise SystemExit(selftest())
    sys.argv = [sys.argv[0]] + rest

    # LC.main 의 parse_args 결과에 위 인자를 실어 준다. LC 가 vars(args) 를
    # config.json 에 그대로 덤프하므로(l.614) 설정이 런 폴더에 남는다.
    # 첫 호출에서 스스로 원복한다 — LeRobot 내부 파서는 건드리지 않는다.
    _orig = argparse.ArgumentParser.parse_args

    def _patched(self, *a, **kw):
        argparse.ArgumentParser.parse_args = _orig
        ns = _orig(self, *a, **kw)
        for k, v in vars(opt).items():
            # skip_eval 은 args 에 싣는다 — LC 가 vars(args) 를 config.json 에
            # 덤프하므로(l.614), SR 의 빈 칸이 "롤아웃을 안 돌린 것"인지
            # "롤아웃이 실패한 것"인지 나중에 구분할 수 있어야 한다.
            if k != "selftest":
                setattr(ns, k, v)
        return ns

    argparse.ArgumentParser.parse_args = _patched
    LC.anchor_loss = make_anchor_loss(opt)
    if opt.skip_eval:
        # 학습 루프는 R[stage][t] = rollout_sr(...) 로 결과를 받아 적는다.
        # None 을 돌려주면 그 칸이 빈 칸으로 남고(원래 실패 시 거동과 같다)
        # write_report 도 그대로 돈다. 체크포인트 저장은 영향받지 않는다.
        LC.rollout_sr = lambda *a, **k: None
        logging.info("[Lotus_V7_dea] --skip_eval: 롤아웃 전부 건너뜀 "
                     "(체크포인트만 남긴다)")
    logging.info(f"[Lotus_V7_dea] form={opt.anchor_form} β={opt.beta} "
                 f"ε_w={opt.eps_w} space={opt.dir_space} reduce={opt.anchor_reduce} "
                 f"P={opt.pose_dims or LC.POSE_DIMS}")
    try:
        LC.main()
    finally:
        argparse.ArgumentParser.parse_args = _orig


if __name__ == "__main__":
    main()
