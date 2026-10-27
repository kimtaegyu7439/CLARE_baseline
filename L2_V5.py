#!/usr/bin/env python
"""L2_V5 — 방향 항의 계수를 **teacher 크기의 역제곱**으로 주는 앵커. 두 셋팅 공용.

    # CLARE 셋팅 (libero_goal 등을 0..9 순차 학습) — 전 태스크 K=100
    python L2_V5.py --setting clare --anchor_c 0.1 --suite libero_goal ...

    # LOTUS 셋팅 (base joint + stage) — base K=100, lifelong 태스크당 K=25
    python L2_V5.py --setting lotus --anchor_c 0.1 --suite 50 ...

    기본값: λ_lvl=1.0 (앵커 손실 앞의 계수, L = L_FM + λ_lvl·L_anchor)
            clare K=100 / lotus base K=100 + lifelong K=25

    python L2_V5.py --selftest            # 모델 없이 수식 검증

════════════════════════════════════════════════════════════════════════════
무엇이 V5 인가 — 기존 V5 팔에서 두 가지를 더한 것
════════════════════════════════════════════════════════════════════════════
기존 V5(= L2_V4.py 에 --anchor_weight speedinv --anchor_c <c> --anchor_wnorm none)는
**제곱 잔차 전체**를 teacher 속도의 역제곱으로 가중했다.

    ρ_T = ‖ṽ_T[0:6]‖,   w = 1/(ρ_T² + c)
    L_j = Σ w · ‖v_S − v_T‖²                 ← 잔차 전체에 곱한다
    그래디언트 = Σ_j ∇L_j                      ← 과거 태스크에 대한 **합**

이 파일은 거기서 두 가지를 바꾼다.

  (1) **1/N 추가.** 기존에는 태스크마다 λ 를 걸고 즉시 backward 해서 실효 앵커
      세기가 과거 수 N 에 비례해 커졌다. CLARE 는 N 이 1→9, LOTUS LIBERO-50 은
      25→45 로 늘어나므로, 스테이지가 갈수록 앵커가 저절로 세진다. 1/N 으로
      과거 태스크의 **평균**을 만들어 그 성장을 없앤다.

  (2) **배치평균 정규화 제거.** V4 는 w̃ = w/mean(w) 로 배치평균을 1 로 맞췄다.
      그러면 c 를 바꿔도 전체 세기는 그대로이고 가중의 *모양*만 바뀐다.
      여기서는 나누지 않으므로 c 가 모양과 세기를 **같이** 지배한다
      (기존 V5 의 anchor_wnorm=none 과 같은 선택이다).

그리고 가중을 잔차 전체가 아니라 **방향 항에만** 건다 — V6/V7 의 분해를 쓴다.

    ‖p_S − p_T‖² = (r_S − r_T)² + 2·r_S·r_T·(1 − cos) = m + 2·r_S·r_T·e

    팔          m 계수    e 계수
    MSE(v3)     1/N       2·r_S·r_T / N
    V6 unifmd   1/N       β / N
    V6 dea      1/N       β·w_j            (Σ_j w_j = 1)
    ★ 이 파일    1/N       1/(r_T² + c) / N

    L_j = (1/N) · [ m.mean() + (1/(r_T²+c)) · e 의 평균 ]

왜 이 모양인가: MSE 의 방향 계수 2·r_S·r_T 는 **속도가 빠른 구간일수록 크다.**
그래서 느린 구간(정밀 조작)의 방향은 사실상 감독을 못 받는다. 1/(r_T²+c) 는
그 의존을 **뒤집어** 느린 구간에 큰 계수를 준다. 기존 V4/V5 의 동기를 방향
항에만 정확히 적용한 것이다.

★ 계수에 상한이 저절로 생긴다. r_T ≥ 0 이므로 coef ∈ (0, 1/c] 다. 그래서
  V4/V5 가 쓰던 w_clip(median·20)이 필요 없다 — 그건 배치 통계라 (2) 와도
  어긋난다. c 가 곧 상한이다.

★ 2-pass 가 아니다. dea 의 w_j 는 Σ_m e_m 으로 정규화돼 모든 과거 태스크의 e 를
  먼저 알아야 했지만, 여기 계수는 그 태스크의 r_T 만 쓴다. 그래서 1-pass 로
  끝나고 **student forward 가 dea 의 절반**이다 (LIBERO-50 에서 큰 차이다).

════════════════════════════════════════════════════════════════════════════
c 를 어떻게 잡나 — 공간마다 다르다
════════════════════════════════════════════════════════════════════════════
--dir_space 가 정하는 공간에서 r_T 를 재므로 c 의 의미도 그 공간에 묶인다
(실측 teacher 노름 r₀: 속도 공간 2.49, 액션 공간 0.82).

    공간        r₀     r₀²     MSE 의 방향계수 2r₀²    coef=1/(r₀²+c) 가 그와 같아지는 c
    속도(vel)   2.49   6.20    12.4                  불가능 (1/12.4=0.08 < r₀²)
    액션(act)   0.82   0.67     1.35                  c ≈ 0.07

즉 액션 공간에서 c≈0.07 이면 **전형적인 r 에서 MSE 와 같은 세기**가 되고, c 를
키우면 약해지고 줄이면 세진다. 속도 공간에서는 어떤 c 로도 MSE 세기에 못 맞춘다
(역수 구조라 r 이 클 때 계수가 작아지므로) — 그래서 기본은 act 다.

기존 V5 가 쓴 c = 0.01/0.05/0.1 은 **속도 공간 + 잔차 전체 가중** 기준이라
여기 c 와 같은 값이 아니다. 같은 숫자를 그대로 옮기지 마라.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import torch


# ════════════════════════════════════════════════════════════════════════════
#  공통 — 방향 항 계수
# ════════════════════════════════════════════════════════════════════════════
def warmup_mult(step_in_stage: int, steps: int, frac: float) -> float:
    """스테이지 안에서 앵커 세기를 0 → 1 로 올리는 배율.

    frac=0 이면 항상 1 (warmup 없음). frac=0.25 면 스테이지의 처음 25% 스텝
    동안 선형으로 올라가 그 뒤 1 에 머문다.

    ── 왜 필요한가 (L2_V5_warmup) ──────────────────────────────────────────
    FWT 는 스테이지 **끝**에서 잰다. 그런데 LOTUS 의 lifelong 태스크는 10 demo
    뿐이라 FM 신호가 약하고, 앵커가 초반부터 세게 걸리면 새 태스크 학습을
    누른다. 실측으로 object t9(과거 3개와 장면·동작이 같고 대상 물체만 다름)의
    FWT 가 seq-FT 95 → V5 65 / dea 30~40 으로 무너졋다. 초반에는 앵커를 풀어
    새 태스크를 seq-FT 속도로 배우게 하고, 앵커는 뒤에서 드리프트만 막게 한다.
    """
    if frac <= 0:
        return 1.0
    warm = max(1, int(round(steps * frac)))
    return min(1.0, step_in_stage / warm)


def speedinv_coef(r_t: torch.Tensor, c: float) -> torch.Tensor:
    """방향 항의 계수 1/(r_T² + c). (B,H) -> (B,H).

    detach: 계수는 학습 대상이 아니다. 빼면 "r_T 를 키워 계수를 낮추는" 경로로
    그래디언트가 흐른다 — teacher 쪽이라 실제로는 상수지만, 습관적으로 끊어
    둔다(V6 의 2·r_S·r_T 계수도 같은 이유로 detach 한다).

    배치평균으로 나누지 않는다 (위 (2)). 그래서 c 가 세기까지 지배한다.
    w_clip 도 쓰지 않는다 — r_T ≥ 0 이므로 coef ≤ 1/c 로 이미 유계다.
    """
    return 1.0 / (r_t.detach().pow(2) + c)


# ════════════════════════════════════════════════════════════════════════════
#  CLARE 셋팅 — L2_V6.DEAAnchor 의 loss() 만 교체
# ════════════════════════════════════════════════════════════════════════════
def _clare_anchor_class():
    """지연 import — lotus 로 돌릴 때 CLARE 드라이버를 끌어올 이유가 없다."""
    import L2_V4_seq as B1
    import L2_V6
    from L2_V6 import split_mag_dir

    class SpeedInvAnchor(L2_V6.DEAAnchor):
        """질의 생성·teacher 부트스트랩·코드북은 DEAAnchor 를 그대로 쓰고
        손실 한 군데만 바꾼다. 바뀌는 것은 방향 항의 계수뿐이다."""

        name = "L2_V5_speedinv_dir"

        def describe(self):
            return (f"L2_V5 — 방향계수 1/(r_T²+c)  c={self.a.anchor_c} "
                    f"({getattr(self.a, 'dir_space', 'act')} 공간), reduce="
                    f"{self.a.anchor_reduce}, K={self.K}, λ_lvl={self.lam_lvl}, "
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
            c = float(self.a.anchor_c)
            inv_N = (1.0 / N) if self.a.anchor_reduce == "mean" else 1.0
            # ★ warmup: 태스크 k 가 바뀌면 스텝 카운터를 0 으로. 한 스텝에 한 번 불린다.
            if getattr(self, "_warm_k", None) != k:
                self._warm_k, self._warm_step = k, 0
            wmult = warmup_mult(self._warm_step, int(self.a.steps_per_task),
                                float(getattr(self.a, "anchor_warmup", 0.0)))
            self._warm_step += 1

            qs = [self._query(policy, batch, x_t, t, k, instructions, j, n,
                              st_shape, tcol, device) for j in js]
            logs = {"m": [], "e": [], "w": []}
            for j, q in zip(js, qs):
                v_s = self._student_v(policy, q, t)
                p_s, p_t = self._pair(v_s, q, t)
                m, e, r_s, r_t, valid = split_mag_dir(p_s, p_t, P)   # 각 (B,H)
                coef = speedinv_coef(r_t, c) * inv_N * float(self.a.beta)   # β: 방향 항 배율
                L_m = inv_N * m.mean()
                L_e = torch.where(valid, coef * e, torch.zeros_like(e)).mean()
                (B1.LAMBDA_ANCHOR * self.lam_lvl * wmult * (L_m + L_e)).backward()
                logs["m"].append(float(m.mean().detach()))
                logs["e"].append(float(e[valid].mean().detach()) if valid.any() else 0.0)
                logs["w"].append(float(coef.mean()))

            self.step += 1
            self.t_anchor += time.perf_counter() - t0w
            if self.step % self.a.log_every_anchor == 0:
                rec = {"task": k, "step": self.step, "form": "speedinv_dir",
                       "c": c, "N": N, "warm_mult": round(wmult, 4),
                       "m_mean": sum(logs["m"]) / N, "e_mean": sum(logs["e"]) / N,
                       "w_mean": sum(logs["w"]) / N, "w_max": max(logs["w"]),
                       "ms_per_step": 1000 * self.t_anchor / max(self.step, 1)}
                self.log.write(json.dumps(rec) + "\n")
                self.log.flush()
            return torch.zeros((), device=device)

    return B1, SpeedInvAnchor


def run_clare(opt) -> None:
    from pathlib import Path

    B1, SpeedInvAnchor = _clare_anchor_class()
    orig = B1.parse_args

    def parse_v5():
        args = orig()
        for kk, vv in vars(opt).items():
            setattr(args, kk, vv)
        return args

    args = parse_v5()
    B1._forbid_time_bin()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor = SpeedInvAnchor(args)
    json.dump({
        "arm": "L2_V5 — 방향계수 1/(r_T²+c)",
        "setting": "clare",
        "anchor_c": args.anchor_c, "dir_space": args.dir_space,
        "anchor_reduce": args.anchor_reduce, "pose_dims": args.pose_dims,
        "anchor_warmup": args.anchor_warmup, "beta": args.beta,
        "sampler_mode": args.sampler_mode, "codebook_k": args.codebook_k,
        "lambda_level": args.lambda_level, "seed": args.seed,
        "eval_seed": args.eval_seed, "eval_episodes": args.eval_episodes,
        "steps_per_task": args.steps_per_task,
        "anchor_wnorm_unused": getattr(args, "anchor_wnorm", None),
        "note": "m 계수 1/N, e 계수 1/(r_T²+c)/N. 배치평균 정규화 없음, w_clip 없음 "
                "(coef <= 1/c 로 유계). 1-pass. "
                "★ anchor_wnorm 은 쓰이지 않는다 — speed_weights() 를 호출하지 않고 "
                "계수를 직접 만든다. 배치 가중 제거가 플래그가 아니라 코드 경로로 보장된다.",
    }, (out_dir / "v5_config.json").open("w"), indent=2, ensure_ascii=False)
    print(f"[V5] {anchor.describe()}", flush=True)
    logging.info(f"[V5] {anchor.describe()}")
    B1._train_loop(args, anchor)


# ════════════════════════════════════════════════════════════════════════════
#  LOTUS 셋팅 — Lotus_l2cb_cl.anchor_loss 를 교체
# ════════════════════════════════════════════════════════════════════════════
def _lotus_anchor_loss(opt):
    # ★ 드라이버로 Lotus_L3_V7dea 를 쓴다 — lifelong 코드북 K 를 태스크당 고정하는
    #   --lifelong_k 가 거기에만 있다. Lotus_l2cb_cl 은 K = k_per_demo x demo 라
    #   lifelong 이 20 으로 고정돼 25 를 만들 수 없다.
    import Lotus_L3_V7dea as LC
    from L2_V6 import split_mag_dir, to_action

    # ★ warmup 상태. anchor_loss 는 한 스텝에 정확히 한 번 불린다. 스테이지가
    #   바뀌면 books 의 크기가 바뀌므로(태스크 수) 그걸로 경계를 잡는다.
    #   (anchor_loss 시그니처에 step 이 없어서 드라이버를 안 고치고 이렇게 센다.)
    wstate = {"N": None, "step": 0}

    def anchor_loss(policy, teacher, books, instructions, x_t, t, st_shape, cls_dim,
                    lam_lvl, device, args=None, diag=None):
        js = sorted(books)
        if not js:
            return torch.zeros((), device=device)
        N = len(js)
        if wstate["N"] != N:
            wstate["N"], wstate["step"] = N, 0
        steps = int(getattr(args, "steps", 10000)) if args else 10000
        wmult = warmup_mult(wstate["step"], steps, float(opt.anchor_warmup))
        wstate["step"] += 1
        n = x_t.shape[0]
        tcol = t[:, None, None]
        c = float(opt.anchor_c)
        P = opt.pose_dims if opt.pose_dims else LC.POSE_DIMS
        inv_N = (1.0 / N) if opt.anchor_reduce == "mean" else 1.0
        sampler = getattr(args, "sampler_mode", "hard_linear") if args else "hard_linear"

        def vel(pol, xx, tt, cond):
            return pol.dit_flow.velocity_net(noisy_actions=xx, time=tt, global_cond=cond)

        lvl = []
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
            tl_S = LC.cond_tail(policy, bb, cls_j)
            cond_S = LC.make_cond(LC.encode_lang(policy, past), tl_S)
            v_s = vel(policy, x_j, t, cond_S)
            # ★ 두 벡터를 **같은** 공간으로. 한쪽만 옮기면 분해 항등식이 깨진다.
            if opt.dir_space == "act":
                p_s, p_t = to_action(v_s, x_j, t), to_action(v_t, x_j, t)
            else:
                p_s, p_t = v_s, v_t
            m, e, r_s, r_t, valid = split_mag_dir(p_s, p_t, P)
            coef = speedinv_coef(r_t, c) * inv_N * float(opt.beta)       # β: 방향 항 배율
            L_j = inv_N * m.mean() + torch.where(
                valid, coef * e, torch.zeros_like(e)).mean()
            (LC.LAMBDA_ANCHOR * lam_lvl * wmult * L_j).backward()
            lvl.append(L_j.detach())
            if diag is not None:
                # LC 의 로그가 w_mean/w_max 쌍을 함께 읽고 함께 비운다.
                # ★ 계수에 warmup 배율을 곱해 찍는다 — 로그의 anc_w 가 실제로 걸린
                #   세기를 보여야 한다. 초반 스텝에서 0 에 가깝게 보이는 게 정상이다.
                diag.setdefault("w_mean", []).append(float(coef.mean()) * wmult)
                diag.setdefault("w_max", []).append(float(coef.max()) * wmult)
        # 반환(로그용)에도 배율을 곱한다 — 실제로 backward 된 크기와 맞춘다.
        return (lam_lvl * wmult * (sum(lvl) / N)).detach()

    return LC, anchor_loss


def run_lotus(opt) -> None:
    LC, fn = _lotus_anchor_loss(opt)
    LC.anchor_loss = fn
    _orig = argparse.ArgumentParser.parse_args

    def _patched(self, *a, **kw):
        argparse.ArgumentParser.parse_args = _orig      # 첫 호출에서 원복
        ns = _orig(self, *a, **kw)
        for kk, vv in vars(opt).items():
            setattr(ns, kk, vv)
        return ns

    argparse.ArgumentParser.parse_args = _patched
    # ★ print 로 찍는다. 드라이버가 init_logging() 을 부르기 **전**이라
    #   logging.info 는 핸들러가 없어 조용히 삼켜진다 (실측: train.log 에 안 남음).
    #   설정 자체는 드라이버가 config.json 에 덤프하므로 기록은 그쪽이 정본이다.
    print(f"[V5] LOTUS — 방향계수 β/(r_T²+c)  c={opt.anchor_c}  β={opt.beta}  "
          f"space={opt.dir_space}  reduce={opt.anchor_reduce}  "
          f"λ_lvl={opt.lambda_level}  lifelong_K={opt.lifelong_k}  warmup={opt.anchor_warmup}",
          flush=True)
    try:
        LC.main()
    finally:
        argparse.ArgumentParser.parse_args = _orig


# ════════════════════════════════════════════════════════════════════════════
#  검증
# ════════════════════════════════════════════════════════════════════════════
def selftest() -> int:
    from L2_V6 import split_mag_dir, to_action
    torch.manual_seed(0)
    B, H, A = 32, 16, 7
    v_s = torch.randn(B, H, A, dtype=torch.float64)
    v_t = torch.randn(B, H, A, dtype=torch.float64)
    bad = 0
    print("L2_V5 자체 검증")

    for P in (6, A):
        m, e, r_s, r_t, valid = split_mag_dir(v_s, v_t, P)
        lhs = (v_s[..., :P] - v_t[..., :P]).pow(2).sum(-1)
        err = (lhs - (m + 2.0 * r_s * r_t * e)).abs().max().item()
        ok = err < 1e-9
        bad += not ok
        print(f"  ① 분해 항등식 P={P}: 최대오차 {err:.2e}  {'OK' if ok else '✗'}")

    # ② 계수가 1/c 로 유계인가 (w_clip 이 필요 없다는 근거)
    m, e, r_s, r_t, valid = split_mag_dir(v_s, v_t, 6)
    for c in (0.01, 0.1, 1.0):
        co = speedinv_coef(r_t, c)
        ok = bool((co <= 1.0 / c + 1e-12).all()) and bool((co > 0).all())
        bad += not ok
        print(f"  ② c={c:<5} coef 범위 [{co.min():.4f}, {co.max():.4f}]  "
              f"상한 1/c={1/c:.1f}  {'OK' if ok else '✗'}")

    # ③ r_T 가 커지면 계수가 작아진다 (속도 역제곱의 핵심)
    r_slow = torch.full((4, 4), 0.1, dtype=torch.float64)
    r_fast = torch.full((4, 4), 3.0, dtype=torch.float64)
    cs, cf = speedinv_coef(r_slow, 0.1), speedinv_coef(r_fast, 0.1)
    ok = float(cs.mean()) > float(cf.mean())
    bad += not ok
    print(f"  ③ 느린 구간 계수 {cs.mean():.3f} > 빠른 구간 {cf.mean():.4f}  "
          f"({cs.mean()/cf.mean():.0f}배)  {'OK' if ok else '✗'}")

    # ④ 1/N: 과거 수가 늘어도 앵커 평균 세기가 일정한가
    for N in (1, 9, 25, 45):
        tot = sum((1.0 / N) * float(speedinv_coef(r_t, 0.1).mean()) for _ in range(N))
        ref = float(speedinv_coef(r_t, 0.1).mean())
        ok = abs(tot - ref) < 1e-9
        bad += not ok
        print(f"  ④ N={N:<3} 1/N 합 {tot:.6f} == 단일 {ref:.6f}  {'OK' if ok else '✗'}")

    # ⑤ act 공간 변환이 가역·모양 보존
    t = torch.rand(B, dtype=torch.float64) * 0.98
    x = torch.randn(B, H, A, dtype=torch.float64)
    a = to_action(v_s, x, t)
    back = (a - x) / (1 - t[:, None, None])
    ok = a.shape == v_s.shape and (back - v_s).abs().max().item() < 1e-9
    bad += not ok
    print(f"  ⑤ to_action 가역·모양 보존 : {'OK' if ok else '✗'}")

    # ⑥ warmup 배율: 0→1 선형, 끝나면 1 고정, frac=0 이면 항상 1
    ws = [warmup_mult(s, 10000, 0.25) for s in (0, 1250, 2500, 5000, 9999)]
    ok = (abs(ws[0] - 0.0) < 1e-12 and abs(ws[1] - 0.5) < 1e-12 and ws[2] == 1.0
          and ws[3] == 1.0 and ws[4] == 1.0 and warmup_mult(0, 10000, 0.0) == 1.0)
    bad += not ok
    print(f"  ⑥ warmup(0.25, 10k): step 0→{ws[0]:.2f}  1250→{ws[1]:.2f}  "
          f"2500→{ws[2]:.2f}  5000→{ws[3]:.2f}  frac=0→{warmup_mult(0,10000,0.0):.0f}"
          f"  {'OK' if ok else '✗'}")
    # ⑦ β 는 방향 항에만 선형으로 걸린다 (크기 항 불변)
    m_, e_, r_s_, r_t_, v_ = split_mag_dir(v_s, v_t, 6)
    base = float(speedinv_coef(r_t_, 0.1).mean())
    ok = abs(float((speedinv_coef(r_t_, 0.1) * 0.5).mean()) - 0.5 * base) < 1e-12
    bad += not ok
    print(f"  ⑦ β=0.5 → 방향 계수 평균 {0.5*base:.4f} = 0.5 × {base:.4f}  {'OK' if ok else '✗'}")
    print()
    print("전부 통과" if bad == 0 else f"✗ {bad} 항목 불일치")
    return 1 if bad else 0


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--setting", choices=["clare", "lotus"], default="clare",
                   help="clare = libero_* 를 0..9 순차 학습(L2_V4_seq 드라이버). "
                        "lotus = base joint + stage(Lotus_l2cb_cl 드라이버)")
    p.add_argument("--anchor_c", type=float, default=0.1,
                   help="방향 계수의 분모 상수. coef = 1/(r_T² + c) 이고 상한이 1/c 다. "
                        "액션 공간에서 c≈0.07 이면 전형적 r 에서 MSE 와 같은 세기가 "
                        "된다(2r₀²=1.35). 작을수록 세고 느린 구간에 더 쏠린다. "
                        "★ 기존 V5 의 c(0.01~0.1)는 속도 공간·잔차 전체 기준이라 "
                        "같은 값이 아니다")
    p.add_argument("--dir_space", choices=["vel", "act"], default="act",
                   help="m 과 e 를 분해하는 공간. act = 끝점 액션(공통 노이즈가 빠져 "
                        "각도가 t 와 무관). vel 은 역수 구조상 MSE 세기에 못 맞춘다")
    p.add_argument("--anchor_reduce", choices=["mean", "sum"], default="mean",
                   help="mean = 과거 태스크에 1/N (이 버전의 정의). "
                        "sum = 기존 V4/V5 거동(과거 수에 비례)")
    p.add_argument("--pose_dims", type=int, default=0,
                   help="0 이면 드라이버의 POSE_DIMS(6, gripper 제외). 7 로 두면 전 축")
    p.add_argument("--beta", type=float, default=1.0,
                   help="방향 항 배율. L_j = (1/N)[ m + β·coef·e ],  coef = 1/(r_T²+c). "
                        "1.0 = 기존 V5. <1 이면 방향 항을 크기 항보다 약하게 — object 처럼 "
                        "새 태스크와의 충돌이 방향(e)에서 날 때 쓴다. >1 이면 e 강조")
    p.add_argument("--anchor_warmup", type=float, default=0.0,
                   help="L2_V5_warmup: 스테이지 안에서 앵커 세기를 처음 이 비율의 스텝 "
                        "동안 0→1 로 선형 상승 (예 0.25 = 처음 25%%). 0 = 끔. "
                        "FWT 는 스테이지 끝에서 재므로, 초반에 새 태스크를 앵커 없이 "
                        "배우게 두고 뒤에서 드리프트만 막는다")
    # ── 앵커 세기 / 코드북 ─────────────────────────────────────────────────
    p.add_argument("--lambda_level", type=float, default=1.0,
                   help="앵커 손실 앞에 곱하는 계수. 최종 손실 L = L_FM + λ_lvl·L_anchor "
                        "다 (LAMBDA_ANCHOR=1.0). ★ 드라이버 기본값은 3.0 인데 여기서는 "
                        "1.0 로 덮는다")
    p.add_argument("--codebook_k", type=int, default=None,
                   help="모든 태스크의 코드북 셀 수를 이 값으로 고정. "
                        "clare 기본 100 (드라이버 기본 96 을 덮는다). "
                        "lotus 는 0 — base 는 규칙(K=k_per_demo x demo=100)을 쓰고 "
                        "lifelong 은 --lifelong_k 로 따로 정한다")
    p.add_argument("--lifelong_k", type=int, default=None,
                   help="lotus 전용. lifelong 태스크의 코드북 셀 수를 **태스크당** "
                        "이 값으로 고정 (기본 25). base 는 영향 없음")
    p.add_argument("--selftest", action="store_true",
                   help="모델 없이 수식만 검증하고 끝낸다")
    opt, rest = p.parse_known_args()
    if opt.selftest:
        raise SystemExit(selftest())
    # ★ DEAAnchor 를 상속하는데 그 쪽 describe()/loss() 가 anchor_form/beta/eps_w 를
    #   읽는다. 둘 다 덮었으니 지금은 안 쓰이지만, 상위 클래스가 바뀌어도
    #   AttributeError 로 죽지 않게 무해한 값을 실어 둔다.
    opt.anchor_form = "speedinv_dir"
    opt.eps_w = 0.0            # DEAAnchor 상속 경로가 읽을 수 있어 무해한 값
    # 셋팅별 코드북 기본값. 두 셋팅의 K 규칙이 다르므로 여기서 갈라 둔다.
    if opt.setting == "clare":
        if opt.codebook_k is None:
            opt.codebook_k = 100          # 전 태스크 K=100 (드라이버 기본 96 을 덮는다)
        delattr(opt, "lifelong_k")        # clare 드라이버에는 없는 인자다
    else:
        if opt.codebook_k is None:
            opt.codebook_k = 0            # 0 = base 는 규칙(K=100)
        if opt.lifelong_k is None:
            opt.lifelong_k = 25           # lifelong 만 태스크당 25
    setting = opt.setting
    delattr(opt, "setting")
    delattr(opt, "selftest")
    sys.argv = [sys.argv[0]] + rest
    if setting == "clare":
        run_clare(opt)
    else:
        run_lotus(opt)


if __name__ == "__main__":
    main()
