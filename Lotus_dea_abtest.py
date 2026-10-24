#!/usr/bin/env python
"""Lotus_dea_abtest — 이식한 dea 앵커를 **원본 앵커와 같은 입력에서** 대조한다.

    python Lotus_dea_abtest.py --suite goal --seed 42
    (내부에서 Lotus_l2cb_cl.main() 을 --steps 1 --n_stages 1 로 돌린다)

왜 필요한가
──────────────────────────────────────────────────────────────────────────────
Lotus_V7_dea.py --selftest 은 **합성 텐서**로 분해 항등식만 확인한다. 그건
수식이 맞다는 증거이지, 이식한 함수가 실제 policy/teacher/코드북에서 원본과
같은 것을 계산한다는 증거가 아니다. 이 스크립트가 그 구멍을 메운다.

무엇을 비교하는가
──────────────────────────────────────────────────────────────────────────────
stage1 의 **첫 스텝**에서, 똑같은 (x_t, t, books, instructions) 에 대해

    ① LC.anchor_loss        원본 (--anchor_weight mse = plain 제곱 앵커)
    ② sqerr, reduce=sum, P=A   이식판. ①과 **수학적으로 항등**이어야 한다
    ③ dea β,  reduce=mean, P=6  실제로 돌리는 설정 (참고용)

★ ②/① 의 기대값은 **A(=액션 차원, 7)** 다. 원본은 제곱오차를 (H·A) 로 나누고
  (= B,H,A 전체 평균), 이식판은 A 축을 노름으로 접은 뒤 (B,H) 로만 나누기
  때문이다. Lotus_V7_dea.selftest 의 ② 항목이 이 배수를 합성 텐서에서 이미
  확인했고, 여기서는 실데이터에서 같은 배수가 나오는지를 본다.

★★ 비교는 반드시 **eval 모드**에서 한다. 정책에 dropout 이 있고(config
   dropout=0.1, 입력단 nn.Dropout(0.2)), train 모드면 forward 가 난수를
   소비한다. 그런데 두 구현의 난수 소비 **순서가 다르다**:

     LC 원본 : for j: [표본, eps0, eps1, teacher fwd, student fwd]
     이식판   : for j: [표본, eps0, eps1, teacher fwd]  그 뒤  for j: [student fwd]

   원본은 student forward 를 샘플링 사이에 끼워 돌리고, 이식판은 (dea 의
   2-pass 때문에) 질의를 먼저 다 만들고 나중에 돌린다. 그래서 train 모드로
   비교하면 같은 시드를 심어도 **서로 다른 질의**를 보게 되고, 배수가 7 에서
   벗어난다 (실측 6.04 — 이식 버그처럼 보이지만 테스트 결함이었다).
   eval 모드면 forward 가 난수를 쓰지 않아 두 구현의 질의가 완전히 같아진다.

★ 난수: 태스크마다 (코드북 표본, eps0, eps1) 을 같은 순서로 뽑으므로, eval
  모드에서 호출 직전에 같은 시드를 심으면 질의가 일치하고 남는 차이는 손실
  계산뿐이다.
"""
from __future__ import annotations

import argparse
import logging
import sys

import torch

import Lotus_l2cb_cl as LC
from Lotus_V7_dea import make_anchor_loss

SEED_CMP = 12345


class _Opt:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _run_one(fn, args, kwargs):
    torch.manual_seed(SEED_CMP)
    torch.cuda.manual_seed_all(SEED_CMP)
    out = fn(*args, **kwargs)
    return float(out)


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--beta", type=float, default=3.0)
    known, rest = ap.parse_known_args()
    # LC.main 을 짧게 돌린다. 첫 앵커 호출에서 비교하고 즉시 끝낸다.
    sys.argv = [sys.argv[0], "--steps", "1", "--n_stages", "1",
                "--skip_base_eval", "--sampler_mode", "hard_linear",
                "--eval_episodes", "1", "--eval_batch_size", "1"] + rest

    orig = LC.anchor_loss
    state = {"done": False}

    def probe(policy, teacher, books, instructions, x_t, t, st_shape, cls_dim,
              lam_lvl, device, args=None, diag=None):
        if state["done"]:
            return orig(policy, teacher, books, instructions, x_t, t, st_shape,
                        cls_dim, lam_lvl, device, args=args, diag=diag)
        state["done"] = True
        A = x_t.shape[-1]
        # ★ dropout 을 끈다. 안 끄면 두 구현의 난수 소비 순서 차이로 질의가
        #   달라져 배수가 흔들린다(위 docstring).
        was_train = policy.training
        policy.eval()
        if teacher is not None:
            teacher.eval()
        # ★★ stage1 step0 에서는 teacher = snapshot(policy) 라 두 속도장이
        #    **완전히 같다**. 그대로 비교하면 잔차가 0 이고 손실이 1e-8 수준
        #    수치 노이즈가 되어 비율이 무의미해진다(실측 23.6).
        #    그래서 student 를 조금 흔들어 잔차를 실제 크기로 만든다. eval
        #    모드이므로 forward 는 여전히 난수를 쓰지 않아 질의는 일치한다.
        g = torch.Generator(device="cpu").manual_seed(777)
        with torch.no_grad():
            for prm in policy.parameters():
                if prm.dtype.is_floating_point:
                    noise = torch.randn(prm.shape, generator=g, dtype=torch.float32)
                    prm.add_(noise.to(prm.device, prm.dtype) * (0.02 * prm.detach().float().std().clamp_min(1e-6)).to(prm.dtype))
        N = len(books)
        pos = (policy, teacher, books, instructions, x_t, t, st_shape, cls_dim,
               lam_lvl, device)
        logging.info(f"[abtest] 비교 시작 — N={N} 과거태스크, A={A} 액션차원, "
                     f"B={x_t.shape[0]}, H={x_t.shape[1]}, λ_lvl={lam_lvl}")

        a_mse = _run_one(orig, pos, dict(args=_Opt(sampler_mode="hard_linear",
                                                   anchor_weight="mse"), diag=None))
        sq = make_anchor_loss(_Opt(anchor_form="sqerr", beta=0.0, eps_w=1e-3,
                                   dir_space="vel", anchor_reduce="sum",
                                   pose_dims=A))
        a_sq = _run_one(sq, pos, dict(args=_Opt(sampler_mode="hard_linear"), diag=None))
        de = make_anchor_loss(_Opt(anchor_form="dea", beta=known.beta, eps_w=1e-3,
                                   dir_space="act", anchor_reduce="mean",
                                   pose_dims=0))
        a_dea = _run_one(de, pos, dict(args=_Opt(sampler_mode="hard_linear"), diag=None))

        if a_mse < 1e-6:
            print(f"★ 원본 손실이 {a_mse:.2e} 로 너무 작다 — 비교가 무의미하다. "
                  f"student 를 더 흔들어야 한다.")
        ratio = a_sq / a_mse if a_mse else float("nan")
        err = abs(ratio - A) / A
        print()
        print("=" * 78)
        print("  이식 대조 — stage1 첫 스텝, 같은 입력·같은 시드")
        print("=" * 78)
        print(f"  ① 원본 mse 앵커            {a_mse:.8f}")
        print(f"  ② sqerr (sum, P=A={A})      {a_sq:.8f}")
        print(f"  ③ dea β={known.beta} (mean, P=6)   {a_dea:.8f}   ← 실제 돌리는 설정")
        print()
        print(f"  ②/① = {ratio:.6f}   기대 A = {A}   상대오차 {err:.2e}")
        if was_train:
            policy.train()
        ok = err < 1e-4
        print(f"  판정: {'통과 — 이식이 원본과 항등' if ok else '★ 불일치 — 이식에 문제가 있다'}")
        print(f"  (eval 모드 비교. dropout 켜진 채로 비교하면 난수 순서 차이로 "
              f"배수가 6 근처로 나온다 — 테스트 결함이지 이식 문제가 아니다)")
        print(f"  참고: ③/① = {a_dea / a_mse if a_mse else float('nan'):.4f} "
              f"(dea 는 항등일 필요가 없다. 계수가 다른 다른 팔이다)")
        print("=" * 78)
        raise SystemExit(0 if ok else 1)

    LC.anchor_loss = probe
    LC.main()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
