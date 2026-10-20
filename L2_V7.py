#!/usr/bin/env python
"""L2_V7 — 크기·방향 분해를 **액션 공간**에서 하는 앵커.

    python L2_V7.py --anchor_form unifmd --beta 2.0 ...
    python L2_V7.py --anchor_form dea    --beta 2.0 ...

V6 와 수식은 같다. 분해하는 **공간**만 다르다 (--dir_space act 고정).

────────────────────────────────────────────────────────────────────────────
왜 액션 공간인가
────────────────────────────────────────────────────────────────────────────
선형 경로 x_t = (1−t)ε + t·a 에서 속도는 v = a − ε 다. 앵커는 student 와
teacher 를 **같은 x_t** 에서 평가하므로 두 속도가 **같은 ε 를 공유**한다.

    ‖ε‖ ≈ √6 = 2.45        ‖a‖ ≈ 0.85        ‖v‖ ≈ 2.49  (실측)

즉 속도의 대부분이 공통 노이즈다. 차이 벡터 v_s − v_t = a_s − a_t 는 ε 가
상쇄되지만, **두 벡터 사이 각도**는 상쇄되지 않는다 — 큰 공통 성분을 공유하면
서로 다른 부분이 아무리 달라도 거의 같은 방향을 가리킨다.

    액션이 직각(90°)으로 틀어져 있어도 속도 각도는 27° 로 보인다 (7.8 배 압축)

압축 정도는 t 에 따라 변한다 (실측):

    t      0.1    0.3    0.5    0.7    0.9
    배율   6.4x   4.2x   2.6x   1.6x   1.0x

t 가 균등 샘플링되므로 **같은 정책 불일치가 t 에 따라 6 배 다른 e 로 들어온다.**
sqerr 은 계수 2·r_s·r̄ 가 t 와 함께 커져 자동 보정되지만, β 를 쓰는 unifmd/dea
에는 그 보정이 없다 — 즉 노트가 제안하는 두 팔에서만 생기는 왜곡이다.

    a = x_t + (1 − t)·v

로 되돌리면 공통 성분이 빠져 각도가 t 와 무관해진다 (실측 e_act 는 전 구간 일정).

★ m 과 e 는 **반드시 같은 공간**에서 뽑는다. 섞으면
  ‖·‖² = m + 2·r·r·e 항등식이 깨지고(1.582 vs 1.471) sqerr 이 MSE 를 복원하지
  못한다. V7 은 둘 다 액션 공간이다.

★ β 보정. 노트의 β ≈ 2r₀² 는 공간마다 다르다:
      속도 공간  r₀ = 2.49  ->  2r₀² = 12.4
      액션 공간  r₀ = 0.82  ->  2r₀² = 1.35
  V6 가 쓴 β=2 는 속도 공간에서 보정값의 1/6 이었다. 액션 공간에서는 1.35 에
  가까우므로 β=2 가 대체로 맞는 값이 된다.
"""
from __future__ import annotations

import sys

import L2_V6


def main() -> None:
    # --dir_space 를 act 로 고정한다. 사용자가 명시해도 act 로 덮는다
    # (vel 로 돌리고 싶으면 L2_V6.py 를 쓰면 된다).
    argv = [a for a in sys.argv[1:]]
    out = []
    skip = False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a == "--dir_space":
            skip = True
            continue
        if a.startswith("--dir_space="):
            continue
        out.append(a)
    sys.argv = [sys.argv[0]] + out + ["--dir_space", "act"]
    L2_V6.main()


if __name__ == "__main__":
    main()
