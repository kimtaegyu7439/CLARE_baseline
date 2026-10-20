#!/usr/bin/env python
"""L2_V9 — V8 의 OMP Directional Alignment 손실을 **액션 공간**에서 낸다.

    python L2_V9.py --anchor_form mse_da --lambda_da 1.0 ...   # 팔 A
    python L2_V9.py --anchor_form mag_da --lambda_da 1.0 ...   # 팔 B

V8 과 수식은 같다. 재는 **공간**만 다르다 (--dir_space act 고정).
V6 -> V7 과 똑같은 관계다.

────────────────────────────────────────────────────────────────────────────
무엇이 바뀌나
────────────────────────────────────────────────────────────────────────────
선형 경로 x_t = (1−t)ε + t·a 에서 v = a − ε 이고, 앵커는 student 와 teacher 를
**같은 x_j** 에서 평가하므로 두 속도가 **같은 ε 를 공유**한다.

    ‖ε[0:6]‖ ≈ √6 = 2.45    ‖a[0:6]‖ ≈ 0.85    ‖v‖ ≈ 2.49   (실측)

즉 속도의 87~89% 가 공통 노이즈다. 차이 벡터는 ε 가 상쇄되지만(v_s−v_T =
a_s−a_T) **각도는 상쇄되지 않는다** — 둘 다 −ε 쪽을 가리키기 때문이다.
실측으로 e_act/e_vel 이 t<0.2 에서 14 배, t>0.8 에서 0.2 배, 전체 2.2 배였다.

    a = x_j + (1 − t)·v

로 되돌려 공통 성분을 뺀 끝점에서 각도를 잰다.

────────────────────────────────────────────────────────────────────────────
두 팔 (V8 과 1:1 대응, v -> a)
────────────────────────────────────────────────────────────────────────────
                                              (과거 태스크 j 마다, 1/N 평균)
    mse_da   L = L_MSE + ‖a_s − a_T‖²              + λ_DA · L_DA(a)
    mag_da   L = L_MSE + (‖a_s‖₂ − ‖a_T‖₂)²        + λ_DA · L_DA(a)

    L_DA(a) = −log((cos α + 1)/2),  cos α = (a_T·a_s)/((‖a_T‖+ε_dir)·‖a_s‖)

★ 유지 항도 같이 액션으로 옮긴다. V7 에서 확인했듯 m 과 e 를 섞으면
  ‖·‖² = m + 2·r·r·e 항등식이 깨진다(1.582 vs 1.471).

★ mse_da 의 유지 항은 **속도 버전의 (1−t)² 배**다:
      a_s − a_T = (1−t)(v_s − v_T)   (공통 x_j 가 빠진다)
  즉 팔 A 는 V8 팔 A 에 t 의존 재가중이 붙은 것이고, 새로운 양은 아니다.
  반면 mag_da 의 크기 항은 다르다 — a = x_j + (1−t)v 가 **공통 오프셋 x_j** 를
  더하므로 ‖a_s‖−‖a_T‖ ≠ (1−t)(‖v_s‖−‖v_T‖) 다. 방향 항도 당연히 다르다.
  그래서 V9 에서 실제로 새로 보는 것은 **방향·크기의 분해 방식**이다.
"""
from __future__ import annotations

import sys

import L2_V8


def main() -> None:
    # --dir_space 를 act 로 고정한다. 사용자가 명시해도 act 로 덮는다
    # (vel 로 돌리고 싶으면 L2_V8.py 를 쓰면 된다).
    argv = list(sys.argv[1:])
    out = []
    skip = False
    for a in argv:
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
    L2_V8.main()


if __name__ == "__main__":
    main()
