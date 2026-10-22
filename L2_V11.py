#!/usr/bin/env python
"""L2_V11 — 앵커를 **배치당 과거 태스크 1 회**만 계산한다 (액션 공간).

    python L2_V11.py --beta 3.0 --eps_w 1e-3 ...

V7(= V6 + --dir_space act) 의 dea 앵커와 수식이 완전히 같다.
바뀌는 것은 **앵커를 몇 개의 표본으로 추정하느냐** 하나뿐이다.

────────────────────────────────────────────────────────────────────────────
무엇이 바뀌나
────────────────────────────────────────────────────────────────────────────
기존(V7)
    과거 태스크 j 마다 코드북에서 n = B(=32) 개를 **독립으로** 뽑는다.
        k* ~ Cat(π) 가 32 개, ε 가 32 개 (L2_V4_seq.py:1261 multinomial(pi, n))
    배치의 32 행이 각각 다른 (s̃, õ) 조건에서 앵커를 받는다.

V11
    과거 태스크 j 마다 **1 개만** 뽑아 그 배치 전체의 앵커로 쓴다.
        k* 1 개, ε 1 개, 그리고 t·ε₀·ε₁ 도 1 행치만 쓴다.
    태스크 하나당 손실이 스칼라 하나이고, 그것이 그 배치의 앵커다.

그래서 **앵커 연산이 배치 크기만큼 줄어든다**. teacher/student forward 가
태스크당 (B,H,A) 가 아니라 (1,H,A) 로 돈다. 학습 손실(FM)은 그대로 B=32 다.

────────────────────────────────────────────────────────────────────────────
무엇을 기대할 수 있나 — 그리고 무엇을 조심해야 하나
────────────────────────────────────────────────────────────────────────────
앵커는 여전히 **불편추정량**이다. 기댓값은 V7 과 같고 수렴 지점도 같다.
달라지는 것은 분산이다. 몬테카를로 평균의 분산으로 쓰면

    V7    Var ≈ σ²_so/B + σ²_tε/B
    V11   Var ≈ σ²_so/1 + σ²_tε/1

σ²_so(조건 좌표에서 오는 분산)가 B 배, σ²_tε 도 B 배 커진다.

★ σ²_so 가 지배적이다. goal seed7 코드북 3 개를 분산분해해 보면
    s(16d)    셀 추첨(between) 비중 96.5 ~ 97.2%
    o(3072d)  셀 추첨(between) 비중 69 ~ 73%
  즉 조건 좌표 분산의 대부분이 k* ~ Cat(π) 추첨에서 온다. 유효 셀 수가
  60~73 개(π 최대 0.03~0.06)라, 한 스텝에 과거 태스크 j 를 **수십 개 모드
  중 딱 한 곳**에서만 앵커하게 된다.

커버리지 자체는 문제가 아니다 — 10,000 step 이면 셀당 100 회 넘게 방문하고,
쿠폰 수집으로 전 셀 1 회가 ~440 step 이다. 문제는 **스텝별 그래디언트 노이즈**다.

또 chunk backward 라 앵커 그래디언트가 과거 태스크에 대한 **합**이므로,
스테이지가 깊어질수록 독립 표본이 N 개 생겨 분산이 상대적으로 완화된다.
초기 스테이지(k=1,2)가 가장 불리하다.

────────────────────────────────────────────────────────────────────────────
구현
────────────────────────────────────────────────────────────────────────────
DEAAnchor 를 상속해 loss() 앞단에서 **앵커 입력만** 1 행으로 자른다.
 · 코드북 수집(_collect)은 반드시 **전체 배치**로 한다. 여기서 1 행만 넣으면
   코드북 자체가 망가진다 — 그래서 플래그로 상위 호출의 재수집을 막는다.
 · self.cls 는 (B*4,768) 이므로 1 행에 맞춰 앞 4 개만 넘긴다.
 · --dir_space 는 act 로 고정한다(V7 과 같은 공간). 지정해도 덮어쓴다.
"""
from __future__ import annotations

import sys

import torch

import L2_V6


class BatchOnceDEAAnchor(L2_V6.DEAAnchor):
    """과거 태스크당 표본 1 개로 앵커를 추정한다. 나머지는 DEAAnchor 그대로."""

    name = "L2_V11_batch_once"

    #: 앵커에 쓰는 행 수. 1 이면 "배치당 한 번".
    ANCHOR_ROWS = 1

    def describe(self):
        return (f"L2_V11 — 배치당 과거 태스크 1 회 앵커 (act), "
                f"form={self.a.anchor_form}, β={self.a.beta}, "
                f"코드북 {len(self.books)}개")

    def _collect(self, batch, cls, device):
        # 상위 loss() 가 또 부르는 것을 막는다. 수집은 V11.loss 에서 전체 배치로
        # 이미 한 번 했다. 두 번 넣으면 저수지 표본이 편향된다.
        if getattr(self, "_no_collect", False):
            return None
        return super()._collect(batch, cls, device)

    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        # ── 1) 수집은 전체 배치로. 여기만 원본과 같다 ──────────────────────
        cls = getattr(self, "cls", None)
        if cls is None:
            cls = L2_V6.B1.rgb_cls(policy, batch)
        self.cls = cls
        self._collect(batch, cls, device)

        if k == 0 or self.teacher is None or not self.books:
            return torch.zeros((), device=device)

        # ── 2) 앵커 입력만 m 행으로 자른다 ────────────────────────────────
        B = x_t.shape[0]
        m = min(self.ANCHOR_ROWS, B)
        if m >= B:                                   # 자를 게 없으면 원본 그대로
            return super().loss(policy, batch, tail, x_t, t, k,
                                instructions, rng, args, device)

        sub = {}
        for key, v in batch.items():
            # 배치 축이 B 인 텐서만 자른다. 메타데이터(문자열 등)는 그대로 둔다.
            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B:
                sub[key] = v[:m]
            else:
                sub[key] = v
        sub_tail = tail[:m] if torch.is_tensor(tail) and tail.shape[0] == B else tail
        # cls 는 (B*4,768) — 카메라 2 x 프레임 2 = 행당 4 개다
        per_row = cls.shape[0] // B
        sub_cls = cls[: m * per_row]

        saved_cls, self.cls = self.cls, sub_cls
        self._no_collect = True
        try:
            out = super().loss(policy, sub, sub_tail, x_t[:m], t[:m], k,
                               instructions, rng, args, device)
        finally:
            self._no_collect = False
            self.cls = saved_cls
        return out


def main() -> None:
    # --dir_space 를 act 로 고정한다 (V7 과 같은 공간).
    argv, out, skip = sys.argv[1:], [], False
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

    # main() 이 DEAAnchor(args) 를 직접 만들므로, 그 이름만 바꿔 끼운다.
    orig = L2_V6.DEAAnchor
    L2_V6.DEAAnchor = BatchOnceDEAAnchor
    try:
        L2_V6.main()
    finally:
        L2_V6.DEAAnchor = orig


if __name__ == "__main__":
    main()
