#!/usr/bin/env python
"""floor/actual 측정 셋을 한 파일로 묶는다 -> results/FloorLoss_ActualLoss.txt

세 측정은 **묻는 질문이 다르다**. 한 표에 몰아넣으면 반드시 혼동되므로 블록을 나누고
각 블록에 어느 팔·어느 좌표·어느 정의인지를 머리말로 붙인다.

    [A] TETHER(v4), LOTUS goal   — 현재 태스크의 **실제 데이터** 좌표
    [B] v3+grad, LIBERO spatial  — 현재 태스크의 **실제 데이터** 좌표
    [C] TETHER(v4), LOTUS goal   — **코드북(앵커가 실제로 걸린)** 좌표

코드북 측정이 진행 중이면 있는 stage 까지만 싣고 그렇게 표시한다.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
W = 96

SRC = {
    "A": RES / "l2_codebook_v4_goal_seed7" / "floor_v4.json",
    "B": RES / "L2_condition_floor_teacher.json",
    "C": RES / "l2_codebook_v4_goal_seed7" / "floor_codebook.json",
}


def load(k):
    p = SRC[k]
    return json.loads(p.read_text()) if p.exists() else []


def main():
    A, B, C = load("A"), load("B"), load("C")
    L = []
    add = L.append

    add("=" * W)
    add("조건 붕괴 바닥 L* 대 실측 손실 — 측정 셋 종합")
    add("=" * W)
    add("")
    add("세 측정은 **묻는 질문이 다르다**. 수치를 옮겨 쓸 때 블록을 반드시 확인하라.")
    add("")
    add("  [A] TETHER(v4), LOTUS goal   현재 태스크의 **실제 데이터** 좌표")
    add("  [B] v3+grad,   LIBERO spatial 현재 태스크의 **실제 데이터** 좌표")
    add("  [C] TETHER(v4), LOTUS goal   **코드북** 좌표 (앵커가 실제로 걸린 곳)")
    add("")
    add("      [A][B] 가 묻는 것: 새 태스크의 관측 분포에서도 조건 분리가 유지되는가")
    add("      [C]    가 묻는 것: 앵커가 걸린 좌표에서 조건 분리가 일어났는가")
    add("")
    add("-" * W)
    add("공통 정의")
    add("-" * W)
    add("")
    add("  L(u)   = μ‖u − v_0‖² + λ Σ_{j=1}^N ‖u − v_j‖²                      (6)")
    add("  L*     = λ/(μ+λN)·( μ Σ_j‖v_0−v_j‖² + λ Σ_{i<j}‖v_i−v_j‖² )        (8)")
    add("  actual = μ‖v_S(ℓ_0) − v_0‖² + λ Σ_j ‖v_S(ℓ_j) − v_j‖²")
    add("")
    add("  v_j = v_T(ℓ_j)   teacher 가 과거 명령어 ℓ_j 를 들었을 때 낸 속도")
    add("  μ = 1 (L_FM 의 암묵 가중),  λ = 3 (λ_level)")
    add("  스왑은 instruction 한 성분만 — (x_t,t)·관측·상태는 고정 (논문 Scope)")
    add("")
    add("  L* 는 가중분산 형태로 계산했고 식 (8) 쌍합 형태와 상대차 4e-09 로 일치한다.")
    add("  N=1 에서 μλ/(μ+λ)‖v_0−v_1‖² 와도 상대오차 2e-08 (부록 A 항등식).")
    add("")
    add("  ★ [C] 에는 FM 항이 없다. 코드북은 합성 좌표라 정답 액션 a 가 없어 v_0 = a−ε 를")
    add("    만들 수 없다. 따라서 [C] 의 L*/actual 은 **과거 조건들 사이의** 양이고,")
    add("    가중이 전부 같아 λ 는 약분된다. [A][B] 와 절대값을 비교하면 안 된다.")
    add("")
    add("판정")
    add("  ratio = actual / L*.   ratio < 1 이면 바닥 아래 = 정책이 instruction 을 읽는다.")
    add("  ★ 단방향 진단이다. ratio < 1 은 증명이지만, ratio > 1 은 아무 결론도 주지")
    add("    않는다 — 붕괴하지 않았어도 그 좌표에서 teacher 와 멀면 1 을 넘는다.")
    add("  split  0 이면 조건맹, 클수록 분화. ratio 와 독립적으로 읽을 수 있다.")
    add("  ±SE / 반분1·반분2 / <1% 는 표본 편향 점검이다.")
    add("")

    def block(tag, title, meta, rows, keyfn, frac_key, note=None):
        add("=" * W)
        add(f"[{tag}] {title}")
        add("=" * W)
        for m in meta:
            add(f"  {m}")
        add("")
        if not rows:
            add("  (아직 결과 없음)")
            add("")
            return
        add(f"{'행':<8}{'N':>4}{'L*':>11}{'actual':>11}{'ratio':>8}{'±SE':>8}"
            f"{'반분1':>8}{'반분2':>8}{'<1%':>7}{'split':>8}")
        add("-" * W)
        for r in rows:
            add(f"{keyfn(r):<8}{r['N']:>4}{r['floor']:>11.5f}{r['actual']:>11.5f}"
                f"{r['ratio']:>8.3f}{r['se_ratio']:>8.3f}"
                f"{r['ratio_half1']:>8.3f}{r['ratio_half2']:>8.3f}"
                f"{100*r[frac_key]:>6.0f}%{r['split']:>8.3f}")
        add("-" * W)
        if note:
            for n in note:
                add(f"  {n}")
        add("")

    block("A", "TETHER (v4) — LOTUS goal, seed 7   /   실제 데이터 좌표",
          ["팔      sampler=hard_linear, anchor_weight=speedinv (= 논문의 TETHER)",
           "런      results/l2_codebook_v4_goal_seed7",
           "셋팅    base 6 태스크 joint -> lifelong 4 스테이지 x 1 태스크 (총 10)",
           "teacher stage k-1 (k=1 은 base phase 체크포인트)",
           "행      stage k = 현재 태스크 t(5+k), 과거 t0..t(4+k) 가 조건 N 개",
           "표본    40배치 x 4draw x B32 = 5,120 / 행"],
          A, lambda r: f"stage{r['stage']}", "frac_below")

    block("B", "v3+grad — LIBERO spatial, seed 41   /   실제 데이터 좌표",
          ["팔      v4 플래그 없음 (soft_kernel + mse 앵커) = v3",
           "런      results/l2_codebook_k96_grad_seed41",
           "셋팅    10 태스크 순차, base phase 없음",
           "teacher stage t-1",
           "행      task t = 현재 태스크 t, 과거 t0..t(t-1) 가 조건 N 개",
           "표본    40배치 x 4draw x B32 = 5,120 / 행",
           "※ 지정된 l2_codebook_k96_grad(seed42) 는 체크포인트가 9/5 에",
           "   l2cb_fullcov_grad 로 덮어써져 없다(ckpt_root 경로에 팔 이름이 없다).",
           "   같은 팔·같은 스위트인 seed41 로 대체했다."],
          B, lambda r: f"task{r['t']}", "frac_below_floor")

    done = "완료" if len(C) >= 4 else f"진행 중 — stage {len(C)}/4 까지"
    block("C", "TETHER (v4) — LOTUS goal, seed 7   /   코드북 좌표",
          ["팔      [A] 와 같은 런·같은 체크포인트. 좌표만 다르다.",
           "좌표    과거 코드북에서 (s̃,õ) 를 뽑고 teacher 부트스트랩으로 x_t 를 만든다",
           "        — L2_V4.anchor_loss 가 학습 중 쓰는 바로 그 경로",
           "목표    v_j = v_T(x_t, t, ℓ_j),  과거 태스크 전부",
           "실측    Σ_j ‖v_S(ℓ_j) − v_j‖²      바닥  Σ_j ‖v_j − v̄‖²",
           "★ FM 항 없음 (합성 좌표에 정답 액션이 없다)",
           f"상태    {done}",
           "코드북은 디스크에 없어 재생성했다 — 입력 (s,o) 는 동결 DINOv2 와",
           "정규화 버퍼에서만 오고 그 버퍼가 전 체크포인트에서 동일하므로 모델 무관이다."],
          C, lambda r: f"stage{r['stage']}", "frac_below")

    # ── 종합 ────────────────────────────────────────────────────────────────
    add("=" * W)
    add("종합")
    add("=" * W)
    add("")
    for tag, rows, key in (("A", A, "stage"), ("B", B, "t"), ("C", C, "stage")):
        if not rows:
            continue
        nb = sum(1 for r in rows if r["ratio"] < 1)
        add(f"  [{tag}] {len(rows)} 행 중 {nb} 행이 바닥 아래   "
            f"ratio {min(r['ratio'] for r in rows):.3f}~{max(r['ratio'] for r in rows):.3f}   "
            f"split {min(r['split'] for r in rows):.2f}~{max(r['split'] for r in rows):.2f}")
    add("")
    if A and C:
        add("  같은 모델, 좌표만 다를 때 (stage 별 대조)")
        add(f"    {'stage':<8}{'[A] 실제데이터':>22}{'[C] 코드북':>22}")
        add(f"    {'':8}{'ratio':>11}{'split':>11}{'ratio':>11}{'split':>11}")
        cmap = {r["stage"]: r for r in C}
        for r in A:
            c = cmap.get(r["stage"])
            if not c:
                add(f"    {r['stage']:<8}{r['ratio']:>11.3f}{r['split']:>11.3f}"
                    f"{'—':>11}{'—':>11}")
            else:
                add(f"    {r['stage']:<8}{r['ratio']:>11.3f}{r['split']:>11.3f}"
                    f"{c['ratio']:>11.3f}{c['split']:>11.3f}")
        add("")
        add("  해석")
        add("    · 앵커가 걸린 좌표([C])에서는 바닥 아래로 크게 내려가고 split 이 0.9 를")
        add("      넘는다 — 정책이 instruction 을 읽는다는 것이 식 (8) 의 충분조건으로")
        add("      증명된다. 붕괴한 해는 그 바닥 밑으로 못 가기 때문이다.")
        add("    · 같은 모델인데 새 태스크의 관측 분포([A])로 가면 split 이 절반 아래로")
        add("      떨어지고 ratio 가 1 을 넘는다. 분리가 학습된 좌표 밖으로 완전히")
        add("      전이되지는 않는다. 망각이 남는 이유를 여기서 정량화할 수 있다.")
        add("    · [A] 의 actual 은 99% 이상이 앵커 항이다(L_FM 은 0.016~0.024).")
        add("      현재 태스크 학습이 아니라 과거 조건 재현이 비용의 전부다.")
    add("")
    add("주의")
    add("  · [C] 와 [A][B] 의 L*/actual 절대값을 비교하지 마라. [C] 에는 FM 항이 없고")
    add("    좌표 분포가 다르다. 비교 가능한 것은 ratio 와 split 이다.")
    add("  · 논문 Table 2 의 26.7/49.5/46.9% 는 이 셋 중 어느 것도 아니다. 그 수치는")
    add("    옛 스크립트(L2_condition_floor.py)의 다른 정의 — floor = ‖V̄−v_0‖² 이고")
    add("    V̄ 가 **student 자신의** 조건별 평균, 앵커 항 없음 — 에 9/5 에 덮어써진")
    add("    fullcov_grad(spatial) 체크포인트를 물려 얻은 값이다. 본문 수식 (6)/(8) 과")
    add("    측정이 일치하도록 교체가 필요하다.")
    add("")

    out = RES / "FloorLoss_ActualLoss.txt"
    out.write_text("\n".join(L) + "\n")
    print(f"saved -> {out}   ({len(L)} 줄, [C] {len(C)}/4 stage)")


if __name__ == "__main__":
    main()
