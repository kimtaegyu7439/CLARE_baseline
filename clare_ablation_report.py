#!/usr/bin/env python
"""TETHER ablation 집계 — results/TETHER_ablation.txt.

네 팔. 1~3 은 앵커 가중을 w=1 로 **고정한 채** 조건 좌표 (s̃,õ) 생성 절차만 바꾼다.
4 는 3 에 크기 정규화를 더한 Full TETHER 라, 3 -> 4 가 정규화의 순수 효과다.

  1 indep    Independent Gaussian  + 정규화 없음
  2 shuffle  Shuffled pairing      + 정규화 없음
  3 joint    Joint reconstruction  + 정규화 없음
  4 tether   Joint reconstruction  + 크기 정규화   (Full TETHER)

바뀌는 것은 L2_V4_seq._sample() 한 군데뿐이고, teacher-matching objective·query
예산·λ·합산 방식·x_t 생성·추론 설정은 전부 공통이다.

지표 (순차 셋팅이라 태스크 t 는 스테이지 t 에서 배운다)
  Retention  마지막 완성 행에서 **과거** 태스크들의 평균 SR    ← 보존
  NewTask    마지막 완성 행의 최신 태스크 SR                   ← 신규 학습
  AvgSR      마지막 완성 행 전체 평균
  FWT/NBT/AUC = LOTUS 표와 같은 정의

    python clare_ablation_report.py
"""
from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
OUT = RES / "TETHER_ablation.txt"

SEEDS = [7, 42, 1000]
# (태그, 표시 이름, 조건 생성, 가중, 한 줄 설명)
ARMS = [
    ("indep",   "1 IndepGaussian", "Independent Gaussian", "w=1",
     "s,o 를 각자의 주변 가우시안에서 독립 추출 — 셀도 A_k 도 짝도 없다"),
    ("shuffle", "2 Shuffled",      "Shuffled pairing",     "w=1",
     "joint 와 똑같이 뽑고 묶음 4 안에서 o 만 자리 교환 — 주변 분포는 그대로"),
    ("joint",   "3 Joint",         "Joint reconstruction", "w=1",
     "TETHER 의 조건 좌표, 가중만 끈 것"),
    ("tether",  "4 Full TETHER",   "Joint reconstruction", "정규화",
     "3 에 크기 정규화를 더한 것 — 3 과의 차이가 정규화의 순수 효과다"),
]


def read_matrix(d: Path):
    p = d / "sr_matrix.csv"
    if not p.exists():
        return None, 0
    body = [ln for ln in p.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    if len(body) < 2:
        return None, 0
    n_task = len(body[0].split(",")) - 1
    R = [[float(c) if c.strip() else None for c in ln.split(",")[1:][:n_task]]
         for ln in body[1:]]
    return (R or None), n_task


def stats(R, n_task):
    fwt, nbt, auc = [], [], []
    for t in range(n_task):
        col = [R[k][t] for k in range(t, len(R)) if R[k][t] is not None]
        if not col:
            continue
        fwt.append(col[0])
        auc.append(sum(col) / len(col))
        if len(col) > 1:
            nbt.append(col[0] - sum(col[1:]) / len(col[1:]))
    avg = row = ret = new = None
    for k in range(len(R)):
        if all(R[k][t] is not None for t in range(k + 1)):
            avg, row = sum(R[k][t] for t in range(k + 1)) / (k + 1), k
    if row is not None:
        new = R[row][row]
        if row >= 1:
            ret = sum(R[row][t] for t in range(row)) / row
    m = lambda v: (sum(v) / len(v)) if v else None                 # noqa: E731
    return {"AvgSR": avg, "row": row, "Retention": ret, "NewTask": new,
            "FWT": m(fwt), "NBT": m(nbt), "AUC": m(auc)}


def main() -> None:
    data: dict[str, list] = {}
    n_task_seen = 0
    for tag, *_ in ARMS:
        for sd in SEEDS:
            d = RES / f"tether_ab_{tag}_seed{sd}"
            R, n_task = read_matrix(d)
            if not R:
                data.setdefault(tag, []).append((sd, None, None, 0, 0))
                continue
            n_task_seen = max(n_task_seen, n_task)
            filled = sum(1 for r in R for c in r if c is not None)
            data.setdefault(tag, []).append(
                (sd, stats(R, n_task), R, filled, n_task * (n_task + 1) // 2))

    nt = n_task_seen or 4
    L = ["=" * 92,
         "TETHER ablation — 추정 조건 좌표가 바뀌면 성능이 어떻게 변하는가",
         "=" * 92, "",
         "주장",
         "  같은 teacher-matching objective 와 같은 query 예산을 써도, joint",
         "  reconstruction 이 독립 샘플링·shuffled pairing 보다 과거 task 수행을 더 잘",
         "  보존하고 신규 task 학습도 유지한다.", "",
         "네 팔",
         f"  {'1 IndepGaussian':<18} Independent Gaussian  + 정규화 없음 (w=1)",
         f"  {'2 Shuffled':<18} Shuffled pairing      + 정규화 없음 (w=1)",
         f"  {'3 Joint':<18} Joint reconstruction  + 정규화 없음 (w=1)",
         f"  {'4 Full TETHER':<18} Joint reconstruction  + 크기 정규화",
         "",
         "  1~3 은 가중을 w=1 로 고정한 채 **조건 좌표 생성 절차만** 바꾼 비교다.",
         "  4 는 3 에 정규화를 더한 것이라, 3 -> 4 차이가 정규화의 순수 효과다.", "",
         "조건 좌표 생성 절차 — L2_V4_seq._sample() 한 군데만 갈린다",
         "  Independent Gaussian  s̃ ~ N(mean_s, std_s²),  õ ~ N(mean_o, std_o²) 를",
         "        **각자 따로** 뽑는다. 셀도 A_k 도 짝도 없다. 평균·분산은 그 태스크에서",
         "        모은 (s,o) 저수지의 주변 통계다.",
         "  Shuffled pairing  joint 와 **똑같이** 뽑은 뒤 크기 4 묶음 안에서 õ 만 자리를",
         "        바꾼다(완전 순열). 주변 분포는 그대로고 결합만 깨지므로 '분포가 달라져서'",
         "        라는 반론이 막힌다.",
         "  Joint reconstruction  원본 TETHER. k* ~ Cat(π), s̃ = μ_{s,k*} + σ_{s,k*}⊙ε,",
         "        õ = m_{k*} + A_{k*}(z̃ − z̄_{k*}).", "",
         "앵커 가중",
         "  w=1      보통 제곱오차. reduce_level = 원소 전체 평균.",
         "  정규화   w = 1/(‖ṽ_T[0:6]‖² + c), median x20 절단, 배치 평균 1 정규화.",
         "           pose 6 차원에만 곱하고 gripper 는 그대로.", "",
         "네 팔 사이에 고정한 것",
         "  · 시드별 동일한 출발 체크포인트(libero_90 사전학습)와 task 순서·demo",
         "  · 태스크당 step 수, optimizer, lr 스케줄",
         "  · 과거 task 당 anchor query 수(n = batch = 32), task 샘플링 비율(매 스텝 전체 과거)",
         "  · 앵커 합산 방식(청크 backward = 과거에 대한 합)과 λ_lvl = 3.0",
         "  · action 좌표 생성 절차(teacher 1-step 오일러 â 로 x_t 를 다시 깖)와 추론 설정",
         "  · 앵커 가중 — 1~3 에서만 같다(w=1). 4 는 여기만 다르다.", "",
         f"프로토콜  libero_spatial, task 0~{nt - 1} 순차, 태스크당 10,000 step / 50 demo",
         "          칸당 20 rollout, eval seed 7 고정, train seed 7 / 42 / 1000", "",
         "지표  — SR matrix 의 **마지막 완성 행** 하나에서 읽는다.",
         "        순차 셋팅이라 마지막 행 = 네 태스크를 다 배운 뒤의 상태이고,",
         "        그 행의 t0..t(K-2) 가 과거, t(K-1) 이 방금 배운 신규 태스크다.",
         "",
         "  Retention  마지막 행에서 **과거** 태스크들의 평균 SR    ← 보존, 높을수록 좋음",
         "  NewTask    마지막 행의 **신규** 태스크 SR               ← 신규 학습, 높을수록 좋음",
         "  AvgSR      마지막 행 전체 평균 (과거 + 신규)",
         "",
         "        예) 마지막 행이  t0=45  t1=0  t2=75  t3=100 이면",
         "            Retention = (45+0+75)/3 = 40.0    NewTask = 100    AvgSR = 55.0",
         "        AvgSR 에는 신규가 섞여 보존력이 희석된다. 이 ablation 의 주장이",
         "        정확히 '과거 보존'에 관한 것이라 Retention 을 따로 뽑았다.",
         "",
         "  FWT  mean_t R[t][t]                    막 배웠을 때의 성능 (습득)",
         "  NBT  mean_t mean_{k>t} (R[t][t] − R[k][t])   낮을수록 좋음, + 면 망각",
         "  AUC  mean_t mean_{k>=t} R[k][t]        배운 뒤 구간 전체의 평균", ""]

    # ── 팔별 상세 ────────────────────────────────────────────────────────────
    for tag, name, cond, wt, desc in ARMS:
        L += ["-" * 92, f"{name}   {cond}  +  {wt}   —  {desc}", "-" * 92]
        rows = data.get(tag, [])
        if not any(m for _, m, *_ in rows):
            L += ["  (아직 SR 칸이 없다)", ""]
            continue
        for sd, m, R, fil, tot in rows:
            if m is None:
                L.append(f"  seed {sd:<5} 결과 없음")
                continue
            f1 = lambda v: "  -  " if v is None else f"{v:5.1f}"      # noqa: E731
            fs = lambda v: "  -  " if v is None else f"{v:+5.1f}"     # noqa: E731
            nk = len(R[0])
            L.append(f"  seed {sd:<5} "
                     + ("완료" if fil >= tot else f"진행 {fil}/{tot}칸"))
            # 행 = 태스크 k 를 배운 직후, 열 = 평가 태스크. 아래 삼각형이다.
            L.append(f"    {'stage':<8}" + "".join(f"{'t'+str(t):>6}" for t in range(nk)))
            for k, row in enumerate(R):
                mark = "  <- 마지막 완성 행" if k == m["row"] else ""
                L.append(f"    {k:<8}" + "".join(
                    (f"{c:6.0f}" if c is not None else f"{'.':>6}") for c in row) + mark)
            L.append(f"    지표  Retention {f1(m['Retention'])}   "
                     f"NewTask {f1(m['NewTask'])}   AvgSR {f1(m['AvgSR'])}   "
                     f"FWT {f1(m['FWT'])}   NBT {fs(m['NBT'])}   AUC {f1(m['AUC'])}")
            L.append("")
        L.append("")

    # ── 요약표: 완주한 시드들의 평균 ± 표준편차 ──────────────────────────────
    def agg(tag, key):
        v = [m[key] for _, m, _, f, t in data.get(tag, [])
             if m is not None and m[key] is not None and t and f >= t]
        return (mean(v), pstdev(v) if len(v) > 1 else 0.0, len(v)) if v else None

    L += ["-" * 92, "요약 — 완주한 시드 평균 ± 표준편차 (괄호는 완주 시드 수)", "-" * 92,
          f"{'팔':<18}{'Retention↑':>20}{'NewTask↑':>20}{'AvgSR↑':>20}"]
    for tag, name, *_ in ARMS:
        cells = []
        for key in ("Retention", "NewTask", "AvgSR"):
            a = agg(tag, key)
            cells.append(f"{'-':>20}" if a is None
                         else f"{a[0]:>12.1f} ±{a[1]:4.1f}({a[2]})")
        L.append(f"{name:<18}" + "".join(cells))
    L.append("")

    # ── 주장 판정 ────────────────────────────────────────────────────────────
    L += ["-" * 92, "판정", "-" * 92]
    j = agg("joint", "Retention")
    g = agg("indep", "Retention")
    s = agg("shuffle", "Retention")
    tt = agg("tether", "Retention")
    if j is None or (g is None and s is None):
        L.append("  [조건 좌표] 아직 판정 불가 — 완주한 팔이 모자란다")
    else:
        others = [(n, x) for n, x in (("IndepGauss", g), ("Shuffled", s)) if x]
        gap = "  ".join(f"vs {n} {j[0]-x[0]:+.1f}%p" for n, x in others)
        win = all(j[0] > x[0] for _, x in others)
        L.append(f"  [조건 좌표, 가중 w=1 고정]  Joint Retention {j[0]:.1f}%   {gap}"
                 + ("   → 주장대로" if win else "   → **주장과 어긋남**"))
    if j is None or tt is None:
        L.append("  [정규화] 아직 판정 불가 — joint 또는 tether 가 안 끝났다")
    else:
        L.append(f"  [정규화의 순수 효과]  3 Joint {j[0]:.1f}%  →  4 Full TETHER "
                 f"{tt[0]:.1f}%   ({tt[0]-j[0]:+.1f}%p)")
    jn, tn = agg("joint", "NewTask"), agg("tether", "NewTask")
    if jn and tn:
        L.append(f"  [신규 학습 유지]  3 Joint {jn[0]:.1f}%  →  4 Full TETHER "
                 f"{tn[0]:.1f}%   ({tn[0]-jn[0]:+.1f}%p)")
    L += ["",
          "  ★ 판정은 완주한 시드 평균만으로 한 것이다. 칸당 20 rollout 이라 칸 하나의",
          "    이항 표준오차가 최대 ±11%p 고, 시드 3개 평균도 ±6%p 급이다. 몇 %p 차이는",
          "    확정할 수 없다 — 위 표의 시드 간 표준편차를 같이 보아야 한다.", "",
          "읽을 때 주의",
          "  · 1행(스테이지 0)은 사전학습 + task 0 만 배운 상태다. 앵커가 걸릴 과거가 없어",
          "    네 팔이 같아야 정상이고, 다르면 구현 차이를 의심해야 한다.",
          "  · Shuffled pairing 의 묶음 크기는 4 다(--shuffle_group). 묶음 안에서 완전",
          "    순열을 써서 원래 짝이 남지 않게 했다.",
          "  · Independent Gaussian 만 주변 분포가 joint 와 정확히 같지는 않다(셀 혼합이",
          "    빠져 단일 가우시안이 된다). 분포까지 맞춘 대조군은 Shuffled pairing 쪽이다.",
          "  · 1 IndepGaussian 의 seed 7 / 42 는 팔 이름을 바꾸기 전 실행(tether_ab_A_*)을",
          "    그대로 옮긴 것이다. 설정(indep_gauss + w=1)이 동일해 재실행하지 않았다.",
          ""]

    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
