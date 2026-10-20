#!/usr/bin/env python
"""CLARE(순차) 셋팅 libero_goal — 앵커 분모 상수 c 스윕 -> results/Clare_TETHER_MAR_sweep.txt

    python CLARE_MAR_report.py

읽는 것  results/clare_mar_goal_<tag>_seed<seed>/sr_matrix.csv
         tag = median | c0.1 | c0.01

sr_matrix.csv 는 L2_V4_seq.py 가 SR 칸을 잴 때마다 다시 쓰므로, 도는 런을
건드리지 않고 읽기만 해도 실시간 표가 된다.

행렬·지표 정의는 clare_v4_report.py 와 공유한다 — 같은 셋팅의 표가 서로 다른
계산을 쓰는 일이 없게 하려는 것이다.
"""
from __future__ import annotations

import pathlib
from pathlib import Path

from clare_v4_report import metrics, read_matrix, seeds_of

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
OUT = RES / "Clare_TETHER_MAR_sweep.txt"

# (폴더 태그, 표시 이름, 설명)
# (폴더, 표시 이름, 설명)  — 폴더는 results/clare_<폴더>_seed7
ARMS = [
    ("mar_goal_median", "V4  c = median",  "c_j = kappa x median ||a_j[0:6]||^2 (태스크마다 0.61~0.79). w 를 배치평균 1 로 정규화"),
    ("mar_goal_c0.1",   "V4  c = 0.1",     "c 고정. w 를 배치평균 1 로 정규화"),
    ("mar_goal_c0.01",  "V4  c = 0.01",    "c 고정. w 를 배치평균 1 로 정규화"),
    ("v5_goal_c0.1",      "V5  c = 0.1",      "w~_h 를 **정규화 없이** 그대로 사용 (칸당 20 rollout)"),
    ("v5_goal_c0.05",     "V5  c = 0.05",     "w~_h 를 **정규화 없이** 그대로 사용 (칸당 20 rollout)"),
    ("v5_goal_c0.01",     "V5  c = 0.01",     "w~_h 를 **정규화 없이** 그대로 사용 (칸당 20 rollout)"),
    ("v5_goal_c0.1_n50",  "V5  c = 0.1  n50", "위와 같은 체크포인트, 칸당 **50 rollout** (seed 7~56)"),
    ("v5_goal_c0.05_n50", "V5  c = 0.05 n50", "위와 같은 체크포인트, 칸당 **50 rollout** (seed 7~56)"),
    ("v5_goal_c0.01_n50", "V5  c = 0.01 n50", "위와 같은 체크포인트, 칸당 **50 rollout** (seed 7~56)"),
]
SEEDS = [7]
W = 92

# ── V5 시드 집계 ─────────────────────────────────────────────────────────────
# 폴더 이름이 시드마다 다르다. seed 7 은 20회 + 30회를 합쳐 n50 을 따로 만들었고,
# seed 42/1000 은 처음부터 N_EVAL=50 으로 돌렸다 (config.json eval_episodes=50 확인).
V5_CS = ["0.1", "0.05", "0.01"]
V5_SEEDS = [7, 42, 1000]


def v5_dir(c: str, seed: int) -> pathlib.Path:
    """(c, seed) -> results/ 아래 폴더. 전부 칸당 50 rollout 이다."""
    if seed == 7:
        return RES / f"clare_v5_goal_c{c}_n50_seed{seed}"
    return RES / f"clare_v5_goal_c{c}_seed{seed}"


def agg(vals: list) -> str:
    """평균 ± 표본표준편차. 완주한 시드만 넘겨라."""
    v = [x for x in vals if x is not None]
    if not v:
        return f"{'-':>14}"
    if len(v) == 1:
        return f"{v[0]:>8.1f}{'':>6}"
    mu = sum(v) / len(v)
    sd = (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5
    return f"{mu:>8.1f} ±{sd:>4.1f}"


def main() -> None:
    L: list[str] = []
    add = L.append

    add("=" * W)
    add("CLARE(순차) 셋팅 · libero_goal — 앵커 가중 분모 상수 c 스윕  [v4 / TETHER]")
    add("=" * W)
    add("")
    add("무엇을 바꾸는가")
    add("  v4 앵커 가중      w = 1 / ( ||v~_T[0:6]||^2 + c )")
    add("  를 쓰고, w 는 median(w) x w_clip 에서 자른 뒤 배치 평균 1 로 정규화해")
    add("  pose 6 축에만 곱한다(gripper 제외). 바꾸는 건 분모의 c 하나뿐이다.")
    add("")
    add("  c 가 작아지면 분모가 작아져 w 가 커지고, 특히 **느린 프레임**(||v~_T|| 가")
    add("  작은 구간)의 가중이 가파르게 커진다. 즉 c 는 '느린 구간을 얼마나 더")
    add("  보호할 것인가' 의 손잡이다. 다만 w_clip 과 평균 1 정규화가 뒤에 있어")
    add("  절대 크기가 아니라 **가중의 모양**이 바뀐다.")
    add("")
    add("  V4  w = w~_h / mean(w~)   배치 평균 1 로 정규화 -> 앵커 **크기**가 c 와 무관,")
    add("                              가중의 **모양**만 바뀐다.")
    add("  V5  w = w~_h              정규화 없음 -> 모양과 **크기**가 같이 바뀐다.")
    add("                              w ~ 1/(rho^2+c) 이므로 사실상 λ 를 c 에 묶는 것과 같다.")
    add("")
    for tag, name, desc in ARMS:
        add(f"  {name:<16} {desc}")
    add("")
    add("프로토콜  태스크당 10,000 step, 50 demo, 10 task = 55칸, 코드북 K=96")
    add("          평가 칸당 **20 rollout**, eval seed 7 고정, train seed 7")
    add("          libero_90 사전학습 체크포인트에서 출발")
    add("")
    add("  ★ 칸당 20 rollout 이면 개별 칸의 이항 표준오차가 최대 ±11%p, 50 이면 ±7%p 다.")
    add("    n50 은 기존 20 회(eval seed 7~26) 에 30 회(seed 27~56) 를 더해 가중평균한 것이다:")
    add("        SR_50 = (SR_20 x 20 + SR_30 x 30) / 50")
    add("    학습은 다시 하지 않았다 — **같은 체크포인트**를 더 많은 초기 상태로 굴린 것이다.")
    add("    따라서 n20 과 n50 의 차이는 방법 차이가 아니라 **측정 잡음이 줄어든 것**이다.")
    add("")
    add("지표  FWT↑ 배운 직후 / NBT↓ 이후 잃은 양(+면 망각) / AUC↑ 전 구간 평균")
    add("      AvgSR = 마지막 **완성된** 행의 평균 (진행 중이면 행 번호를 괄호에)")
    add("")

    rows = []
    for tag, name, _desc in ARMS:
        for seed in SEEDS:
            d = RES / f"clare_{tag}_seed{seed}"
            R, n_task = read_matrix(d)
            add("-" * W)
            add(f"{name}   train seed {seed}   ({d.name})")
            add("-" * W)
            if not R:
                add("  아직 SR 칸이 하나도 없다")
                add("")
                rows.append((name, seed, None, "미시작", 0, 55))
                continue
            m = metrics(R, n_task)
            filled = sum(1 for r in R for c in r if c is not None)
            total = n_task * (n_task + 1) // 2
            done = filled >= total
            add(f"  {'stage':<7}" + "".join(f"{'t'+str(t):>6}" for t in range(n_task)))
            for k in range(len(R)):
                add(f"  {k:<7}" + "".join(
                    (f"{R[k][t]:6.0f}" if R[k][t] is not None else f"{'':>6}")
                    for t in range(n_task)))
            f1 = lambda v: "-" if v is None else f"{v:.1f}"      # noqa: E731
            fs = lambda v: "-" if v is None else f"{v:+.1f}"     # noqa: E731
            add(f"  지표  AvgSR {f1(m['AvgSR'])}"
                + (f" (행{m['row']})" if m["row"] is not None and not done else "")
                + f"   FWT↑ {f1(m['FWT'])}   NBT↓ {fs(m['NBT'])}   AUC↑ {f1(m['AUC'])}"
                + f"   [{'완주' if done else f'진행 중 {filled}/{total}칸'}]")
            add("")
            rows.append((name, seed, m, "완주" if done else f"{filled}/{total}칸", filled, total))

    add("=" * W)
    add("대조")
    add("=" * W)
    add(f"  {'팔':<20}{'seed':>5}{'AvgSR':>9}{'FWT↑':>9}{'NBT↓':>9}{'AUC↑':>9}   상태")
    for name, seed, m, state, _f, _t in rows:
        if m is None:
            add(f"  {name:<20}{seed:>5}" + "".join(f"{'-':>9}" for _ in range(4)) + f"   {state}")
            continue
        f1 = lambda v: f"{'-':>9}" if v is None else f"{v:9.1f}"      # noqa: E731
        fs = lambda v: f"{'-':>9}" if v is None else f"{v:+9.1f}"     # noqa: E731
        add(f"  {name:<20}{seed:>5}" + f1(m["AvgSR"]) + f1(m["FWT"])
            + fs(m["NBT"]) + f1(m["AUC"]) + f"   {state}")
    add("")

    # ── V5 시드 집계 (c 별 평균 ± 표준편차) ─────────────────────────────
    add("=" * W)
    add("V5 시드 집계 — c 별 평균 ± 표본표준편차  (칸당 50 rollout, train seed 7/42/1000)")
    add("=" * W)
    add(f"  {'c':<8}{'시드':>6}{'AvgSR':>15}{'FWT↑':>15}{'NBT↓':>15}{'AUC↑':>15}   미완주")
    for c in V5_CS:
        per = {}
        partial = []
        for seed in V5_SEEDS:
            d = v5_dir(c, seed)
            R, n_task = read_matrix(d)
            if not R:
                partial.append(f"seed{seed} 0칸"); continue
            filled = sum(1 for r in R for x in r if x is not None)
            total = n_task * (n_task + 1) // 2
            if filled < total:
                # ★ 미완주 시드는 집계에서 뺀다. AvgSR 이 '마지막 완성 행' 기준이라
                #   행 수가 다른 값을 섞으면 평균이 의미를 잃는다.
                partial.append(f"seed{seed} {filled}/{total}칸"); continue
            per[seed] = metrics(R, n_task)
        ks = sorted(per)
        add(f"  {c:<8}{len(ks):>6}"
            + agg([per[k]["AvgSR"] for k in ks])
            + agg([per[k]["FWT"] for k in ks])
            + agg([per[k]["NBT"] for k in ks])
            + agg([per[k]["AUC"] for k in ks])
            + ("   " + ", ".join(partial) if partial else "   -"))
        for k in ks:
            m = per[k]
            add(f"    {'':<6}seed{k:<5}"
                + f"{m['AvgSR']:>8.1f}{'':>6}" + f"{m['FWT']:>8.1f}{'':>6}"
                + f"{m['NBT']:>+8.1f}{'':>6}" + f"{m['AUC']:>8.1f}{'':>6}")
    add("")
    add("  ★ 표준편차는 **완주한 시드만** 모아 낸 표본표준편차(n−1)다. 미완주 시드는")
    add("    AvgSR 이 '마지막 완성 행' 기준이라 행 수가 다른 값을 섞으면 평균이")
    add("    의미를 잃으므로 오른쪽 칸에 이름만 적고 집계에서 뺀다.")
    add("  ★ 시드 2 개로 낸 표준편차는 그 자체가 매우 불안정하다(자유도 1). 시드 3 개가")
    add("    모두 차기 전에는 '대략 이 정도 흔들린다' 이상으로 읽지 마라.")
    add("")

    add("읽을 때 주의")
    add("  · 시드가 1 개다. 팔 간 차이가 시드 변동(기존 CLARE 측정에서 AvgSR")
    add("    표준편차 2.2%p)보다 작으면 c 의 효과로 읽으면 안 된다.")
    add("  · 0행(스테이지 0)은 사전학습 + task 0 만 배운 상태다. 앵커가 걸릴")
    add("    과거가 없으므로 세 팔이 같아야 정상이다 — 다르면 구현을 의심해야 한다.")
    add("  · 진행 중인 런의 AvgSR 은 마지막 완성 행 기준이라 최종값이 아니다.")
    add("  · 실제로 쓰인 c 는 각 런 로그의 '[L2CB][v4] ... c=...' 줄에서 확인된다.")
    add("")

    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
