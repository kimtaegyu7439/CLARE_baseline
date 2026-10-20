#!/usr/bin/env python
"""V5 세 팔의 시드 간 평균 ± 표준편차 -> Clare_TETHER_MAR_sweep.txt 끝에 덧붙인다.

    python CLARE_v5_seed_summary.py            표를 만들어 파일 끝에 붙인다
    python CLARE_v5_seed_summary.py --print    출력만

읽는 것
    seed 7     results/clare_v5_goal_c<C>_n50_seed7   (20회 + 30회 합친 n=50)
    seed 42    results/clare_v5_goal_c<C>_seed42      (처음부터 n=50)
    seed 1000  results/clare_v5_goal_c<C>_seed1000    (처음부터 n=50)

  세 시드 모두 **칸당 50 rollout, eval seed 7~56** 으로 같은 초기 상태를 본다.
  다른 것은 **학습 시드**뿐이라, 여기서 나오는 편차가 곧 학습의 재현성이다.

★ 완주(55칸)한 시드만 평균에 넣는다. 반쯤 찬 행렬의 AvgSR 은 "마지막 완성 행"
  기준이라 완주한 런의 마지막 행과 같은 것을 재고 있지 않다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from clare_v4_report import metrics, read_matrix

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
OUT = RES / "Clare_TETHER_MAR_sweep.txt"
ARMS = [("0.1", "V5  c = 0.1"), ("0.05", "V5  c = 0.05"), ("0.01", "V5  c = 0.01")]
SEEDS = [7, 42, 1000]
KEYS = (("AvgSR", False), ("FWT", False), ("NBT", True), ("AUC", False))
W = 92


def rundir(c: str, seed: int) -> Path:
    # seed 7 은 20회+30회를 합쳐 만든 n50 폴더에 있다
    return RES / (f"clare_v5_goal_c{c}_n50_seed7" if seed == 7
                  else f"clare_v5_goal_c{c}_seed{seed}")


def ms(v: list[float]):
    """(평균, 표본표준편차 ddof=1, n). 1 개면 편차는 None."""
    if not v:
        return None, None, 0
    mu = sum(v) / len(v)
    if len(v) < 2:
        return mu, None, 1
    return mu, (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5, len(v)


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--print", dest="only_print", action="store_true")
    args = a.parse_args()

    L: list[str] = ["", "=" * W,
                    "V5 시드 요약 — libero_goal, 칸당 50 rollout (eval seed 7~56 고정)",
                    "=" * W, "",
                    "  학습 시드만 7 / 42 / 1000 으로 바꿨다. 평가 조건(초기 상태 50개)은 같다.",
                    "  따라서 ± 는 **학습의 재현성**이고, 롤아웃 잡음은 이미 50회로 눌려 있다.",
                    ""]
    rows = []
    for c, name in ARMS:
        per: dict[str, list[float]] = {k: [] for k, _ in KEYS}
        done, pend = [], []
        for s in SEEDS:
            d = rundir(c, s)
            R, n = read_matrix(d)
            if not R:
                pend.append(f"seed{s} 미시작"); continue
            filled = sum(1 for r in R for v in r if v is not None)
            if filled < n * (n + 1) // 2:
                pend.append(f"seed{s} {filled}/{n*(n+1)//2}칸"); continue
            m = metrics(R, n)
            done.append(s)
            for k, _ in KEYS:
                if isinstance(m.get(k), (int, float)):
                    per[k].append(m[k])
        rows.append((name, done, pend, per))

    L.append(f"  {'팔':<14}{'완주':>5}   " + "".join(f"{k:>16}" for k, _ in KEYS) + "   비고")
    L.append("  " + "-" * (W - 2))
    for name, done, pend, per in rows:
        cells = ""
        for k, signed in KEYS:
            mu, sd, n = ms(per[k])
            if mu is None:
                cells += f"{'-':>16}"; continue
            s = ("%+.1f" if signed else "%.1f") % mu
            cells += f"{(s + (f' ± {sd:.1f}' if sd is not None else f' (n={n})')):>16}"
        note = ", ".join(pend) if pend else ""
        L.append(f"  {name:<14}{len(done):>5}   " + cells + (f"   {note}" if note else ""))
    L.append("")
    L.append("  시드별 원값")
    L.append(f"  {'팔':<14}{'seed':>6}" + "".join(f"{k:>9}" for k, _ in KEYS))
    for c, name in ARMS:
        for s in SEEDS:
            R, n = read_matrix(rundir(c, s))
            if not R:
                L.append(f"  {name:<14}{s:>6}" + "".join(f"{'-':>9}" for _ in KEYS)); continue
            filled = sum(1 for r in R for v in r if v is not None)
            m = metrics(R, n)
            tail = "" if filled >= n * (n + 1) // 2 else f"  ({filled}/{n*(n+1)//2}칸)"
            L.append(f"  {name:<14}{s:>6}"
                     + "".join((f"{m[k]:>9.1f}" if isinstance(m.get(k), (int, float)) else f"{'-':>9}")
                               for k, _ in KEYS) + tail)
    L += ["",
          "  읽을 때",
          "  · ± 는 표본표준편차(ddof=1)다. 시드 3 개뿐이라 이 값 자체도 흔들린다.",
          "  · 팔 간 차이가 ± 보다 작으면 c 의 효과로 읽으면 안 된다.",
          "  · NBT 는 FWT 가 낮으면 자동으로 좋아진다(떨어질 자리가 없다).",
          "    FWT 가 비슷한 팔끼리만 NBT 를 맞대 비교하라.",
          ""]

    txt = "\n".join(L)
    print(txt)
    if not args.only_print:
        base = OUT.read_text() if OUT.exists() else ""
        marker = "V5 시드 요약"
        if marker in base:
            base = base[:base.index("=" * W + "\n" + marker) if ("=" * W + "\n" + marker) in base
                        else base.index(marker)].rstrip("=\n ") + "\n"
        OUT.write_text(base.rstrip() + "\n" + txt + "\n")
        print(f"\n-> {OUT} 끝에 기록")


if __name__ == "__main__":
    main()
