#!/usr/bin/env python
"""V5 의 20 회 + 추가 30 회를 합쳐 칸당 50 회 SR 행렬을 만든다.

    python CLARE_v5_merge50.py            합치고 결과를 출력
    python CLARE_v5_merge50.py --check    상태만 본다

    results/clare_<arm>_seed7          칸당 20 회 (eval seed 7~26)
  + results/clare_<arm>_seed7_extra30  칸당 30 회 (eval seed 27~56)
  -> results/clare_<arm>_seed7_n50     칸당 50 회

합치는 법
    SR 은 성공률(%)이므로 횟수로 가중평균한다.

        SR_50 = (SR_20 x 20 + SR_30 x 30) / 50

    두 구간의 eval 시드가 겹치지 않으므로(7~26 vs 27~56) 50 개의 서로 다른
    초기 상태에 대한 성공률이 된다. 같은 시드를 두 번 세는 일이 없다.

★ 한쪽이라도 비어 있는 칸은 비워 둔다. 20 회만 있는 칸을 50 회인 척하면
  다른 칸과 다른 것을 재고 있게 된다.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
ARMS = ["v5_goal_c0.01", "v5_goal_c0.05", "v5_goal_c0.1"]
N_BASE, N_EXTRA = 20, 30
K = 10


def read(d: Path) -> list[list[float | None]]:
    R: list[list[float | None]] = [[None] * K for _ in range(K)]
    p = d / "sr_matrix.csv"
    if not p.exists():
        return R
    body = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
    for line in body[1:]:
        c = line.split(",")
        k = int(c[0])
        if k >= K:
            continue
        for i, v in enumerate(c[1:K + 1]):
            if v.strip():
                R[k][i] = float(v)
    return R


def write(d: Path, R, src: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    # config/metrics 류는 20 회 쪽을 복사해 둔다(리포트가 seeds_of 로 읽는다)
    for name in ("config.json", "metrics.json", "instructions.json"):
        s = src / name
        if s.exists():
            shutil.copy(s, d / name)
    head = "after_stage," + ",".join(f"stage{t}(task{t})" for t in range(K))
    lines = [f"# merged: {N_BASE} (eval seed 7-26) + {N_EXTRA} (eval seed 27-56) = {N_BASE+N_EXTRA} rollouts/cell",
             head]
    for k in range(K):
        cells = [("" if R[k][t] is None else f"{R[k][t]:.6g}") for t in range(K)]
        lines.append(str(k) + "," + ",".join(cells))
    (d / "sr_matrix.csv").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    print(f"{'팔':<18}{'20회':>8}{'30회':>8}{'합쳐짐':>9}   비고")
    for arm in ARMS:
        d20 = RES / f"clare_{arm}_seed7"
        d30 = RES / f"clare_{arm}_seed7_extra30"
        R20, R30 = read(d20), read(d30)
        n20 = sum(1 for r in R20 for v in r if v is not None)
        n30 = sum(1 for r in R30 for v in r if v is not None)

        R50: list[list[float | None]] = [[None] * K for _ in range(K)]
        both = 0
        for k in range(K):
            for t in range(k + 1):
                if R20[k][t] is not None and R30[k][t] is not None:
                    R50[k][t] = (R20[k][t] * N_BASE + R30[k][t] * N_EXTRA) / (N_BASE + N_EXTRA)
                    both += 1
        note = "" if both == 55 else f"★ {55 - both}칸 부족 — 30회가 덜 끝남"
        print(f"  {arm:<16}{n20:>8}{n30:>8}{both:>9}   {note}")
        if not a.check and both:
            write(RES / f"clare_{arm}_n50_seed7", R50, d20)   # 리포트가 clare_<tag>_seed7 로 읽는다
    if not a.check:
        print(f"\n-> results/clare_<arm>_seed7_n50/sr_matrix.csv 에 기록")


if __name__ == "__main__":
    main()
