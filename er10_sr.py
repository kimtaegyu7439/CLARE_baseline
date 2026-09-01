#!/usr/bin/env python
"""ER 10-task (칸당 20 롤아웃) SR 행렬 모음 — results/ER_10task_SR.txt.

네 스위트의 ER 결과가 서로 다른 형식/위치에 흩어져 있어서 한 파일로 모은다.

    libero_spatial   B_mod_none_null.txt 안의 "ER (55/55 칸, 완료)" 블록
                     (전용 디렉터리가 ER_10task/ 이고 파일 이름이 4-task 시절
                      ER_task0123_SR.txt 로 남아 있어 여기서 읽지 않는다)
    나머지 3개       ER_<suite>_10task/sr_matrix.csv

집계는 원본 문서 값을 베끼지 않고 칸에서 다시 계산한다(검산 겸).

    python er10_sr.py
"""
from __future__ import annotations

import re
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
OUT = RES / "ER_10task_SR.txt"
K = 10

SUITES = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]


def from_b_mod() -> dict[tuple[int, int], float]:
    """B_mod_none_null.txt 의 ER 블록. 'after\\task' 머리줄 뒤 K 행."""
    txt = (RES / "B_mod_none_null.txt").read_text().splitlines()
    i = next(n for n, l in enumerate(txt)
             if l.startswith("ER ") and "55/55" in l)
    j = next(n for n in range(i, i + 6) if txt[n].lstrip().startswith("after"))
    cells = {}
    for k, line in enumerate(txt[j + 1:j + 1 + K]):
        f = line.split()
        assert int(f[0]) == k, line
        for t, v in enumerate(f[1:]):
            if v != ".":
                cells[(k, t)] = float(v)
    return cells


def from_csv(path: Path) -> dict[tuple[int, int], float]:
    cells = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split(",")
        if not f[0].strip().isdigit():
            continue
        k = int(f[0])
        for t, v in enumerate(f[1:]):
            if v.strip():
                cells[(k, t)] = float(v)
    return cells


def stats(c: dict[tuple[int, int], float]) -> dict:
    last = [c[(K - 1, t)] for t in range(K)]
    diag = [c[(t, t)] for t in range(K)]
    return {"avg": sum(last) / K,
            "acq": sum(diag) / K,
            "bwt": sum(last[t] - diag[t] for t in range(K - 1)) / (K - 1),
            "n": len(c)}


def block(name: str, c: dict, s: dict, src: str) -> list[str]:
    L = ["-" * 84,
         f"{name}   ({s['n']}/55 칸)   AvgSR {s['avg']:.1f}   "
         f"BWT {s['bwt']:+.1f}   습득 {s['acq']:.1f}",
         f"출처 {src}",
         "-" * 84,
         "after\\task " + "".join(f"{t:>6}" for t in range(K))]
    for k in range(K):
        L.append(f"{k:>10} " + "".join(
            (f"{c[(k,t)]:>6.0f}" if (k, t) in c else f"{'.':>6}") for t in range(K)))
    return L + [""]


def main() -> None:
    got = {"libero_spatial": (from_b_mod(), "results/B_mod_none_null.txt 의 ER 블록")}
    for s in SUITES[1:]:
        p = RES / f"ER_{s}_10task" / "sr_matrix.csv"
        got[s] = (from_csv(p), f"results/ER_{s}_10task/sr_matrix.csv")

    L = ["=" * 84,
         "ER (experience replay) — 스위트별 10 태스크, 칸당 20 롤아웃",
         "=" * 84, "",
         "프로토콜  5000 step/task, 45 에피소드(뒤 5개 hold-out), seed 42,",
         "          배치 32 = 현재 태스크 24 + 리플레이 버퍼 8, 과거 태스크당 5 에피소드.",
         "          K1/R1x 팔과 같은 조건이다(그쪽은 과거 원시 데이터를 안 쓴다).",
         "측정      시뮬레이터 롤아웃 성공률(%), 칸당 20 에피소드.",
         "          이항 표준오차가 칸당 최대 ±11%p 다. 행/열 평균으로 읽어야 한다.",
         "",
         "행 = 스테이지 k (태스크 k 까지 학습한 시점), 열 = 평가 태스크 j. 하삼각 55칸.",
         "AvgSR = 마지막 행 평균,  습득 = 대각 평균,  BWT = 마지막행 − 대각 (task9 제외).",
         "집계값은 이 스크립트가 칸에서 다시 계산한 것이다(er10_sr.py).", "",
         "-" * 84, "요약", "-" * 84,
         f"{'스위트':<18}{'AvgSR':>8}{'BWT':>8}{'습득':>8}", ""]
    for s in SUITES:
        c, _ = got[s]
        st = stats(c)
        L.append(f"{s:<18}{st['avg']:>8.1f}{st['bwt']:>+8.1f}{st['acq']:>8.1f}")
    L += ["", "주의  results/ER_libero_40_SR.txt 는 40 태스크를 한 줄로 이어붙인 별개 실험이다",
          "      (820칸, AvgSR 55.8). 위 표와 직접 비교하면 안 된다.", ""]

    for s in SUITES:
        c, src = got[s]
        L += block(s, c, stats(c), src)

    OUT.write_text("\n".join(L) + "\n")
    print(f"saved -> {OUT}")
    for s in SUITES:
        st = stats(got[s][0])
        print(f"  {s:<16} {st['n']}/55 칸  AvgSR {st['avg']:.1f}  "
              f"BWT {st['bwt']:+.1f}  습득 {st['acq']:.1f}")


if __name__ == "__main__":
    main()
