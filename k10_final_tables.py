#!/usr/bin/env python
"""K10 체인 Phase 2 마무리 — 3팔 SR 표 + 한 장 요약.

    python k10_final_tables.py

읽는 것   results/{K10L,K7b,K10LB}/sr_matrix.csv   (B1 이 칸마다 갱신하므로 미완이어도 읽힌다)
쓰는 것   각 팔의 sr_table.{csv,md}  +  results/K10/summary.md

md 하단에는 참고값(R13=79.5, K1=77.0)과 **task1 열의 stage 별 궤적**을 R13 과 나란히 놓는다.
10 태스크에서 무너지는 지점이 매번 task1 이었기 때문이다
(R13 70 / K1 30 / R12 20 / R11 15 / R10 5 — results/B_mod_none_null.txt).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
RES = REPO / "results"

ARMS = [
    ("K10L", "R13 + Langevin 합성"),
    ("K7b", "R13 + 잔차-EMA task 배분"),
    ("K10LB", "K10-L + K7b 결합"),
]
REF = {"R13": 79.5, "K1": 77.0, "ER": 86.0}
R13_DIR = RES / "R13_10task"


def read_sr(d: Path):
    p = d / "sr_matrix.csv"
    if not p.exists():
        return None, 0
    cells, K = {}, 0
    for line in p.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split(",")
        if not f[0].strip().isdigit():
            continue
        k = int(f[0]); K = max(K, len(f) - 1)
        for t, v in enumerate(f[1:]):
            if v.strip():
                cells[(k, t)] = float(v)
    return cells, K


def summarize(cells, K):
    last = [cells.get((K - 1, t)) for t in range(K)]
    diag = [cells.get((t, t)) for t in range(K)]
    done = all(v is not None for v in last)
    acq = [v for v in diag if v is not None]
    avg = sum(last) / K if done else None
    bwt = (sum(last[t] - diag[t] for t in range(K - 1)) / (K - 1)
           if done and all(v is not None for v in diag) else None)
    return {"avg": avg, "bwt": bwt, "acq": (sum(acq) / len(acq)) if acq else None,
            "filled": len(cells), "total": K * (K + 1) // 2}


def write_arm(name: str, subtitle: str, cells, K, s):
    d = RES / name
    with (d / "sr_table.csv").open("w") as f:
        f.write("after_task," + ",".join(f"task{t}" for t in range(K)) + "\n")
        for k in range(K):
            f.write(f"{k}," + ",".join(
                f"{cells[(k,t)]:.1f}" if (k, t) in cells else "" for t in range(K)) + "\n")
    L = [f"# {name} — {subtitle} (LIBERO-spatial, {K} task, 20 rollout/칸)", "",
         "| after task | " + " | ".join(f"task{t}" for t in range(K)) + " |",
         "|---" * (K + 1) + "|"]
    for k in range(K):
        L.append(f"| {k} | " + " | ".join(
            f"{cells[(k,t)]:.0f}" if (k, t) in cells else "" for t in range(K)) + " |")
    head = (f"**AvgSR (마지막 행 평균) = {s['avg']:.1f}**" if s["avg"] is not None
            else f"**진행 중 — {s['filled']}/{s['total']} 칸**")
    if s["bwt"] is not None:
        head += f"   BWT {s['bwt']:+.1f}"
    if s["acq"] is not None:
        head += f"   습득 {s['acq']:.1f}"
    L += ["", head, "",
          "참고값  " + "   ".join(f"{k} = {v}" for k, v in REF.items())]
    L += ["", "## task1 궤적 (stage 별)", "",
          "| stage | " + " | ".join(str(k) for k in range(K)) + " |",
          "|---" * (K + 1) + "|"]
    r13, K13 = read_sr(R13_DIR)
    L.append(f"| {name} | " + " | ".join(
        f"{cells[(k,1)]:.0f}" if (k, 1) in cells else "" for k in range(K)) + " |")
    if r13:
        L.append("| R13 | " + " | ".join(
            f"{r13[(k,1)]:.0f}" if (k, 1) in r13 else "" for k in range(K)) + " |")
    (d / "sr_table.md").write_text("\n".join(L) + "\n")
    return L


def main() -> None:
    (RES / "K10").mkdir(parents=True, exist_ok=True)
    rows, blocks = [], []
    for name, sub in ARMS:
        cells, K = read_sr(RES / name)
        if cells is None:
            rows.append((name, sub, None)); continue
        s = summarize(cells, K)
        write_arm(name, sub, cells, K, s)
        rows.append((name, sub, s))
        blocks.append((name, cells, K, s))
        a = f"{s['avg']:.1f}" if s["avg"] is not None else f"진행중 {s['filled']}/{s['total']}"
        print(f"  {name:<7} {sub:<28} AvgSR {a}")

    L = ["# K10 체인 — 3팔 요약 (LIBERO-spatial 10 task)", "",
         "R13 프로토콜 그대로: 5000 step/task, batch 32, rolling teacher, p_drop=0, w=1,",
         "칸당 20 rollout, 55칸. 과거 원시 데이터 미사용.", "",
         "| 팔 | 내용 | 칸 | AvgSR | BWT | 습득 |", "|---|---|---|---|---|---|"]
    for name, sub, s in rows:
        if s is None:
            L.append(f"| {name} | {sub} | 미시작 | | | |"); continue
        a = f"{s['avg']:.1f}" if s["avg"] is not None else "진행중"
        b = f"{s['bwt']:+.1f}" if s["bwt"] is not None else "—"
        q = f"{s['acq']:.1f}" if s["acq"] is not None else "—"
        L.append(f"| {name} | {sub} | {s['filled']}/{s['total']} | {a} | {b} | {q} |")
    L += ["", "참고값 (같은 프로토콜) " + "   ".join(f"{k} = {v}" for k, v in REF.items()), ""]

    r13, K13 = read_sr(R13_DIR)
    if blocks:
        K = blocks[0][2]
        L += ["## task1 궤적 — 10 태스크에서 매번 무너지던 열", "",
              "| stage | " + " | ".join(str(k) for k in range(K)) + " |",
              "|---" * (K + 1) + "|"]
        for name, cells, Kk, _ in blocks:
            L.append(f"| {name} | " + " | ".join(
                f"{cells[(k,1)]:.0f}" if (k, 1) in cells else "" for k in range(K)) + " |")
        if r13:
            L.append("| R13 | " + " | ".join(
                f"{r13[(k,1)]:.0f}" if (k, 1) in r13 else "" for k in range(K)) + " |")
        L.append("")
        for name, cells, Kk, s in blocks:
            L += [f"## {name}", "",
                  "| after task | " + " | ".join(f"task{t}" for t in range(Kk)) + " |",
                  "|---" * (Kk + 1) + "|"]
            for k in range(Kk):
                L.append(f"| {k} | " + " | ".join(
                    f"{cells[(k,t)]:.0f}" if (k, t) in cells else "" for t in range(Kk)) + " |")
            L.append("")
    p = RES / "K10" / "summary.md"
    p.write_text("\n".join(L) + "\n")
    print(f"\nsaved -> {p}")
    for name, _, _ in ARMS:
        if (RES / name / "sr_table.md").exists():
            print(f"       -> {RES/name/'sr_table.md'}")


if __name__ == "__main__":
    main()
