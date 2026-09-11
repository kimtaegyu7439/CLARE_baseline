#!/usr/bin/env python
"""ER 5k 체크포인트의 롤아웃 재측정 집계 — 같은 가중치, 시드만 다름."""
from __future__ import annotations
import json, math, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
N = 4
SEEDS = [("42", REPO / "results/ER_task0123/er_results.jsonl")] + [
    (s, REPO / f"results/ER_reprobe/seed{s}/er_results.jsonl") for s in ("43", "44", "45")]


def cells(p):
    c = {}
    if not p.exists():
        return c
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("run_tag") == "er" and r.get("sr") is not None:
            c[(r["stage"], r["probe_task"])] = float(r["sr"])
    return c


def metrics(c):
    last = [c.get((N - 1, t)) for t in range(N)]
    diag = [c.get((t, t)) for t in range(N)]
    if any(v is None for v in last + diag):
        return None
    return (sum(last) / N,
            sum(last[t] - diag[t] for t in range(N - 1)) / (N - 1))


runs = [(s, cells(p)) for s, p in SEEDS]
have = [(s, c) for s, c in runs if c]

L = ["=" * 78,
     "ER (5000 steps/task) — 같은 체크포인트, 롤아웃 시드만 바꿔 재측정",
     "=" * 78, "",
     "가중치는 outputs/ER/libero_spatial/seed42 로 고정. --seed 가 eval_policy 의",
     "start_seed 로 들어가므로(E0.py:264) 초기 상태와 ODE 노이즈만 달라진다.",
     "칸당 20 롤아웃. seed 42 행은 원래 측정값(results/ER_task0123)이다.", ""]

for s, c in have:
    m = metrics(c)
    L += ["-" * 78,
          f"seed {s}" + (f"   AvgSR {m[0]:.1f}   BWT {m[1]:+.1f}" if m else "   (미완)"),
          "-" * 78,
          "after\\task " + "".join(f"{t:>7d}" for t in range(N))]
    for k in range(N):
        L.append(f"{k:>10d} " + "".join(
            f"{c[(k, t)]:7.0f}" if (k, t) in c else "      ." for t in range(k + 1)))
    L.append("")

L += ["=" * 78, "칸별 요약 — 시드 간 평균 ± 표준편차 (n = 참여한 시드 수)", "=" * 78, "",
      f"{'stage':>6}{'task':>6}" + "".join(f"{'s'+s:>8}" for s, _ in have)
      + f"{'평균':>9}{'표준편차':>10}{'범위':>10}"]
agg = {}
for k in range(N):
    for t in range(k + 1):
        vs = [c[(k, t)] for _, c in have if (k, t) in c]
        if not vs:
            continue
        mu = sum(vs) / len(vs)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vs) / (len(vs) - 1)) if len(vs) > 1 else 0.0
        agg[(k, t)] = (mu, sd)
        row = f"{k:>6}{t:>6}"
        for _, c in have:
            row += f"{c[(k,t)]:8.0f}" if (k, t) in c else f"{'.':>8}"
        L.append(row + f"{mu:>9.1f}{sd:>10.1f}{max(vs)-min(vs):>10.0f}")

L += ["", "-" * 78, "최종 행(stage 3) 평균 SR", "-" * 78,
      "      " + "".join(f"{'task'+str(t):>10}" for t in range(N))]
L.append("평균  " + "".join(
    f"{agg[(N-1,t)][0]:10.1f}" if (N - 1, t) in agg else f"{'.':>10}" for t in range(N)))
L.append("표준편차" + "".join(
    f"{agg[(N-1,t)][1]:10.1f}" if (N - 1, t) in agg else f"{'.':>10}" for t in range(N)))

ms = [metrics(c) for _, c in have if metrics(c)]
if ms:
    a = [m[0] for m in ms]
    mu = sum(a) / len(a)
    sd = math.sqrt(sum((v - mu) ** 2 for v in a) / (len(a) - 1)) if len(a) > 1 else 0.0
    L += ["", f"AvgSR: " + ", ".join(f"{v:.1f}" for v in a) + f"   -> {mu:.1f} ± {sd:.1f}"]

rep = "\n".join(L)
out = REPO / "results/ER_reprobe"; out.mkdir(parents=True, exist_ok=True)
(out / "report.txt").write_text(rep)
print(rep + f"\n\nsaved -> {out/'report.txt'}")
