#!/usr/bin/env python
"""K0 summary.json -> 사람이 읽는 summary.txt.

    python K0_summary.py results/K0_10task [results/K0 ...]

핵심은 "상한 대비 (bin 평균)" 표다 — rank 를 얼마로 잡아야 하는지가 여기서 읽힌다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def build(d: Path) -> str:
    j = json.loads((d / "summary.json").read_text())
    G, R = j["grid"], j["ranks"]
    tasks = sorted(int(k[4:]) for k in G if k.startswith("task") and "-" not in k)
    D = j["d"]

    def binmean(nm, r):
        g = [x for x in G[nm] if x]
        return float(np.mean([x["rho"][str(r)] for x in g]))

    def binmin(nm, r):
        g = [x for x in G[nm] if x]
        return float(np.min([x["rho"][str(r)] for x in g]))

    ref = {r: binmean("task0-B", r) for r in R}
    W = 7
    L = ["=" * 78,
         f"K0 — 공유 기저 전이 검사   {j['suite']}, task 0..{max(tasks)}, "
         f"bin {j['n_bins']}개", "=" * 78, "",
         "기저는 task 0 을 에피소드 단위로 A/B 로 가른 뒤 A 로만 만들었다.",
         "rho_k(tau; r) = mean‖W_r^T x‖² / mean‖x‖²,  x 는 (k,tau) 자기 평균 중심화.",
         "상한 = task0-B (같은 태스크 held-out)   하한 = r/3072 (무작위 부분공간)",
         "",
         f"기저 W(3072, {max(R)})  task0-A {j['basis']['n_frames_A']} 프레임  "
         f"‖WᵀW−I‖ = {j['basis']['orth_err']:.2e}",
         "task0-A 자신의 누적 설명분산  "
         + "   ".join(f"r={r} {j['basis']['cum_evr'][str(r)]*100:.1f}%" for r in R),
         "",
         "-" * 78,
         "상한 대비 (bin 평균, %)",
         "-" * 78,
         f"{'':<12}" + "".join(f"{('r='+str(r)):>{W}}" for r in R)]
    for k in tasks:
        L.append(f"task{k:<8}" + "".join(
            f"{binmean(f'task{k}', r)/ref[r]*100:{W}.0f}" for r in R))
    L += ["", f"{'task0-B':<12}" + "".join(f"{100:{W}.0f}" for r in R) + "   <- 상한 자신",
          "",
          "-" * 78,
          "상한 대비 (bin 최솟값, %)   — 판정은 이 값으로 한다",
          "-" * 78,
          f"{'':<12}" + "".join(f"{('r='+str(r)):>{W}}" for r in R)]
    for k in tasks:
        L.append(f"task{k:<8}" + "".join(
            f"{binmin(f'task{k}', r)/ref[r]*100:{W}.0f}" for r in R))
    L += ["",
          "-" * 78,
          "절대값 rho (%, bin 평균)",
          "-" * 78,
          f"{'':<12}" + "".join(f"{('r='+str(r)):>{W}}" for r in R)]
    for nm in [f"task{k}" for k in tasks] + ["task0-B"]:
        L.append(f"{nm:<12}" + "".join(f"{binmean(nm, r)*100:{W}.1f}" for r in R))
    L.append(f"{'무작위(하한)':<10}" + "".join(f"{r/D*100:{W}.1f}" for r in R))

    L += ["",
          "-" * 78,
          "판정 (r=256 기준)   OK >= 상한의 90% / 경계 70~90% / 기각 <70% 또는 bin<60%",
          "-" * 78,
          f"{'':<10}{'rho 평균':>10}{'rho 최소':>10}{'상한의':>9}"
          f"{'ā':>8}{'offdiag':>10}   판정"]
    for k in tasks:
        v = j["verdict"][f"task{k}"]
        g = [x for x in G[f"task{k}"] if x]
        L.append(f"task{k:<6}{v['rho256_mean']*100:9.1f}%{v['rho256_min']*100:9.1f}%"
                 f"{v['frac_of_ceiling']*100:8.1f}%"
                 f"{np.mean([x['abar'] for x in g]):8.3f}"
                 f"{np.mean([x['offdiag'] for x in g]):10.3f}   {v['verdict']}")
    gb = [x for x in G["task0-B"] if x]
    L.append(f"{'task0-B':<10}{binmean('task0-B',256)*100:9.1f}%"
             f"{binmin('task0-B',256)*100:9.1f}%{100.0:8.1f}%"
             f"{np.mean([x['abar'] for x in gb]):8.3f}"
             f"{np.mean([x['offdiag'] for x in gb]):10.3f}   상한")

    L += ["",
          "ā       = 상위 10 고유벡터의 eigenvalue 가중 정렬도 mean‖W_256^T v_i‖²",
          "offdiag = corr(W_256^T x) 의 비대각 에너지 비율. 클수록 회전 후에도 좌표가 얽혀 있다.",
          ""]
    notes = j.get("low_alignment_notes") or []
    L.append("정렬도 낮은 상위 고유벡터 (a_i < 0.5): "
             + ("없음" if not notes else ""))
    for n in notes:
        L.append("  " + n)
    L += ["",
          "그림  fig1_rho.png (메인, r=256)  fig2_rank_sweep.png  fig3_leak_and_corr.png",
          ""]
    return "\n".join(L)


def main() -> None:
    dirs = [Path(x) for x in (sys.argv[1:] or ["results/K0_10task"])]
    for d in dirs:
        if not (d / "summary.json").exists():
            print(f"skip {d} — summary.json 없음"); continue
        txt = build(d)
        (d / "summary.txt").write_text(txt + "\n")
        print(txt)
        print(f"saved -> {d/'summary.txt'}\n")


if __name__ == "__main__":
    main()
