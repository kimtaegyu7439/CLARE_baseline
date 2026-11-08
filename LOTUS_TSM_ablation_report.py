#!/usr/bin/env python
"""LOTUS_TSM_ablation_report — results/lotus_tsm_<팔>_<suite>_seed<N> 집계.

    python LOTUS_TSM_ablation_report.py              # results/LOTUS_TSM_ablation.txt 갱신 + 출력
    SUITE=object python LOTUS_TSM_ablation_report.py # 한 스위트만

팔 순서  time_bin → state_mean → state_map → tether  (Lotus_TSM_ablation.py 의 사다리)
지표     lifelong 태스크만 (base 열 제외). FWT↑ / NBT↓(+면 망각) / AUC↑ / AvgSR↑(마지막 행)
         시드 평균 ± 표본표준편차(ddof=1). 완주한 시드만 집계에 넣고 진행 중은 * 로 따로 보인다.
읽기/지표 함수는 LOTUS_MAR_sweep_report 의 것을 그대로 쓴다 (같은 정의로 다른 표와 비교 가능).
"""
from __future__ import annotations

import json
import os
import re
import statistics as st
import time
from pathlib import Path

from LOTUS_MAR_sweep_report import fmt, lifelong, progress, read_R

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
OUT = RES / "LOTUS_TSM_ablation.txt"
PAT = re.compile(r"^lotus_tsm_(?P<mode>time_bin|state_mean|state_map|tether)_(?P<suite>goal|object|50)_seed(?P<seed>\d+)$")
ORDER = ["time_bin", "state_mean", "state_map", "tether"]
LABEL = {"time_bin": "Time-bin Gaussian", "state_mean": "State cluster + mean obs",
         "state_map": "State cluster + local map A", "tether": "TETHER (= state_map + MAR)"}
SUITE_NAME = {"goal": "libero_goal", "object": "libero_object", "50": "libero_50"}


def discover():
    runs = []
    if not RES.exists():
        return runs
    for d in sorted(RES.iterdir()):
        m = PAT.match(d.name) if d.is_dir() else None
        if not m or (d / "INVALID.md").exists():
            continue
        g = m.groupdict()
        arm = {}
        f = d / "tsm_arm.json"
        if f.exists():
            try:
                arm = json.loads(f.read_text())
            except Exception:
                arm = {}
        runs.append(dict(dir=d, mode=g["mode"], suite=g["suite"], seed=int(g["seed"]), arm=arm))
    return runs


def pm(vals, sign=False):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return f"{mu:+.1f}" if sign else f"{mu:.1f}"
    sd = st.stdev(vals)
    return (f"{mu:+.1f} ± {sd:.1f}") if sign else (f"{mu:.1f} ± {sd:.1f}")


def main() -> None:
    only = os.environ.get("SUITE")
    runs = discover()
    if only:
        runs = [r for r in runs if r["suite"] == only]
    L = []
    add = L.append
    add("=" * 100)
    add("TSM 구성요소 분해 — LOTUS 셋팅, lifelong 태스크만  (Lotus_TSM_ablation.py)")
    add("=" * 100)
    add(f"갱신 {time.strftime('%F %T')}   런 {len(runs)}개 발견")
    add("")
    add("  팔            셀      s̃                 õ                      앵커")
    add("  time_bin      진행률 K구간   N(μ_s,σ_s)      N(m_k, σ_o,k)           sqerr(MSE)")
    add("  state_mean    state k-means  N(μ_s,σ_s)      m_k                     sqerr(MSE)")
    add("  state_map     state k-means  N(μ_s,σ_s)      m_k + A_k(z̃−z̄_k)        sqerr(MSE)")
    add("  tether        = state_map                                            MAR 1/(r_T²+c)")
    add("")
    add("지표  FWT↑ 배운 직후 / NBT↓ 이후 잃은 양(+면 망각) / AUC↑ 전 구간 평균 / AvgSR↑ 마지막 행")
    add("      t_last = 마지막 lifelong 태스크의 FWT.  칸이 덜 찼으면 * (최종값 아님).")
    add("      ± 는 완주 시드 간 표본표준편차(ddof=1). 칸당 20 rollout → 칸당 이항 SE ±9.7%p.")
    if not runs:
        add("")
        add("  (아직 런이 없다 — results/lotus_tsm_* 폴더가 생기면 채워진다)")
    for suite in [s for s in ("object", "goal", "50") if any(r["suite"] == s for r in runs)]:
        add("")
        add("-" * 100)
        add(SUITE_NAME[suite])
        add("-" * 100)
        add(f"  {'팔':<12}{'seed':>6}{'AvgSR':>8}{'FWT↑':>8}{'NBT↓':>8}{'AUC↑':>8}{'t_last':>8}"
            f"{'칸':>9}  상태")
        agg = {}
        for mode in ORDER:
            for r in sorted((x for x in runs if x["suite"] == suite and x["mode"] == mode),
                            key=lambda x: x["seed"]):
                R, la, src = read_R(r["dir"])
                m = lifelong(R, la) if R else None
                star = "" if (m and m["done"]) else "*"
                cells = f"{m['have']}/{m['need']}" if m else "-"
                status = "완주" if (m and m["done"]) else progress(r["dir"])
                add(f"  {mode:<12}{r['seed']:>6}{fmt(m['AvgSR']) if m else '-':>8}"
                    f"{fmt(m['FWT']) if m else '-':>8}{fmt(m['NBT'], True) if m else '-':>8}"
                    f"{fmt(m['AUC']) if m else '-':>8}{fmt(m['t9']) if m else '-':>7}{star:<1}"
                    f"{cells:>9}  {status}")
                if m and m["done"]:
                    a = agg.setdefault(mode, {"AvgSR": [], "FWT": [], "NBT": [], "AUC": [], "t9": [], "n": 0})
                    for k in ("AvgSR", "FWT", "NBT", "AUC", "t9"):
                        a[k].append(m[k])
                    a["n"] += 1
        add("")
        add("  집계 (완주한 시드만, 평균 ± sd)")
        add(f"  {'팔':<12}{'n':>3}{'AvgSR':>14}{'FWT↑':>14}{'NBT↓':>14}{'AUC↑':>14}{'t_last':>14}")
        for mode in ORDER:
            a = agg.get(mode)
            if not a:
                add(f"  {mode:<12}{0:>3}{'-':>14}{'-':>14}{'-':>14}{'-':>14}{'-':>14}")
                continue
            add(f"  {mode:<12}{a['n']:>3}{pm(a['AvgSR']):>14}{pm(a['FWT']):>14}"
                f"{pm(a['NBT'], True):>14}{pm(a['AUC']):>14}{pm(a['t9']):>14}")
        # 논문 표용 (라벨 + 4 지표)
        add("")
        add("  표 (복붙용)")
        add(f"  {'Method':<30}{'AvgSR':>14}{'FWT':>14}{'NBT':>14}{'AUC':>14}")
        for mode in ORDER:
            a = agg.get(mode)
            if not a:
                continue
            add(f"  {LABEL[mode]:<30}{pm(a['AvgSR']):>14}{pm(a['FWT']):>14}"
                f"{pm(a['NBT'], True):>14}{pm(a['AUC']):>14}")
        # 사다리 차이 (인접 팔, 평균 차)
        means = {m: (sum(agg[m]["AvgSR"]) / len(agg[m]["AvgSR"])) for m in ORDER if agg.get(m)}
        steps = [(ORDER[i], ORDER[i + 1]) for i in range(len(ORDER) - 1)
                 if ORDER[i] in means and ORDER[i + 1] in means]
        if steps:
            add("")
            add("  사다리 (AvgSR 평균 차)")
            for a_, b_ in steps:
                add(f"    {a_:<12}→ {b_:<12}{means[b_] - means[a_]:+.1f}")
    add("")
    add("=" * 100)
    add("SR 행렬 (완주/진행 중 모두)")
    add("=" * 100)
    for r in sorted(runs, key=lambda x: (x["suite"], ORDER.index(x["mode"]), x["seed"])):
        R, la, src = read_R(r["dir"])
        if not R:
            continue
        idx = [t for t in range(len(la)) if la[t] > 0]
        if not idx:
            continue
        add("")
        add(f"{SUITE_NAME[r['suite']]}  {r['mode']}  seed{r['seed']}   ({r['dir'].name})   출처 {src}")
        add("stage " + "".join(f"{'t'+str(t):>6}" for t in idx))
        for k in range(len(R)):
            row = "".join(f"{'' if (t >= len(R[k]) or R[k][t] is None) else int(round(R[k][t])):>6}"
                          for t in idx)
            add(f"{k:<6}{row}")
    txt = "\n".join(L) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
