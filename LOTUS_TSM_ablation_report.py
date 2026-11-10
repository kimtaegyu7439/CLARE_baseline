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
PAT = re.compile(r"^lotus_tsm_(?P<mode>indep_gauss|time_bin|state_mean|state_gauss|state_map|tether)_(?P<suite>goal|object|50)_seed(?P<seed>\d+)$")
ORDER = ["indep_gauss", "time_bin", "state_mean", "state_gauss", "state_map", "tether"]
LABEL = {"indep_gauss": "Indep. Gaussian (state, obs)", "time_bin": "Time-bin Gaussian",
         "state_mean": "State cluster + mean obs", "state_gauss": "State cluster + Gaussian obs",
         "state_map": "State cluster + local map A", "tether": "TETHER (= state_map + MAR)"}
KEYS = ("AvgSR", "FWT", "NBT", "AUC")
LL_OF = {"goal": (6, 7, 8, 9), "object": (6, 7, 8, 9), "50": tuple(range(25, 50))}
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


def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def _sd(v):
    v = [x for x in v if x is not None]
    if len(v) < 2:
        return None
    m = _mean(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def _g(x, w=8, p=1, sign=False):
    if x is None:
        return "-".rjust(w)
    return f"{x:>+{w}.{p}f}" if sign else f"{x:>{w}.{p}f}"


def decompose(per):
    """{학습시드: [노이즈 평가값..]} -> (σ_noise, σ_train, SE, 팔 평균, S, n̄)  (LOTUS_MLR_noise_report 와 같은 식)
    σ_noise 학습 시드 안 pooled sd, σ_train = sqrt(max(0, sd(시드평균)² − σ_noise²/n̄)),
    SE = sqrt(σ_train²/S + σ_noise²/(S·n̄))."""
    per = {s: [x for x in v if x is not None] for s, v in per.items()}
    per = {s: v for s, v in per.items() if v}
    if not per:
        return (None,) * 4 + (0, 0.0)
    num = sum(sum((x - _mean(v)) ** 2 for x in v) for v in per.values())
    den = sum(len(v) - 1 for v in per.values())
    s_noise = (num / den) ** 0.5 if den > 0 else None
    means = [_mean(v) for v in per.values()]
    nbar = _mean([len(v) for v in per.values()])
    S = len(means)
    s_btw = _sd(means)
    if s_btw is None or s_noise is None:
        s_train = None
        se = s_noise / (nbar * S) ** 0.5 if s_noise is not None else None
    else:
        s_train = max(0.0, s_btw ** 2 - s_noise ** 2 / nbar) ** 0.5
        se = (s_train ** 2 / S + s_noise ** 2 / (S * nbar)) ** 0.5
    return s_noise, s_train, se, _mean(means), S, nbar


def noise_rows(d: Path, suite: str):
    """<run>/eval_seeds/seed<s>.json (Lotus_eval_seeds.py) -> [dict(noise, AvgSR, FWT, NBT, AUC, t..)], 노이즈 시드 순."""
    out = []
    ed = d / "eval_seeds"
    if not ed.is_dir():
        return out
    for jf in sorted(ed.glob("seed*.json")):
        try:
            j = json.loads(jf.read_text())
        except Exception:
            continue
        if j.get("suffix"):                     # _r1 (재현 확인용) 은 뺀다
            continue
        mt = j.get("metrics") or {}
        if mt.get("AvgSR") is None:
            continue
        r = {"noise": int(j["seed"]), **{k: mt.get(k) for k in KEYS}}
        last = j["R"][-1]
        for t in LL_OF[suite]:
            r[f"t{t}"] = last[t] if t < len(last) else None
        out.append(r)
    return sorted(out, key=lambda r: r["noise"])


def noise_section(add, runs, suite):
    """평가-노이즈 시드 재평가 절. eval_seeds json 이 하나도 없으면 아무것도 쓰지 않는다."""
    per_run = {}                                # (mode, seed) -> rows
    for r in runs:
        if r["suite"] != suite:
            continue
        rows = noise_rows(r["dir"], suite)
        if rows:
            per_run[(r["mode"], r["seed"])] = rows
    if not per_run:
        return
    add("")
    add("  ── 평가-노이즈 시드 재평가 (Lotus_eval_seeds.py: 같은 체크포인트, 노이즈 시드 s 만 바꿔 다시 굴림) ──")
    add("     s 가 정하는 것: init state 묶음 (s*20)%50 부터 20개 (s mod 5 로 5 종류) + 정책 flow 잡음 Generator(s*1000+t)")
    add("     σ_noise 평가 한 번의 흔들림(학습 시드 안, pooled)  σ_train 학습 시드를 바꿔 남는 차이(σ_noise 를 뺀 뒤, 음수면 0)")
    add("     SE 팔 평균의 표준오차 = √(σ_train²/S + σ_noise²/(S·n̄)).  두 팔 격차가 유의하려면 대략 2·√2·SE 보다 커야 한다.")
    add("")
    add(f"  {'팔':<12}{'지표':>6}{'S':>3}{'n̄':>6}{'평균':>8}{'σ_noise':>9}{'σ_train':>9}{'SE':>7}{'필요격차':>9}")
    for mode in ORDER:
        seeds = sorted(s for (m, s) in per_run if m == mode)
        if not seeds:
            continue
        for k in KEYS:
            per = {s: [r[k] for r in per_run[(mode, s)]] for s in seeds}
            s_noise, s_train, se, mu, S, nbar = decompose(per)
            add(f"  {mode:<12}{k:>6}{S:>3}{nbar:>6.1f}{_g(mu, 8, 1, k == 'NBT')}{_g(s_noise, 9)}{_g(s_train, 9)}"
                f"{_g(se, 7, 2)}{_g(None if se is None else 2 * 2 ** 0.5 * se, 9)}")
    add("")
    add(f"  {'팔':<12}{'seed':>6}{'n':>4}{'AvgSR':>14}{'FWT↑':>14}{'NBT↓':>14}{'AUC↑':>14}   (노이즈 시드 평균 ± sd)")
    for mode in ORDER:
        for (m, s), rows in sorted(per_run.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
            if m != mode:
                continue
            cells = []
            for k in KEYS:
                mu, sdv = _mean([r[k] for r in rows]), _sd([r[k] for r in rows])
                cells.append((f"{mu:+.1f}" if k == "NBT" else f"{mu:.1f}") + (f" ± {sdv:.1f}" if sdv is not None else ""))
            add(f"  {mode:<12}{s:>6}{len(rows):>4}" + "".join(f"{c:>14}" for c in cells))
    ll = LL_OF[suite]
    add("")
    add(f"  생값 (한 줄 = 런 × 노이즈 시드)")
    tcols = "".join(f"{'t'+str(t):>5}" for t in ll) if len(ll) <= 8 else ""
    add(f"  {'팔':<12}{'seed':>6}{'noise':>6}{'AvgSR':>7}{'FWT':>7}{'NBT':>7}{'AUC':>7}{tcols}")
    for (m, s), rows in sorted(per_run.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
        for r in rows:
            tv = "".join(f"{'' if r.get(f't{t}') is None else int(round(r[f't{t}'])):>5}" for t in ll) if tcols else ""
            add(f"  {m:<12}{s:>6}{r['noise']:>6}{_g(r['AvgSR'], 7)}{_g(r['FWT'], 7)}{_g(r['NBT'], 7, 1, True)}{_g(r['AUC'], 7)}{tv}")


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
    add("  indep_gauss   없음(1셀)      N(μ_s,σ_s)      N(m, σ_o)   서로 독립   sqerr(MSE)")
    add("  time_bin      진행률 K구간   N(μ_s,σ_s)      N(m_k, σ_o,k)           sqerr(MSE)")
    add("  state_mean    state k-means  N(μ_s,σ_s)      m_k                     sqerr(MSE)")
    add("  state_gauss   state k-means  N(μ_s,σ_s)      N(m_k, σ_o,k 원시)      sqerr(MSE)")
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
        noise_section(add, runs, suite)
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
