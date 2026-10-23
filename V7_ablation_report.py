#!/usr/bin/env python
"""V7 dea β 애블레이션 (시드 3개) — results/V7_ablation_study_seed3.txt

    python V7_ablation_report.py
    python V7_ablation_report.py --betas "0.5 2.0 5.0 10.0 20.0" --seeds "7 42 1000"
    python V7_ablation_report.py --no-write

results/clare_v7_dea_b<β>_goal_seed<N>/sr_matrix.csv 를 읽는다. 폴더가 없거나
롤아웃 전이면 학습 진행률(l2cb.jsonl 의 task·step)을 대신 보여준다.

내는 것
  1. β x 시드 표 (AvgSR / FWT / NBT / AUC)
  2. β 별 시드 평균 ± 표본표준편차(ddof=1) — **완주한 시드만**
  3. 런별 SR 행렬

★ 집계는 완주한 런만 쓴다. 진행 중인 런의 AvgSR 은 마지막으로 다 채워진 행
  기준이라 최종값이 아니고, 그걸 다른 시드의 최종값과 평균하면 행이 섞인다.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

from clare_v4_report import metrics, read_matrix

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"
OUT = RESULTS / "V7_ablation_study_seed3.txt"


def tag_of(beta: float, seed: int) -> str:
    """표준 태그. 같은 β 를 접미사 없이 돌린 런(기본 실행)도 받아준다."""
    cands = [f"clare_v7_dea_b{beta:g}_goal_seed{seed}",
             f"clare_v7_dea_b{beta:.1f}_goal_seed{seed}"]
    if abs(beta - 2.0) < 1e-9:
        cands.append(f"clare_v7_dea_goal_seed{seed}")
    for c in cands:
        if (RESULTS / c / "sr_matrix.csv").exists():
            return c
    for c in cands:
        if (RESULTS / c).is_dir():
            return c
    return cands[0]


def cells(d: Path) -> int:
    p = d / "sr_matrix.csv"
    if not p.exists():
        return 0
    b = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return sum(1 for l in b[1:] for c in l.split(",")[1:] if c.strip())


def train_progress(d: Path):
    """롤아웃 전 런의 학습 진행률. l2cb.jsonl 마지막 줄 기준."""
    jl = d / "l2cb.jsonl"
    if not jl.exists():
        return None
    try:
        cfg = json.loads((d / "config.json").read_text())
        nt, ns = int(cfg.get("num_tasks") or 10), int(cfg.get("steps_per_task") or 10000)
    except Exception:
        nt, ns = 10, 10000
    last = None
    try:
        with jl.open() as fh:
            for line in fh:
                if line.strip():
                    last = line.strip()
    except Exception:
        return None
    if not last:
        return None
    try:
        r = json.loads(last)
        t, s = int(r.get("task", 0)), int(r.get("step", 0))
    except Exception:
        return None
    return dict(task=t, n_task=nt, step=s, n_step=ns,
                pct=100.0 * (t * ns + s) / max(1, nt * ns),
                age_h=(time.time() - jl.stat().st_mtime) / 3600.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", default="0.5 2.0 5.0 10.0 20.0")
    ap.add_argument("--seeds", default="7 42 1000")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    betas = [float(x) for x in a.betas.split()]
    seeds = [int(x) for x in a.seeds.split()]
    NEED = 55

    L: list[str] = []
    add = L.append
    bar = "=" * 96
    add(bar)
    add("V7 dea (액션 공간) β 애블레이션 — libero_goal, 시드 3개")
    add(bar)
    add("")
    add("프로토콜  10,000 step/task · 50 demo · 10 task(55칸) · 칸당 50 rollout")
    add("          batch 32 · K=96 · λ_lvl=3.0 · eval seed 7 · --dir_space act")
    add(f"β         {', '.join(format(b, 'g') for b in betas)}")
    add(f"시드      {', '.join(map(str, seeds))}")
    add("")
    add("지표  FWT↑ 배운 직후 / NBT↓ 이후 잃은 양(+면 망각) / AUC↑ 전 구간 평균")
    add("      AvgSR = 마지막으로 다 채워진 행의 평균. 진행 중이면 * 를 붙인다.")
    add("")

    data: dict[tuple[float, int], dict] = {}
    add("-" * 96)
    add("β x 시드")
    add("-" * 96)
    add(f"  {'β':>6}{'seed':>7}{'AvgSR':>9}{'FWT↑':>9}{'NBT↓':>9}{'AUC↑':>9}   상태")
    for b in betas:
        for s in seeds:
            tag = tag_of(b, s)
            d = RESULTS / tag
            R, nt = read_matrix(d)
            if not R:
                pr = train_progress(d)
                if pr is None:
                    note = "미시작" if not d.is_dir() else "롤아웃 전"
                else:
                    stale = "" if pr["age_h"] < 0.5 else f" ⚠{pr['age_h']:.1f}h"
                    note = (f"학습 중 t{pr['task']}/{pr['n_task'] - 1} "
                            f"{pr['step']}/{pr['n_step']} ({pr['pct']:.0f}%){stale}")
                add(f"  {b:>6g}{s:>7}" + f"{'-':>9}" * 4 + f"   {note}")
                continue
            m = metrics(R, nt)
            n = cells(d)
            done = n >= NEED
            data[(b, s)] = dict(m=m, done=done, n=n, tag=tag)
            avg = f"{m['AvgSR']:9.1f}" if done else f"{m['AvgSR']:8.1f}*"
            add(f"  {b:>6g}{s:>7}{avg}{m['FWT']:9.1f}{m['NBT']:+9.1f}{m['AUC']:9.1f}"
                f"   {'완주' if done else f'{n}/{NEED}칸'}")
        add("")
    add("  * 진행 중인 런의 AvgSR 은 마지막 완성 행 기준이라 최종값이 아니다")
    add("")

    # ── β 별 시드 집계 ──────────────────────────────────────────────────
    add("=" * 96)
    add("β 별 시드 평균 ± 표준편차  (완주한 시드만. ± 는 시드 간 표본표준편차 ddof=1)")
    add("=" * 96)
    add(f"  {'β':>6}{'n':>3}{'시드':>14}{'AvgSR':>15}{'FWT↑':>15}{'NBT↓':>15}{'AUC↑':>15}")
    for b in betas:
        g = [(s, data[(b, s)]) for s in seeds
             if (b, s) in data and data[(b, s)]["done"]]
        if not g:
            add(f"  {b:>6g}{0:>3}{'-':>14}" + f"{'-':>15}" * 4)
            continue
        cols = []
        for key in ("AvgSR", "FWT", "NBT", "AUC"):
            vals = [r["m"][key] for _, r in g if r["m"][key] is not None]
            mu = st.mean(vals)
            sign = "+" if key == "NBT" and mu >= 0 else ""
            txt = (f"{sign}{mu:.1f} ± {st.stdev(vals):.1f}" if len(vals) > 1
                   else f"{sign}{mu:.1f}")
            cols.append(txt.rjust(15))
        add(f"  {b:>6g}{len(g):>3}{','.join(str(s) for s, _ in g):>14}" + "".join(cols))
    add("")
    nd = sum(1 for v in data.values() if v["done"])
    add(f"진행  완주 {nd} / 계획 {len(betas) * len(seeds)} 런")
    add("")
    add("읽을 때 주의")
    add("  · 칸당 50 rollout 이라 개별 칸의 이항 표준오차가 최대 ±7%p 다.")
    add("    AvgSR 은 10 칸 평균이라 ±2%p 안팎 — 그보다 작은 차이는 효과로 읽지 마라.")
    add("  · n=2 의 ± 는 분포 추정이 아니라 두 값 간격을 √2 로 나눈 값에 가깝다.")
    add("  · 액션 공간의 β 보정값은 2r₀²≈1.35 다(L2_V7.py 주석). β 20 은 그 15 배라")
    add("    방향 항이 크기 항을 압도하는 구간을 보는 셈이다.")
    add("")

    # ── SR 행렬 ─────────────────────────────────────────────────────────
    add("=" * 96)
    add("SR 행렬   행 = 스테이지 k 학습 직후, 열 = 평가 태스크 t")
    add("=" * 96)
    add("")
    for b in betas:
        for s in seeds:
            tag = tag_of(b, s)
            d = RESULTS / tag
            R, nt = read_matrix(d)
            add("-" * 96)
            if not R:
                add(f"β{b:g} seed{s}   ({tag})   — sr_matrix.csv 없음")
                add("-" * 96)
                add("")
                continue
            n = cells(d)
            done = n >= NEED
            add(f"β{b:g} seed{s}   ({tag})   " + ("완주" if done else f"진행 중 {n}/{NEED}칸"))
            add("-" * 96)
            add(f"{'stage':<8}" + "".join(f"{'t' + str(t):>6}" for t in range(nt)))
            for k in range(len(R)):
                add(f"{k:<8}" + "".join(
                    (f"{R[k][t]:6.0f}" if R[k][t] is not None else f"{'':>6}")
                    for t in range(nt)))
            m = metrics(R, nt)
            f1 = lambda v: "-" if v is None else f"{v:.1f}"      # noqa: E731
            fs = lambda v: "-" if v is None else f"{v:+.1f}"     # noqa: E731
            add(f"  지표  AvgSR {f1(m['AvgSR'])}"
                + (f" (행{m['row']})" if m["row"] is not None and not done else "")
                + f"   FWT↑ {f1(m['FWT'])}   NBT↓ {fs(m['NBT'])}   AUC↑ {f1(m['AUC'])}")
            add("")

    out = "\n".join(L)
    print(out)
    if not a.no_write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(out + "\n", encoding="utf-8")
        print(f"\n기록 -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
