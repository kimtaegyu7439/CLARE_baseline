#!/usr/bin/env python
"""TETHER(V7 dea) 배치 24 vs 32 — results/TETHER_forward_24sample.txt

    python TETHER_forward_24sample_report.py
    python TETHER_forward_24sample_report.py --beta 2.0 --no-write

왜 보는가
    ER 은 배치 32 를 현재 24 + 버퍼 8 로 쓴다(L2_er_cl.py:181). 새 태스크를
    배우는 데 쓰는 표본이 24 개다. TETHER 는 32 개를 전부 새 태스크에 쓰므로,
    둘을 맞대면 "새 태스크 표본 수"가 다른 채로 비교하는 셈이 된다.
    그 축을 맞춘 대조군(batch 24)을 batch 32 와 나란히 놓는다.

★ 읽을 때 반드시 같이 볼 것
    배치를 줄이면 **앵커 표본도 같이 줄어든다**. 앵커는 과거 태스크당 n=B 개를
    뽑으므로(L2_V6.py:341 -> L2_V4_seq.py:1261 의 multinomial(pi, n)), B=24 면
    앵커도 24 개다. 앵커 몬테카를로 분산이 32/24 = 1.33 배(sd 1.15 배) 커진다.
    따라서 두 열의 차이는 "새 태스크 표본 24 vs 32" 단독 효과가 아니라
    "새 태스크 표본 + 앵커 표본"이 같이 줄어든 효과다.

지표는 clare_v4_report 의 것을 그대로 쓴다(LOTUS 표와 같은 정의).
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
NEED = 55


def tag_of(beta: float, seed: int, batch: int) -> str:
    """태그 철자가 한 가지가 아니다 — 실제로 있는 것을 고른다.

    · β 접미사 없는 기본 실행:  clare_v7_dea_goal_seed7        (β2.0)
    · 스윕이 만드는 것:          clare_v7_dea_b2.0_goal_seed42  (TAGSUF="_b${BETA}")
    · 사람이 2 로 적은 경우:     clare_v7_dea_b2_goal_seed42
    셋 다 후보에 넣고 sr_matrix.csv 가 있는 첫 번째를 쓴다. 하나도 없으면
    가장 그럴듯한 이름을 돌려줘 '미시작' 로 보이게 한다.
    """
    if batch != 32:
        return f"clare_v7_dea_b{batch}bs_goal_seed{seed}"
    cands = [f"clare_v7_dea_b{format(beta, '.1f')}_goal_seed{seed}",
             f"clare_v7_dea_b{format(beta, 'g')}_goal_seed{seed}"]
    if abs(beta - 2.0) < 1e-9:                      # 기본 실행은 접미사가 없다
        cands.insert(0, f"clare_v7_dea_goal_seed{seed}")
    for c in cands:
        if (RESULTS / c / "sr_matrix.csv").exists():
            return c
    return cands[0]


def train_progress(tag: str):
    """sr_matrix.csv 가 아직 없는 런의 **학습 진행률**. 롤아웃 전에도 보이게.

    l2cb.jsonl 마지막 줄의 (task, step) 과 config 의 (num_tasks, steps_per_task)
    로 전체 대비 몇 %인지 센다. 학습이 끝나고 롤아웃이 시작되면 sr_matrix.csv
    가 생기므로 그때부터는 칸 수로 바뀐다.
    """
    d = RESULTS / tag
    cfg_p, jl_p = d / "config.json", d / "l2cb.jsonl"
    if not jl_p.exists():
        return None
    try:
        cfg = json.loads(cfg_p.read_text())
        n_task = int(cfg.get("num_tasks") or 10)
        n_step = int(cfg.get("steps_per_task") or 10000)
    except Exception:
        n_task, n_step = 10, 10000
    last = None
    try:
        with jl_p.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last = line
    except Exception:
        return None
    if not last:
        return None
    try:
        r = json.loads(last)
        t, st_ = int(r.get("task", 0)), int(r.get("step", 0))
    except Exception:
        return None
    frac = (t * n_step + st_) / max(1, n_task * n_step)
    mt = jl_p.stat().st_mtime
    return dict(task=t, n_task=n_task, step=st_, n_step=n_step,
                pct=100.0 * frac, mtime=mt)


def sr_matrix_block(tag: str, title: str, L: list) -> None:
    """런 하나의 SR 행렬을 clare_v4_report.py 와 같은 양식으로 찍는다.

    행 = 스테이지 k 학습 직후, 열 = 평가 태스크 t. CLARE 순차 셋팅이라
    태스크 t 를 배운 스테이지가 t 이므로 아래 삼각형이다. 1 행(스테이지 0)은
    앵커가 걸릴 과거가 없어 어떤 방법을 써도 같아야 정상이다.
    """
    d = RESULTS / tag if "RESULTS" in globals() else RES / tag
    R, n = read_matrix(d)
    L.append("-" * 92)
    if not R:
        L.append(f"{title}   — sr_matrix.csv 없음 (롤아웃 전)")
        L.append("-" * 92)
        L.append("")
        return
    filled = sum(1 for r in R for v in r if v is not None)
    total = n * (n + 1) // 2
    done = filled >= total
    L.append(f"{title}   " + ("완료" if done else f"진행 중 {filled}/{total}칸"))
    L.append("-" * 92)
    L.append(f"{'stage':<8}" + "".join(f"{'t' + str(t):>6}" for t in range(n)))
    for k in range(len(R)):
        L.append(f"{k:<8}" + "".join(
            (f"{R[k][t]:6.0f}" if R[k][t] is not None else f"{'':>6}")
            for t in range(n)))
    m = metrics(R, n)
    f1 = lambda v: "-" if v is None else f"{v:.1f}"        # noqa: E731
    fs = lambda v: "-" if v is None else f"{v:+.1f}"       # noqa: E731
    L.append(f"  지표  AvgSR {f1(m['AvgSR'])}"
             + (f" (행{m['row']})" if m["row"] is not None and not done else "")
             + f"   FWT↑ {f1(m['FWT'])}   NBT↓ {fs(m['NBT'])}   AUC↑ {f1(m['AUC'])}")
    L.append("")

def load(tag: str):
    d = RESULTS / tag
    R, n = read_matrix(d)
    if not R:
        return None
    m = metrics(R, n)
    filled = sum(1 for r in R for v in r if v is not None)
    return dict(tag=tag, m=m, filled=filled, done=filled >= NEED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--seeds", default="7 42 1000")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split()]

    L: list[str] = []
    add = L.append
    bar = "=" * 94
    add(bar)
    add(f"TETHER (V7 dea, 액션 공간) — 배치 24 vs 32   β={a.beta:g}, libero_goal, 10,000 step/task")
    add(bar)
    add("")
    add("무엇을 보는가")
    add("  ER 은 배치 32 를 현재 24 + 버퍼 8 로 쓴다 — 새 태스크를 배우는 표본이 24 개다.")
    add("  TETHER 는 32 개를 전부 새 태스크에 쓴다. 그 축을 맞춘 대조군이 batch 24 다.")
    add("  step 수(10,000), demo 수(50), 태스크 수(10), 롤아웃(칸당 50), eval seed(7),")
    add("  K(96), λ_lvl(3.0)은 전부 동일하다. 바뀐 것은 배치뿐이다.")
    add("")
    add("★ 배치를 줄이면 앵커 표본도 같이 줄어든다")
    add("  앵커는 과거 태스크당 n=B 개를 뽑는다(L2_V6.py:341 -> multinomial(pi, n)).")
    add("  B=24 면 앵커도 24 개라 몬테카를로 분산이 32/24=1.33 배(sd 1.15 배) 커진다.")
    add("  두 열의 차이를 '새 태스크 표본 수' 단독 효과로 읽으면 안 된다.")
    add("")
    add("지표  FWT↑ 배운 직후 / NBT↓ 이후 잃은 양(+면 망각) / AUC↑ 전 구간 평균")
    add("      AvgSR = 마지막으로 다 채워진 행의 평균. 진행 중이면 * 를 붙인다.")
    add("")

    rows = {24: {}, 32: {}}
    add("-" * 94)
    add("시드별")
    add("-" * 94)
    add(f"  {'seed':>5}{'배치':>6}{'AvgSR':>9}{'FWT↑':>9}{'NBT↓':>9}{'AUC↑':>9}   상태   런")
    for seed in seeds:
        for batch in (24, 32):
            tag = tag_of(a.beta, seed, batch)
            r = load(tag)
            if r is None:
                pr = train_progress(tag)
                if pr is None:
                    add(f"  {seed:>5}{batch:>6}" + f"{'-':>9}" * 4 + f"   미시작   {tag}")
                else:
                    age = (time.time() - pr["mtime"]) / 60.0
                    stale = "" if age < 30 else f" (갱신 {age/60:.1f}h 전 — 멈췄을 수 있다)"
                    add(f"  {seed:>5}{batch:>6}" + f"{'-':>9}" * 4 +
                        f"   학습 중 task {pr['task']}/{pr['n_task'] - 1}"
                        f" step {pr['step']}/{pr['n_step']} ({pr['pct']:.0f}%){stale}")
                    add(f"  {'':>11}{'':>36}{tag}")
                continue
            rows[batch][seed] = r
            m = r["m"]
            avg = f"{m['AvgSR']:9.1f}" if r["done"] else f"{m['AvgSR']:8.1f}*"
            state = "완주" if r["done"] else f"{r['filled']}/{NEED}칸"
            add(f"  {seed:>5}{batch:>6}{avg}{m['FWT']:9.1f}{m['NBT']:+9.1f}"
                f"{m['AUC']:9.1f}   {state}   {tag}")
        add("")
    add("  * 진행 중인 런의 AvgSR 은 마지막 완성 행 기준이라 최종값이 아니다")
    add("")

    # ── SR 행렬 ────────────────────────────────────────────────────────
    add("=" * 94)
    add("SR 행렬   행 = 스테이지 k 학습 직후, 열 = 평가 태스크 t")
    add("=" * 94)
    add("")
    for seed in seeds:
        for batch in (24, 32):
            t = tag_of(a.beta, seed, batch)
            sr_matrix_block(t, f"seed {seed} / batch {batch}   ({t})", L)

    # ── 시드 집계 (완주한 런만) ─────────────────────────────────────────
    add("-" * 94)
    add("시드 평균 ± 표준편차  (완주한 런만. ± 는 시드 간 표본표준편차 ddof=1)")
    add("-" * 94)
    agg = {}
    add(f"  {'배치':<6}{'n':>3}{'시드':>12}{'AvgSR':>15}{'FWT↑':>15}{'NBT↓':>15}{'AUC↑':>15}")
    for batch in (24, 32):
        g = [r for r in rows[batch].values() if r["done"]]
        if not g:
            add(f"  {batch:<6}{0:>3}{'-':>12}" + f"{'-':>15}" * 4)
            continue
        sd_list = sorted(s for s, r in rows[batch].items() if r["done"])
        cells, store = [], {}
        for key in ("AvgSR", "FWT", "NBT", "AUC"):
            vals = [r["m"][key] for r in g if r["m"][key] is not None]
            mu = st.mean(vals)
            store[key] = mu
            sign = "+" if key == "NBT" and mu >= 0 else ""
            txt = f"{sign}{mu:.1f} ± {st.stdev(vals):.1f}" if len(vals) > 1 else f"{sign}{mu:.1f}"
            cells.append(txt.rjust(15))
        agg[batch] = store
        add(f"  {batch:<6}{len(g):>3}{','.join(map(str, sd_list)):>12}" + "".join(cells))
    add("")

    # ── 차이 ────────────────────────────────────────────────────────────
    if 24 in agg and 32 in agg:
        add("-" * 94)
        add("차이 (24 − 32)")
        add("-" * 94)
        n24 = len([r for r in rows[24].values() if r["done"]])
        n32 = len([r for r in rows[32].values() if r["done"]])
        add(f"  {'AvgSR':>9}{'FWT↑':>9}{'NBT↓':>9}{'AUC↑':>9}")
        d = [agg[24][k] - agg[32][k] for k in ("AvgSR", "FWT", "NBT", "AUC")]
        add(f"  {d[0]:+9.1f}{d[1]:+9.1f}{d[2]:+9.1f}{d[3]:+9.1f}")
        add("")
        if n24 != n32:
            add(f"  ⚠ 완주 시드 수가 다르다 (24:{n24}, 32:{n32}) — 같은 시드끼리 맞춘 비교가 아니다.")
        add("  칸당 50 rollout 이라 개별 칸의 이항 표준오차가 최대 ±7%p 다.")
        add("  AvgSR 은 10 칸 평균이라 ±2%p 안팎. 그보다 작은 차이는 효과로 읽지 마라.")
    else:
        add("-" * 94)
        add("차이 — 양쪽 다 완주한 시드가 생기면 여기에 나온다.")
        add("-" * 94)

    out = "\n".join(L)
    print(out)
    if not a.no_write:
        p = RESULTS / "TETHER_forward_24sample.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(out + "\n", encoding="utf-8")
        print(f"\n기록 -> {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
