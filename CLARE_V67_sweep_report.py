#!/usr/bin/env python
"""V6/V7 β 스윕 리포트 — results/ 를 **자동 탐색**해 시드 집계까지 낸다.

    python CLARE_V67_sweep_report.py                  # 표준 출력 + 파일 기록
    python CLARE_V67_sweep_report.py --form dea       # form 지정 (기본 dea)
    python CLARE_V67_sweep_report.py --suite goal
    python CLARE_V67_sweep_report.py --no-write       # 파일 안 쓰고 화면만

왜 따로 만드는가
    CLARE_V6_report.py 는 런 태그를 하드코딩한 목록으로 들고 있고 전부 _seed7
    이다. β 스윕(seed 42/1000)은 그 목록에 없어 리포트에 **안 나타난다**.
    여기서는 results/clare_v{6,7}_<form>_b<β>_goal_seed<N>/ 를 glob 으로 긁어
    새 런이 생기면 자동으로 들어오게 한다.

읽는 것   results/clare_v[67]_<form>_b*_<suite>_seed*/sr_matrix.csv
쓰는 것   results/CLARE_V67_beta_sweep_SR.txt

지표는 clare_v4_report 의 것을 그대로 쓴다 — LOTUS 표와 같은 정의다.
    FWT = 막 배운 직후,  NBT = 그 뒤 잃은 양(+면 망각),  AUC = 전 구간 평균
    AvgSR = 마지막으로 **다 채워진** 행의 평균 (진행 중이면 행 번호를 적는다)

★ 시드 집계는 **완주한 런만** 쓴다. 진행 중인 런의 AvgSR 은 마지막 완성 행
  기준이라 최종값이 아니고, 그걸 다른 시드의 최종값과 평균하면 행이 섞인다.
  ± 는 시드 간 표본표준편차(ddof=1)다. n=2 면 분포 추정이 아니라 두 값 간격을
  √2 로 나눈 값에 가깝다는 걸 염두에 두고 읽어라.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import time
from pathlib import Path

from clare_v4_report import metrics, read_matrix

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"
SPACE = {"v6": "vel", "v7": "act"}          # 분해 공간. L2_V7 이 act 로 고정한다.


def discover(form: str, suite: str):
    """results/ 를 긁어 (ver, beta, seed, dir) 목록을 돌려준다."""
    pat = re.compile(
        rf"^clare_(v6|v7)_{re.escape(form)}_b([0-9.]+)_{re.escape(suite)}_seed(\d+)$")
    out = []
    if not RESULTS.is_dir():
        return out
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if m:
            out.append((m.group(1), float(m.group(2)), int(m.group(3)), d))
    # ver -> beta -> seed 순
    out.sort(key=lambda r: (r[0], r[1], r[2]))
    return out


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

def train_progress(d):
    """sr_matrix.csv 가 없는 런의 학습 진행률. l2cb.jsonl 의 (task, step) 기준."""
    jl = d / "l2cb.jsonl"
    if not jl.exists():
        return None
    try:
        cfg = json.loads((d / "config.json").read_text())
        n_task = int(cfg.get("num_tasks") or 10)
        n_step = int(cfg.get("steps_per_task") or 10000)
    except Exception:
        n_task, n_step = 10, 10000
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
        t, st_ = int(r.get("task", 0)), int(r.get("step", 0))
    except Exception:
        return None
    return dict(task=t, n_task=n_task, step=st_, n_step=n_step,
                pct=100.0 * (t * n_step + st_) / max(1, n_task * n_step),
                age_h=(time.time() - jl.stat().st_mtime) / 3600.0)


def load(rows):
    """각 런의 지표와 완주 여부를 붙인다."""
    recs = []
    for ver, beta, seed, d in rows:
        R, n = read_matrix(d)
        if not R:
            recs.append(dict(ver=ver, beta=beta, seed=seed, dir=d, m=None,
                             filled=0, total=0, done=False))
            continue
        m = metrics(R, n)
        filled = sum(1 for r in R for v in r if v is not None)
        total = n * (n + 1) // 2
        recs.append(dict(ver=ver, beta=beta, seed=seed, dir=d, m=m,
                         filled=filled, total=total, done=filled >= total))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default="dea")
    ap.add_argument("--suite", default="goal")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()

    recs = load(discover(a.form, a.suite))
    L: list[str] = []
    add = L.append

    bar = "=" * 92
    add(bar)
    add(f"V6/V7 β 스윕 — form={a.form}, suite=libero_{a.suite}  (자동 탐색)")
    add(bar)
    if not recs:
        add("")
        add(f"  results/clare_v[67]_{a.form}_b*_{a.suite}_seed*/ 에 해당하는 런이 없다.")
        add("  스윕을 아직 안 돌렸거나 --form / --suite 가 다르다.")
        print("\n".join(L))
        return

    add("")
    add("지표  FWT↑ 배운 직후 / NBT↓ 이후 잃은 양(+면 망각) / AUC↑ 전 구간 평균")
    add("      AvgSR = 마지막으로 다 채워진 행의 평균. 진행 중이면 (행N) 을 적는다.")
    add("      V6 = 속도 공간 분해, V7 = 액션 공간 분해 (L2_V7 이 --dir_space act 고정)")
    add("")

    # ── 런별 ────────────────────────────────────────────────────────────
    add("-" * 92)
    add("런별")
    add("-" * 92)
    add(f"  {'런':<34}{'공간':>5}{'AvgSR':>9}{'FWT↑':>9}{'NBT↓':>9}{'AUC↑':>9}   상태")
    f1 = lambda v: f"{'-':>9}" if v is None else f"{v:9.1f}"       # noqa: E731
    fs = lambda v: f"{'-':>9}" if v is None else f"{v:+9.1f}"      # noqa: E731
    for r in recs:
        tag = f"{r['ver']} β{r['beta']:g} seed{r['seed']}"
        if r["m"] is None:
            pr = train_progress(r["dir"])
            if pr is None:
                add(f"  {tag:<34}{SPACE[r['ver']]:>5}" + f1(None) * 4 + "   미시작")
            else:
                stale = "" if pr["age_h"] < 0.5 else f"  ⚠{pr['age_h']:.1f}h 정지?"
                add(f"  {tag:<34}{SPACE[r['ver']]:>5}" + f1(None) * 4 +
                    f"   학습 중 t{pr['task']}/{pr['n_task'] - 1}"
                    f" step {pr['step']}/{pr['n_step']} ({pr['pct']:.0f}%){stale}")
            continue
        m = r["m"]
        state = "완주" if r["done"] else f"진행 중 {r['filled']}/{r['total']}칸"
        avg = f1(m["AvgSR"])
        if m["row"] is not None and not r["done"]:
            avg = f"{m['AvgSR']:8.1f}*"
        add(f"  {tag:<34}{SPACE[r['ver']]:>5}{avg}" + f1(m["FWT"])
            + fs(m["NBT"]) + f1(m["AUC"]) + f"   {state}")
    add("  * 진행 중인 런의 AvgSR 은 마지막 완성 행 기준이라 최종값이 아니다")
    add("")

    # ── SR 행렬 ────────────────────────────────────────────────────────
    add("=" * 92)
    add("SR 행렬   행 = 스테이지 k 학습 직후, 열 = 평가 태스크 t")
    add("=" * 92)
    add("")
    for r in recs:
        sr_matrix_block(r["dir"].name,
                        f"{r['ver']} β{r['beta']:g} seed{r['seed']}   ({r['dir'].name})", L)

    # ── 시드 집계 (완주한 런만) ─────────────────────────────────────────
    add("-" * 92)
    add("시드 평균 ± 표준편차  (완주한 런만. ± 는 시드 간 표본표준편차 ddof=1)")
    add("-" * 92)
    groups: dict[tuple[str, float], list] = {}
    for r in recs:
        if r["done"] and r["m"] is not None:
            groups.setdefault((r["ver"], r["beta"]), []).append(r)
    if not groups:
        add("  아직 완주한 런이 없다 — 55/55 칸을 채운 런이 생기면 여기에 나온다.")
    else:
        add(f"  {'팔':<16}{'공간':>5}{'n':>3}{'시드':>14}"
            f"{'AvgSR':>15}{'FWT↑':>15}{'NBT↓':>15}{'AUC↑':>15}")
        for (ver, beta) in sorted(groups):
            g = groups[(ver, beta)]
            seeds = ",".join(str(x["seed"]) for x in sorted(g, key=lambda y: y["seed"]))
            cells = []
            for key in ("AvgSR", "FWT", "NBT", "AUC"):
                vals = [x["m"][key] for x in g if x["m"][key] is not None]
                if not vals:
                    cells.append(f"{'-':>15}")
                    continue
                mu = st.mean(vals)
                sign = "+" if key == "NBT" and mu >= 0 else ""
                if len(vals) > 1:
                    txt = f"{sign}{mu:.1f} ± {st.stdev(vals):.1f}"
                else:
                    txt = f"{sign}{mu:.1f}"          # n=1 은 ±0.0 을 쓰지 않는다
                cells.append(txt.rjust(15))
            add(f"  {ver + ' β' + format(beta, 'g'):<16}{SPACE[ver]:>5}{len(g):>3}"
                f"{seeds:>14}" + "".join(cells))
    add("")

    # ── 진행 현황 ───────────────────────────────────────────────────────
    done = sum(1 for r in recs if r["done"])
    add("-" * 92)
    add(f"진행  완주 {done} / 발견 {len(recs)} 런")
    miss = [r for r in recs if not r["done"]]
    if miss:
        add("  미완주: " + ", ".join(
            f"{r['ver']}β{r['beta']:g}s{r['seed']}({r['filled']}/{r['total']})"
            for r in miss))
    add("")
    add("읽을 때 주의")
    add("  · 칸당 50 rollout 이라 개별 칸의 이항 표준오차가 최대 ±7%p 다.")
    add("  · n=2 의 ± 는 분포 추정이 아니라 두 값 간격을 √2 로 나눈 값에 가깝다.")
    add("  · β 의 적정 스케일은 공간마다 다르다 — 보정값 2r₀² 가 vel 12.4, act 1.35")
    add("    이므로 같은 β 를 두 공간에 걸면 V6 쪽이 전 구간 과소일 수 있다.")

    out = "\n".join(L)
    print(out)
    if not a.no_write:
        p = RESULTS / "CLARE_V67_beta_sweep_SR.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(out + "\n", encoding="utf-8")
        print(f"\n기록 -> {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
