#!/usr/bin/env python
"""L2_V5 (액션 공간) LOTUS 결과를 한 파일로 — results/LOTUS_act_V5_SR.txt

    python LOTUS_act_V5_report.py
    python LOTUS_act_V5_report.py --cs "0.1 0.01" --seeds 7
    python LOTUS_act_V5_report.py --no-write

실시간 갱신:
    REPORT=LOTUS_act_V5_report.py INTERVAL=300 bash bash/report_loop.sh start

앵커  m 계수 1/N,  e 계수 1/(r_T² + c)/N   (액션 공간, 배치평균 정규화 없음)
      λ_lvl 은 앵커 손실 앞의 계수 — L = L_FM + λ_lvl·L_anchor

읽는 것 (우선순위)
    <런>/sr_rollout.json   롤아웃 워커가 **칸마다** 갱신하는 durable 기록
    <런>/metrics.json      학습+롤아웃 런이 끝에 쓰는 것
    <런>/train.log         아직 롤아웃 전이면 학습 진행률(stage/step)

★ 지표는 **lifelong 태스크만**으로 낸다. LOTUS 의 base 는 여러 태스크를 한꺼번에
  joint 로 20만 step 학습한 구간이고 모든 방법이 **같은 base 체크포인트**에서
  출발한다. base 칸은 방법을 구분하지 못하면서 평균만 올린다 — libero_50 은
  50 태스크 중 25 개가 base 라 AvgSR 이 10~16%p 부풀려진다. 그래서 학습도
  --skip_base_eval 로 base 를 애초에 재지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
OUT = RES / "LOTUS_act_V5_SR.txt"
SUITES = ("goal", "object", "50")
SPACE = os.environ.get("SPACE", "act")


def run_dir(c: float, k: int, suite: str, seed: int) -> Path:
    """c 표기가 여러 가지다 (런처는 "0.1"/"0.01", 파이썬 :g 도 같다). 있는 쪽을 쓴다."""
    for cs in (f"{c:g}", f"{c}", f"{c:.2f}"):
        d = RES / f"lotus_v5{SPACE}_c{cs}_k{k}_{suite}_seed{seed}"
        if d.is_dir():
            return d
    return RES / f"lotus_v5{SPACE}_c{c:g}_k{k}_{suite}_seed{seed}"


def read_R(d: Path):
    """(R, learned_at, 출처) — 없으면 (None, None, None)."""
    p = d / "sr_rollout.json"
    if p.exists():
        try:
            R = json.loads(p.read_text())["R"]
            la = None
            for f in ("metrics.json", "metrics_train.json"):
                q = d / f
                if q.exists():
                    la = json.loads(q.read_text()).get("learned_at")
                    if la:
                        break
            if R:
                return R, la, "sr_rollout.json"
        except Exception:
            pass
    for f in ("metrics.json", "metrics_train.json"):
        q = d / f
        if q.exists():
            try:
                j = json.loads(q.read_text())
                if j.get("R"):
                    return j["R"], j.get("learned_at"), f
            except Exception:
                pass
    return None, None, None


def lifelong(R, la):
    """base(learned_at == 0)를 뺀 지표. 칸이 하나도 없으면 None.

    완성도를 요구하지 않는다 — 진행 중인 런도 보여야 하므로 **채워진 칸으로만**
    계산하고, 몇 칸인지 함께 돌려준다. 그래서 중간 값은 최종값이 아니다.
    """
    if not R or not la:
        return None
    idx = [t for t in range(len(la)) if la[t] > 0]
    need = sum(1 for k in range(len(R)) for t in idx if la[t] <= k)
    have = sum(1 for k in range(len(R)) for t in idx
               if t < len(R[k]) and R[k][t] is not None)
    if have == 0:
        return None
    fwt, nbt, auc = [], [], []
    for t in idx:
        col = [R[k][t] for k in range(la[t], len(R))
               if t < len(R[k]) and R[k][t] is not None]
        if not col:
            continue
        fwt.append(col[0])
        auc.append(sum(col) / len(col))
        if len(col) > 1:
            nbt.append(col[0] - sum(col[1:]) / len(col[1:]))
    last = [R[-1][t] for t in idx if t < len(R[-1]) and R[-1][t] is not None]
    mean = lambda v: (sum(v) / len(v)) if v else None      # noqa: E731
    return {"AvgSR": mean(last), "FWT": mean(fwt), "NBT": mean(nbt),
            "AUC": mean(auc), "have": have, "need": need,
            "done": have >= need > 0, "n_task": len(idx)}


def progress(d: Path):
    """학습 진행률. train.log 의 마지막 step 줄 기준."""
    f = d / "train.log"
    if not f.exists():
        return None
    last = cb = None
    try:
        for line in f.read_text(errors="ignore").splitlines():
            if " step=" in line:
                last = line
            elif "코드북 task" in line:
                cb = line
    except Exception:
        return None
    age = (time.time() - f.stat().st_mtime) / 3600.0
    if last:
        m = re.search(r"stage(\d+) step=\s*(\d+)", last)
        if m:
            return dict(stage=int(m.group(1)), step=int(m.group(2)), age_h=age)
    if cb:
        return dict(stage=0, step=-1, age_h=age)     # 코드북 빌드 중
    return dict(stage=0, step=-2, age_h=age)         # 로딩 중


def fmt(v, sign=False):
    if v is None:
        return "-"
    return f"{v:+.1f}" if sign else f"{v:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs", default=os.environ.get("CS", "0.1 0.01"),
                    help="비교할 c 값 목록")
    ap.add_argument("--k", type=int, default=int(os.environ.get("LIFELONG_K", "25")))
    ap.add_argument("--seeds", default=os.environ.get("SEEDS", "7"))
    ap.add_argument("--lam", default=os.environ.get("LAM", "1.0"))
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    cs = [float(x) for x in a.cs.split()]
    seeds = [int(x) for x in a.seeds.split()]

    L: list[str] = []
    add = L.append
    W = 96
    add("=" * W)
    add(f"L2_V5 — 방향계수 1/(r_T² + c), {SPACE} 공간 · LOTUS 셋팅 lifelong")
    add("=" * W)
    add(f"갱신 {time.strftime('%F %T')}")
    add("")
    add("앵커  m 계수 1/N,  e 계수 1/(r_T² + c)/N")
    add("      ‖p_S − p_T‖² = (r_S−r_T)² + 2·r_S·r_T·(1−cos) = m + 2·r_S·r_T·e 에서")
    add("      방향 항의 계수만 teacher 크기의 역제곱으로 바꾼 것이다. MSE 의 계수")
    add("      2·r_S·r_T 는 빠른 구간일수록 커서 느린 구간(정밀 조작)의 방향을")
    add("      사실상 감독하지 못한다 — 그 의존을 뒤집는다.")
    add("      배치평균 정규화 없음 · w_clip 없음(coef ≤ 1/c 로 유계) · 1-pass")
    add("")
    add(f"설정  L2_V5.py --setting lotus --dir_space {SPACE}   c = "
        + ", ".join(f"{c:g}" for c in cs))
    add(f"      λ_lvl={a.lam} (앵커 손실 앞 계수, L = L_FM + λ_lvl·L_anchor)")
    add(f"      lifelong 코드북 K={a.k} (태스크당 고정), base K = k_per_demo x demo = 100")
    add("      lifelong stage 당 10,000 step · 태스크당 10 demo · batch 32")
    add(f"      평가 칸당 20 rollout · eval seed 7 · train seed {', '.join(map(str, seeds))}")
    add("")
    add("★ 지표는 lifelong 태스크만으로 낸다. base 는 모든 팔이 같은 체크포인트에서")
    add("  출발하므로 구분력이 없고 평균만 올린다 — 학습도 base 를 재지 않는다.")
    add("")
    add("지표  FWT↑ 배운 직후 / NBT↓ 이후 잃은 양(+면 망각) / AUC↑ 전 구간 평균")
    add("      AvgSR↑ 마지막 행 평균.  칸이 덜 찼으면 * (최종값이 아니다)")
    add("")

    add("-" * W)
    add("c x 스위트 x 시드")
    add("-" * W)
    add(f"  {'c':>7}{'suite':>9}{'seed':>6}{'AvgSR':>9}{'FWT↑':>9}{'NBT↓':>9}"
        f"{'AUC↑':>9}{'칸':>8}   상태")
    for c in cs:
        for suite in SUITES:
            for sd in seeds:
                d = run_dir(c, a.k, suite, sd)
                if not d.is_dir():
                    continue          # 안 돌린 조합은 표에 넣지 않는다
                R, la, src = read_R(d)
                m = lifelong(R, la)
                pr = progress(d)
                if pr:
                    if pr["step"] == -2:
                        note = "시작 중"
                    elif pr["step"] == -1:
                        note = "코드북 빌드 중"
                    else:
                        note = f"학습 stage{pr['stage']} {pr['step']}/10000"
                    if pr["age_h"] > 1.0:
                        note += f" ⚠{pr['age_h']:.1f}h 정지?"
                else:
                    note = "폴더만 있음"
                if m is None:
                    add(f"  {c:>7g}{suite:>9}{sd:>6}" + f"{'-':>9}" * 4
                        + f"{'-':>8}   {note}")
                    continue
                star = "" if m["done"] else "*"
                add("  {:>7g}{:>9}{:>6}{:>9}{:>9}{:>9}{:>9}{:>8}   {}".format(
                    c, suite, sd, fmt(m["AvgSR"]) + star, fmt(m["FWT"]),
                    fmt(m["NBT"], True), fmt(m["AUC"]),
                    "{}/{}".format(m["have"], m["need"]),
                    "완주" if m["done"] else note))
        add("")

    # ── c 비교 (같은 스위트·시드에서 c 끼리) ────────────────────────────────
    add("=" * W)
    add("c 비교 — 같은 스위트/시드에서만 맞댄다")
    add("=" * W)
    for suite in SUITES:
        rows = []
        for c in cs:
            for sd in seeds:
                d = run_dir(c, a.k, suite, sd)
                if not d.is_dir():
                    continue
                R, la, _ = read_R(d)
                m = lifelong(R, la)
                if m:
                    rows.append((c, sd, m))
        if not rows:
            continue
        add("")
        add(f"  libero_{suite}")
        add(f"    {'c':>7}{'seed':>6}{'AvgSR':>9}{'FWT↑':>9}{'NBT↓':>9}{'AUC↑':>9}   비고")
        for c, sd, m in rows:
            star = "" if m["done"] else "*"
            add("    {:>7g}{:>6}{:>9}{:>9}{:>9}{:>9}   {}".format(
                c, sd, fmt(m["AvgSR"]) + star, fmt(m["FWT"]),
                fmt(m["NBT"], True), fmt(m["AUC"]),
                "완주" if m["done"] else f"{m['have']}/{m['need']}칸"))
    add("")

    # ── SR 행렬 ─────────────────────────────────────────────────────────────
    add("=" * W)
    add("SR 행렬   행 = 스테이지(0 = base 직후), 열 = 스위트 태스크")
    add("         base 열은 비어 있다 — 재지 않는다(--skip_base_eval)")
    add("=" * W)
    add("")
    for c in cs:
        for suite in SUITES:
            for sd in seeds:
                d = run_dir(c, a.k, suite, sd)
                if not d.is_dir():
                    continue
                R, la, src = read_R(d)
                add("-" * W)
                if not R:
                    add(f"c={c:g} {suite} seed{sd}   ({d.name})   — SR 없음")
                    add("-" * W)
                    add("")
                    continue
                m = lifelong(R, la)
                tail = "   {}/{}칸".format(m["have"], m["need"]) if m else ""
                add(f"c={c:g} {suite} seed{sd}   ({d.name})   출처 {src}" + tail)
                add("-" * W)
                nt = max(len(r) for r in R)
                add(f"{'stage':<7}" + "".join(f"{'t'+str(t):>5}" for t in range(nt)))
                for k in range(len(R)):
                    add(f"{k:<7}" + "".join(
                        (f"{R[k][t]:5.0f}" if t < len(R[k]) and R[k][t] is not None
                         else f"{'':>5}") for t in range(nt)))
                if m:
                    add(f"  지표  AvgSR {fmt(m['AvgSR'])}   FWT↑ {fmt(m['FWT'])}   "
                        f"NBT↓ {fmt(m['NBT'], True)}   AUC↑ {fmt(m['AUC'])}"
                        + ("" if m["done"] else "   (진행 중)"))
                add("")

    add("=" * W)
    add("읽을 때 주의")
    add("  · 칸당 20 rollout 이라 개별 칸의 이항 표준오차가 최대 ±11%p 다.")
    add("    goal/object 는 lifelong 태스크가 4 개뿐이라 AvgSR 표준오차가 ±5%p 안팎 —")
    add("    시드 1 개로 c 를 고르면 그 폭 안의 차이는 읽어낼 수 없다.")
    add("  · c 는 **공간에 묶인다**. act 공간 r₀=0.82 이므로 c≈0.07 이면 전형적 r 에서")
    add("    MSE 의 방향 계수 2r₀²=1.35 와 같은 세기다. c 가 작을수록 세고 느린")
    add("    구간으로 더 쏠린다 (상한 1/c: c=0.1 -> 10, c=0.01 -> 100).")
    add("  · 기존 V5 의 c(0.01~0.1)는 속도 공간·잔차 전체 가중 기준이라 같은 값이 아니다.")
    add("  · * 가 붙은 값은 칸이 덜 찬 중간값이다. 최종값과 섞어 읽지 마라.")
    add("")

    out = "\n".join(L)
    print(out)
    if not a.no_write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(out + "\n", encoding="utf-8")
        print(f"\n기록 -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
