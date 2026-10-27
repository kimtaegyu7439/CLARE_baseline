#!/usr/bin/env python
"""LOTUS V5 (λ, β, c) 스윕을 한 파일로 — results/LOTUS_MAR_sweep.txt

    python LOTUS_MAR_sweep_report.py
    python LOTUS_MAR_sweep_report.py --no-write
    REPORT=LOTUS_MAR_sweep_report.py INTERVAL=300 bash bash/report_loop.sh start

폴더 이름을 **그대로 파싱**해서 런을 찾는다 — 어떤 (λ, β, c, warm) 조합이든
`results/lotus_v5act_c<c>[_lam<λ>][_b<β>][_warm<w>]_k<K>_scope<LL|ALL>_<suite>_seed<N>`
이면 잡힌다. 생략된 접미사는 기본값(λ=1, β=1, warm=0)이다.

    V5 앵커   L_j = (1/N)[ m + β·e/(r_T²+c) ]   (액션 공간, scope=lifelong)
    λ         앵커 손실 앞 계수 (L = L_FM + λ·L_anchor)
    β         방향 항 배율 (1 = 기존 V5)
    c         방향 계수의 분모 상수 (상한 1/c)

지표는 lifelong 태스크만으로 낸다. 진행 중인 런은 report.txt 에서 읽어 * 를 붙인다.
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
OUT = RES / "LOTUS_MAR_sweep.txt"
PAT = re.compile(r"^lotus_v5(?P<space>act|vel)_c(?P<c>[0-9.]+)"
                 r"(?:_lam(?P<lam>[0-9.]+))?(?:_b(?P<beta>[0-9.]+))?(?:_warm(?P<warm>[0-9.]+))?"
                 r"_k(?P<k>\d+)_scope(?P<scope>LL|ALL)_(?P<suite>goal|object|50)_seed(?P<seed>\d+)$")


def discover():
    runs = []
    for d in sorted(RES.iterdir()):
        if not d.is_dir():
            continue
        m = PAT.match(d.name)
        if not m or (d / "INVALID.md").exists():
            continue
        g = m.groupdict()
        runs.append(dict(dir=d, suite=g["suite"], seed=int(g["seed"]), c=float(g["c"]),
                         lam=float(g["lam"] or 1.0), beta=float(g["beta"] or 1.0),
                         warm=float(g["warm"] or 0.0), k=int(g["k"]), scope=g["scope"],
                         space=g["space"]))
    return runs


# ── 읽기 (LOTUS_act_V5_report 와 같은 규칙) ────────────────────────────────
def read_report_txt(d: Path):
    p = d / "report.txt"
    if not p.exists():
        return None, None
    txt = p.read_text(errors="ignore")
    i = txt.find("SR matrix")
    if i < 0:
        return None, None
    lines = txt[i:].splitlines()
    hdr = next((l for l in lines if l.strip().startswith("stage")), None)
    if hdr is None:
        return None, None
    cols = [(mm.start(), int(mm.group(1))) for mm in re.finditer(r"t(\d+)", hdr)]
    if not cols:
        return None, None
    nt = max(t for _, t in cols) + 1
    rows = {}
    for l in lines[lines.index(hdr) + 1:]:
        mm = re.match(r"^(\d+)\s", l)
        if not mm:
            if l.startswith("---") and rows:
                break
            continue
        k = int(mm.group(1))
        row = [None] * nt
        for pos, t in cols:
            cell = l[pos - 1:pos + 5].strip() if pos else l[:6].strip()
            if cell and re.fullmatch(r"-?\d+(\.\d+)?", cell):
                row[t] = float(cell)
        rows[k] = row
    if not rows:
        return None, None
    R = [rows.get(k, [None] * nt) for k in range(max(rows) + 1)]
    la = [0] * nt
    for t in range(nt):
        ks = [k for k in range(len(R)) if R[k][t] is not None]
        la[t] = min(ks) if ks else 0
    return R, la


def read_R(d: Path):
    for f in ("metrics.json",):
        q = d / f
        if q.exists():
            try:
                j = json.loads(q.read_text())
                if j.get("R") and j.get("learned_at"):
                    return j["R"], j["learned_at"], f
            except Exception:
                pass
    R, la = read_report_txt(d)
    if R:
        return R, la, "report.txt (진행 중)"
    return None, None, None


def lifelong(R, la):
    if not R or not la:
        return None
    idx = [t for t in range(len(la)) if la[t] > 0]
    need = sum(1 for k in range(len(R)) for t in idx if la[t] <= k)
    have = sum(1 for k in range(len(R)) for t in idx if t < len(R[k]) and R[k][t] is not None)
    if have == 0:
        return None
    fwt, nbt, auc = [], [], []
    for t in idx:
        col = [R[k][t] for k in range(la[t], len(R)) if t < len(R[k]) and R[k][t] is not None]
        if not col:
            continue
        fwt.append(col[0]); auc.append(sum(col) / len(col))
        if len(col) > 1:
            nbt.append(col[0] - sum(col[1:]) / len(col[1:]))
    last = [R[-1][t] for t in idx if t < len(R[-1]) and R[-1][t] is not None]
    mean = lambda v: (sum(v) / len(v)) if v else None      # noqa: E731
    return {"AvgSR": mean(last), "FWT": mean(fwt), "NBT": mean(nbt), "AUC": mean(auc),
            "have": have, "need": need, "done": have >= need > 0,
            "t9": R[la[idx[-1]]][idx[-1]] if idx and idx[-1] < len(R[la[idx[-1]]]) else None}


def progress(d: Path):
    f = d / "train.log"
    if not f.exists():
        return "폴더만"
    last = None
    for line in f.read_text(errors="ignore").splitlines():
        if " step=" in line:
            last = line
    age = (time.time() - f.stat().st_mtime) / 3600
    if last:
        mm = re.search(r"stage(\d+) step=\s*(\d+)", last)
        if mm:
            s = f"학습 stage{mm.group(1)} {mm.group(2)}/10000"
            return s + (f" ⚠{age:.1f}h 정지?" if age > 1.0 else "")
    return "시작/코드북"


def fmt(v, sign=False):
    return "-" if v is None else (f"{v:+.1f}" if sign else f"{v:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default=os.environ.get("OUT", ""))
    a = ap.parse_args()
    out_path = (RES / a.out) if a.out else OUT
    runs = discover()
    W = 100
    L: list[str] = []
    add = L.append
    add("=" * W)
    add("LOTUS V5 (λ, β, c) 스윕 — 액션 공간, scope=lifelong, lifelong 태스크만")
    add("=" * W)
    add(f"갱신 {time.strftime('%F %T')}   런 {len(runs)}개 발견")
    add("")
    add("V5 앵커   L_j = (1/N)[ m + β·e/(r_T²+c) ]")
    add("  λ  앵커 손실 앞 계수 (L = L_FM + λ·L_anchor)     β  방향 항 배율 (1 = 기존)")
    add("  c  방향 계수 분모 상수 (상한 1/c)                 warm  스테이지 내 앵커 warmup 비율")
    add("")
    add("지표  FWT↑ 배운 직후 / NBT↓ 이후 잃은 양(+면 망각) / AUC↑ 전 구간 평균 / AvgSR↑ 마지막 행")
    add("      t9 = 마지막 lifelong 태스크의 FWT (최종값과 같다 — object 의 병목 지표)")
    add("      칸이 덜 찼으면 * (최종값이 아니다)")
    add("")

    for suite in ("goal", "object", "50"):
        rs = [r for r in runs if r["suite"] == suite]
        if not rs:
            continue
        add("-" * W)
        add(f"libero_{suite}")
        add("-" * W)
        add(f"  {'λ':>5}{'β':>5}{'c':>6}{'warm':>5}{'seed':>6}{'AvgSR':>8}{'FWT↑':>8}{'NBT↓':>8}"
            f"{'AUC↑':>8}{'t9':>6}{'칸':>7}   상태")
        groups: dict = {}
        for r in sorted(rs, key=lambda r: (r["lam"], r["beta"], r["c"], r["warm"], r["seed"])):
            R, la, src = read_R(r["dir"])
            m = lifelong(R, la)
            key = (r["lam"], r["beta"], r["c"], r["warm"])
            if m is None:
                add(f"  {r['lam']:>5g}{r['beta']:>5g}{r['c']:>6g}{r['warm']:>5g}{r['seed']:>6}"
                    + f"{'-':>8}" * 4 + f"{'-':>6}{'-':>7}   {progress(r['dir'])}")
                continue
            groups.setdefault(key, []).append((r["seed"], m))
            star = "" if m["done"] else "*"
            add("  {:>5g}{:>5g}{:>6g}{:>5g}{:>6}{:>8}{:>8}{:>8}{:>8}{:>6}{:>7}   {}".format(
                r["lam"], r["beta"], r["c"], r["warm"], r["seed"], fmt(m["AvgSR"]) + star,
                fmt(m["FWT"]), fmt(m["NBT"], True), fmt(m["AUC"]), fmt(m["t9"]),
                "{}/{}".format(m["have"], m["need"]),
                "완주" if m["done"] else progress(r["dir"])))
        add("")
        add(f"  집계 (완주한 시드만, ± ddof=1)")
        add(f"    {'λ':>5}{'β':>5}{'c':>6}{'warm':>5}{'n':>3}{'AvgSR':>14}{'FWT↑':>14}{'NBT↓':>14}{'AUC↑':>14}{'t9':>12}")
        for key in sorted(groups):
            g = [(s, m) for s, m in groups[key] if m["done"]]
            if not g:
                continue
            cols = []
            for kk in ("AvgSR", "FWT", "NBT", "AUC", "t9"):
                vals = [m[kk] for _, m in g if m[kk] is not None]
                if not vals:
                    cols.append("-".rjust(14 if kk != "t9" else 12)); continue
                mu = st.mean(vals); sg = "+" if kk == "NBT" and mu >= 0 else ""
                txt = f"{sg}{mu:.1f} ± {st.stdev(vals):.1f}" if len(vals) > 1 else f"{sg}{mu:.1f}"
                cols.append(txt.rjust(14 if kk != "t9" else 12))
            add(f"    {key[0]:>5g}{key[1]:>5g}{key[2]:>6g}{key[3]:>5g}{len(g):>3}" + "".join(cols))
        add("")

    add("=" * W)
    add("SR 행렬")
    add("=" * W)
    for r in sorted(runs, key=lambda r: (r["suite"], r["lam"], r["beta"], r["c"], r["warm"], r["seed"])):
        R, la, src = read_R(r["dir"])
        add("-" * W)
        head = f"{r['suite']} λ={r['lam']:g} β={r['beta']:g} c={r['c']:g} warm={r['warm']:g} seed{r['seed']}   ({r['dir'].name})"
        if not R:
            add(head + "   — SR 없음"); add("-" * W); add(""); continue
        m = lifelong(R, la)
        add(head + f"   출처 {src}" + ("   {}/{}칸".format(m["have"], m["need"]) if m else ""))
        add("-" * W)
        nt = max(len(x) for x in R)
        add(f"{'stage':<7}" + "".join(f"{'t'+str(t):>5}" for t in range(nt)))
        for k in range(len(R)):
            add(f"{k:<7}" + "".join((f"{R[k][t]:5.0f}" if t < len(R[k]) and R[k][t] is not None else f"{'':>5}") for t in range(nt)))
        if m:
            add(f"  지표  AvgSR {fmt(m['AvgSR'])}   FWT↑ {fmt(m['FWT'])}   NBT↓ {fmt(m['NBT'], True)}   AUC↑ {fmt(m['AUC'])}"
                + ("" if m["done"] else "   (진행 중)"))
        add("")
    out = "\n".join(L)
    print(out)
    if not a.no_write:
        out_path.write_text(out + "\n", encoding="utf-8")
        print(f"\n기록 -> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
