#!/usr/bin/env python
"""CLARE(순차) 셋팅 TETHER(v4) 전 런 -> results/Clare_setting_TETHER_SR.txt

    python CLARE_TETHER_report.py
    python CLARE_TETHER_report.py --out results/다른이름.txt

읽는 것  results/clare_v4_*_seed*/sr_matrix.csv 와 config.json
    폴더 이름이 두 규칙으로 섞여 있다 — config.json 에서 suite/K/seed 를 읽어
    이름 규칙에 의존하지 않는다.
        clare_v4_<suite>_seed<seed>          스윕 이전 런 (K=96)
        clare_v4_<suite>_k<K>_seed<seed>     K 스윕

★ sr_matrix.csv 는 L2_V4_seq.py 가 SR 칸을 잴 때마다 다시 쓴다. 그래서 도는 런을
  건드리지 않고 파일만 읽어도 실시간 표가 된다.

★ 완주하지 않은 런도 **빼지 않고** 표시한다. 채운 칸 수를 함께 적어 두면 중간
  결과를 완성본으로 오독하지 않는다. 다만 시드 요약(평균 ± 편차)은 **완주한 런만**
  으로 낸다 — 반쯤 찬 행렬의 지표는 다른 런과 같은 것을 재고 있지 않다
  (AvgSR 은 마지막 완성 행 기준, NBT 는 이후 스테이지가 적어 작게 나온다).

행렬·지표 정의는 clare_v4_report 에서 가져온다. 같은 셋팅의 두 표가 서로 다른
계산을 쓰는 일이 없게 하려는 것이다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clare_v4_report import metrics, read_matrix

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
KEYS = ("AvgSR", "FWT", "NBT", "AUC")
SUITE_ORDER = ("libero_spatial", "libero_goal", "libero_object", "libero_10")


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(RES / "Clare_setting_TETHER_SR.txt"))
    p.add_argument("--glob", default="clare_v4_*seed*")
    p.add_argument("--ks", default=None,
                   help="쉼표로 구분한 K 목록만 남긴다. 생략하면 전부.")
    return p.parse_args()


def collect(pattern: str) -> list[dict]:
    """(suite, K, seed, R, 칸수, 완주여부) 목록. 폴더 이름이 아니라 config 를 믿는다."""
    out = []
    for d in sorted(RES.glob(pattern)):
        cp = d / "config.json"
        if not (d.is_dir() and cp.exists()):
            continue
        try:
            c = json.loads(cp.read_text())
        except Exception:
            continue
        if c.get("arm") not in (None, "v4"):      # v3 / seq_ft 팔은 제외
            continue
        R, n = read_matrix(d)
        if not R:
            continue
        filled = sum(1 for r in R for v in r if v is not None)
        total = n * (n + 1) // 2
        out.append({"dir": d.name, "suite": c.get("suite"), "K": c.get("codebook_k"),
                    "seed": c.get("seed"), "R": R, "n": n, "filled": filled,
                    "total": total, "done": filled >= total,
                    "m": metrics(R, n)})
    return out


def ms(v: list[float]):
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else None
    return m, sd


def main():
    a = parse()
    runs = collect(a.glob)
    if a.ks:
        keep = {int(x) for x in a.ks.split(",") if x.strip()}
        runs = [r for r in runs if r["K"] in keep]
    L: list[str] = []
    add = L.append

    add("=" * 96)
    add("CLARE(순차) 셋팅 — TETHER / v4 전 런 SR")
    add("=" * 96)
    add("")
    add("  셋팅  base phase 가 없다. libero_90 사전학습 체크포인트에서 출발해 task 0 부터")
    add("        하나씩 순차로 배운다. 태스크 t 는 스테이지 t 에서 배우므로 SR 행렬은")
    add("        아래삼각형(10 task = 55칸)이고, 0 행은 앵커가 걸릴 과거가 없다.")
    add("")
    add("  프로토콜  태스크당 10,000 step, 태스크당 50 demo(hold-out 없음)")
    add("            평가 칸당 100 rollout(n_envs=50), eval seed 7 고정")
    add("            v4 = hard_linear 샘플러 + speedinv 앵커 가중, λ_lvl = 3.0")
    add("")
    add("  지표  FWT↑ 배운 직후 / NBT↓ 이후 잃은 양(+면 망각) / AUC↑ 전 구간 평균")
    add("        AvgSR = 마지막 **완성된** 행의 평균")
    add("")

    # ── 런 전체 ──────────────────────────────────────────────────────────────
    add("-" * 96)
    add("전체 런")
    add("-" * 96)
    add(f"  {'suite':<16}{'K':>5}{'seed':>7}{'AvgSR':>9}{'FWT↑':>9}"
        f"{'NBT↓':>9}{'AUC↑':>9}   상태")
    key = lambda r: (SUITE_ORDER.index(r["suite"]) if r["suite"] in SUITE_ORDER else 9,
                     r["K"] or 0, r["seed"] or 0)               # noqa: E731
    for r in sorted(runs, key=key):
        m = r["m"]
        f1 = lambda v: f"{'-':>9}" if v is None else f"{v:9.1f}"      # noqa: E731
        fs = lambda v: f"{'-':>9}" if v is None else f"{v:+9.1f}"     # noqa: E731
        add(f"  {r['suite']:<16}{r['K']:>5}{r['seed']:>7}"
            + f1(m["AvgSR"]) + f1(m["FWT"]) + fs(m["NBT"]) + f1(m["AUC"])
            + ("   완주" if r["done"] else f"   진행 중 {r['filled']}/{r['total']}칸"))
    add("")

    # ── (suite, K) 별 시드 요약 — 완주한 런만 ───────────────────────────────
    add("-" * 96)
    add("시드 요약 (완주한 런만, 평균 ± 시드 간 표본표준편차 ddof=1)")
    add("-" * 96)
    add(f"  {'suite':<16}{'K':>5}{'시드':>5}" + "".join(f"{k:>15}" for k in KEYS))
    agg: dict = {}
    for r in runs:
        if r["done"]:
            agg.setdefault((r["suite"], r["K"]), []).append(r["m"])
    for (suite, K) in sorted(agg, key=lambda t: (SUITE_ORDER.index(t[0])
                                                 if t[0] in SUITE_ORDER else 9, t[1])):
        got = agg[(suite, K)]
        row = f"  {suite:<16}{K:>5}{len(got):>5}"
        for k in KEYS:
            v = [g[k] for g in got if isinstance(g.get(k), (int, float))]
            if not v:
                row += f"{'-':>15}"; continue
            m, sd = ms(v)
            body = f"{m:>+9.1f}" if k == "NBT" else f"{m:>9.1f}"
            row += body + (f" ±{sd:4.1f}" if sd is not None else "      ")
        add(row)
    add("")

    add("=" * 96)
    add("주의")
    add("  · SR 은 시뮬 롤아웃 성공률이다. 칸당 100 회라 한 칸의 이항 표준오차가 최대")
    add("    ±5%p, 10칸 평균도 ±1.6%p 다. 그보다 작은 차이는 확정할 수 없다.")
    add("  · ± 는 완주한 train seed 간 표본표준편차다. 시드가 1 개면 적지 않는다 —")
    add("    그 줄의 수치는 재현성 정보가 없다는 뜻이다.")
    add("  · K 는 코드북 셀 수다. K=96 은 스윕 이전에 돌린 기준점이라 폴더 이름에 k 가 없다.")
    if a.ks:
        add(f"  · 이 표는 K = {a.ks} 로 **걸러진 것**이다. 다른 K 의 런도 results/ 에 있다 —")
        add("    전부 보려면 --ks 없이 돌려라.")
    add("")

    txt = "\n".join(L)
    Path(a.out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
