#!/usr/bin/env python
"""K10 게이트 리포트 — results/K10/report.md + selected.json.

partial.json 이 2개만 있어도 동작한다(누락 GPU 는 공란).

판정
    PASS    d̂_after ≤ 1.4  AND  diversity ≥ 0.7
    STRONG  d̂_after ≤ 1.2  AND  diversity ≥ 0.7
selected.json = diversity ≥ 0.7 인 조합 중 d̂_after 최소 {arm, T_mode, coords}.
없으면 default {prod, anneal, collective} + "GATE_FAILED" 플래그 (체인은 계속).

핵심 대조 세 개를 표 아래에 명시한다.
    anneal vs T0            노이즈가 기여했는가
    prod   vs 최선 단독      교집합이 기여했는가
    collective vs full      좌표계가 기여했는가
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"
K10 = RES / "K10"
DEFAULT = {"arm": "prod", "T_mode": "anneal", "coords": "collective"}


def load():
    rows, base = [], {}
    for g in sorted(K10.glob("gpu*/partial.json")):
        d = json.loads(g.read_text())
        base.update(d.get("baseline", {}))
        for key, c in d.get("combos", {}).items():
            if c.get("status") != "ok":
                rows.append({"key": key, "arm": d["arm"], "status": c.get("status", "?"),
                             "tmode": None, "coords": None, "bin": None})
                continue
            rows.append({"key": key, "arm": d["arm"], "status": "ok",
                         "tmode": c["tmode"], "coords": c["coords"], "bin": c["bin"],
                         "d_before": c["d_hat_before"], "d_after": c["d_hat_after"],
                         "div": c["diversity"],
                         "ew_b": c["Ehat_wit_before"], "ew_a": c["Ehat_wit_after"],
                         "eu_b": c["Ehat_U_before"], "eu_a": c["Ehat_U_after"],
                         "sec": c["sec"]})
    return rows, base


def agg(rows):
    """(arm, tmode, coords) 로 bin 평균."""
    out = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        k = (r["arm"], r["tmode"], r["coords"])
        out.setdefault(k, []).append(r)
    res = {}
    for k, v in out.items():
        res[k] = {m: float(np.mean([x[m] for x in v]))
                  for m in ("d_before", "d_after", "div", "ew_b", "ew_a", "eu_b", "eu_a")}
        res[k]["n_bins"] = len(v)
        res[k]["sec"] = float(np.sum([x["sec"] for x in v]))
    return res


def main() -> None:
    K10.mkdir(parents=True, exist_ok=True)
    rows, base = load()
    A = agg(rows)
    have = sorted({k[0] for k in A})
    L = ["# K10 게이트 — Langevin 표본이 manifold 에 접근하는가", "",
         "판정  PASS = d̂_after ≤ 1.4 AND diversity ≥ 0.7   |   STRONG = d̂ ≤ 1.2 동일 조건",
         "d̂ = med NN(b → 실제, 같은 bin) / 실제 LOO NN 중앙값.  bin {2,5,8} 평균.", ""]
    if base:
        L += ["기준선 (실제 프레임 대비)", "",
              "| bin | d_real | 가우시안 d̂ | 가우시안 Ê_wit | 가우시안 Ê_U |", "|---|---|---|---|---|"]
        for b, v in sorted(base.items()):
            L.append(f"| {b} | {v['d_real']:.1f} | {v['d_hat_gauss']:.3f} | "
                     f"{v['Ehat_wit_gauss']:.2f} | {v['Ehat_U_gauss']:.2f} |")
        L.append("")
    missing = [x for x in ("wit", "U", "prod") if x not in have]
    if missing:
        L += [f"⚠ 누락 arm: {', '.join(missing)} (partial.json 없음 — 공란 처리)", ""]

    L += ["| arm | T | coords | d̂_before | d̂_after | diversity | Ê_wit b→a | Ê_U b→a | 판정 | s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(A):
        v = A[k]
        pas = v["d_after"] <= 1.4 and v["div"] >= 0.7
        strong = v["d_after"] <= 1.2 and v["div"] >= 0.7
        tag = "**STRONG**" if strong else ("PASS" if pas else "—")
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {v['d_before']:.3f} | {v['d_after']:.3f} | "
                 f"{v['div']:.3f} | {v['ew_b']:.2f}→{v['ew_a']:.2f} | "
                 f"{v['eu_b']:.2f}→{v['eu_a']:.2f} | {tag} | {v['sec']:.0f} |")
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        L += ["", f"비정상 조합 {len(bad)}개: "
                  + ", ".join(f"{r['key']}({r['status']})" for r in bad[:12])]

    # ── 핵심 대조 3개 ────────────────────────────────────────────────────────
    L += ["", "## 핵심 대조", ""]

    def get(arm, tm, co="collective"):
        return A.get((arm, tm, co))

    for arm in have:
        t0, an = get(arm, "T0"), get(arm, "anneal")
        if t0 and an:
            L.append(f"- **노이즈 기여** ({arm}): T0 d̂ {t0['d_after']:.3f} → "
                     f"anneal {an['d_after']:.3f}  "
                     f"({an['d_after']-t0['d_after']:+.3f}, diversity "
                     f"{t0['div']:.2f}→{an['div']:.2f})")
    best_single = None
    for arm in ("wit", "U"):
        for tm in ("T0", "const", "anneal"):
            v = get(arm, tm)
            if v and (best_single is None or v["d_after"] < best_single[1]["d_after"]):
                best_single = ((arm, tm), v)
    best_prod = None
    for tm in ("T0", "const", "anneal"):
        v = get("prod", tm)
        if v and (best_prod is None or v["d_after"] < best_prod[1]["d_after"]):
            best_prod = (("prod", tm), v)
    if best_single and best_prod:
        L.append(f"- **교집합 기여**: 최선 단독 {best_single[0]} d̂ "
                 f"{best_single[1]['d_after']:.3f}  vs  최선 prod {best_prod[0]} d̂ "
                 f"{best_prod[1]['d_after']:.3f}  "
                 f"({best_prod[1]['d_after']-best_single[1]['d_after']:+.3f})")
    cf, ff = get("prod", "anneal", "collective"), get("prod", "anneal", "full")
    if cf and ff:
        L.append(f"- **좌표계 기여** (prod×anneal): collective d̂ {cf['d_after']:.3f}  vs  "
                 f"full {ff['d_after']:.3f}  ({ff['d_after']-cf['d_after']:+.3f})")

    # ── selected ────────────────────────────────────────────────────────────
    cand = [(v["d_after"], k) for k, v in A.items() if v["div"] >= 0.7]
    if cand:
        cand.sort()
        d, k = cand[0]
        sel = {"arm": k[0], "T_mode": k[1], "coords": k[2], "d_hat_after": d,
               "diversity": A[k]["div"], "gate_failed": False,
               "pass": d <= 1.4, "strong": d <= 1.2}
        L += ["", f"**선택: arm={k[0]}, T={k[1]}, coords={k[2]}**  "
                  f"(d̂ {d:.3f}, diversity {A[k]['div']:.3f}, "
                  f"{'STRONG' if d <= 1.2 else ('PASS' if d <= 1.4 else '문턱 미달이나 최선')})"]
    else:
        sel = {**DEFAULT, "gate_failed": True, "pass": False, "strong": False,
               "note": "diversity ≥ 0.7 인 조합이 없다 — default 로 진행"}
        L += ["", "**GATE_FAILED — diversity ≥ 0.7 인 조합 없음. "
                  f"default {DEFAULT} 로 Phase 2 진행.**"]
    json.dump(sel, (K10 / "selected.json").open("w"), indent=2, ensure_ascii=False)
    (K10 / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsaved -> {K10/'report.md'}, {K10/'selected.json'}")


if __name__ == "__main__":
    main()
