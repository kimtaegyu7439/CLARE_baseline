#!/usr/bin/env python
"""CLARE(순차) 셋팅 libero_goal 코드북 셀 수 K 스윕 -> results/Clare_setting_libero_goal.txt

    python CLARE_goal_K_report.py

읽는 것
    results/clare_v4_goal_k<K>_seed<seed>/sr_matrix.csv      스윕 (K = 90/100/110/120)
    results/clare_v4_goal_seed<seed>/sr_matrix.csv           기존 K=96 — 기준점으로 함께 싣는다

★ sr_matrix.csv 는 L2_V4_seq.py 가 **SR 칸을 잴 때마다** 다시 쓴다. 그래서 이
  스크립트는 도는 런을 건드리지 않고 파일만 읽어도 실시간 표가 된다 — LOTUS 쪽처럼
  로그를 파싱해 합성할 필요가 없다.

행렬·지표 정의는 clare_v4_report.py 와 공유한다(거기서 import). 같은 셋팅의 두 표가
서로 다른 계산을 쓰는 일이 없게 하려는 것이다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from clare_v4_report import metrics, read_matrix, seeds_of

REPO = Path(__file__).resolve().parent
RES = REPO / "results"

# ★ 기본값은 원래 스윕(K 90~120, 시드 3개) 그대로라 인자 없이 부르면 동작이 안 바뀐다.
#   다른 스윕을 같은 형식으로 뽑을 때만 --ks / --seeds / --out 을 준다.
_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--ks", default="90,100,110,120", help="코드북 셀 수 목록")
_ap.add_argument("--seeds", default="7,42,1000", help="train seed 목록")
_ap.add_argument("--out", default=None, help="출력 파일 (기본 Clare_setting_libero_goal.txt)")
_ap.add_argument("--base_k", type=int, default=96,
                 help="폴더 이름에 k 가 없는 기존 런의 K. 목록에 없으면 무시된다")
_a = _ap.parse_args()

OUT = Path(_a.out) if _a.out else RES / "Clare_setting_libero_goal.txt"
KS = [int(x) for x in _a.ks.split(",") if x.strip()]
BASE_K = _a.base_k               # 이미 돌려 둔 기존 런 (폴더 이름에 k 가 없다)
SEEDS = [int(x) for x in _a.seeds.split(",") if x.strip()]
W = 96


def rundir(k: int, seed: int) -> Path:
    """K=96 은 스윕 이전에 돌린 것이라 폴더 이름에 k 가 없다."""
    return RES / (f"clare_v4_goal_seed{seed}" if k == BASE_K
                  else f"clare_v4_goal_k{k}_seed{seed}")


def ms(vals):
    """(평균, 표본표준편차 ddof=1, n). 값이 1 개면 편차는 None."""
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    mu = sum(v) / len(v)
    if len(v) < 2:
        return mu, None, 1
    return mu, (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5, len(v)


def cell(mu, sd, n, signed=False):
    if mu is None:
        return f"{'-':>15}"
    s = ("%+.1f" if signed else "%.1f") % mu
    return f"{(s + (f' ± {sd:.1f}' if sd is not None else f' (n={n})')):>15}"


def main() -> None:
    L: list[str] = []
    add = L.append

    add("=" * W)
    add("CLARE(순차) 셋팅 · libero_goal — 코드북 셀 수 K 스윕   [v4 / TETHER]")
    add("=" * W)
    add("")
    add("무엇을 재는가")
    add("  코드북 셀 수 K 하나만 바꾸고 나머지는 전부 고정했다. K 가 커지면 과거 태스크의")
    add("  상태·관측 분포를 더 잘게 쪼개 앵커를 걸지만, 셀당 표본이 줄어 (μ, Σ) 추정이")
    add("  거칠어진다. 그 사이 어디가 좋은지 보려는 표다.")
    add("")
    add("셋팅  base phase 가 없다. libero_90 사전학습 체크포인트에서 출발해 task 0 부터")
    add("      하나씩 순차로 배운다. 태스크 t 는 스테이지 t 에서 배우므로 SR 행렬은")
    add("      아래 삼각형이고, 0행은 앵커가 걸릴 과거가 없어 K 와 무관해야 정상이다.")
    add("      사전학습: ~/Models/dit_flow_mt_libero_90_pretrain")
    add("")
    add("프로토콜  태스크당 10,000 step, 태스크당 50 demo(hold-out 없음), 10 task = 55칸")
    add("          평가 칸당 100 rollout(n_envs=50 x 2배치), eval seed 7 고정")
    add("          v4 = hard_linear 샘플러 + speedinv 앵커 가중, λ_lvl = 3.0")
    add("")
    add("지표  FWT↑ 배운 직후 / NBT↓ 이후 잃은 양(+면 망각) / AUC↑ 전 구간 평균")
    add("      AvgSR = 마지막 **완성된** 행의 평균 (진행 중이면 행 번호를 괄호에 적는다)")
    add("")
    add(f"★ K={BASE_K} 는 스윕 이전에 같은 프로토콜로 돌려 둔 기존 런이다. 기준점으로 함께 싣는다.")
    add("")

    per_k: dict[int, list] = {}
    for k in sorted(set(KS + [BASE_K])):
        add("-" * W)
        add(f"K = {k}" + ("   (기존 런 — 기준점)" if k == BASE_K else ""))
        add("-" * W)
        got = []
        for seed in SEEDS:
            d = rundir(k, seed)
            R, n_task = read_matrix(d)
            if not R:
                add(f"  train seed {seed:<5}  아직 SR 칸 없음   ({d.name})")
                add("")
                continue
            m = metrics(R, n_task)
            filled = sum(1 for r in R for c in r if c is not None)
            total = n_task * (n_task + 1) // 2
            done = filled >= total
            tr, ev = seeds_of(d)
            got.append((seed, m, done))

            add(f"  train seed {seed} / eval seed {ev}   "
                + ("완주" if done else f"진행 중 {filled}/{total}칸")
                + f"   ({d.name})")
            add("    " + f"{'stage':<7}" + "".join(f"{'t'+str(t):>6}" for t in range(n_task)))
            for i in range(len(R)):
                add("    " + f"{i:<7}" + "".join(
                    (f"{R[i][t]:6.0f}" if R[i][t] is not None else f"{'':>6}")
                    for t in range(n_task)))
            f1 = lambda v: "-" if v is None else f"{v:.1f}"        # noqa: E731
            fs = lambda v: "-" if v is None else f"{v:+.1f}"       # noqa: E731
            add(f"    지표  AvgSR {f1(m['AvgSR'])}"
                + (f" (행{m['row']})" if m["row"] is not None and not done else "")
                + f"   FWT↑ {f1(m['FWT'])}   NBT↓ {fs(m['NBT'])}   AUC↑ {f1(m['AUC'])}")
            add("")
        per_k[k] = got
        if not got:
            add("")

    # ── K 별 요약 ────────────────────────────────────────────────────────────
    # 완주한 시드만 평균에 넣는다. 진행 중인 런의 AvgSR 은 마지막 완성 행 기준이라
    # 완주한 런의 최종 행과 같은 것을 재고 있지 않다 — 섞으면 평균이 오염된다.
    add("=" * W)
    add("K 별 요약  (완주한 시드만. ± 는 시드 간 표본표준편차 ddof=1)")
    add("=" * W)
    add(f"  {'K':>5}{'n':>4}   " + "".join(f"{h:>15}" for h in ("AvgSR", "FWT↑", "NBT↓", "AUC↑"))
        + "   상태")
    for k in sorted(set(KS + [BASE_K])):
        got = per_k.get(k) or []
        done = [(s, m) for s, m, d in got if d]
        pend = [f"seed{s} {m['AvgSR']:.0f}%(행{m['row']})" if m["AvgSR"] is not None
                else f"seed{s} 0칸"
                for s, m, d in got if not d]
        miss = [s for s in SEEDS if s not in [g[0] for g in got]]
        tail = ""
        if pend:
            tail += "   진행 중 " + ", ".join(pend)
        if miss:
            tail += "   미시작 " + ", ".join(f"seed{s}" for s in miss)
        if not done:
            add(f"  {k:>5}{0:>4}   " + "".join(f"{'-':>15}" for _ in range(4)) + tail)
            continue
        row = ""
        n_used = 0
        for key, signed in (("AvgSR", False), ("FWT", False), ("NBT", True), ("AUC", False)):
            mu, sd, n = ms([m[key] for _, m in done])
            n_used = max(n_used, n)
            row += cell(mu, sd, n, signed)
        add(f"  {k:>5}{n_used:>4}   " + row + tail)
    add("")

    # 완주한 K 가 둘 이상이면 어느 쪽이 나은지 한 줄로 짚어 준다.
    ready = [(k, ms([m["AvgSR"] for _, m in [(s, mm) for s, mm, d in (per_k.get(k) or []) if d]]))
             for k in sorted(set(KS + [BASE_K]))]
    ready = [(k, t) for k, t in ready if t[0] is not None]
    if len(ready) >= 2:
        best = max(ready, key=lambda x: x[1][0])
        add(f"  현재까지 AvgSR 최고: K={best[0]}  ({best[1][0]:.1f}"
            + (f" ± {best[1][1]:.1f}" if best[1][1] is not None else "")
            + f", 완주 {best[1][2]}시드)")
        sds = [t[1] for _, t in ready if t[1] is not None]
        if sds:
            add(f"  참고 — 시드 간 편차가 {min(sds):.1f}~{max(sds):.1f}%p 다. K 간 차이가 이보다")
            add("         작으면 그건 K 효과가 아니라 시드 잡음으로 읽어야 한다.")
        add("")

    add("읽을 때 주의")
    add("  · 칸당 100 rollout 이라 개별 칸의 이항 표준오차가 최대 ±5%p 다. AvgSR 은 10 칸")
    add("    평균이라 ±1.6%p 수준이지만, 시드 간 편차는 그보다 크게 나오는 게 보통이다.")
    add("  · 0행(스테이지 0)은 사전학습 + task 0 만 배운 상태다. 앵커가 걸릴 과거가 없어")
    add("    K 와 무관해야 하고, K 별로 크게 다르면 구현이나 시드 처리를 의심해야 한다.")
    add("  · 진행 중인 런의 AvgSR 은 마지막 완성 행 기준이라 최종값이 아니다. 행 번호를")
    add("    괄호에 적어 뒀다. 요약표의 평균에는 완주한 시드만 들어간다.")
    add("  · K 는 k-means 요청 셀 수다. 병합 후 실제로 살아남은 셀(K_eff)은 이보다 작을 수")
    add("    있다 — 각 런의 l2cb.jsonl 에 Ke 로 찍힌다.")
    add("")

    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
