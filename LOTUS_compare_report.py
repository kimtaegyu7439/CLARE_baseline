#!/usr/bin/env python
"""LOTUS 셋팅 세 방법 대조 — **lifelong 구간만** -> results/LOTUS_ER_SEQ_TETHER_SR.txt

    python LOTUS_compare_report.py

읽는 것  각 런의 metrics.json 안의 R (SR 행렬) 과 learned_at
    seq-FT   results/lotus_seq_<suite>_seed<seed>
    ER       results/lotus_er_<suite>_seed<seed>
    TETHER   results/l2_codebook_v4_*        (폴더 이름이 옛 규칙이라 아래 표로 건다)

★ base 태스크(learned_at == 0)를 **뺀다.** LOTUS 의 base 는 여러 태스크를 한꺼번에
  joint 로 20만 step 학습한 구간이라, 그 SR 은 순차 학습의 성능이 아니라 base
  사전학습 성능이다. 세 방법이 **같은 base 체크포인트**에서 출발하므로 base 칸은
  방법을 구분하지 못하면서 평균만 끌어올린다. libero_50 은 50 태스크 중 25 개가
  base 라 그 영향이 특히 크다(AvgSR 이 10~16%p 씩 부풀려진다).

★ 지표 정의는 L2_V4.metrics / LOTUS_seq_cl_report.recompute 와 같다.
    FWT_t = R[learned_at(t)][t]                    배운 직후
    NBT_t = mean_{k>learned} (FWT_t − R[k][t])      이후 잃은 양 (+면 망각)
    AUC_t = mean_{k>=learned} R[k][t]               시퀀스 내내 낸 평균
    AvgSR = 마지막 행 평균
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
SUITES = ("object", "goal", "50")
SEEDS = (7, 42, 1000)
KEYS = ("AvgSR", "FWT", "NBT", "AUC")

# TETHER(v4) 는 폴더 이름이 옛 규칙이라 명시적으로 건다. seed42 만 _hard_sf_* 다.
TETHER_DIR = {
    ("goal", 7): "l2_codebook_v4_goal_seed7",
    ("goal", 42): "l2_codebook_v4_hard_sf_k96",
    ("goal", 1000): "l2_codebook_v4_goal_seed1000",
    ("object", 7): "l2_codebook_v4_object_seed7",
    ("object", 42): "l2_codebook_v4_hard_sf_object_k96",
    ("object", 1000): "l2_codebook_v4_object_seed1000",
    ("50", 7): "l2_codebook_v4_50_seed7",
    ("50", 42): "l2_codebook_v4_hard_sf_50_k96",
    ("50", 1000): "l2_codebook_v4_50_seed1000",
}
# TETHER-dea (V7): 액션 공간 분해 + 방향오차 인지 가중. Lotus_V7_dea.py 가
# results/lotus_v7_dea_b<β>_<suite>_seed<seed> 로 쓴다. β 는 BETA_DEA 로 고른다
# (환경변수 BETA_DEA 또는 --beta).
BETA_DEA = float(os.environ.get("BETA_DEA", "3.0"))


def dea_dir(suite: str, seed: int) -> str:
    """β 표기가 두 가지다 — 런처는 BETA=3.0 을 그대로 폴더명에 넣어 "b3.0" 이
    되는데, 파이썬 f"{3.0:g}" 는 "b3" 을 만든다. 실제로 있는 쪽을 쓴다.
    (안 맞으면 리포트가 런을 영원히 못 찾고 조용히 "-" 로만 나온다.)"""
    cands = [f"lotus_v7_dea_b{BETA_DEA:.1f}_{suite}_seed{seed}",
             f"lotus_v7_dea_b{BETA_DEA:g}_{suite}_seed{seed}"]
    for c in cands:
        if (RES / c).is_dir():
            return c
    return cands[0]


METHODS = (("seq-FT", lambda s, d: f"lotus_seq_{s}_seed{d}"),
           ("ER", lambda s, d: f"lotus_er_{s}_seed{d}"),
           ("TETHER", lambda s, d: TETHER_DIR[(s, d)]),
           ("TETHER-dea", dea_dir))


def lifelong(R, la) -> dict | None:
    """base(learned_at == 0)를 뺀 지표. 칸이 덜 찼으면 None.

    ★ 완성도는 **지표에 실제로 쓰는 칸**(lifelong, la[t] > 0)만 센다. 전에는
      base 칸까지 요구했는데, 정작 아래에서 base 는 제외하고 계산한다. 그래서
      --skip_base_eval 로 돌린 런은 lifelong 칸이 다 찼어도 영원히 None 이
      나왔다 (libero_50 은 스테이지마다 base 25 칸이 더 붙어 그게 전체 롤아웃
      시간의 대부분이라, 빼고 돌리는 게 정상 운용이다).
      base 칸을 채운 기존 런은 조건이 느슨해진 것이므로 수치가 바뀌지 않는다.
    """
    idx = [t for t in range(len(la)) if la[t] > 0]
    need = sum(1 for k in range(len(R)) for t in idx if la[t] <= k)
    have = sum(1 for k in range(len(R)) for t in idx
               if t < len(R[k]) and R[k][t] is not None)
    if not (have >= need > 0):
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
            "AUC": mean(auc), "n_task": len(idx)}


def load(dirname: str) -> dict | None:
    p = RES / dirname / "metrics.json"
    if not p.exists():
        return None
    try:
        m = json.loads(p.read_text())
    except Exception:
        return None
    R, la = m.get("R") or [], m.get("learned_at") or []
    return lifelong(R, la) if R and la else None


def ms(v: list[float]):
    """(평균, 표본표준편차 ddof=1). 시드가 1 개면 편차는 None."""
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else None
    return m, sd


def cell(v: list[float], signed: bool) -> str:
    if not v:
        return f"{'-':>14}"
    m, sd = ms(v)
    body = f"{m:>+8.1f}" if signed else f"{m:>8.1f}"
    return body + (f" ±{sd:4.1f}" if sd is not None else "      ")


def main() -> None:
    L: list[str] = []
    add = L.append
    add("=" * 94)
    add("LOTUS 셋팅 lifelong 대조 — seq-FT / ER / TETHER  (base 제외)")
    add("=" * 94)
    add("")
    add("  ★ base 태스크를 뺀 값이다. LOTUS 의 base 는 여러 태스크를 한꺼번에 joint 로")
    add("    20만 step 학습한 구간이고, 세 방법이 **같은 base 체크포인트**에서 출발한다.")
    add("    그래서 base 칸은 방법을 구분하지 못하면서 평균만 끌어올린다 — libero_50 은")
    add("    50 태스크 중 25 개가 base 라 AvgSR 이 10~16%p 부풀려진다.")
    add("")
    add("  프로토콜  lifelong stage 당 10,000 step, 태스크당 10 demo")
    add("            평가 칸당 20 rollout, eval seed 7 고정 (train seed 만 7/42/1000)")
    add("")
    add("  지표  FWT↑ 배운 직후 / NBT↓ 이후 잃은 양(+면 망각) / AUC↑ 전 구간 평균")
    add("        AvgSR↑ 마지막 행 평균.  모두 lifelong 태스크에 대해서만 낸다.")
    add("")

    for suite in SUITES:
        rows = []
        for name, fn in METHODS:
            got = []
            for sd in SEEDS:
                try:
                    m = load(fn(suite, sd))
                except KeyError:
                    m = None
                if m:
                    got.append((sd, m))
            rows.append((name, got))
        n_ll = next((g[0][1]["n_task"] for _n, g in rows if g), "-")
        add("-" * 94)
        add(f"libero_{suite}   (lifelong 태스크 {n_ll}개)")
        add("-" * 94)
        add(f"  {'method':<10}{'시드':>4}" + "".join(f"{k:>14}" for k in KEYS))
        for name, got in rows:
            if not got:
                add(f"  {name:<10}{0:>4}" + "".join(f"{'-':>14}" for _ in KEYS))
                continue
            line = f"  {name:<10}{len(got):>4}"
            for k in KEYS:
                v = [m[k] for _s, m in got if isinstance(m.get(k), (int, float))]
                line += cell(v, signed=(k == "NBT"))
            add(line)
        add("")
        # 시드별 원값 — 평균만 보고 판단하지 않도록 함께 적는다
        add(f"  시드별 AvgSR")
        for name, got in rows:
            if got:
                add(f"    {name:<10}" + "   ".join(
                    f"seed{s}={m['AvgSR']:.1f}" for s, m in got))
        add("")

    add("=" * 94)
    add("주의")
    add("  · SR 은 시뮬 롤아웃 성공률이다. 칸당 20 회라 ±5%p 단위로만 움직인다.")
    add("  · ± 는 train seed 간 표본표준편차(ddof=1)다. 시드가 3 개뿐이라 이 값도 흔들린다.")
    add("  · object/goal 은 lifelong 태스크가 4 개뿐이라 편차가 크다. 망각(NBT) 관련")
    add("    주장은 태스크가 25 개인 libero_50 을 근거로 삼는 편이 안전하다.")
    add("  · TETHER 는 sampler_mode=hard_linear, anchor_weight=speedinv, lambda_level=3.0.")
    add("")

    txt = "\n".join(L)
    (RES / "LOTUS_ER_SEQ_TETHER_SR.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
