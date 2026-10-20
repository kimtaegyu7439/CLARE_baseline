#!/usr/bin/env python
"""LOTUS 셋팅 v4 팔만 모은 SR — results/LOTUS_v4_SR_matrix.txt.

v4 = hard_linear 샘플러 + **앵커 속도 정규화**
     · 샘플러  k* ~ Cat(π);  s̃ = μ_{s,k*} + σ_{s,k*}⊙ε
               õ = m_{k*} + A_{k*}(z̃ − z̄_{k*})
               v3 의 커널 가중 w·대역폭 h·관측 잡음 σ_o 를 쓰지 않는다.
     · 앵커    w = 1/(‖ṽ_T[0:6]‖² + c),  median·20 에서 자르고 배치 평균 1 로 정규화.
               pose 6 차원에만 곱하고 gripper 는 그대로. FM 손실은 기존 MSE 다.

★ 샘플러만 바꾼 v4- (anchor_weight=mse) 는 **이 표에 넣지 않는다.**
  여기는 앵커 정규화까지 켠 팔만 모은다. v4- 는 LOTUS_SR_matrix.txt 에 있다.

시드는 두 가지가 따로 논다 — 표에 둘 다 적는다.
    train seed   학습 RNG + 코드북 k-means. 팔 사이 비교의 변동 원인.
    eval seed    롤아웃 초기 상태 20개. **전부 7 로 고정**해 평가 조건을 맞췄다.

    python lotus_v4_report.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
OUT = RES / "LOTUS_v4_SR_matrix.txt"
COLS_PER_BLOCK = 25

# (스위트, 결과 디렉터리)  — train seed 는 config.json 에서 읽는다
ARMS = [
    ("goal",   "l2_codebook_v4_hard_sf_k96"),
    ("goal",   "l2_codebook_v4_goal_seed7"),
    ("goal",   "l2_codebook_v4_goal_seed1000"),
    ("object", "l2_codebook_v4_hard_sf_object_k96"),
    ("object", "l2_codebook_v4_object_seed7"),
    ("object", "l2_codebook_v4_object_seed1000"),
    ("50",     "l2_codebook_v4_hard_sf_50_k96"),
    ("50",     "l2_codebook_v4_50_seed7"),
    ("50",     "l2_codebook_v4_50_seed1000"),
]


def read_matrix(d: Path):
    p = d / "report.txt"
    if not p.exists():
        return None, 0
    rows, n_task, seen = [], 0, False
    for line in p.read_text().splitlines():
        if line.startswith("stage ") and "t0" in line:
            n_task = len(re.findall(r"\bt\d+", line))
            seen = True
            continue
        if not seen:
            continue
        if not re.match(r"^\d+\s{2,}", line):
            if rows:
                break
            continue
        body = line[8:]
        cells = [body[i:i + 6].strip() for i in range(0, 6 * n_task, 6)]
        rows.append([float(c) if c else None for c in cells])
    return (rows or None), n_task


def learned_at(d: Path, n_task: int, n_row: int) -> list[int]:
    p = d / "metrics.json"
    if p.exists():
        j = json.loads(p.read_text())
        if "learned_at" in j:
            return j["learned_at"]
    got = {}
    rp = d / "report.txt"
    if rp.exists():
        for line in rp.read_text().splitlines():
            m = re.match(r"\s*t(\d+)\s+stage(\d+)", line)
            if m:
                got[int(m.group(1))] = int(m.group(2))
    if len(got) == n_task:
        return [got[t] for t in range(n_task)]
    n_base = n_task - (n_row - 1)
    return [0] * n_base + [i + 1 for i in range(n_task - n_base)]


def metrics(R, la, n_task, only=None):
    """only 가 주어지면 그 태스크들만으로 지표를 낸다.

    ★ AvgSR 의 '마지막 완성 행'은 **전체** 태스크 기준으로 고른다. 부분집합으로
      고르면 lifelong 만 찬 중간 행이 뽑혀 base 를 포함한 표와 다른 시점을
      비교하게 된다. 행을 고른 뒤 평균만 부분집합으로 낸다.
    """
    sel = set(range(n_task)) if only is None else set(only)
    fwt, nbt, auc = [], [], []
    for t in sorted(sel):
        col = [R[k][t] for k in range(la[t], len(R)) if R[k][t] is not None]
        if not col:
            continue
        fwt.append(col[0])
        auc.append(sum(col) / len(col))
        if len(col) > 1:
            nbt.append(col[0] - sum(col[1:]) / len(col[1:]))
    avg, row = None, None
    for k in range(len(R)):
        need = [t for t in range(n_task) if la[t] <= k]
        if need and all(R[k][t] is not None for t in need):
            got = [t for t in need if t in sel]
            row = k
            avg = (sum(R[k][t] for t in got) / len(got)) if got else None
    mean = lambda v: (sum(v) / len(v)) if v else None            # noqa: E731
    return {"AvgSR": avg, "row": row, "FWT": mean(fwt),
            "NBT": mean(nbt), "AUC": mean(auc)}


def seeds_of(d: Path) -> tuple:
    p = d / "config.json"
    if not p.exists():
        return ("?", "?")
    j = json.loads(p.read_text())
    return (j.get("seed", "?"), j.get("eval_seed", "?"))


def main() -> None:
    L = ["=" * 92,
         "LOTUS 셋팅 — v4 (hard_linear 샘플러 + 앵커 속도 정규화) 전용",
         "=" * 92, "",
         "v4 가 v3 에서 바꾼 것 두 가지",
         "  1. 샘플러   k* ~ Cat(π);  s̃ = μ_{s,k*} + σ_{s,k*}⊙ε",
         "              õ = m_{k*} + A_{k*}(z̃ − z̄_{k*})",
         "              v3 의 커널 가중 w · 대역폭 h · 관측 잡음 σ_o 를 쓰지 않는다.",
         "  2. 앵커     w = 1/(‖ṽ_T[0:6]‖² + c),  c = kappa x median‖a[0:6]‖²",
         "              median(w)x20 에서 자르고 배치 평균 1 로 정규화, pose 6 차원에만",
         "              곱한다(gripper 제외). FM 손실은 **기존 MSE 그대로**다.",
         "",
         "  ★ 샘플러만 바꾼 v4- (anchor_weight=mse) 는 이 표에 없다.",
         "    LOTUS_SR_matrix.txt 를 보라.", "",
         "시드  train = 학습 RNG + 코드북 k-means.  eval = 롤아웃 초기 상태 20개.",
         "      eval 은 전부 7 로 고정해 평가 조건을 맞췄고, train 만 바꿔 변동을 잰다.", "",
         "프로토콜  base phase 200,000 step(joint, 태스크당 50 demo) 체크포인트에서 출발",
         "          lifelong stage 당 10,000 step, 태스크당 10 demo",
         "          평가 칸당 20 rollout, 코드북 K = 2 x demo, 전 프레임 수집", "",
         "구조  goal / object   base 6 task + 4 stage x 1 task = 10 task, 40칸",
         "      LIBERO-50       base 25 task + 5 stage x 5 task = 50 task, 225칸", "",
         "지표 (LIBERO 벤치마크 Liu et al. 2023)",
         "  FWT = mean_t R[배운 스테이지][t]                높을수록 좋음",
         "  NBT = mean_t mean_{k>배운} (FWT_t − R[k][t])    **낮을수록** 좋음, + 면 망각",
         "  AUC = mean_t mean_{k>=배운} R[k][t]             높을수록 좋음",
         "  AvgSR = 마지막 **완성된** 행의 평균", ""]

    rows = []
    for suite, d in ARMS:
        R, n_task = read_matrix(RES / d)
        tr, ev = seeds_of(RES / d)
        if not R:
            L += [f"[{suite}  train seed {tr}] 결과 없음 ({d})", ""]
            continue
        la = learned_at(RES / d, n_task, len(R))
        m = metrics(R, la, n_task)
        # lifelong = 스테이지 1 이후에 배운 태스크. base 태스크는 la[t]==0 이다.
        life = [t for t in range(n_task) if la[t] > 0]
        m_life = metrics(R, la, n_task, only=life)
        filled = sum(1 for r in R for c in r if c is not None)
        total = sum(1 for t in range(n_task) for k in range(len(R)) if la[t] <= k)
        done = filled >= total
        rows.append((suite, tr, ev, m, done, filled, total, m_life, len(life)))

        L += ["-" * 92,
              f"{suite}   train seed {tr} / eval seed {ev}   ({d})   "
              + ("완료" if done else f"진행 중 {filled}/{total}칸"),
              "-" * 92]
        for c0 in range(0, n_task, COLS_PER_BLOCK):
            c1 = min(c0 + COLS_PER_BLOCK, n_task)
            if n_task > COLS_PER_BLOCK:
                L.append(f"  (태스크 t{c0}~t{c1-1})")
            L.append(f"{'stage':<8}" + "".join(f"{'t'+str(t):>6}" for t in range(c0, c1)))
            for k in range(len(R)):
                L.append(f"{k:<8}" + "".join(
                    (f"{R[k][t]:6.0f}" if R[k][t] is not None else f"{'':>6}")
                    for t in range(c0, c1)))
        f1 = lambda v: "-" if v is None else f"{v:.1f}"          # noqa: E731
        fs = lambda v: "-" if v is None else f"{v:+.1f}"         # noqa: E731
        L.append(f"  지표  AvgSR {f1(m['AvgSR'])}"
                 + (f" (행{m['row']})" if m["row"] is not None and not done else "")
                 + f"   FWT↑ {f1(m['FWT'])}   NBT↓ {fs(m['NBT'])}   AUC↑ {f1(m['AUC'])}")
        L.append("")

    if rows:
        L += ["-" * 92, "v4 전체 대조", "-" * 92,
              f"{'스위트':<10}{'train':>7}{'eval':>6}{'AvgSR':>8}{'FWT↑':>8}"
              f"{'NBT↓':>8}{'AUC↑':>8}  상태"]
        for suite, tr, ev, m, done, filled, total, _ml, _nl in rows:
            f1 = lambda v: f"{'-':>8}" if v is None else f"{v:8.1f}"      # noqa: E731
            fs = lambda v: f"{'-':>8}" if v is None else f"{v:+8.1f}"     # noqa: E731
            L.append(f"{suite:<10}{str(tr):>7}{str(ev):>6}" + f1(m["AvgSR"])
                     + f1(m["FWT"]) + fs(m["NBT"]) + f1(m["AUC"])
                     + "  " + ("완료" if done else f"진행 중 {filled}/{total}칸"))
        L.append("")
        # ── 스위트별 시드 요약: 평균 ± 표본표준편차 ──────────────────────────
        # ★ ddof=1 (표본표준편차). 시드가 3 개뿐이라 이 값 자체도 꽤 흔들린다 —
        #   신뢰구간이 아니라 "재현성이 이 정도 폭"이라는 눈금이다. 완주한 팔만 쓴다:
        #   반쯤 찬 행렬의 AvgSR 은 다른 런과 같은 것을 재고 있지 않다.
        KS = (("AvgSR", "AvgSR", False), ("FWT", "FWT↑", False),
              ("NBT", "NBT↓", True), ("AUC", "AUC↑", False))
        agg = {}
        for suite, _tr, _ev, m, done, _f, _t, _ml, _nl in rows:
            if done:
                agg.setdefault(suite, []).append(m)
        if agg:
            L += ["시드 요약 (완주한 것만) — 평균 ± 시드 간 표본표준편차",
                  f"{'스위트':<10}{'시드':>5}" + "".join(f"{lbl:>16}" for _k, lbl, _s in KS)]
            for suite, ms_ in agg.items():
                line = f"{suite:<10}{len(ms_):>5}"
                for key, _lbl, signed in KS:
                    v = [x[key] for x in ms_ if x.get(key) is not None]
                    if not v:
                        line += f"{'-':>16}"; continue
                    mu = sum(v) / len(v)
                    if len(v) < 2:      # 시드 1 개면 편차가 정의되지 않는다
                        line += (f"{mu:>+10.1f}" if signed else f"{mu:>10.1f}") + "      "
                        continue
                    sd = (sum((y - mu) ** 2 for y in v) / (len(v) - 1)) ** 0.5
                    line += (f"{mu:>+10.1f}" if signed else f"{mu:>10.1f}") + f" ±{sd:4.1f}"
                L.append(line)
            L.append("")
        # ── lifelong 태스크만 따로 ────────────────────────────────────────
        # base 태스크는 사전학습이 이미 푼 것이라 전체 평균에 섞이면 lifelong 에서
        # 실제로 무슨 일이 일어났는지가 가려진다. 같은 R 행렬에서 열만 골라 다시 낸다.
        agg_l, nl_of = {}, {}
        for suite, _tr, _ev, _m, done, _f, _t, ml, nl in rows:
            if done:
                agg_l.setdefault(suite, []).append(ml)
                nl_of[suite] = nl
        if agg_l:
            L += ["lifelong 태스크만 (base 제외) — 평균 ± 시드 간 표본표준편차",
                  f"{'스위트':<10}{'시드':>5}{'태스크':>7}"
                  + "".join(f"{lbl:>16}" for _k, lbl, _s in KS)]
            for suite, ms_ in agg_l.items():
                line = f"{suite:<10}{len(ms_):>5}{nl_of[suite]:>7}"
                for key, _lbl, signed in KS:
                    v = [x[key] for x in ms_ if x.get(key) is not None]
                    if not v:
                        line += f"{'-':>16}"; continue
                    mu = sum(v) / len(v)
                    if len(v) < 2:
                        line += (f"{mu:>+10.1f}" if signed else f"{mu:>10.1f}") + "      "
                        continue
                    sd = (sum((y - mu) ** 2 for y in v) / (len(v) - 1)) ** 0.5
                    line += (f"{mu:>+10.1f}" if signed else f"{mu:>10.1f}") + f" ±{sd:4.1f}"
                L.append(line)
            L.append("")

        # 같은 스위트의 시드별 폭
        by = {}
        for suite, tr, _, m, done, _, _, _ml, _nl in rows:
            if done and m["AvgSR"] is not None:
                by.setdefault(suite, []).append((tr, m["AvgSR"]))
        spread = [(s, v) for s, v in by.items() if len(v) > 1]
        if spread:
            L += ["시드 변동 (완주한 것만)"]
            for s, v in spread:
                vals = [x for _, x in v]
                L.append(f"  {s:<8} " + "  ".join(f"seed{t}={x:.1f}" for t, x in v)
                         + f"   폭 {max(vals)-min(vals):.1f}%p")
            L.append("")

    L += ["읽을 때 주의",
          "  · ± 는 완주한 train seed 간 표본표준편차(ddof=1)다. 시드 3 개짜리라 이 값도",
          "    흔들린다 — 신뢰구간이 아니고, 아래의 롤아웃 분산도 그 안에 섞여 있다.",
          "  · 칸당 20 rollout 이라 개별 칸의 이항 표준오차가 최대 ±11%p, 10칸 평균도",
          "    ±3.5%p 다. 시드 하나로 잰 수 %p 차이는 확정할 수 없다 — 그래서 train",
          "    seed 를 42 / 7 / 1000 으로 반복한다.",
          "  · 진행 중인 팔의 AvgSR 은 **마지막 완성 행** 기준이라 최종값이 아니다.",
          "  · 앵커는 chunk backward 라 실효 가중이 과거 태스크 수에 비례한다.",
          "    goal/object 는 stage 4 에서 λ_lvl 의 9 배, LIBERO-50 은 stage 5 에서 45 배다.",
          "    스위트를 가로질러 비교할 때 감안해야 한다.", ""]

    OUT.write_text("\n".join(L) + "\n")
    print(f"saved -> {OUT}")
    for suite, tr, ev, m, done, filled, total, _ml, _nl in rows:
        f = lambda v: "-" if v is None else f"{v:.1f}"                   # noqa: E731
        print(f"  {suite:<8} train{str(tr):>5} eval{str(ev):>3}  "
              f"AvgSR {f(m['AvgSR']):>5}  FWT {f(m['FWT']):>5}  NBT {f(m['NBT']):>5}  "
              f"AUC {f(m['AUC']):>5}  " + ("완료" if done else f"{filled}/{total}칸"))


if __name__ == "__main__":
    main()
