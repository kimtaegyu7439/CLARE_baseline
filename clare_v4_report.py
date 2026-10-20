#!/usr/bin/env python
"""CLARE(순차) 셋팅 v4 팔만 모은 SR — results/CLARE_v4_SR_matrix.txt.

LOTUS_v4_SR_matrix.txt 와 같은 양식이되 셋팅이 다르다. LOTUS 는 base joint 단계에서
출발해 stage 로 태스크를 더하지만, 여기는 **원본 CLARE 셋팅**이다 — libero_90
사전학습 체크포인트에서 출발해 task 0 부터 하나씩 순차로 배운다. 그래서 태스크 t 는
항상 스테이지 t 에서 배운다(learned_at[t] = t)이고 SR 행렬은 아래 삼각형이다.

v4 = hard_linear 샘플러 + **앵커 속도 정규화**  (= 논문의 TETHER)

행렬은 각 런의 sr_matrix.csv 에서 읽는다. report.txt 를 다시 파싱하는 LOTUS 쪽과
달리 여기는 원본 수치를 그대로 쓰므로 반올림이 두 번 들어가지 않는다.

    python clare_v4_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
OUT = RES / "CLARE_v4_SR_matrix.txt"

# (표에 쓸 스위트 이름, 결과 디렉터리)  — train/eval seed 는 metrics.json 에서 읽는다
ARMS = [
    ("libero_10",      "clare_v4_10_seed7"),
    ("libero_10",      "clare_v4_10_seed42"),
    ("libero_10",      "clare_v4_10_seed1000"),
    ("libero_goal",    "clare_v4_goal_seed7"),
    ("libero_goal",    "clare_v4_goal_seed42"),
    ("libero_goal",    "clare_v4_goal_seed1000"),
    ("libero_spatial", "clare_v4_spatial_seed7"),
    ("libero_spatial", "clare_v4_spatial_seed42"),
    ("libero_spatial", "clare_v4_spatial_seed1000"),
]


def read_matrix(d: Path):
    """sr_matrix.csv -> (R, n_task). R[k][i] = 스테이지 k 학습 후 스테이지 i 의 SR(%).

    첫 줄은 '# task_order: ...' 주석, 둘째 줄이 헤더다. 행/열 모두 스테이지
    인덱스다(L2_V4_seq.write_outputs). --task_order 를 주지 않은 실행이라
    스테이지 i = task i 로 같다.
    """
    p = d / "sr_matrix.csv"
    if not p.exists():
        return None, 0
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    body = [ln for ln in lines if not ln.startswith("#")]
    if len(body) < 2:
        return None, 0
    n_task = len(body[0].split(",")) - 1
    R = []
    for ln in body[1:]:
        cells = ln.split(",")[1:]
        R.append([float(c) if c.strip() else None for c in cells[:n_task]])

    # ★ 병렬 칸 워커(cell_worker.py)가 잰 칸을 겹쳐 읽는다. 학습 런이 종료할 때
    #   sr_matrix.csv 를 제 메모리의 R 로 다시 쓰기 때문에, 워커는 CSV 를 건드리지
    #   않고 cells_extra.jsonl 에만 적는다(끝난 뒤 merge_cells.py 로 합친다).
    #   여기서 겹쳐 읽어야 도는 중에도 표가 실시간으로 맞는다.
    #   CSV 에 이미 값이 있으면 그쪽이 우선이다 — 학습 런이 쓴 값이 정본이다.
    ex = d / "cells_extra.jsonl"
    if ex.exists():
        for ln in ex.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            k, i, sr = o.get("row"), o.get("col"), o.get("sr")
            if sr is None or k is None or i is None:
                continue
            if 0 <= k < len(R) and 0 <= i < n_task and R[k][i] is None:
                R[k][i] = float(sr)
    return (R or None), n_task


def seeds_of(d: Path) -> tuple:
    p = d / "metrics.json"
    if not p.exists():
        p = d / "config.json"
        if not p.exists():
            return ("?", "?")
        j = json.loads(p.read_text())
        return (j.get("seed", "?"), j.get("eval_seed", j.get("seed", "?")))
    j = json.loads(p.read_text()).get("config", {})
    return (j.get("seed", "?"), j.get("eval_seed", j.get("seed", "?")))


def metrics(R, n_task):
    """LOTUS 표와 **같은 정의**. 순차 셋팅이라 태스크 t 를 배운 스테이지는 t 다."""
    fwt, nbt, auc = [], [], []
    for t in range(n_task):
        col = [R[k][t] for k in range(t, len(R)) if R[k][t] is not None]
        if not col:
            continue
        fwt.append(col[0])                       # 막 배웠을 때
        auc.append(sum(col) / len(col))
        if len(col) > 1:
            nbt.append(col[0] - sum(col[1:]) / len(col[1:]))
    # AvgSR 은 **마지막으로 다 채워진 행**의 평균이다. 진행 중인 런에서 최종값인
    # 척하지 않으려고 행 번호를 같이 적는다.
    avg, row = None, None
    for k in range(len(R)):
        need = [t for t in range(n_task) if t <= k]
        if need and all(R[k][t] is not None for t in need):
            avg, row = sum(R[k][t] for t in need) / len(need), k
    mean = lambda v: (sum(v) / len(v)) if v else None            # noqa: E731
    return {"AvgSR": avg, "row": row, "FWT": mean(fwt),
            "NBT": mean(nbt), "AUC": mean(auc)}


def main() -> None:
    L = ["=" * 92,
         "CLARE(순차) 셋팅 — v4 (hard_linear 샘플러 + 앵커 속도 정규화) 전용",
         "=" * 92, "",
         "LOTUS 셋팅과 무엇이 다른가",
         "  LOTUS  base 를 joint 로 먼저 배우고 stage 로 태스크를 더한다.",
         "  CLARE  base phase 가 없다. **libero_90 사전학습 체크포인트**에서 출발해",
         "         task 0 부터 하나씩 순차로 배운다. 태스크 t 는 스테이지 t 에서",
         "         배우므로 SR 행렬이 아래 삼각형이고, 1행은 앵커가 걸릴 과거가",
         "         없어 어떤 방법을 써도 같아야 정상이다.",
         "         사전학습: ~/Models/dit_flow_mt_libero_90_pretrain", "",
         "v4 가 v3 에서 바꾼 것 두 가지",
         "  1. 샘플러   k* ~ Cat(π);  s̃ = μ_{s,k*} + σ_{s,k*}⊙ε",
         "              õ = m_{k*} + A_{k*}(z̃ − z̄_{k*})",
         "              v3 의 커널 가중 w · 대역폭 h · 관측 잡음 σ_o 를 쓰지 않는다.",
         "  2. 앵커     w = 1/(‖ṽ_T[0:6]‖² + c),  c = kappa x median‖a[0:6]‖²",
         "              median(w)x20 에서 자르고 배치 평균 1 로 정규화, pose 6 차원에만",
         "              곱한다(gripper 제외). FM 손실은 **기존 MSE 그대로**다.", "",
         "시드  train = 학습 RNG + 코드북 k-means.  eval = 롤아웃 초기 상태 100개.",
         "      eval 은 전부 7 로 고정해 평가 조건을 맞췄고, train 만 바꿔 변동을 잰다.", "",
         "프로토콜  태스크당 10,000 step, 태스크당 50 demo(hold-out 없음 — CLARE 원본과 동일)",
         "          평가 칸당 100 rollout(n_envs=50 x 2배치), 코드북 K = 96, 8000쌍 수집",
         "          스위트당 10 task = 55칸", "",
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
            L += [f"[{suite}  {d}] 결과 없음 — 아직 SR 칸이 하나도 안 나왔다", ""]
            continue
        m = metrics(R, n_task)
        filled = sum(1 for r in R for c in r if c is not None)
        total = n_task * (n_task + 1) // 2           # 아래 삼각형
        done = filled >= total
        rows.append((suite, tr, ev, m, done, filled, total))

        L += ["-" * 92,
              f"{suite}   train seed {tr} / eval seed {ev}   ({d})   "
              + ("완료" if done else f"진행 중 {filled}/{total}칸"),
              "-" * 92,
              f"{'stage':<8}" + "".join(f"{'t'+str(t):>6}" for t in range(n_task))]
        for k in range(len(R)):
            L.append(f"{k:<8}" + "".join(
                (f"{R[k][t]:6.0f}" if R[k][t] is not None else f"{'':>6}")
                for t in range(n_task)))
        f1 = lambda v: "-" if v is None else f"{v:.1f}"          # noqa: E731
        fs = lambda v: "-" if v is None else f"{v:+.1f}"         # noqa: E731
        L.append(f"  지표  AvgSR {f1(m['AvgSR'])}"
                 + (f" (행{m['row']})" if m["row"] is not None and not done else "")
                 + f"   FWT↑ {f1(m['FWT'])}   NBT↓ {fs(m['NBT'])}   AUC↑ {f1(m['AUC'])}")
        L.append("")

    if rows:
        L += ["-" * 92, "v4 전체 대조 (CLARE 셋팅)", "-" * 92,
              f"{'스위트':<14}{'train':>7}{'eval':>6}{'AvgSR':>8}{'FWT↑':>8}"
              f"{'NBT↓':>8}{'AUC↑':>8}  상태"]
        for suite, tr, ev, m, done, filled, total in rows:
            f1 = lambda v: f"{'-':>8}" if v is None else f"{v:8.1f}"      # noqa: E731
            fs = lambda v: f"{'-':>8}" if v is None else f"{v:+8.1f}"     # noqa: E731
            L.append(f"{suite:<14}{str(tr):>7}{str(ev):>6}" + f1(m["AvgSR"])
                     + f1(m["FWT"]) + fs(m["NBT"]) + f1(m["AUC"])
                     + "  " + ("완료" if done else f"진행 중 {filled}/{total}칸"))
        L.append("")
        by = {}
        for suite, tr, _, m, done, _, _ in rows:
            if done and m["AvgSR"] is not None:
                by.setdefault(suite, []).append((tr, m["AvgSR"]))
        spread = [(s, v) for s, v in by.items() if len(v) > 1]
        if spread:
            L += ["시드 변동 (완주한 것만)"]
            for s, v in spread:
                vals = [x for _, x in v]
                L.append(f"  {s:<14} " + "  ".join(f"seed{t}={x:.1f}" for t, x in v)
                         + f"   폭 {max(vals)-min(vals):.1f}%p")
            L.append("")

    if rows:
        # ── 시드 평균 ± 표준편차 ──────────────────────────────────────────────
        # 완주한 런만 넣는다. 진행 중인 런의 AvgSR 은 **마지막 완성 행** 기준이라
        # 완주한 런의 최종 행과 같은 것을 재고 있지 않다 — 섞으면 평균이 오염된다.
        def _ms(vals):
            v = [x for x in vals if x is not None]
            if not v:
                return None, None, 0
            mu = sum(v) / len(v)
            if len(v) < 2:
                return mu, None, len(v)
            sd = (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5
            return mu, sd, len(v)

        order, agg, pend = [], {}, {}
        for suite, tr, _, m, done, _, _ in rows:
            if suite not in order:
                order.append(suite)
            (agg if done else pend).setdefault(suite, []).append(m if done else tr)

        if agg:
            L += ["시드 평균 ± 표준편차  (완주한 시드만. ± 는 시드 간 표본표준편차 ddof=1)",
                  f"  {'스위트':<14}{'n':>3}   " + "".join(
                      f"{k:>15}" for k in ("AvgSR", "FWT↑", "NBT↓", "AUC↑"))]
            for suite in order:
                ms = agg.get(suite)
                if not ms:
                    L.append(f"  {suite:<14}{0:>3}   " + "".join(f"{'-':>15}" for _ in range(4)))
                    continue
                cells, n_used = "", 0
                for key, signed in (("AvgSR", False), ("FWT", False),
                                    ("NBT", True), ("AUC", False)):
                    mu, sd, n = _ms([m[key] for m in ms])
                    n_used = max(n_used, n)
                    if mu is None:
                        cells += f"{'-':>15}"
                    elif sd is None:
                        cells += f"{(('%+.1f' if signed else '%.1f') % mu) + ' (n=1)':>15}"
                    else:
                        cells += f"{(('%+.1f' if signed else '%.1f') % mu) + f' ± {sd:.1f}':>15}"
                tail = ""
                if suite in pend:
                    tail = f"   ★ seed {', '.join(str(t) for t in pend[suite])} 진행 중 — 제외"
                L.append(f"  {suite:<14}{n_used:>3}   " + cells + tail)
            L.append("")

    L += ["읽을 때 주의",
          "  · 칸당 100 rollout 이라 개별 칸의 이항 표준오차가 최대 ±5%p 다. LOTUS 표",
          "    (칸당 20)보다 촘촘하므로 두 표의 칸을 직접 맞대 비교하면 안 된다.",
          "  · 1행(스테이지 0)은 사전학습 + task 0 만 배운 상태다. 앵커가 걸릴 과거가",
          "    없으므로 어떤 CL 방법을 써도 같아야 하고, 다르면 구현 차이를 의심해야 한다.",
          "  · 진행 중인 런의 AvgSR 은 **마지막 완성 행** 기준이라 최종값이 아니다.",
          "    행 번호를 괄호에 적어 뒀다.",
          "  · 앵커는 chunk backward 라 실효 가중이 과거 태스크 수에 비례한다.",
          "    스테이지 9 에서 λ_lvl 의 9 배다.", ""]

    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
