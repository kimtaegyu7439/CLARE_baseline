#!/usr/bin/env python
"""LOTUS 셋팅 전 팔 SR 모음 — results/LOTUS_SR_matrix.txt.

각 run 의 results/<name>/report.txt 에서 SR 행렬을 읽어 한 파일로 모은다.
report.txt 는 칸이 채워질 때마다 갱신되므로 **도는 중에 실행해도 된다**.
미완 팔은 "진행 중 x/y칸" 으로 표시하고 집계는 채워진 칸으로만 낸다.

지표는 LIBERO 벤치마크(Liu et al. 2023) 정의로, L2_SR_matrix.txt ·
ER_10task_SR.txt · tools/plot_sr_matrix.py 와 같은 식이다. 단 LOTUS 셋팅은
base 가 여러 태스크를 **joint 로** 배우므로 "배운 스테이지"가 태스크마다 다르다.

    FWT_t = R[배운 스테이지][t]                     base 태스크는 전부 stage 0
    NBT_t = mean_{k>배운} (FWT_t − R[k][t])          + 면 망각
    AUC_t = mean_{k>=배운} R[k][t]
    AvgSR = 마지막 **완성된** 행의 평균

    python lotus_sr_report.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
OUT = RES / "LOTUS_SR_matrix.txt"

# (표시명, 결과 디렉터리, 스위트, 설명)
ARMS = [
    ("v3  goal",        "lotus_l2cb_goal",                  "goal",
     "soft_kernel 샘플러 + MSE 앵커 (= l2_codebook v3 그대로)"),
    ("v3  object",      "lotus_l2cb_object",                "object",
     "soft_kernel 샘플러 + MSE 앵커"),
    ("v4  goal",        "l2_codebook_v4_hard_sf_k96",       "goal",
     "hard_linear 샘플러 + 앵커 속도 정규화"),
    ("v4  object",      "l2_codebook_v4_hard_sf_object_k96", "object",
     "hard_linear 샘플러 + 앵커 속도 정규화"),
    ("v4- goal",        "l2_codebook_v4_hard_only_k96",     "goal",
     "hard_linear 샘플러만 (앵커는 MSE) — 샘플러/가중 효과 분리용"),
    ("v3  50",          "lotus_l2cb_50",                    "50",
     "soft_kernel 샘플러 + MSE 앵커"),
    ("v4  50",          "l2_codebook_v4_hard_sf_50_k96",    "50",
     "hard_linear 샘플러 + 앵커 속도 정규화"),
]

# 태스크가 많은 스위트는 열을 잘라 여러 블록으로 찍는다(LIBERO-50 은 50열).
COLS_PER_BLOCK = 25


def read_matrix(d: Path):
    """report.txt 에서 SR 행렬을 읽는다. 반환 (R, n_task).

    행 = 스테이지, 열 = 태스크. 아직 안 잰 칸은 None.
    """
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
        m = re.match(r"^(\d+)\s{2,}(.*)$", line)
        if not m:
            if rows:
                break
            continue
        # 고정폭 6칸. 빈 칸은 아직 안 잰 것이다.
        body = line[8:]
        cells = [body[i:i + 6].strip() for i in range(0, 6 * n_task, 6)]
        rows.append([float(c) if c else None for c in cells])
    return (rows or None), n_task


def learned_at(d: Path, n_task: int, n_row: int) -> list[int]:
    """태스크별 '배운 스테이지'. metrics.json 이 있으면 그대로, 없으면 report 로 추정."""
    p = d / "metrics.json"
    if p.exists():
        j = json.loads(p.read_text())
        if "learned_at" in j:
            return j["learned_at"]
    # report.txt 의 "t{n}  stage{k}" 줄에서 읽는다
    rp = (d / "report.txt")
    got = {}
    if rp.exists():
        for line in rp.read_text().splitlines():
            m = re.match(r"\s*t(\d+)\s+stage(\d+)", line)
            if m:
                got[int(m.group(1))] = int(m.group(2))
    if len(got) == n_task:
        return [got[t] for t in range(n_task)]
    # 마지막 수단: base 는 stage0, 이후 한 스테이지에 하나씩
    n_base = n_task - (n_row - 1)
    return [0] * n_base + [i + 1 for i in range(n_task - n_base)]


def metrics(R, la, n_task):
    fwt, nbt, auc = [], [], []
    for t in range(n_task):
        col = [R[k][t] for k in range(la[t], len(R)) if R[k][t] is not None]
        if not col:
            continue
        fwt.append(col[0])
        auc.append(sum(col) / len(col))
        if len(col) > 1:
            nbt.append(col[0] - sum(col[1:]) / len(col[1:]))
    # AvgSR 은 **완성된** 마지막 행에서만 낸다
    avg, last_full = None, None
    for k in range(len(R)):
        need = [t for t in range(n_task) if la[t] <= k]
        if all(R[k][t] is not None for t in need):
            avg, last_full = sum(R[k][t] for t in need) / len(need), k
    mean = lambda v: (sum(v) / len(v)) if v else None            # noqa: E731
    return {"AvgSR": avg, "row": last_full, "FWT": mean(fwt),
            "NBT": mean(nbt), "AUC": mean(auc)}


def main() -> None:
    L = ["=" * 88,
         "LOTUS 셋팅 lifelong — L2_codebook 계열 팔 모음",
         "=" * 88, "",
         "프로토콜  base phase 200,000 step (joint, 태스크당 50 demo) 체크포인트에서 출발",
         "          lifelong 4 stage x 1 task x 10 demo, stage 당 10,000 step",
         "          평가 칸당 20 rollout, eval seed 7, 학습·k-means seed 42",
         "          코드북 K = 2 x demo (base 50 demo -> 100, lifelong 10 demo -> 20),",
         "          전 프레임 수집(저수지 없음), λ_lvl=3.0, chunk backward", "",
         "행 = 스테이지 (0 = base 직후), 열 = 스위트 태스크.",
         "  goal / object   base 6 태스크(t0~t5) 를 stage 0 에서 joint,",
         "                  stage k 가 t(5+k) 하나씩. 4 stage, 총 10 task, 평가 40칸.",
         "  LIBERO-50       base 25 태스크(t0~t24) 를 stage 0 에서 joint,",
         "                  stage k 가 5개씩. 5 stage, 총 50 task, 평가 225칸.",
         "                  열이 50개라 25개씩 두 블록으로 나눠 찍는다.",
         "", "지표 (LIBERO 벤치마크 Liu et al. 2023 — L2_SR_matrix.txt 와 같은 정의)",
         "  FWT = mean_t R[배운 스테이지][t]                높을수록 좋음",
         "  NBT = mean_t mean_{k>배운} (FWT_t − R[k][t])    **낮을수록** 좋음, + 면 망각",
         "  AUC = mean_t mean_{k>=배운} R[k][t]             높을수록 좋음",
         "  AvgSR = 마지막 **완성된** 행의 평균", "",
         "  ★ base 태스크는 배운 스테이지가 전부 0 이라 FWT 가 base phase 성능이다.",
         "    stage 태스크는 각자 배운 직후 값이 FWT 가 된다.", ""]

    rows = []
    for name, d, suite, desc in ARMS:
        R, n_task = read_matrix(RES / d)
        if not R:
            L += [f"[{name}] 결과 없음 ({d})", ""]
            continue
        la = learned_at(RES / d, n_task, len(R))
        m = metrics(R, la, n_task)
        filled = sum(1 for r in R for c in r if c is not None)
        total = sum(1 for t in range(n_task) for k in range(len(R)) if la[t] <= k)
        done = filled >= total
        rows.append((name, suite, m, done, filled, total, desc))

        L += ["-" * 88,
              f"{name}   ({d})   " + ("완료" if done else f"진행 중 {filled}/{total}칸"),
              f"  {desc}", "-" * 88]
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
        L += ["-" * 88, "팔 대조", "-" * 88,
              f"{'팔':<12}{'스위트':<9}{'AvgSR':>8}{'FWT↑':>8}{'NBT↓':>8}{'AUC↑':>8}  상태"]
        for name, suite, m, done, filled, total, _ in rows:
            f1 = lambda v: f"{'-':>8}" if v is None else f"{v:8.1f}"      # noqa: E731
            fs = lambda v: f"{'-':>8}" if v is None else f"{v:+8.1f}"     # noqa: E731
            L.append(f"{name:<12}{suite:<9}" + f1(m["AvgSR"]) + f1(m["FWT"])
                     + fs(m["NBT"]) + f1(m["AUC"])
                     + "  " + ("완료" if done else f"진행 중 {filled}/{total}칸"))
        L.append("")

    L += ["읽을 때 주의",
          "  · 칸당 20 rollout 이라 개별 칸의 이항 표준오차가 최대 ±11%p, 10칸 평균도",
          "    ±3.5%p 다. 시드 하나(42)뿐이라 팔 사이 수 %p 차이는 확정할 수 없다.",
          "  · 진행 중인 팔의 AvgSR 은 **마지막 완성 행** 기준이라 최종값이 아니다.",
          "    스테이지가 더 붙으면 대체로 내려간다.",
          "  · 앵커는 chunk backward 라 실효 가중이 과거 태스크 수에 비례한다.",
          "    goal/object 는 stage 4 에서 λ_lvl 의 9 배(base 6 + stage 3),",
          "    LIBERO-50 은 stage 5 에서 45 배(base 25 + stage 20)다. 스위트 간",
          "    비교에서 이 차이를 감안해야 한다.",
          "  · LIBERO-50 은 goal/object 보다 평가 칸이 5.6 배(225 vs 40)라 훨씬 오래 걸린다.", ""]

    OUT.write_text("\n".join(L) + "\n")
    print(f"saved -> {OUT}")
    for name, suite, m, done, filled, total, _ in rows:
        f = lambda v: "-" if v is None else f"{v:.1f}"                    # noqa: E731
        print(f"  {name:<12}{suite:<9} AvgSR {f(m['AvgSR']):>5}  FWT {f(m['FWT']):>5}  "
              f"NBT {f(m['NBT']):>5}  AUC {f(m['AUC']):>5}  "
              + ("완료" if done else f"{filled}/{total}칸"))


if __name__ == "__main__":
    main()
