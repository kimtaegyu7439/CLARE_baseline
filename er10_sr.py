#!/usr/bin/env python
"""ER 10-task (칸당 20 롤아웃) SR 행렬 모음 — results/ER_10task_SR.txt.

네 스위트의 ER 결과가 서로 다른 형식/위치에 흩어져 있어서 한 파일로 모은다.

    libero_spatial   B_mod_none_null.txt 안의 "ER (55/55 칸, 완료)" 블록
                     (전용 디렉터리가 ER_10task/ 이고 파일 이름이 4-task 시절
                      ER_task0123_SR.txt 로 남아 있어 여기서 읽지 않는다)
    나머지 3개       ER_<suite>_10task/sr_matrix.csv

집계는 원본 문서 값을 베끼지 않고 칸에서 다시 계산한다(검산 겸).

    python er10_sr.py
"""
from __future__ import annotations

import re
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
OUT = RES / "ER_10task_SR.txt"
K = 10

SUITES = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]


def from_b_mod() -> dict[tuple[int, int], float]:
    """B_mod_none_null.txt 의 ER 블록. 'after\\task' 머리줄 뒤 K 행."""
    txt = (RES / "B_mod_none_null.txt").read_text().splitlines()
    i = next(n for n, l in enumerate(txt)
             if l.startswith("ER ") and "55/55" in l)
    j = next(n for n in range(i, i + 6) if txt[n].lstrip().startswith("after"))
    cells = {}
    for k, line in enumerate(txt[j + 1:j + 1 + K]):
        f = line.split()
        assert int(f[0]) == k, line
        for t, v in enumerate(f[1:]):
            if v != ".":
                cells[(k, t)] = float(v)
    return cells


def from_csv(path: Path) -> dict[tuple[int, int], float]:
    cells = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split(",")
        if not f[0].strip().isdigit():
            continue
        k = int(f[0])
        for t, v in enumerate(f[1:]):
            if v.strip():
                cells[(k, t)] = float(v)
    return cells


def stats(c: dict[tuple[int, int], float]) -> dict:
    """AvgSR/BWT 에 LIBERO 지표(FWT/NBT/AUC)를 더한다.

    정의는 l2_family_report.py · L2_codebook_grad_report.py · tools/plot_sr_matrix.py
    와 동일하다(LIBERO 벤치마크, Liu et al. 2023). 세 파일의 숫자를 그대로
    나란히 놓고 비교할 수 있어야 하므로 식을 공유한다.

        FWT_t = R[t][t]                        갓 배운 직후 (= '습득', 대각)
        NBT_t = mean_{k>t} (R[t][t] − R[k][t])   배운 뒤 잃은 양 (+면 망각)
        AUC_t = mean_{k>=t} R[k][t]            시퀀스 내내 낸 평균
    """
    last = [c[(K - 1, t)] for t in range(K)]
    diag = [c[(t, t)] for t in range(K)]
    fwt, nbt, auc = [], [], []
    for t in range(K):
        col = [c[(k, t)] for k in range(t, K) if (k, t) in c]
        if not col:
            continue
        fwt.append(col[0])
        auc.append(sum(col) / len(col))
        if len(col) > 1:                       # 마지막 태스크는 NBT 가 정의되지 않는다
            nbt.append(col[0] - sum(col[1:]) / len(col[1:]))
    mean = lambda v: (sum(v) / len(v)) if v else None          # noqa: E731
    return {"avg": sum(last) / K,
            "acq": sum(diag) / K,
            "bwt": sum(last[t] - diag[t] for t in range(K - 1)) / (K - 1),
            "fwt": mean(fwt), "nbt": mean(nbt), "auc": mean(auc),
            "n": len(c)}


def block(name: str, c: dict, s: dict, src: str) -> list[str]:
    L = ["-" * 84,
         f"{name}   ({s['n']}/55 칸)   AvgSR {s['avg']:.1f}   FWT↑ {s['fwt']:.1f}   "
         f"NBT↓ {s['nbt']:+.1f}   AUC↑ {s['auc']:.1f}   BWT↑ {s['bwt']:+.1f}",
         f"출처 {src}",
         "-" * 84,
         "after\\task " + "".join(f"{t:>6}" for t in range(K))]
    for k in range(K):
        L.append(f"{k:>10} " + "".join(
            (f"{c[(k,t)]:>6.0f}" if (k, t) in c else f"{'.':>6}") for t in range(K)))
    return L + [""]


def main() -> None:
    got = {"libero_spatial": (from_b_mod(), "results/B_mod_none_null.txt 의 ER 블록")}
    for s in SUITES[1:]:
        p = RES / f"ER_{s}_10task" / "sr_matrix.csv"
        got[s] = (from_csv(p), f"results/ER_{s}_10task/sr_matrix.csv")

    L = ["=" * 84,
         "ER (experience replay) — 스위트별 10 태스크, 칸당 20 롤아웃",
         "=" * 84, "",
         "프로토콜  5000 step/task, 45 에피소드(뒤 5개 hold-out), seed 42,",
         "          배치 32 = 현재 태스크 24 + 리플레이 버퍼 8, 과거 태스크당 5 에피소드.",
         "          K1/R1x 팔과 같은 조건이다(그쪽은 과거 원시 데이터를 안 쓴다).",
         "측정      시뮬레이터 롤아웃 성공률(%), 칸당 20 에피소드.",
         "          이항 표준오차가 칸당 최대 ±11%p 다. 행/열 평균으로 읽어야 한다.",
         "",
         "행 = 스테이지 k (태스크 k 까지 학습한 시점), 열 = 평가 태스크 j. 하삼각 55칸.",
         "집계값은 이 스크립트가 칸에서 다시 계산한 것이다(er10_sr.py).", "",
         "지표 (LIBERO 벤치마크 Liu et al. 2023 — L2_SR_matrix.txt 와 같은 정의)",
         "  AvgSR = mean_t R[9][t]                          높을수록 좋음",
         "  FWT   = mean_t R[t][t]                          높을수록 좋음",
         "          예전 표기의 '습득'(대각 평균)과 같은 값이다. 이름만 바꿨다.",
         "  NBT   = mean_t mean_{k>t} (R[t][t] − R[k][t])   **낮을수록** 좋음, +면 망각",
         "  AUC   = mean_t mean_{k>=t} R[k][t]              높을수록 좋음",
         "  BWT   = mean_t (R[9][t] − R[t][t])              높을수록 좋음 (task9 제외)",
         "",
         "  NBT 와 BWT 는 다르다. BWT 는 마지막 행 한 줄만 보고, NBT 는 배운 뒤의",
         "  모든 스테이지를 평균한다. 그래서 NBT ≈ −BWT 이지만 같지 않다.", "",
         "-" * 84, "요약", "-" * 84,
         f"{'스위트':<18}{'AvgSR':>8}{'FWT↑':>8}{'NBT↓':>8}{'AUC↑':>8}{'BWT↑':>8}", ""]
    for s in SUITES:
        c, _ = got[s]
        st = stats(c)
        L.append(f"{s:<18}{st['avg']:>8.1f}{st['fwt']:>8.1f}{st['nbt']:>+8.1f}"
                 f"{st['auc']:>8.1f}{st['bwt']:>+8.1f}")
    L += ["", "주의  results/ER_libero_40_SR.txt 는 40 태스크를 한 줄로 이어붙인 별개 실험이다",
          "      (820칸, AvgSR 55.8). 위 표와 직접 비교하면 안 된다.", ""]

    for s in SUITES:
        c, src = got[s]
        L += block(s, c, stats(c), src)

    OUT.write_text("\n".join(L) + "\n")
    print(f"saved -> {OUT}")
    for s in SUITES:
        st = stats(got[s][0])
        print(f"  {s:<16} {st['n']}/55 칸  AvgSR {st['avg']:.1f}  FWT {st['fwt']:.1f}  "
              f"NBT {st['nbt']:+.1f}  AUC {st['auc']:.1f}  BWT {st['bwt']:+.1f}")


if __name__ == "__main__":
    main()
