#!/usr/bin/env python
"""AICP baseline SR 행렬 — results/aicp_libero_SR.txt.

clare.py 는 SR 을 json 으로 남기지 않는다. eval/ 에는 영상만 있고, 성공률은
MetricsTracker 로 logging 에만 찍힌다(clare.py:1018, 1045, 1059).

    success_Libero_Spatial_Task_0:95.0  success_Libero_Spatial_Task_1:80.0 ...

각 python 호출(=스테이지 k)의 평가는 태스크 0..k 를 한 줄에 전부 찍으므로,
그 줄 하나가 SR 행렬의 한 **행**이 된다. 행 번호는 그 줄에 등장한 최대 task 인덱스다.
같은 행이 여러 번 찍히면(중간 평가) 마지막 것을 쓴다.

    python aicp_sr.py                    # 있는 로그만으로 갱신 (도는 중에 실행해도 된다)
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOGS = REPO / "results" / "aicp"
OUT = REPO / "results" / "aicp_libero_SR.txt"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
TARGET = 10                       # 스위트당 목표 태스크 수 (하삼각 55칸)


def seed_of(tag: str) -> tuple[str, str]:
    """로그 태그 -> (학습 시드, 롤아웃 시드).

    태그 없음  = 2026-08-30 에 시작한 첫 배치. aicp_queue.sh 에 seed 7 을 넘겼고
                 그때는 clare.py 가 start_seed=cfg.seed 였으므로 롤아웃도 7.
    seed42     = clare.py 에 EVAL_SEED 를 넣은 뒤의 배치. 학습만 42, 롤아웃은
                 aicp_env.sh 의 EVAL_SEED=7 로 고정.
    """
    if not tag:
        return "7", "7"
    m = re.match(r"seed(\d+)$", tag)
    return (m.group(1) if m else "?"), "7"
PAT = re.compile(r"success_(Libero_[A-Za-z0-9]+_Task_(\d+)):\s*([0-9.]+)")

# ── libero_40 (연결 시퀀스) ────────────────────────────────────────────────
# 태스크 이름이 4개 벤치마크에 걸쳐 있어 j 만으로는 열이 충돌한다
# (Libero_10_Task_5 와 Libero_Goal_Task_5 가 둘 다 5). 벤치마크별 offset 을 더한다.
BLOCKS40 = [("Libero_10", "libero_10", 0), ("Libero_Goal", "libero_goal", 10),
            ("Libero_Spatial", "libero_spatial", 20), ("Libero_Object", "libero_object", 30)]
OFF40 = {b: o for b, _, o in BLOCKS40}
PAT40 = re.compile(r"success_(Libero_(?:10|Goal|Spatial|Object))_Task_(\d+):\s*([0-9.]+)")
N40 = 40


def suite_logs():
    """[(suite, tag, path)]. 태그 없는 것이 먼저, 그다음 태그순.

    같은 스위트를 다른 시드로 다시 돌릴 때 LOG_TAG 로 로그가 갈리므로
    (libero_object.log / libero_object_seed42.log) 둘 다 표로 낸다.
    """
    out = []
    for s in SUITES:
        base = LOGS / f"{s}.log"
        if base.exists():
            out.append((s, "", base))
        for p in sorted(LOGS.glob(f"{s}_*.log")):
            out.append((s, p.stem[len(s) + 1:], p))
    return out


def parse(log: Path):
    """{row: {task: sr}}. row = 그 평가 줄의 최대 task 인덱스."""
    rows: dict[int, dict[int, float]] = {}
    if not log.exists():
        return rows
    for line in log.read_text(errors="ignore").splitlines():
        hits = PAT.findall(line)
        if not hits:
            continue
        cells = {int(j): float(v) for _, j, v in hits}
        rows[max(cells)] = cells          # 나중 것이 앞의 것을 덮는다
    return rows


def parse40(log: Path):
    """{row: {col: sr}}. col = 40 태스크 순서에서의 위치, row = 그 줄의 최대 col."""
    rows: dict[int, dict[int, float]] = {}
    if not log.exists():
        return rows
    for line in log.read_text(errors="ignore").splitlines():
        hits = PAT40.findall(line)
        if not hits:
            continue
        cells = {OFF40[b] + int(j): float(v) for b, j, v in hits}
        rows[max(cells)] = cells
    return rows


def block40(rows) -> list[str]:
    n = max(rows) + 1 if rows else 0
    if not rows:
        return ["[libero_40] 결과 없음 (아직 평가 전이거나 로그 없음)", ""]
    filled = sum(len(v) for v in rows.values())
    full = N40 * (N40 + 1) // 2
    done = n >= N40
    L = ["-" * 78,
         f"libero_40   ({filled}/{full} 칸 · "
         + (f"완료)" if done else f"**진행 중 (스테이지 {n}/{N40})**)"),
         "-" * 78,
         "학습 순서: libero_10(0-9) -> goal(10-19) -> spatial(20-29) -> object(30-39)",
         "칸당 50 에피소드. 이항 표준오차 ±7%p — 개별 칸이 아니라 행/블록 평균으로 읽어야 한다.",
         ""]
    last = rows.get(n - 1, {})
    diag = {k: rows.get(k, {}).get(k) for k in range(n)}
    # 블록별 분해 (마지막 완성 행 기준)
    if len(last) == n:
        L += ["블록                태스크    최종    습득    망각", ""]
        for name, low, off in BLOCKS40:
            idx = [c for c in range(off, min(off + 10, n))]
            if not idx:
                continue
            fin = [last[c] for c in idx if c in last]
            acq = [diag[c] for c in idx if diag.get(c) is not None]
            if fin and acq:
                mf, ma = sum(fin) / len(fin), sum(acq) / len(acq)
                L.append(f"{low:<18} {off:>3}-{min(off+9, n-1):<3}  {mf:6.1f}  {ma:6.1f}  {mf-ma:+7.1f}")
        avg = sum(last.values()) / n
        acq_all = [v for v in diag.values() if v is not None]
        bwt = (sum(last[c] - diag[c] for c in range(n - 1)
                   if c in last and diag.get(c) is not None) / max(1, n - 1))
        tail = "" if done else "  ← 최종 아님(스테이지 %d/%d)" % (n, N40)
        L += ["", f"스테이지 {n-1} 행평균 {avg:.1f}   BWT {bwt:+.1f}   "
                  f"습득(대각) {sum(acq_all)/len(acq_all):.1f}{tail}"]
    L += ["", "SR matrix   행 = 스테이지, 열 = 태스크(위 순서)",
          "LIBERO_40\t" + "\t".join(str(c) for c in range(n))]
    for k in range(n):
        r = rows.get(k, {})
        L.append(f"{k}\t" + "\t".join(
            (f"{r[c]:.0f}" if c in r else "") for c in range(k + 1)))
    L.append("")
    return L


def main() -> None:
    L = ["=" * 78,
         "AICP baseline — /home/sa090180/Models/aicp_clare_pretrain 을 --policy.path 로 쓴",
         "CLARE 연속학습의 SR 행렬", "=" * 78, "",
         "행 = 스테이지 k (태스크 k 까지 학습한 시점), 열 = 평가 태스크 j. 하삼각.",
         "값 = 성공률(%). 칸당 50 에피소드(N_EVAL=50), 20000 step/task.",
         "출처: results/aicp/libero_*.log 의 success_<Task_j> 로그.", "",
         "── 시드 표기 ────────────────────────────────────────────────────────────",
         "블록 제목의 (train seed X / rollout seed Y) 를 보고 읽어야 한다.",
         "",
         "  train seed 7  / rollout 7   첫 배치(2026-08-30 시작). 그때 clare.py 는",
         "                              start_seed=cfg.seed 였으므로 둘이 같다.",
         "  train seed 42 / rollout 7   clare.py:1014 를 EVAL_SEED 환경변수를 읽도록",
         "                              고친 뒤의 배치. 학습만 42 로 바꾸고 롤아웃",
         "                              에피소드 시드는 7~56 으로 고정했다.",
         "                              -> 두 배치의 차이는 **학습 변동성**이다.",
         "                                 평가 조건(초기 상태 50개)은 같다.",
         "",
         "libero_40 은 4개 스위트를 한 줄로 이어붙인 별개 실험이다(40 스테이지).", ""]
    any_data = False
    for s, tag, log in suite_logs():
        rows = parse(log)
        n = max(rows) + 1 if rows else 0
        tr_s, ev_s = seed_of(tag)
        label = f"{s}   (train seed {tr_s} / rollout seed {ev_s})"
        head = f"{s.upper()}"
        if not rows:
            L += [f"[{label}] 결과 없음 (아직 평가 전이거나 로그 없음)", ""]
            continue
        any_data = True
        filled = sum(len(v) for v in rows.values())
        full = TARGET * (TARGET + 1) // 2
        done = n >= TARGET
        prog = (f"{filled}/{full} 칸 · 완료" if done
                else f"{filled}/{full} 칸 · **진행 중 (스테이지 {n}/{TARGET})**")
        L += ["-" * 78, f"{label}   ({prog})", "-" * 78,
              head + "\t" + "\t".join(str(j) for j in range(n))]
        for k in range(n):
            r = rows.get(k, {})
            L.append(f"{k}\t" + "\t".join(
                (f"{r[j]:.0f}" if j in r else "") for j in range(k + 1)))
        last = rows.get(n - 1, {})
        diag = {k: rows.get(k, {}).get(k) for k in range(n)}
        if len(last) == n and all(v is not None for v in diag.values()):
            avg = sum(last.values()) / n
            acq = sum(diag.values()) / n
            bwt = sum(last[j] - diag[j] for j in range(n - 1)) / max(1, n - 1)
            if done:
                L += ["", f"AvgSR {avg:.1f}   BWT {bwt:+.1f}   습득(대각) {acq:.1f}"]
            else:
                # ★ 부분 집계다. 최종 AvgSR 과 같은 것처럼 읽으면 안 된다 —
                #   태스크가 더 붙으면 값이 내려간다.
                L += ["", f"스테이지 {n-1} 시점 부분 집계 (최종 아님):",
                      f"  행평균 {avg:.1f}   BWT {bwt:+.1f}   습득(대각) {acq:.1f}",
                      f"  남은 스테이지 {TARGET-n}개가 붙으면 행평균은 더 내려간다."]
        else:
            L += ["", "진행 중 — 마지막 행이 미완이라 집계 생략"]
        L.append("")
    rows40 = parse40(LOGS / "libero_40.log")
    if rows40:
        any_data = True
    L += block40(rows40)
    if not any_data:
        L += ["(아직 어떤 스위트도 평가 결과가 없다)", ""]
    OUT.write_text("\n".join(L) + "\n")
    print(f"saved -> {OUT}")
    for s, tag, log in suite_logs():
        rows = parse(log)
        n = max(rows) + 1 if rows else 0
        name = f"{s} (train {seed_of(tag)[0]})"
        print(f"  {name:24} {sum(len(v) for v in rows.values()):>3} 칸  "
              f"(스테이지 {n}/{TARGET})")
    r40 = parse40(LOGS / "libero_40.log")
    print(f"  {'libero_40':16} {sum(len(v) for v in r40.values()):>3} 칸  "
          f"(스테이지 {max(r40)+1 if r40 else 0}/{N40})")


if __name__ == "__main__":
    main()
