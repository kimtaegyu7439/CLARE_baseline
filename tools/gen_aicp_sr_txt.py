#!/usr/bin/env python
"""results/aicp/*.log -> results/aicp_libero_SR.txt 를 기존 양식 그대로 다시 쓴다.

지표 정의는 기존 파일에서 역산해 맞췄다 (완료 블록 값이 한 자리도 안 틀리는지
--check 로 확인할 수 있다):
    AvgSR / 행평균  = **마지막 스테이지 행**의 평균 (전체 칸 평균이 아니다)
    BWT             = mean_{t<last} ( R[last,t] - R[t,t] )
    습득(대각)      = mean_k R[k,k]
"""
import datetime as dt
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/sa090180/clare")
LOGS = ROOT / "results/aicp"
OUT = ROOT / "results/aicp_libero_SR.txt"

SUITE_OFF = {"10": 0, "Goal": 10, "Spatial": 20, "Object": 30}
PAT = re.compile(r"success_Libero_([A-Za-z0-9]+)_Task_(\d+):([0-9.]+)")


def parse(path: Path, n: int, forty: bool = False):
    rows = {}
    with open(path, "rb") as f:
        for raw in f:
            hits = PAT.findall(raw.decode("utf-8", "replace"))
            if not hits:
                continue
            d = {(SUITE_OFF[s] if forty else 0) + int(j): float(v) for s, j, v in hits}
            rows[max(d)] = d                     # 같은 스테이지가 여러 줄이면 마지막 줄
    m = np.full((n, n), np.nan)
    for k, d in rows.items():
        for t, v in d.items():
            m[k, t] = v
    return m, (max(rows) if rows else -1)


def stats(m, last):
    row = m[last, : last + 1]
    avg = float(np.nanmean(row))
    diag = float(np.nanmean([m[k, k] for k in range(last + 1)]))
    bw = [m[last, t] - m[t, t] for t in range(last)]
    return avg, (float(np.mean(bw)) if bw else float("nan")), diag


def matrix_txt(m, last, hdr):
    out = [hdr + "\t" + "\t".join(str(t) for t in range(last + 1))]
    for k in range(last + 1):
        out.append("\t".join([str(k)] + [f"{m[k, t]:g}" for t in range(k + 1)]))
    return "\n".join(out)


def rule(c="-"):
    return c * 78


BLOCKS = [
    ("libero_spatial", "seed 7",    "libero_spatial.log",          10, "LIBERO_SPATIAL"),
    ("libero_spatial", "seed 1000", "libero_spatial_seed1000.log", 10, "LIBERO_SPATIAL"),
    ("libero_spatial", "seed 42",   "libero_spatial_seed42.log",   10, "LIBERO_SPATIAL"),
    ("libero_object",  "seed 7",    "libero_object.log",           10, "LIBERO_OBJECT"),
    ("libero_object",  "seed 42",   "libero_object_seed42.log",    10, "LIBERO_OBJECT"),
    ("libero_goal",    "seed 7",    "libero_goal.log",             10, "LIBERO_GOAL"),
    ("libero_goal",    "seed 42",   "libero_goal_seed42.log",      10, "LIBERO_GOAL"),
    ("libero_10",      "seed 7",    "libero_10.log",               10, "LIBERO_10"),
    ("libero_10",      "seed 1000", "libero_10_seed1000.log",      10, "LIBERO_10"),
    ("libero_10",      "seed 42",   "libero_10_seed42.log",        10, "LIBERO_10"),
]

FORTY_BLOCKS = [(0, 9, "libero_10"), (10, 19, "libero_goal"),
                (20, 29, "libero_spatial"), (30, 39, "libero_object")]

# 지금 돌고 있는 잡. 상태는 하드코딩하지 않고 로그에서 읽는다 (다시 돌려도 최신이 된다).
RUNNING_LOGS = ["libero_spatial_seed1000.log", "libero_10_seed1000.log", "libero_40.log"]

TS = re.compile(r"^INFO (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def live_status(path: Path):
    """마지막 '학습 step' 줄과 '평가 태스크' 줄 중 나중 것으로 현재 국면을 정한다."""
    step = evaltask = None
    step_ts = eval_ts = ""
    done_ts = []
    with open(path, "rb") as f:
        for raw in f:
            ln = raw.decode("utf-8", "replace")
            t = TS.match(ln)
            t = t.group(1) if t else ""
            if "success_Libero" in ln:
                done_ts.append(t)
            elif "Eval task" in ln:
                evaltask, eval_ts = ln.split("Eval task", 1)[1].strip(), t
            else:
                mo = re.search(r"step:([0-9.]+K?)", ln)
                if mo:
                    step, step_ts = mo.group(1), t
    phase = (f"평가 중 (태스크 {evaltask})" if eval_ts >= step_ts and evaltask
             else f"학습 중 (step {step})")
    # 최근 스테이지 3개 소요 시간의 중앙값으로 스테이지당 시간을 잡는다
    per = None
    if len(done_ts) >= 2:
        ds = [dt.datetime.fromisoformat(x) for x in done_ts[-4:]]
        gaps = [(b - a).total_seconds() for a, b in zip(ds, ds[1:])]
        gaps.sort()
        per = gaps[len(gaps) // 2]
    return phase, per, (done_ts[-1] if done_ts else None)


def eta(last_done, per, n_left):
    """남은 스테이지 수 x 스테이지당 시간. 평가 태스크가 늘어 실제로는 더 걸린다."""
    if per is None or last_done is None:
        return "예상 불가"
    base = dt.datetime.fromisoformat(last_done)
    return (base + dt.timedelta(seconds=per * n_left)).strftime("%m-%d %H:%M")


def main():
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = []
    L.append(rule("="))
    L.append("AICP baseline — /home/sa090180/Models/aicp_clare_pretrain 을 --policy.path 로 쓴")
    L.append("CLARE 연속학습의 SR 행렬")
    L.append(rule("="))
    L.append("")
    L.append("행 = 스테이지 k (태스크 k 까지 학습한 시점), 열 = 평가 태스크 j. 하삼각.")
    L.append("값 = 성공률(%). 칸당 50 에피소드(N_EVAL=50), 20000 step/task.")
    L.append("출처: results/aicp/libero_*.log 의 success_<Task_j> 로그.")
    L.append(f"갱신: {now}  (로그에서 자동 재생성)")
    L.append("")
    L.append("── 지표 정의 ──────────────────────────────────────────────────────────")
    L.append("  AvgSR / 행평균  마지막 스테이지 행의 평균. 전 칸 평균이 아니다.")
    L.append("  BWT             mean_{t<last} ( R[last,t] − R[t,t] ).  −면 망각.")
    L.append("  습득(대각)      mean_k R[k,k].  갓 배운 직후 성능.")
    L.append("")
    L.append("── 시드 표기 ────────────────────────────────────────────────────────────")
    L.append("블록 제목의 (train seed X / rollout seed Y) 를 보고 읽어야 한다.")
    L.append("")
    L.append("  train seed 7  / rollout 7   첫 배치(2026-08-30 시작). 그때 clare.py 는")
    L.append("                              start_seed=cfg.seed 였으므로 둘이 같다.")
    L.append("  train seed 42 / rollout 7   clare.py:1014 를 EVAL_SEED 환경변수를 읽도록")
    L.append("                              고친 뒤의 배치. 학습만 42 로 바꾸고 롤아웃")
    L.append("                              에피소드 시드는 7~56 으로 고정했다.")
    L.append("  train seed 1000/ rollout 7  세 번째 배치(2026-09-04 12:12 시작).")
    L.append("                              aicp_env.sh 의 EVAL_SEED 기본값 7 을 그대로 쓴다.")
    L.append("                              -> 세 배치의 차이는 **학습 변동성**이다.")
    L.append("                                 평가 조건(초기 상태 50개)은 전부 같다.")
    L.append("")
    L.append("libero_40 은 4개 스위트를 한 줄로 이어붙인 별개 실험이다(40 스테이지).")
    L.append("")

    # ── 진행 중인 잡 요약 ────────────────────────────────────────────────
    L.append("── 지금 돌고 있는 것 ────────────────────────────────────────────────────")
    L.append("  ETA = (마지막 스테이지 3개 소요시간의 중앙값) x (남은 스테이지 수).")
    L.append("  뒤 스테이지일수록 평가할 태스크가 늘어 실제로는 이보다 늦어진다.")
    L.append("")
    for name, tag, log, n, _ in BLOCKS + [("libero_40", "seed 7", "libero_40.log", 40, "")]:
        if log not in RUNNING_LOGS:
            continue
        nn = 40 if log == "libero_40.log" else n
        m_, last_ = parse(LOGS / log, nn, forty=(log == "libero_40.log"))
        phase, per, done_ts = live_status(LOGS / log)
        left = nn - (last_ + 1)
        L.append(f"  {name} ({tag})")
        L.append(f"      완료 {last_ + 1}/{nn} 스테이지 (0–{last_}) · 지금은 스테이지 {last_ + 1} {phase}")
        L.append(f"      남은 스테이지 {left}개 · 스테이지당 "
                 f"{('%.1fh' % (per / 3600)) if per else '?'} · ETA {eta(done_ts, per, left)}")
    L.append("")
    L.append("  큐(aicp_queue.sh): GPU1 = spatial -> object,  GPU2 = 10 -> goal  (둘 다 seed 1000)")
    L.append("  즉 spatial/10 이 끝나면 object/goal seed 1000 이 이어서 돈다.")
    L.append("")

    # ── 스위트 블록 ──────────────────────────────────────────────────────
    for name, tag, log, n, hdr in BLOCKS:
        p = LOGS / log
        if not p.exists():
            continue
        m, last = parse(p, n)
        if last < 0:
            continue
        cells = int((~np.isnan(m)).sum())
        total = n * (n + 1) // 2
        done = last + 1 == n
        state = "완료" if done else f"**진행 중 (스테이지 {last + 1}/{n})**"
        L.append(rule())
        L.append(f"{name}   (train {tag} / rollout seed 7)   ({cells}/{total} 칸 · {state})")
        L.append(rule())
        L.append(matrix_txt(m, last, hdr))
        L.append("")
        avg, bwt, diag = stats(m, last)
        if done:
            L.append(f"AvgSR {avg:.1f}   BWT {bwt:+.1f}   습득(대각) {diag:.1f}")
        else:
            L.append(f"스테이지 {last} 시점 부분 집계 (최종 아님):")
            L.append(f"  행평균 {avg:.1f}   BWT {bwt:+.1f}   습득(대각) {diag:.1f}")
            L.append(f"  남은 스테이지 {n - last - 1}개가 붙으면 행평균은 더 내려간다.")
        L.append("")

    # ── libero_40 ─
    # ── libero_40 ───────────────────────
    m40, last40 = parse(LOGS / "libero_40.log", 40, forty=True)
    cells40 = int((~np.isnan(m40)).sum())
    done40 = last40 + 1 == 40
    state40 = "완료" if done40 else f"**진행 중 (스테이지 {last40 + 1}/40)**"
    L.append(rule())
    L.append(f"libero_40   ({cells40}/820 칸 · {state40})")
    L.append(rule())
    L.append("학습 순서: libero_10(0-9) -> goal(10-19) -> spatial(20-29) -> object(30-39)")
    L.append("칸당 50 에피소드. 이항 표준오차 ±7%p — 개별 칸이 아니라 행/블록 평균으로 읽어야 한다.")
    L.append("")
    L.append("블록                태스크    최종    습득    망각")
    L.append("")
    for lo, hi, bname in FORTY_BLOCKS:
        hi_seen = min(hi, last40)
        if lo > last40:
            continue
        fin = float(np.nanmean(m40[last40, lo:hi_seen + 1]))
        acq = float(np.nanmean([m40[k, k] for k in range(lo, hi_seen + 1)]))
        L.append(f"{bname:18s} {lo}-{hi_seen:<7d} {fin:5.1f}   {acq:5.1f}   {fin - acq:+6.1f}")
    L.append("")
    a40, b40, d40 = stats(m40, last40)
    tail = "" if done40 else f"  ← 최종 아님(스테이지 {last40 + 1}/40)"
    L.append(f"스테이지 {last40} 행평균 {a40:.1f}   BWT {b40:+.1f}   습득(대각) {d40:.1f}{tail}")
    L.append("")
    L.append("SR matrix   행 = 스테이지, 열 = 태스크(위 순서)")
    L.append(matrix_txt(m40, last40, "LIBERO_40"))
    L.append("")

    text = "\n".join(L)
    if "--check" in sys.argv:
        sys.stdout.write(text)
        return
    OUT.write_text(text)
    print(f"wrote {OUT}  ({len(text)} bytes)")


if __name__ == "__main__":
    main()
