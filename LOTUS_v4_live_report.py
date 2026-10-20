#!/usr/bin/env python
"""진행 중인 v4 런까지 포함해 LOTUS_v4_SR_matrix_20000.txt 를 갱신한다.

    python LOTUS_v4_live_report.py

왜 필요한가
    Lotus_l2cb_cl.py 는 metrics.json 을 **런이 끝날 때 한 번** 쓴다. 그래서 통합
    리포트(LOTUS_seq_cl_report.py)는 완주한 런만 볼 수 있다. 반면 SR 자체는 칸을
    잴 때마다 로그에 찍힌다. 그 로그를 읽어 metrics.json 과 같은 모양으로 만들어
    두면, 진행 중인 런도 통합 표에 실시간으로 들어온다.

    ★ 돌고 있는 프로세스를 건드리지 않는다. 로그만 읽고, 합성한 파일은 별도
      스테이징 폴더(results/_live_v4k30/)에 쓴다. 런 폴더는 손대지 않으므로
      큐의 완주 검사(is_done)도 영향받지 않는다.

읽는 것
    logs/<tag>.log 의 두 줄 형식
        [LOTUS] suite=goal  base 6 task  4 stage x 1 task  총 10 task
        [LOTUS] SR stage1 task6 (Libero_Goal_Task_6) = 35.0
    완주한 런은 진짜 metrics.json 을 그대로 쓴다(로그 파싱보다 정확하다).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
RES = REPO / "results"
LOGS = REPO / "logs"
PREFIX = "lotus_v4k30_"
STAGE_DIR = RES / "_live_v4k30"
OUT = RES / "LOTUS_v4_SR_matrix_20000.txt"

SPEC_RE = re.compile(r"suite=(\S+)\s+base (\d+) task\s+(\d+) stage x (\d+) task\s+총 (\d+) task")
SR_RE = re.compile(r"SR stage(\d+) task(\d+) \([^)]*\) = ([0-9.]+|None)")


def from_log(tag: str) -> dict | None:
    """로그에서 R · learned_at 을 복원한다. 아직 SR 이 없으면 None."""
    p = LOGS / f"{tag}.log"
    if not p.exists():
        return None
    spec = None
    cells: list[tuple[int, int, float]] = []
    with p.open(errors="ignore") as f:
        for line in f:
            if spec is None:
                m = SPEC_RE.search(line)
                if m:
                    spec = (m.group(1), int(m.group(2)), int(m.group(3)),
                            int(m.group(4)), int(m.group(5)))
            m = SR_RE.search(line)
            if m and m.group(3) != "None":
                cells.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    if spec is None or not cells:
        return None
    suite, n_base, n_stages, per_stage, n_task = spec

    # learned_at: base 는 0, stage s 의 태스크는 s
    learned_at = [0] * n_task
    t = n_base
    for s in range(1, n_stages + 1):
        for _ in range(per_stage):
            if t < n_task:
                learned_at[t] = s
                t += 1

    R: list[list[float | None]] = [[None] * n_task for _ in range(n_stages + 1)]
    for st, tk, v in cells:
        if st < len(R) and tk < n_task:
            R[st][tk] = v
    return {"suite": suite, "R": R, "learned_at": learned_at, "_live": True}


def main() -> int:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    suites = ["object", "goal", "50"]
    seeds = [7, 42, 1000]
    live = done = 0

    for suite in suites:
        for seed in seeds:
            tag = f"{PREFIX}{suite}_seed{seed}"
            dst = STAGE_DIR / tag
            real = RES / tag / "metrics.json"
            if real.exists():                       # 완주한 런 — 진짜 파일을 쓴다
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(real, dst / "metrics.json")
                done += 1
                continue
            m = from_log(tag)                       # 진행 중 — 로그에서 복원
            if m is None:
                if dst.exists():
                    shutil.rmtree(dst)
                continue
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "metrics.json").write_text(json.dumps(m, ensure_ascii=False))
            live += 1

    cmd = [sys.executable, str(REPO / "LOTUS_seq_cl_report.py"),
           "--prefix", f"{STAGE_DIR.name}/{PREFIX}",
           "--out", str(OUT),
           "--title", "LOTUS 셋팅 lifelong — v4/TETHER (셀 30, 태스크당 20,000 step, lifelong 만 평가)",
           "--loss_note", "  손실      L = L_FM(현재) + λ_a·λ_lvl·L_anchor. 코드북 셀 30개(전 태스크 고정), hard_linear + speedinv.",
           "--tail_note", "  · base 태스크는 롤아웃하지 않았다(사전학습 92~94%). 지표는 lifelong 태스크만으로 낸 값이다.\n"
                          "  · 진행 중인 런은 로그에서 읽어 실시간 반영한 것이다 — 칸이 덜 찬 행은 아직 측정 중이다."]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:], file=sys.stderr)
        return r.returncode
    print(f"[live] 갱신 완료 -> {OUT}   완주 {done}개 / 진행 중 {live}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
