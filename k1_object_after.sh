#!/usr/bin/env bash
# K1 4 태스크(results/K1) 가 끝나면 GPU 0 에서 libero_object 10 태스크를 이어 돌린다.
# pgrep 패턴 대신 PID 를 직접 기다린다 — 패턴 매칭은 자기 자신을 잡는다.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
PIDF=results/K1/run.pid
for _ in $(seq 1 20); do [ -f "${PIDF}" ] && break; sleep 10; done
PID=$(cat "${PIDF}" 2>/dev/null)
echo "[$(date '+%F %T')] K1 4task pid=${PID} 대기 시작"
while [ -n "${PID}" ] && kill -0 "${PID}" 2>/dev/null; do sleep 120; done
echo "[$(date '+%F %T')] K1 4task 종료 -> libero_object 10 태스크 시작 (gpu 0)"
bash libero_object.sh 0
