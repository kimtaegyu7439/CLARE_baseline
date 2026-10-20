#!/usr/bin/env bash
# CLARE 셋팅 TETHER 런이 도는 동안 성능 표를 주기적으로 갱신한다.
#
#   setsid nohup bash CLARE_TETHER_watch.sh > logs/CLARE_TETHER_watch.log 2>&1 &
#   INTERVAL=180 REPORT_ARGS="--ks 120" bash CLARE_TETHER_watch.sh
#
# sr_matrix.csv 는 SR 칸마다 갱신되므로 읽기만 하면 실시간 표가 된다. 도는 런은
# 건드리지 않는다. L2_V4_seq 도 큐도 사라지면 한 번 더 쓰고 빠진다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
INTERVAL=${INTERVAL:-300}
REPORT_ARGS=${REPORT_ARGS:-}
while true; do
    python -u CLARE_TETHER_report.py ${REPORT_ARGS} > /dev/null 2>&1 \
        && echo "[$(date '+%F %T')] 갱신" || echo "[$(date '+%F %T')] 갱신 실패"
    if ! pgrep -f "L2_V4_seq.py" > /dev/null 2>&1 \
       && ! pgrep -f "CLARE_goal_K_queue.sh" > /dev/null 2>&1; then
        sleep 60
        if ! pgrep -f "L2_V4_seq.py" > /dev/null 2>&1; then
            python -u CLARE_TETHER_report.py ${REPORT_ARGS} > /dev/null 2>&1
            echo "[$(date '+%F %T')] 런이 모두 끝났다 — 최종본 쓰고 종료"
            break
        fi
    fi
    sleep "${INTERVAL}"
done
