#!/usr/bin/env bash
# R13 연기시험 -> 4태스크 -> 10태스크 를 GPU 0 에서 순차 실행.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
echo "[$(date '+%F %T')] R13 연기시험 종료 대기"
while pgrep -f "R13.py --smoke" > /dev/null; do sleep 30; done
echo "[$(date '+%F %T')] 연기시험 종료"
bash R_queue.sh 0 "" R13 4
bash R_queue.sh 0 "" R13 10
echo "[$(date '+%F %T')] R13 체인 종료"
