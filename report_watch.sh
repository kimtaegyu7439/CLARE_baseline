#!/usr/bin/env bash
# 두 리포트를 20분마다 다시 만든다. 며칠짜리 실행이라 중간 결과를 보려고
# 매번 손으로 돌리지 않아도 되게 해 둔 것이다. 실행 중인 학습은 건드리지 않는다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
while true; do
    python clare_v4_report.py        > /dev/null 2>&1
    python clare_ablation_report.py  > /dev/null 2>&1
    # 학습이 전부 끝나면 한 번 더 쓰고 빠진다
    if ! pgrep -f "L2_V4_seq.py" > /dev/null; then
        sleep 120
        pgrep -f "L2_V4_seq.py" > /dev/null || {
            python clare_v4_report.py > /dev/null 2>&1
            python clare_ablation_report.py > /dev/null 2>&1
            echo "[$(date '+%F %T')] 학습 프로세스 없음 — 리포트 최종본 쓰고 종료"; break; }
    fi
    sleep 1200
done
