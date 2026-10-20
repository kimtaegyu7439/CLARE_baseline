#!/usr/bin/env bash
# V6/V7 롤아웃이 도는 동안 리포트를 주기적으로 갱신한다.
# 큐 스크립트는 롤아웃이 **끝날 때만** 리포트를 부르므로 진행 중에는 표가 멈춘다.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
INTERVAL=${INTERVAL:-300}
while true; do
    python -u CLARE_V6_report.py > /dev/null 2>&1 && echo "[$(date '+%F %T')] 갱신"
    if ! pgrep -f "[C]LARE_rollout_only.py" >/dev/null 2>&1 \
       && ! pgrep -f "[L]2_V6.py" >/dev/null 2>&1 \
       && ! pgrep -f "[L]2_V7.py" >/dev/null 2>&1 \
       && ! pgrep -f "[L]2_V8.py" >/dev/null 2>&1 \
       && ! pgrep -f "[L]2_V9.py" >/dev/null 2>&1 \
       && ! pgrep -f "[C]LARE_chain_v9.sh" >/dev/null 2>&1; then
        sleep 60; python -u CLARE_V6_report.py > /dev/null 2>&1
        echo "[$(date '+%F %T')] 전부 종료 — 최종본 쓰고 빠짐"; break
    fi
    sleep "${INTERVAL}"
done
