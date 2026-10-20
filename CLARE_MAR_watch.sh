#!/usr/bin/env bash
# 롤아웃이 도는 동안 results/Clare_TETHER_MAR_sweep.txt 를 주기적으로 갱신한다.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
INTERVAL=${INTERVAL:-300}
while true; do
    python -u CLARE_MAR_report.py > /dev/null 2>&1 && echo "[$(date '+%F %T')] 갱신"
    if ! pgrep -f "[C]LARE_rollout_only.py" >/dev/null 2>&1 \
       && ! pgrep -f "[C]LARE_rollout_daemon.sh" >/dev/null 2>&1 \
       && ! pgrep -f "[L]2_V4_seq.py" >/dev/null 2>&1 \
       && ! pgrep -f "[C]LARE_v5_seeds.sh" >/dev/null 2>&1 \
       && ! pgrep -f "[C]LARE_rollout_queue.sh" >/dev/null 2>&1 \
       && ! pgrep -f "[C]LARE_chain_after_v8.sh" >/dev/null 2>&1; then
        sleep 60; python -u CLARE_MAR_report.py > /dev/null 2>&1
        echo "[$(date '+%F %T')] 전부 종료 — 최종본 쓰고 빠짐"; break
    fi
    sleep "${INTERVAL}"
done
