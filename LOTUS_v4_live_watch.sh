#!/usr/bin/env bash
# v4 20k 런이 도는 동안 results/LOTUS_v4_SR_matrix_20000.txt 를 주기적으로 갱신한다.
#
#   setsid nohup bash LOTUS_v4_live_watch.sh > logs/LOTUS_v4_live_watch.log 2>&1 &
#   INTERVAL=180 bash LOTUS_v4_live_watch.sh     갱신 주기(초)
#
# 진행 중인 런은 로그에서 SR 을 읽어 반영한다(LOTUS_v4_live_report.py).
# 학습 프로세스가 모두 사라지면 마지막으로 한 번 더 쓰고 빠진다 — 그 시점엔
# 완주한 런의 진짜 metrics.json 이 있으므로 최종본이 된다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
INTERVAL=${INTERVAL:-300}

while true; do
    python -u LOTUS_v4_live_report.py 2>&1 | tail -1
    # 러너도 큐도 없으면 끝난 것이다. 한 번 더 쓰고 종료.
    if ! pgrep -f "Lotus_l2cb_cl.py" > /dev/null 2>&1 \
       && ! pgrep -f "LOTUS_v4_20000_queue.sh" > /dev/null 2>&1; then
        sleep 60
        if ! pgrep -f "Lotus_l2cb_cl.py" > /dev/null 2>&1; then
            python -u LOTUS_v4_live_report.py 2>&1 | tail -1
            echo "[$(date '+%F %T')] v4 런이 모두 끝났다 — 최종본 쓰고 종료"
            break
        fi
    fi
    sleep "${INTERVAL}"
done
