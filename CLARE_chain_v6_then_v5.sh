#!/usr/bin/env bash
# V6 3 팔이 모두 끝나면 멈춰 둔 V5 seed1000 (c0.1, c0.05) 을 GPU 0,1 에 되살린다.
# --resume 이라 끝난 스테이지(0~2)는 보존되고 stage 3 부터 다시 간다.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
while pgrep -f "[C]LARE_V6_queue.sh" > /dev/null 2>&1; do sleep 120; done
echo "[$(date '+%F %T')] V6 종료 — V5 seed1000 재개"
setsid nohup env GPU=0 C=0.1  SEEDS=1000 bash CLARE_v5_seeds.sh > logs/launch_v5resume_gpu0.log 2>&1 &
setsid nohup env GPU=1 C=0.05 SEEDS=1000 bash CLARE_v5_seeds.sh > logs/launch_v5resume_gpu1.log 2>&1 &
