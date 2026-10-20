#!/usr/bin/env bash
# median 롤아웃이 끝나면 GPU 2 -> c0.1, GPU 3 -> c0.01 롤아웃 데몬을 띄운다.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
while pgrep -f "[C]LARE_rollout_only.py" > /dev/null 2>&1; do sleep 60; done
echo "[$(date '+%F %T')] median 롤아웃 종료 — c 팔 롤아웃 데몬 시작"
setsid nohup env GPU=2 ARM=c0.1  bash CLARE_rollout_daemon.sh > logs/daemon_c0.1.log 2>&1 &
setsid nohup env GPU=3 ARM=c0.01 bash CLARE_rollout_daemon.sh > logs/daemon_c0.01.log 2>&1 &
