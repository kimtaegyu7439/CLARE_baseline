#!/usr/bin/env bash
# GPU 1, 2 가 비는 대로 V9(= V8 의 각도·크기 항을 **액션 공간**에서) 2 팔을 띄운다.
#
#   GPU 1  V7 dea 롤아웃 큐가 끝나면        -> V9 mse_da
#   GPU 2  V6 dea ε1e-5 롤아웃이 끝나면     -> V9 mag_da
#
# 기다릴 대상은 pid 로 받는다. pgrep 패턴을 쓰면 이 스크립트 자신의 커맨드라인이
# 패턴에 걸려 자기를 죽이는 사고가 난다.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
ts() { echo "[$(date '+%F %T')] $*"; }
wait_pid() { while kill -0 "$1" 2>/dev/null; do sleep 60; done; }

GPU1_WAIT=${GPU1_WAIT:?GPU1 을 쓰고 있는 pid}
GPU2_WAIT=${GPU2_WAIT:?GPU2 를 쓰고 있는 pid}

( wait_pid "${GPU1_WAIT}"; ts "GPU1 해제 — V9 mse_da 시작"
  exec env GPU=1 FORM=mse_da ENTRY=L2_V9.py VER=v9 CKROOT=CLARE_V9 \
      bash CLARE_V8_queue.sh
) > logs/launch_v9_gpu1.log 2>&1 &

( wait_pid "${GPU2_WAIT}"; ts "GPU2 해제 — V9 mag_da 시작"
  exec env GPU=2 FORM=mag_da ENTRY=L2_V9.py VER=v9 CKROOT=CLARE_V9 \
      bash CLARE_V8_queue.sh
) > logs/launch_v9_gpu2.log 2>&1 &

ts "V9 2 건 예약 — GPU1<-pid ${GPU1_WAIT} (mse_da), GPU2<-pid ${GPU2_WAIT} (mag_da)"
wait
