#!/usr/bin/env bash
# PART 0 — GPU 0 순차 큐: K3 spatial 10task -> K1 --marginal zscore 10task
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
ts() { echo "[$(date '+%F %T')] $*"; }

ts "K3 spatial 10task 시작"
python -u k3.py --out results/K3_spatial_10task --chunk_backward --suite libero_spatial \
    --stats_workers 4 --passthru --num_tasks 10 --suite libero_spatial --num_workers 6 \
    > results/K3_spatial_10task/train.log 2>&1
ts "K3 종료 rc=$?"

ts "K1 --marginal zscore 10task 시작"
python -u k1.py --out results/K1_zscore_10task --chunk_backward --suite libero_spatial \
    --stats_workers 4 --marginal zscore \
    --passthru --num_tasks 10 --suite libero_spatial --num_workers 6 \
    > results/K1_zscore_10task/train.log 2>&1
ts "K1 zscore 종료 rc=$?"
