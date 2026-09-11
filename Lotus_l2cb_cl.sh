#!/usr/bin/env bash
# LOTUS 셋팅 lifelong (L2_codebook v3 앵커).
#   bash Lotus_l2cb_cl.sh <suite: goal|object|50> <gpu> [steps]
#   suite=50 은 base 25 task + 5 stage x 5 task = 50 task 다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
SUITE=${1:?suite (goal|object|50)}; GPU=${2:-0}; STEPS=${3:-10000}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
OUT="results/lotus_l2cb_${SUITE}"; mkdir -p "${OUT}"
echo "[$(date '+%F %T')] LOTUS lifelong  suite=${SUITE}  gpu=${GPU}  steps=${STEPS}  -> ${OUT}"
nohup python -u Lotus_l2cb_cl.py --suite "${SUITE}" --steps "${STEPS}" \
    --eval_seed 7 --num_workers "${NUM_WORKERS:-6}" --out "${OUT}" \
    > "${OUT}/train.log" 2>&1 &
echo $! > "${OUT}/run.pid"; echo "  pid $(cat ${OUT}/run.pid)   로그 ${OUT}/train.log"
