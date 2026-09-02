#!/usr/bin/env bash
#
# L2_codebook — L2 + (s,o) 결합 코드북 앵커 샘플러.
#
#   bash l2_codebook.sh                          기본: GPU 2, libero_spatial 10 task
#   bash l2_codebook.sh 2 L2_codebook 10 libero_spatial
#   K=64 H_SCALE=0.7 bash l2_codebook.sh ...
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

GPU=${1:-2}; NAME=${2:-L2_codebook}; NTASK=${3:-10}; SUITE=${4:-libero_spatial}
for _ in 1 2 3 4; do [ $# -gt 0 ] && shift; done

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
NUM_WORKERS=${NUM_WORKERS:-6}
K=${K:-96}; N_PAIRS=${N_PAIRS:-8000}; H_SCALE=${H_SCALE:-1.0}; XT_MODE=${XT_MODE:-teacher}

OUT="results/${NAME}"; mkdir -p "${OUT}"
echo "[$(date '+%F %T')] L2_codebook 본 실행  gpu=${GPU}  out=${OUT}  K=${K}  n_pairs=${N_PAIRS}  h_scale=${H_SCALE}"
nohup python -u l2_codebook.py --out "${OUT}" --chunk_backward \
    --codebook_k "${K}" --n_pairs "${N_PAIRS}" --h_scale "${H_SCALE}" --xt_mode "${XT_MODE}" "$@" \
    --passthru --num_tasks "${NTASK}" --suite "${SUITE}" --num_workers "${NUM_WORKERS}" \
    > "${OUT}/train.log" 2>&1 &
echo $! > "${OUT}/run.pid"
echo "  pid $(cat "${OUT}/run.pid")   로그 ${OUT}/train.log"
