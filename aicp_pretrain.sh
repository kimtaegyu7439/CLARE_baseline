#!/usr/bin/env bash
#
# AICP base phase — CLARE 정책 from scratch 사전학습 (LIBERO-90, v3.0 데이터셋).
#
#   bash aicp_pretrain.sh              GPU 0, seed 7, 200k step
#   bash aicp_pretrain.sh 0 --smoke    20 스텝 연기시험
#
# 손실은 정책 기본 flow-matching MSE 하나뿐이다. 앵커/패널티 없음.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

GPU=${1:-0}; shift || true
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

OUT=${OUT:-/home/sa090180/Models/aicp_clare_pretrain}
LOG=${LOG:-results/aicp_pretrain}
mkdir -p "${LOG}"

echo "[$(date '+%F %T')] AICP base phase  gpu=${GPU}  out=${OUT}"
nohup python -u aicp_pretrain.py --out "${OUT}" "$@" \
    > "${LOG}/train.log" 2>&1 &
echo $! > "${LOG}/run.pid"
echo "  pid $(cat ${LOG}/run.pid)   로그 ${LOG}/train.log"
echo "진행 확인:  tail -f ${LOG}/train.log"
