#!/usr/bin/env bash
# 학습된 체크포인트로 롤아웃만 돌린다. 행을 나눠 여러 GPU 가 같은 --out 을 채운다.
#   GPU=2 ROWS=0-6 CK=... OUT=... bash CLARE_rollout_queue.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"; export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
GPU=${GPU:?}; ROWS=${ROWS:?}; CK=${CK:?}; OUT=${OUT:?}
N_EVAL=${N_EVAL:-20}; SUITE=${SUITE:-libero_goal}
TAG=$(basename "${OUT}")_rows${ROWS}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "[$(date '+%F %T')] ▶ ${TAG}  GPU ${GPU}" | tee -a logs/CLARE_rollout.queue
python -u CLARE_rollout_only.py --ckpt_dir "${CK}" --out "${OUT}" \
    --suite "${SUITE}" --rows "${ROWS}" \
    --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" >> "logs/rollout_${TAG}.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] ◀ ${TAG} exit=${rc}" | tee -a logs/CLARE_rollout.queue
