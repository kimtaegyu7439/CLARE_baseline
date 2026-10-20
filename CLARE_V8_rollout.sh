#!/usr/bin/env bash
# V8 롤아웃만. 학습은 2026-10-14 22:44/22:58 에 10 stage 전부 끝났는데,
# 큐 스크립트를 **실행 중에 수정**해서(bash 가 바이트 오프셋으로 읽는다)
# line 107 구문 오류로 죽어 롤아웃 단계에 못 들어갔다. 그 부분만 다시 돈다.
#
#   GPU=0 FORM=mse_da bash CLARE_V8_rollout.sh
#   GPU=3 FORM=mag_da bash CLARE_V8_rollout.sh
#
# ★ 이 파일은 돌고 있는 동안 고치지 마라. 고쳐야 하면 복사해서 새 이름으로 써라.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

GPU=${GPU:?GPU}
FORM=${FORM:?FORM (mse_da|mag_da)}
VER=${VER:-v8}
CKROOT=${CKROOT:-CLARE_V8}
SEED=${SEED:-7}
SUITE=${SUITE:-libero_goal}
N_EVAL=${N_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-7}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

TAG="clare_${VER}_${FORM}_goal_seed${SEED}"
OUT="results/${TAG}"
LOG="logs/${TAG}.log"
CKDIR=$(ls -d "${HERE}/outputs/${CKROOT}/${VER}_${FORM}_seed${SEED}"/*_ours 2>/dev/null | head -1)
[ -z "${CKDIR}" ] && { echo "체크포인트 폴더 없음"; exit 1; }

echo "[$(date '+%F %T')] ▶ ${TAG} 롤아웃 ${N_EVAL}회  GPU ${GPU}" | tee -a logs/CLARE_V8.rollout
python -u CLARE_rollout_only.py --ckpt_dir "${CKDIR}" --out "${OUT}" \
    --suite "${SUITE}" --rows all \
    --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
    --eval_seed "${EVAL_SEED}" >> "${LOG}" 2>&1
rc=$?
echo "[$(date '+%F %T')] ◀ ${TAG} 롤아웃 exit=${rc}" | tee -a logs/CLARE_V8.rollout
python -u CLARE_V6_report.py > /dev/null 2>&1
