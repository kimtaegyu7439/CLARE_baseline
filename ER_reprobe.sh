#!/usr/bin/env bash
#
# ER 5k 체크포인트를 재학습 없이 롤아웃만 다시 돌린다.
#
# 학습은 seed 42 로 이미 끝나 있고 가중치는 고정이다. 바뀌는 것은 --seed 뿐이며
# E0.py:264 에서 그 값이 eval_policy(start_seed=cfg.seed) 로 들어간다. 즉 시드를
# 바꾸면 롤아웃 초기 상태와 ODE 노이즈만 달라진다 -> 순수한 롤아웃 표본 변동.
#
# 사용법:  bash ER_reprobe.sh <SEED> <GPU>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"

if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

SEED=${1:?SEED 를 달라}
GPU=${2:-0}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}

CKPT_ROOT=${CKPT_ROOT:-./outputs/ER/libero_spatial/seed42}   # 학습된 5k 가중치 (읽기 전용)
RES_DIR=${RES_DIR:-${HERE}/results/ER_reprobe/seed${SEED}}
WORK=${WORK:-./outputs/ER_reprobe/seed${SEED}}               # 로그용. 체크포인트 디렉토리와 분리한다.
RESULTS=${RES_DIR}/er_results.jsonl
LOG=${RES_DIR}/run.log
NUM_TASKS=${NUM_TASKS:-4}
mkdir -p "${RES_DIR}" "${WORK}"
: > "${RESULTS}"

DATASET_PREFIX=continuallearning/libero_spatial_image_task_
ENV_TASK_PREFIX=Libero_Spatial_Task_
PYTHON=${PYTHON:-python}

echo "[$(date '+%F %T')] ER reprobe seed=${SEED} gpu=${GPU} ckpt=${CKPT_ROOT}" | tee -a "${LOG}"

for k in $(seq 0 $((NUM_TASKS - 1))); do
    ckpt="${CKPT_ROOT}/task_${k}/checkpoints/last/pretrained_model"
    [ -d "${ckpt}" ] || { echo "SKIP ${k}: 없음" | tee -a "${LOG}"; continue; }
    echo "[$(date '+%F %T')] stage ${k} (태스크 0..${k}, 칸당 20 롤아웃)" | tee -a "${LOG}"
    "${PYTHON}" ./lerobot_lsy/src/lerobot/scripts/E0.py \
        --seed="${SEED}" --job_name="ER_reprobe_s${SEED}_t${k}" \
        --output_dir="${WORK}/task_${k}" \
        --dataset.repo_id="${DATASET_PREFIX}${k}" \
        --policy.path="${ckpt}" --policy.push_to_hub=false \
        --reprobe=true --eval_freq=0 \
        --env.type=libero --env.benchmark=libero_spatial \
        --env.task="${ENV_TASK_PREFIX}${k}" \
        --ewc_lambda=0 --run_tag="er" --current_task="${k}" \
        --task_ids="$(seq -s, 0 "${k}")" \
        --dataset_prefix="${DATASET_PREFIX}" --env_task_prefix="${ENV_TASK_PREFIX}" \
        --results_path="${RESULTS}" \
        --holdout_episodes=5 --probe_batches=16 \
        --probe_sr=true --probe_n_episodes=20 --probe_eval_batch_size=20 \
        --wandb.enable=false >>"${LOG}" 2>&1 || echo "FAILED stage ${k}" | tee -a "${LOG}"
done
echo "[$(date '+%F %T')] 완료 -> ${RESULTS}" | tee -a "${LOG}"
