#!/usr/bin/env bash
# task 0 을 20000 스텝 학습하되 중간 체크포인트를 남긴다.
# "학습이 길어질수록 현재 태스크가 속도장을 채운다"를 스텝 축으로 재기 위한 재료.
# 기존 outputs/ 는 건드리지 않는다. -> outputs/B_fill/
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=${1:-3}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}

STEPS=${STEPS:-20000}
SAVE=${SAVE:-1000}
OUT=${OUT:-./outputs/B_fill/task0_s${STEPS}}
EPS="[$(python -c "print(','.join(str(i) for i in range(45)))")]"

python ./lerobot_lsy/src/lerobot/scripts/train.py \
    --seed=42 --job_name="B_fill_task0" --output_dir="${OUT}" \
    --dataset.repo_id=continuallearning/libero_spatial_image_task_0 \
    --dataset.episodes="${EPS}" \
    --policy.path="${PRETRAIN_PATH}" --policy.push_to_hub=false \
    --batch_size=32 --num_workers=8 \
    --steps="${STEPS}" --save_freq="${SAVE}" --log_freq=200 --eval_freq=0 \
    --env.type=libero --env.benchmark=libero_spatial --env.task=Libero_Spatial_Task_0 \
    --wandb.enable=false
