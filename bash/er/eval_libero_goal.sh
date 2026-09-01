#!/usr/bin/env bash
#
# ER 체크포인트 성능 평가 — libero_goal
#
# 학습(bash/er/ER_libero_goal.sh)은 EVAL_FREQ=0으로 시뮬레이터 평가를 끄고 돈다.
# 성능은 학습이 끝난 뒤 이 스크립트로 잰다. 이유는 eval_common.sh 상단 참조
# (평가는 LIBERO 환경을 eval.batch_size개 동시에 띄우므로 학습과 GPU를 다툰다).
#
# 사용법
#   bash bash/er/eval_libero_goal.sh
#   BS_EVAL=50 bash bash/er/eval_libero_goal.sh          # 카드가 넉넉하면 더 빠르게
#   N_EVAL=20  bash bash/er/eval_libero_goal.sh          # 빠른 예비 확인
#   REDO=1     bash bash/er/eval_libero_goal.sh          # 끝난 조합도 다시
#
# 서버/GPU를 나눌 때 — 결과는 조합마다 별도 디렉터리라 나중에 합쳐도 안전하다.
#   서버A: STAGES="0 1 2 3 4" bash bash/er/eval_libero_goal.sh
#   서버B: STAGES="5 6 7 8 9" bash bash/er/eval_libero_goal.sh
#   (특정 태스크만: TASKS="0 3" — 스테이지보다 나중 태스크는 자동 제외)

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-$CUDA_VISIBLE_DEVICES}

BENCH_NAME=libero_goal
NUM_TASKS=${NUM_TASKS:-10}
SEED=${SEED:-42}

# ER_libero_goal.sh 가 쓴 체크포인트 경로. 학습 스크립트의 --output_dir 와 같아야 한다.
CKPT_BASE=${CKPT_BASE:-./outputs/libero_goal/er}

declare -a STAGE_CKPT TASK_BENCH TASK_HANDLE TASK_REPO
for i in $(seq 0 $((NUM_TASKS - 1))); do
    STAGE_CKPT[$i]="${CKPT_BASE}/dit_flow_mt_cl_seed_${SEED}_libero_goal_task_${i}_er/checkpoints/last/pretrained_model"
    TASK_BENCH[$i]="libero_goal"
    TASK_HANDLE[$i]="Libero_Goal_Task_${i}"
    TASK_REPO[$i]="continuallearning/libero_goal_image_task_${i}"
done

source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"
