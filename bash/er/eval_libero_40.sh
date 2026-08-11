#!/usr/bin/env bash
#
# ER 체크포인트 성능 평가 — libero_40
#
# libero_40은 단일 벤치마크가 아니라 **네 벤치마크를 이어 붙인 40스테이지 시퀀스**다:
#
#   스테이지  0..9   libero_10       (Libero_10_Task_0..9)
#            10..19  libero_goal     (Libero_Goal_Task_0..9)
#            20..29  libero_spatial  (Libero_Spatial_Task_0..9)
#            30..39  libero_object   (Libero_Object_Task_0..9)
#
# 그래서 태스크마다 --env.benchmark / --env.task / --dataset.repo_id 가 다르고,
# 체크포인트 디렉터리 이름에도 벤치마크가 들어간다
# (dit_flow_mt_cl_seed_42_libero_40_libero_goal_task_3_er).
#
# 학습(bash/er/ER_libero_40.sh)은 EVAL_FREQ=0으로 시뮬레이터 평가를 끄고 돈다.
# 이유는 eval_common.sh 상단 참조.
#
# 사용법
#   bash bash/er/eval_libero_40.sh
#   REDO=1 bash bash/er/eval_libero_40.sh
#
# ★ 40스테이지 전부면 40×41/2 = 820회 평가다. 서버를 나눠라.
#   서버A: STAGES="$(seq 0 19)"  bash bash/er/eval_libero_40.sh
#   서버B: STAGES="$(seq 20 39)" bash bash/er/eval_libero_40.sh
#   빠른 확인만 하려면 N_EVAL=20 으로 낮춰라.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-$CUDA_VISIBLE_DEVICES}

BENCH_NAME=libero_40
NUM_TASKS=${NUM_TASKS:-40}
SEED=${SEED:-42}
CKPT_BASE=${CKPT_BASE:-./outputs/libero_40/er}

# 벤치마크 블록: 이름 / gym 핸들 접두사 / 각 10태스크.
BLOCK_BENCH=(libero_10      libero_goal        libero_spatial       libero_object)
BLOCK_HANDLE=(Libero_10_Task Libero_Goal_Task  Libero_Spatial_Task  Libero_Object_Task)

declare -a STAGE_CKPT TASK_BENCH TASK_HANDLE TASK_REPO
for i in $(seq 0 $((NUM_TASKS - 1))); do
    b=$((i / 10))                       # 몇 번째 벤치마크 블록인가
    j=$((i % 10))                       # 그 안에서 몇 번째 태스크인가
    bench=${BLOCK_BENCH[$b]}
    STAGE_CKPT[$i]="${CKPT_BASE}/dit_flow_mt_cl_seed_${SEED}_libero_40_${bench}_task_${j}_er/checkpoints/last/pretrained_model"
    TASK_BENCH[$i]="${bench}"
    TASK_HANDLE[$i]="${BLOCK_HANDLE[$b]}_${j}"
    TASK_REPO[$i]="continuallearning/${bench}_image_task_${j}"
done

source "$(dirname "${BASH_SOURCE[0]}")/eval_common.sh"
