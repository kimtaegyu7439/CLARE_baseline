#!/usr/bin/env bash
#
# ER — 임의 LIBERO 스위트의 태스크 0..NUM_TASKS-1 순차 학습.
#
# bash/er/ER_task0123.sh 의 스위트 일반화판이다. 원본은 libero_spatial 이
# DATASET_PREFIX / ENV_TASK_PREFIX / BUFFER_PREFIX / OUT_ROOT 에 박혀 있어
# 다른 스위트에 못 쓴다. 원본은 건드리지 않고 이 파일을 새로 둔다.
#
# 조건은 원본과 문자 그대로 같다:
#   STEPS=5000, 총 배치 32 (현재 24 + 버퍼 8), seed=42,
#   태스크당 50 에피소드 중 뒤 5 개는 hold-out, 과거 태스크당 버퍼 5 에피소드.
#
# 태스크 0 은 버퍼가 없어 ER = 순차 파인튜닝이다. libero_spatial 에는 E0 의
# seq(lam0) task_0 이 있어 그걸 symlink 했지만, 다른 스위트에는 없으므로
# train.py 로 PRETRAIN_PATH 에서 새로 학습한다 — K1 의 태스크 0 과 같은 출발점이다.
#
# 사용법
#   SUITE=libero_goal NUM_TASKS=10 bash bash/er/ER_suite.sh
#   SUITE=libero_object NUM_TASKS=10 OUT_ROOT=./outputs/ER/libero_object/seed42 bash ...
#
# 끝난 스테이지는 out_dir/.done 으로 표시되고 그 스테이지만 건너뛴다.

set -uo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES}}

source "$(dirname "${BASH_SOURCE[0]}")/er_env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ER_PY=./lerobot_lsy/src/lerobot/scripts/er.py
TRAIN_PY=./lerobot_lsy/src/lerobot/scripts/train.py
BUFFER_PY=./lerobot_lsy/src/lerobot/scripts/util/create_er_dataset.py
PYTHON=${PYTHON:-python}

SUITE=${SUITE:?SUITE 를 지정하라 (libero_spatial / libero_goal / libero_object / libero_10)}
SEED=${SEED:-42}
NUM_TASKS=${NUM_TASKS:-10}
STEPS=${STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-24}
REPLAY_BATCH_SIZE=${REPLAY_BATCH_SIZE:-8}
BATCH_SIZE_FIRST=${BATCH_SIZE_FIRST:-32}
NUM_WORKERS=${NUM_WORKERS:-6}
REPLAY_NUM_WORKERS=${REPLAY_NUM_WORKERS:-3}
LOG_FREQ=${LOG_FREQ:-100}
HOLDOUT_EP=${HOLDOUT_EP:-5}
BUFFER_EP=${BUFFER_EP:-5}
TOTAL_EP=${TOTAL_EP:-50}

# "libero_goal" -> "Libero_Goal_Task_"  (B1.suite_prefixes 와 같은 규칙)
ENV_TASK_PREFIX=$(${PYTHON} -c "print('_'.join(w.capitalize() for w in '${SUITE}'.split('_')) + '_Task_')")
DATASET_PREFIX="continuallearning/${SUITE}_image_task_"
BUFFER_PREFIX="er_buffer/${SUITE}_seed${SEED}_ep${BUFFER_EP}_tasks_0_"
OUT_ROOT=${OUT_ROOT:-./outputs/ER/${SUITE}/seed${SEED}}
mkdir -p "${OUT_ROOT}"

EPISODES=$(${PYTHON} -c "print('[' + ','.join(str(i) for i in range(${TOTAL_EP} - ${HOLDOUT_EP})) + ']')")

echo "[ER] suite=${SUITE}  tasks=0..$((NUM_TASKS-1))  steps=${STEPS}  batch=${BATCH_SIZE}+${REPLAY_BATCH_SIZE}"
echo "[ER] dataset=${DATASET_PREFIX}*  env=${ENV_TASK_PREFIX}*  out=${OUT_ROOT}"

build_buffer() {   # build_buffer <k> -> 태스크 0..k-1 버퍼 repo_id
    local k=$1 repo_id="${BUFFER_PREFIX}$((k - 1))"
    local dir="${HF_LEROBOT_HOME}/${repo_id}"
    if [ -f "${dir}/meta/info.json" ]; then echo "${repo_id}"; return 0; fi
    local ids="" j
    for j in $(seq 0 $((k - 1))); do
        [ -n "${ids}" ] && ids="${ids},"
        ids="${ids}${DATASET_PREFIX}${j}"
    done
    echo "[ER] building replay buffer ${repo_id}  (${BUFFER_EP} ep/task, holdout ${HOLDOUT_EP})" >&2
    rm -rf "${dir}"
    ${PYTHON} "${BUFFER_PY}" \
        --repo_ids="${ids}" \
        --num_episodes="${BUFFER_EP}" \
        --merged_repo_id="${repo_id}" \
        --holdout_episodes="${HOLDOUT_EP}" \
        --seed="${SEED}" >&2 || return 1
    echo "${repo_id}"
}

for k in $(seq 0 $((NUM_TASKS - 1))); do
    out_dir="${OUT_ROOT}/task_${k}"
    prev_dir="${OUT_ROOT}/task_$((k - 1))"

    if [ -f "${out_dir}/.done" ] || [ -L "${out_dir}" ]; then
        echo "[ER] skip (done) ${out_dir}"; continue
    fi
    if [ -d "${out_dir}" ]; then
        if [ "${REDO_INCOMPLETE:-1}" = "1" ]; then
            echo "[ER] incomplete stage -> removing and redoing: ${out_dir}"; rm -rf "${out_dir}"
        else
            echo "[ER] incomplete stage left as-is: ${out_dir}"; exit 1
        fi
    fi

    if [ "${k}" -eq 0 ]; then
        echo ""; echo "══ [ER] ${SUITE} task=0 (버퍼 없음, 순차 파인튜닝)  init=${PRETRAIN_PATH}"
        "${PYTHON}" "${TRAIN_PY}" \
            --seed="${SEED}" --job_name="ER_${SUITE}_task_0" --output_dir="${out_dir}" \
            --dataset.repo_id="${DATASET_PREFIX}0" --dataset.episodes="${EPISODES}" \
            --policy.path="${PRETRAIN_PATH}" --policy.push_to_hub=false \
            --batch_size="${BATCH_SIZE_FIRST}" --num_workers="${NUM_WORKERS}" \
            --steps="${STEPS}" --log_freq="${LOG_FREQ}" --save_freq="${STEPS}" --eval_freq=0 \
            --env.type=libero --env.benchmark="${SUITE}" --env.task="${ENV_TASK_PREFIX}0" \
            --wandb.enable=false || { echo "[ER] FAILED task=0"; exit 1; }
        touch "${out_dir}/.done"; continue
    fi

    buffer_repo=$(build_buffer "${k}") || { echo "[ER] FAILED buffer for task=${k}"; exit 1; }
    echo ""; echo "══ [ER] ${SUITE} task=${k}  init=${prev_dir}/checkpoints/last/pretrained_model"
    echo "        replay=${buffer_repo}  (batch ${BATCH_SIZE}+${REPLAY_BATCH_SIZE})"

    "${PYTHON}" "${ER_PY}" \
        --seed="${SEED}" --job_name="ER_${SUITE}_task_${k}" --output_dir="${out_dir}" \
        --dataset.repo_id="${DATASET_PREFIX}${k}" --dataset.episodes="${EPISODES}" \
        --replay_dataset.repo_id="${buffer_repo}" \
        --policy.path="${prev_dir}/checkpoints/last/pretrained_model" \
        --policy.push_to_hub=false \
        --batch_size="${BATCH_SIZE}" --num_workers="${NUM_WORKERS}" \
        --replay_batch_size="${REPLAY_BATCH_SIZE}" \
        --replay_num_workers="${REPLAY_NUM_WORKERS}" \
        --steps="${STEPS}" --log_freq="${LOG_FREQ}" --save_freq="${STEPS}" --eval_freq=0 \
        --env.type=libero --env.benchmark="${SUITE}" --env.task="${ENV_TASK_PREFIX}${k}" \
        --wandb.enable=false || { echo "[ER] FAILED task=${k}"; exit 1; }

    touch "${out_dir}/.done"
done

echo ""; echo "[ER] ${SUITE} done. tree=${OUT_ROOT}/task_{0..$((NUM_TASKS-1))}/checkpoints/last/pretrained_model"
