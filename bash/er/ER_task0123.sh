#!/usr/bin/env bash
#
# ER — libero_spatial 태스크 0..3 순차 학습 (R1에 ER 팔을 붙이기 위한 실행)
#
# E0가 만든 seq(λ=0) / ewc(λ=100) / frozen(λ=inf) 팔과 **같은 조건**에서 ER을 돌린다.
# R1이 읽는 트리 모양을 그대로 맞춘다:  <root>/task_{k}/checkpoints/last/pretrained_model
#
# ER (Chaudhry et al., "On Tiny Episodic Memories in Continual Learning")
#   매 스텝 현재 태스크 배치 + 메모리 버퍼 배치를 뽑아 concat하고 한 번의 gradient step.
#   scripts/er.py가 그 방식 그대로다.
#
# E0와 맞춰 둔 것 (안 맞추면 그림 C의 x축이 팔마다 다른 자로 재진다)
#   STEPS=5000, 총 배치 32, seed=42, 태스크당 뒤 HOLDOUT_EP개 에피소드는 학습에서 제외.
#   총 배치 32 = BATCH_SIZE(현재 태스크) + REPLAY_BATCH_SIZE(버퍼). E0 팔은 32 전부가
#   현재 태스크였으므로, ER은 그중 REPLAY_BATCH_SIZE칸을 과거 태스크에 내주는 셈이다.
#
# ★ 버퍼를 여기서 직접 만드는 이유
#   HF에 있는 continuallearning/libero_spatial_image_task_0_er_new 등은 50개 에피소드
#   전체에서 무작위로 뽑혀 있어 held-out(45..49)이 섞여 있다(실측: 그 버퍼의 5개 중
#   1개가 원본 ep47). 그대로 쓰면 ER만 held-out loss를 자기가 학습한 데이터에서 재게
#   되어 그림 C의 왼쪽 패널이 ER 쪽으로 기운다. 그래서 create_er_dataset.py에
#   --holdout_episodes를 주고 학습 split(0..44)에서만 뽑아 로컬로 만든다.
#
# 태스크 0은 버퍼가 비어 있어 ER = 순차 파인튜닝이다. 그래서 E0의 seq(lam0) task_0
# 체크포인트를 그대로 가리킨다(SEQ_TASK0). 두 팔의 출발점이 문자 그대로 같아져서
# stage>=1의 차이가 온전히 ER 때문이 된다. lam0가 없으면 train.py로 직접 학습한다.
#
# 사용법
#   bash bash/er/ER_task0123.sh
#   NUM_TASKS=4 STEPS=5000 bash bash/er/ER_task0123.sh
#   BUFFER_EP=10 bash bash/er/ER_task0123.sh        # 과거 태스크당 버퍼 에피소드 수
#   REDO_INCOMPLETE=0 bash bash/er/ER_task0123.sh   # 미완료 스테이지를 지우지 않고 멈춤
#
# 끝난 스테이지는 out_dir/.done 으로 표시된다. 이 파일이 있는 스테이지만 건너뛴다.

set -uo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}

# HF_LEROBOT_HOME / HF_HUB_CACHE / PRETRAIN_PATH 를 세팅한다.
source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ER_PY=./lerobot_lsy/src/lerobot/scripts/er.py
TRAIN_PY=./lerobot_lsy/src/lerobot/scripts/train.py
BUFFER_PY=./lerobot_lsy/src/lerobot/scripts/util/create_er_dataset.py
PYTHON=${PYTHON:-python}   # conda clare 환경이 활성화돼 있다고 가정. 아니면 PYTHON=... 로 지정.

# ── 조절할 것들 ───────────────────────────────────────────────────────────────
SEED=${SEED:-42}
NUM_TASKS=${NUM_TASKS:-4}                # 태스크 0..NUM_TASKS-1
STEPS=${STEPS:-5000}                     # 태스크당 학습 스텝 (E0와 동일)
BATCH_SIZE=${BATCH_SIZE:-24}             # 현재 태스크 배치
REPLAY_BATCH_SIZE=${REPLAY_BATCH_SIZE:-8}  # 버퍼 배치. 24+8=32 = E0 팔의 배치
BATCH_SIZE_FIRST=${BATCH_SIZE_FIRST:-32} # 버퍼가 없는 태스크 0
NUM_WORKERS=${NUM_WORKERS:-8}
REPLAY_NUM_WORKERS=${REPLAY_NUM_WORKERS:-4}
LOG_FREQ=${LOG_FREQ:-100}

HOLDOUT_EP=${HOLDOUT_EP:-5}              # 태스크당 50 에피소드 중 뒤 5개는 학습에서 제외
BUFFER_EP=${BUFFER_EP:-5}                # 과거 태스크당 버퍼에 넣을 에피소드 수

OUT_ROOT=${OUT_ROOT:-./outputs/ER/libero_spatial/seed${SEED}}   # R1.sh의 ER_ROOT 기본값
E0_ROOT=${E0_ROOT:-./outputs/E0/libero_spatial/seed_42}
SEQ_TASK0=${SEQ_TASK0:-${E0_ROOT}/lam0/task_0}   # ER의 task_0 = seq의 task_0
mkdir -p "${OUT_ROOT}"

DATASET_PREFIX=continuallearning/libero_spatial_image_task_
ENV_TASK_PREFIX=Libero_Spatial_Task_
BUFFER_PREFIX=er_buffer/libero_spatial_seed${SEED}_ep${BUFFER_EP}_tasks_0_

# 학습 대상 에피소드 목록 "[0,1,...,44]". E0는 샘플러로 갈랐지만 er.py에는 그 통로가
# 없으므로 --dataset.episodes로 준다. 0에서 시작하는 연속 구간이라 LeRobotDataset의
# 재인덱싱 문제(E0.py episode_sampler 주석)에 걸리지 않는다 — 앞을 자를 때만 안전하다.
train_episodes() {   # train_episodes <total>
    ${PYTHON} -c "print('[' + ','.join(str(i) for i in range($1 - ${HOLDOUT_EP})) + ']')"
}
TOTAL_EP=${TOTAL_EP:-50}
EPISODES=$(train_episodes "${TOTAL_EP}")

# ── 버퍼: 태스크 0..k-1에서 학습 split만 골라 하나로 합친다 ───────────────────
# 이미 만들어져 있으면(=meta/info.json 존재) 다시 만들지 않는다. 무작위 추출이라
# 다시 만들면 시드가 같아도 태스크 목록이 바뀌면 구성이 바뀐다.
build_buffer() {   # build_buffer <k>  -> 태스크 0..k-1 을 담은 버퍼 repo_id를 stdout으로
    local k=$1 repo_id="${BUFFER_PREFIX}$((k - 1))"
    local dir="${HF_LEROBOT_HOME}/${repo_id}"
    if [ -f "${dir}/meta/info.json" ]; then
        echo "${repo_id}"
        return 0
    fi
    local ids="" j
    for j in $(seq 0 $((k - 1))); do
        [ -n "${ids}" ] && ids="${ids},"
        ids="${ids}${DATASET_PREFIX}${j}"
    done
    echo "[ER] building replay buffer ${repo_id}  <- ${ids}  (${BUFFER_EP} ep/task, holdout ${HOLDOUT_EP})" >&2
    rm -rf "${dir}"
    ${PYTHON} "${BUFFER_PY}" \
        --repo_ids="${ids}" \
        --num_episodes="${BUFFER_EP}" \
        --merged_repo_id="${repo_id}" \
        --holdout_episodes="${HOLDOUT_EP}" \
        --seed="${SEED}" >&2 || return 1
    echo "${repo_id}"
}

# ── 태스크 0..NUM_TASKS-1 순차 학습 ───────────────────────────────────────────
for k in $(seq 0 $((NUM_TASKS - 1))); do
    out_dir="${OUT_ROOT}/task_${k}"
    prev_dir="${OUT_ROOT}/task_$((k - 1))"

    # 끝까지 간 스테이지만 건너뛴다. 디렉터리 존재만 보면 중간에 죽은 스테이지가
    # 영원히 재실행되지 않는다(E0.sh와 같은 이유).
    if [ -f "${out_dir}/.done" ] || [ -L "${out_dir}" ]; then
        echo "[ER] skip (done) ${out_dir}"
        continue
    fi
    if [ -d "${out_dir}" ]; then
        if [ "${REDO_INCOMPLETE:-1}" = "1" ]; then
            echo "[ER] incomplete stage -> removing and redoing: ${out_dir}"
            rm -rf "${out_dir}"
        else
            echo "[ER] incomplete stage left as-is (REDO_INCOMPLETE=0): ${out_dir}"
            exit 1
        fi
    fi

    # ── 태스크 0: 버퍼가 없으므로 ER = 순차 파인튜닝 ──────────────────────────
    if [ "${k}" -eq 0 ]; then
        if [ -d "${SEQ_TASK0}/checkpoints/last/pretrained_model" ]; then
            echo "[ER] task 0 = seq(lam0) task 0 를 그대로 쓴다 -> symlink ${out_dir}"
            ln -s "$(cd "${SEQ_TASK0}" && pwd)" "${out_dir}"
            continue
        fi
        echo ""
        echo "══ [ER] task=0 (버퍼 없음, 순차 파인튜닝)  init=${PRETRAIN_PATH}"
        "${PYTHON}" "${TRAIN_PY}" \
            --seed="${SEED}" \
            --job_name="ER_task_0" \
            --output_dir="${out_dir}" \
            --dataset.repo_id="${DATASET_PREFIX}0" \
            --dataset.episodes="${EPISODES}" \
            --policy.path="${PRETRAIN_PATH}" \
            --policy.push_to_hub=false \
            --batch_size="${BATCH_SIZE_FIRST}" \
            --num_workers="${NUM_WORKERS}" \
            --steps="${STEPS}" \
            --log_freq="${LOG_FREQ}" \
            --save_freq="${STEPS}" \
            --eval_freq=0 \
            --env.type=libero \
            --env.benchmark=libero_spatial \
            --env.task="${ENV_TASK_PREFIX}0" \
            --wandb.enable=false || { echo "[ER] FAILED task=0"; exit 1; }
        touch "${out_dir}/.done"
        continue
    fi

    # ── 태스크 1..: 버퍼를 만들고 ER로 학습 ───────────────────────────────────
    buffer_repo=$(build_buffer "${k}") || { echo "[ER] FAILED buffer for task=${k}"; exit 1; }

    echo ""
    echo "══ [ER] task=${k}  init=${prev_dir}/checkpoints/last/pretrained_model"
    echo "        replay=${buffer_repo}  (batch ${BATCH_SIZE}+${REPLAY_BATCH_SIZE})"

    "${PYTHON}" "${ER_PY}" \
        --seed="${SEED}" \
        --job_name="ER_task_${k}" \
        --output_dir="${out_dir}" \
        --dataset.repo_id="${DATASET_PREFIX}${k}" \
        --dataset.episodes="${EPISODES}" \
        --replay_dataset.repo_id="${buffer_repo}" \
        --policy.path="${prev_dir}/checkpoints/last/pretrained_model" \
        --policy.push_to_hub=false \
        --batch_size="${BATCH_SIZE}" \
        --num_workers="${NUM_WORKERS}" \
        --replay_batch_size="${REPLAY_BATCH_SIZE}" \
        --replay_num_workers="${REPLAY_NUM_WORKERS}" \
        --steps="${STEPS}" \
        --log_freq="${LOG_FREQ}" \
        --save_freq="${STEPS}" \
        --eval_freq=0 \
        --env.type=libero \
        --env.benchmark=libero_spatial \
        --env.task="${ENV_TASK_PREFIX}${k}" \
        --wandb.enable=false || { echo "[ER] FAILED task=${k}"; exit 1; }

    touch "${out_dir}/.done"
done

echo ""
echo "[ER] done.  tree=${OUT_ROOT}/task_{0..$((NUM_TASKS - 1))}/checkpoints/last/pretrained_model"
echo "[ER] R1에 붙이려면:  ER_ROOT=${OUT_ROOT} bash bash/E0/R1.sh"
