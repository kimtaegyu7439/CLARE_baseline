#!/usr/bin/env bash
#
# E0 — EWC λ 스윕: MSE vs SR 비교 (libero_spatial)
#
# λ마다 태스크 0..NUM_TASKS-1 을 순차 학습하고, 각 태스크가 끝날 때마다
# 지금까지 본 모든 태스크의 held-out MSE와 SR을 재서 하나의 JSONL에 쌓는다.
# 마지막에 MSE / SR 두 패널 그림을 그린다.
#
#   λ=0    순차 파인튜닝 (하한)
#   λ=10, 100, 1000
#   λ=inf  파라미터 완전 동결 (상한)
#
# 사용법
#   bash bash/E0/E0.sh
#   NUM_TASKS=6 bash bash/E0/E0.sh            # 태스크 개수 조절
#   STEPS=2000 bash bash/E0/E0.sh             # 더 빠르게
#   LAMBDAS="0 100 inf" bash bash/E0/E0.sh    # λ 목록 조절
#   REPROBE=1 bash bash/E0/E0.sh              # 재학습 없이 프로브만 다시 (체크포인트 재사용)
#   PLOT_ONLY=1 bash bash/E0/E0.sh            # 이미 쌓인 JSONL로 그림만
#   REDO_INCOMPLETE=0 bash bash/E0/E0.sh      # 미완료 스테이지를 지우지 않고 멈춤
#
# 끝난 스테이지는 out_dir/.done 으로 표시된다. 이 파일이 있는 스테이지만 건너뛴다.

set -uo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}

# HF_LEROBOT_HOME / HF_HUB_CACHE / PRETRAIN_PATH 를 세팅한다.
source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

E0_PY=./lerobot_lsy/src/lerobot/scripts/E0.py
PYTHON=${PYTHON:-python}   # conda clare 환경이 활성화돼 있다고 가정. 아니면 PYTHON=... 로 지정.

# ── 조절할 것들 ───────────────────────────────────────────────────────────────
NUM_TASKS=${NUM_TASKS:-4}                # 태스크 0..NUM_TASKS-1
# LAMBDAS=${LAMBDAS:-"0 10 100 1000 inf"}
# 재실행이 필요한 팔만. 10 = 결과 0줄이라 새로, inf = 동결이 깨졌던 버그 대상.
# 0 / 100 / 1000 은 그대로 유효하므로 건드리지 않는다(넣으면 .done이 없어 지워지고 재학습된다).
LAMBDAS=${LAMBDAS:-"10 inf"}
SEED=${SEED:-42}
STEPS=${STEPS:-5000}                     # 태스크당 학습 스텝
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
LOG_FREQ=${LOG_FREQ:-100}

HOLDOUT_EP=${HOLDOUT_EP:-5}              # 태스크당 50 에피소드 중 뒤 5개는 학습에서 제외
PROBE_BATCHES=${PROBE_BATCHES:-16}       # held-out MSE를 평균 낼 배치 수
PROBE_SR=${PROBE_SR:-true}
PROBE_N_EP=${PROBE_N_EP:-20}             # SR 롤아웃 에피소드 수
PROBE_EVAL_BS=${PROBE_EVAL_BS:-20}

OUT_ROOT=${OUT_ROOT:-./outputs/E0/libero_spatial/seed_${SEED}}
RESULTS=${RESULTS:-${OUT_ROOT}/e0_results.jsonl}
FIGURE=${FIGURE:-${OUT_ROOT}/E0_mse_vs_sr.png}
mkdir -p "${OUT_ROOT}"

DATASET_PREFIX=continuallearning/libero_spatial_image_task_
ENV_TASK_PREFIX=Libero_Spatial_Task_

# ── λ 하나에 대해 태스크 0..NUM_TASKS-1 순차 학습 ─────────────────────────────
run_lambda() {
    local lam=$1 tag="lam$1"
    for k in $(seq 0 $((NUM_TASKS - 1))); do
        local out_dir="${OUT_ROOT}/${tag}/task_${k}"
        local prev_dir="${OUT_ROOT}/${tag}/task_$((k - 1))"

        # 끝까지 간 스테이지만 건너뛴다. E0.py가 프로브까지 마치고 .done을 남긴다.
        # ★ 디렉터리 존재만 보고 건너뛰면 중간에 죽은 스테이지가 영원히 재실행되지 않는다
        #   (λ=10이 task_0/task_1 디렉터리만 남기고 결과 0줄인 채로 굳었던 이유).
        if [ -f "${out_dir}/.done" ]; then
            echo "[E0] skip (done) ${out_dir}"
            continue
        fi
        # 미완료 잔해는 지우고 다시 돈다. 남겨두면 train.py가 FileExistsError로 죽는다.
        if [ -d "${out_dir}" ]; then
            if [ "${REDO_INCOMPLETE:-1}" = "1" ]; then
                echo "[E0] incomplete stage -> removing and redoing: ${out_dir}"
                rm -rf "${out_dir}"
            else
                echo "[E0] incomplete stage left as-is (REDO_INCOMPLETE=0): ${out_dir}"
                return 1
            fi
        fi

        # 첫 태스크는 사전학습 체크포인트, 이후는 직전 태스크 체크포인트에서 출발.
        local policy_path="${PRETRAIN_PATH}"
        [ "${k}" -gt 0 ] && policy_path="${prev_dir}/checkpoints/last/pretrained_model"

        # 직전 태스크가 남긴 Fisher+anchor를 이어받는다(첫 태스크는 없음 -> 페널티 0).
        local extra=()
        [ "${k}" -gt 0 ] && extra+=("--ewc_state_path=${prev_dir}/ewc_state.pt")

        echo ""
        echo "══ [E0] λ=${lam}  task=${k}  init=${policy_path}"

        "${PYTHON}" "${E0_PY}" \
            --seed="${SEED}" \
            --job_name="E0_${tag}_task_${k}" \
            --output_dir="${out_dir}" \
            --dataset.repo_id="${DATASET_PREFIX}${k}" \
            --policy.path="${policy_path}" \
            --policy.push_to_hub=false \
            --batch_size="${BATCH_SIZE}" \
            --num_workers="${NUM_WORKERS}" \
            --steps="${STEPS}" \
            --log_freq="${LOG_FREQ}" \
            --save_freq="${STEPS}" \
            --eval_freq=0 \
            --env.type=libero \
            --env.benchmark=libero_spatial \
            --env.task="${ENV_TASK_PREFIX}${k}" \
            --ewc_lambda="${lam}" \
            --run_tag="${lam}" \
            --current_task="${k}" \
            --task_ids="$(seq -s, 0 "${k}")" \
            --dataset_prefix="${DATASET_PREFIX}" \
            --env_task_prefix="${ENV_TASK_PREFIX}" \
            --results_path="${RESULTS}" \
            --holdout_episodes="${HOLDOUT_EP}" \
            --probe_batches="${PROBE_BATCHES}" \
            --probe_sr="${PROBE_SR}" \
            --probe_n_episodes="${PROBE_N_EP}" \
            --probe_eval_batch_size="${PROBE_EVAL_BS}" \
            --wandb.enable=True \
            "${extra[@]}" || { echo "[E0] FAILED λ=${lam} task=${k}"; return 1; }
    done
}

# ── 재학습 없이 프로브만 다시 돌기 ────────────────────────────────────────────
# 프로브 코드가 바뀌었을 때(시드 고정, SR 판정 수정 등) 쓴다. 체크포인트는 그대로 두고
# JSONL만 새로 만든다. 기존 JSONL은 append 전용이라 반드시 먼저 치워야 한다
# (안 그러면 옛 행과 새 행이 같은 키로 섞여 그림에서 평균돼 버린다).
#
# ★ 학습 목록(LAMBDAS)과 분리한다. LAMBDAS는 "재학습이 필요한 팔"로 좁혀져 있을 수
#   있는데, 프로브는 디스크에 있는 팔을 전부 다시 재야 그림이 완성되기 때문이다.
#   기본값은 OUT_ROOT 밑의 lam* 디렉터리에서 자동으로 찾는다.
REPROBE_LAMBDAS=${REPROBE_LAMBDAS:-"$(ls -d "${OUT_ROOT}"/lam*/ 2>/dev/null \
    | sed 's|.*/lam||; s|/$||' | tr '\n' ' ')"}

reprobe_all() {
    [ -s "${RESULTS}" ] && mv "${RESULTS}" "${RESULTS}.bak" \
        && echo "[E0] previous results -> ${RESULTS}.bak"
    echo "[E0] reprobe arms: ${REPROBE_LAMBDAS}"

    for lam in ${REPROBE_LAMBDAS}; do
        local tag="lam${lam}"
        for k in $(seq 0 $((NUM_TASKS - 1))); do
            local out_dir="${OUT_ROOT}/${tag}/task_${k}"
            local ckpt="${out_dir}/checkpoints/last/pretrained_model"
            [ -d "${ckpt}" ] || { echo "[E0] skip (no ckpt) ${ckpt}"; continue; }

            echo ""
            echo "══ [E0 reprobe] λ=${lam}  task=${k}"
            "${PYTHON}" "${E0_PY}" \
                --seed="${SEED}" \
                --job_name="E0_reprobe_${tag}_task_${k}" \
                --output_dir="${out_dir}" \
                --dataset.repo_id="${DATASET_PREFIX}${k}" \
                --policy.path="${ckpt}" \
                --policy.push_to_hub=false \
                --reprobe=true \
                --eval_freq=0 \
                --env.type=libero \
                --env.benchmark=libero_spatial \
                --env.task="${ENV_TASK_PREFIX}${k}" \
                --ewc_lambda="${lam}" \
                --run_tag="${lam}" \
                --current_task="${k}" \
                --task_ids="$(seq -s, 0 "${k}")" \
                --dataset_prefix="${DATASET_PREFIX}" \
                --env_task_prefix="${ENV_TASK_PREFIX}" \
                --results_path="${RESULTS}" \
                --holdout_episodes="${HOLDOUT_EP}" \
                --probe_batches="${PROBE_BATCHES}" \
                --probe_sr="${PROBE_SR}" \
                --probe_n_episodes="${PROBE_N_EP}" \
                --probe_eval_batch_size="${PROBE_EVAL_BS}" \
                --wandb.enable=false \
                || { echo "[E0] FAILED reprobe λ=${lam} task=${k}"; return 1; }
        done
    done
}

if [ "${REPROBE:-0}" = "1" ]; then
    reprobe_all || exit 1
elif [ "${PLOT_ONLY:-0}" != "1" ]; then
    for lam in ${LAMBDAS}; do
        run_lambda "${lam}"
    done
fi

if [ -s "${RESULTS}" ]; then
    "${PYTHON}" "${E0_PY}" --plot_only --results="${RESULTS}" --out="${FIGURE}"
fi

echo ""
echo "[E0] done.  raw=${RESULTS}"
echo "      per-task figure = ${FIGURE}"
echo "      summary figure  = ${FIGURE%.png}_summary.png"
echo "      table           = ${FIGURE%.png}.csv"
