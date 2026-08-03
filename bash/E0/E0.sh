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
#   PLOT_ONLY=1 bash bash/E0/E0.sh            # 이미 쌓인 JSONL로 그림만

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
LAMBDAS=${LAMBDAS:-"0 10 100 1000 inf"}
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

        # 이미 끝난 스테이지는 건너뛴다(중간에 죽어도 이어서 돌릴 수 있게).
        [ -d "${out_dir}" ] && { echo "[E0] skip ${out_dir}"; continue; }

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

if [ "${PLOT_ONLY:-0}" != "1" ]; then
    for lam in ${LAMBDAS}; do
        run_lambda "${lam}"
    done
fi

if [ -s "${RESULTS}" ]; then
    "${PYTHON}" "${E0_PY}" --plot_only --results="${RESULTS}" --out="${FIGURE}"
fi

echo ""
echo "[E0] done.  raw=${RESULTS}  figure=${FIGURE}  table=${FIGURE%.png}.csv"
