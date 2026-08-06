#!/usr/bin/env bash
#
# H4 — 태스크 간 Fisher 충돌 측정 (libero_spatial)
#
# E0가 "λ를 바꿔 가며 실제로 순차 학습을 돌려서 SR로" 보는 현상을,
# H4는 학습 전에 파라미터 공간에서 직접 잰다. 그래서 한 번만 돌면 된다
# (E0처럼 λ × task 만큼 학습을 반복하지 않는다).
#
# 재는 것
#   [A] 공통 앵커 θ*(사전학습 체크포인트)에서 태스크 0..NUM_TASKS-1 각각의
#       대각 Fisher F_k 와 평균 그래디언트 g_k. 태스크당 한 번, 캐시된다.
#   [B] 1) 중요 파라미터 부공간이 얼마나 겹치는가 (cos / Bhattacharyya / top-p% lift)
#       2) 이전 태스크의 큰 Fisher가 다음 태스크의 update 성분을 얼마나 덮는가
#          (blocked_gain: 새 태스크가 얻을 수 있는 손실 감소 중 보호 좌표에 갇힌 비율)
#       3) λ의 stability-plasticity 파레토가 퇴화했는가
#          (pareto_gain: 0.5 = 좋은 λ가 존재하지 않음, 1.0 = λ로 분리 가능)
#   [C] (MEASURE_STEPS>0) 실제로 θ*에서 λ를 바꿔 가며 짧게 학습해 Δθ를 비교.
#       EWC가 "중요 좌표만" 골라 막는지, 아니면 그냥 전부 줄이는지 확인한다.
#
# 시뮬레이터(gym_libero)가 필요 없다. SR은 E0가 잰다.
#
# 사용법
#   bash bash/E0/H4.sh
#   NUM_TASKS=6 bash bash/E0/H4.sh              # 태스크 개수 조절
#   FISHER_BATCHES=200 bash bash/E0/H4.sh       # Fisher 추정을 더 촘촘히
#   MEASURE_STEPS=0 bash bash/E0/H4.sh          # [C] 실측 생략 (빠름)
#   RECOMPUTE=1 bash bash/E0/H4.sh              # Fisher 캐시 무시하고 다시 재기
#   PLOT_ONLY=1 bash bash/E0/H4.sh              # 이미 쌓인 JSONL로 그림만

set -uo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-1}

# HF_LEROBOT_HOME / HF_HUB_CACHE / PRETRAIN_PATH 를 세팅한다.
source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

H4_PY=./lerobot_lsy/src/lerobot/scripts/H4.py
PYTHON=${PYTHON:-python}   # conda clare 환경이 활성화돼 있다고 가정. 아니면 PYTHON=... 로 지정.

# ── 조절할 것들 ───────────────────────────────────────────────────────────────
NUM_TASKS=${NUM_TASKS:-4}                # 태스크 0..NUM_TASKS-1
SEED=${SEED:-42}
HOLDOUT_EP=${HOLDOUT_EP:-5}              # E0와 같은 분할. Fisher는 학습 에피소드에서만.

# [A] Fisher / 그래디언트 추정
FISHER_BATCHES=${FISHER_BATCHES:-100}    # E0.build_ewc_state와 같은 기본값
FISHER_BATCH_SIZE=${FISHER_BATCH_SIZE:-8}

# [B] 분석
LAMBDAS=${LAMBDAS:-"0,1e-4,1e-3,1e-2,0.03,0.1,0.3,1,3,10,30,100,300,1000,3000,1e4,1e5,1e6,1e7,1e8,inf"}
TOP_P=${TOP_P:-"0.0001,0.001,0.01,0.05,0.1,0.25"}
CURV_DAMPING=${CURV_DAMPING:-1e-3}
LAYER_REPORT=${LAYER_REPORT:-12}

# [C] 실측 (0이면 통째로 생략)
MEASURE_STEPS=${MEASURE_STEPS:-200}
MEASURE_LAMBDAS=${MEASURE_LAMBDAS:-"0,10,100,1000"}
MEASURE_TOP_P=${MEASURE_TOP_P:-0.01}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
LOG_FREQ=${LOG_FREQ:-50}

OUT_ROOT=${OUT_ROOT:-./outputs/H4/libero_spatial/seed_${SEED}}
RESULTS=${RESULTS:-${OUT_ROOT}/h4_results.jsonl}
FIGURE=${FIGURE:-${OUT_ROOT}/H4_fisher_conflict.png}
mkdir -p "${OUT_ROOT}"

DATASET_PREFIX=continuallearning/libero_spatial_image_task_

if [ "${PLOT_ONLY:-0}" != "1" ]; then
    # 같은 JSONL에 두 번 쌓이면 그림에 곡선이 겹쳐 나온다. Fisher 캐시(비싼 쪽)는
    # 그대로 두고 결과 파일만 밀어 둔다.
    [ -s "${RESULTS}" ] && mv "${RESULTS}" "${RESULTS}.bak" && echo "[H4] previous results -> ${RESULTS}.bak"

    extra=()
    [ "${RECOMPUTE:-0}" = "1" ] && extra+=("--recompute=true")

    echo ""
    echo "══ [H4] tasks 0..$((NUM_TASKS - 1))  anchor=${PRETRAIN_PATH}"

    "${PYTHON}" "${H4_PY}" \
        --seed="${SEED}" \
        --job_name="H4_seed_${SEED}" \
        --output_dir="${OUT_ROOT}/run" \
        --dataset.repo_id="${DATASET_PREFIX}0" \
        --policy.path="${PRETRAIN_PATH}" \
        --policy.push_to_hub=false \
        --eval_freq=0 \
        --wandb.enable=false \
        --batch_size="${BATCH_SIZE}" \
        --num_workers="${NUM_WORKERS}" \
        --log_freq="${LOG_FREQ}" \
        --num_tasks="${NUM_TASKS}" \
        --dataset_prefix="${DATASET_PREFIX}" \
        --holdout_episodes="${HOLDOUT_EP}" \
        --fisher_batches="${FISHER_BATCHES}" \
        --fisher_batch_size="${FISHER_BATCH_SIZE}" \
        --stats_dir="${OUT_ROOT}/stats" \
        --lambdas="${LAMBDAS}" \
        --top_p="${TOP_P}" \
        --curv_damping="${CURV_DAMPING}" \
        --layer_report="${LAYER_REPORT}" \
        --measure_steps="${MEASURE_STEPS}" \
        --measure_lambdas="${MEASURE_LAMBDAS}" \
        --measure_top_p="${MEASURE_TOP_P}" \
        --run_tag="h4" \
        --results_path="${RESULTS}" \
        "${extra[@]}" || { echo "[H4] FAILED"; exit 1; }
fi

if [ -s "${RESULTS}" ]; then
    "${PYTHON}" "${H4_PY}" --plot_only --results="${RESULTS}" --out="${FIGURE}"
fi

echo ""
echo "[H4] done.  raw=${RESULTS}  figure=${FIGURE}  table=${FIGURE%.png}.csv"
echo "[H4] Fisher cache=${OUT_ROOT}/stats  (지우지 않으면 다음 실행은 [A]를 건너뛴다)"
