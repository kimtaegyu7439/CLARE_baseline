#!/usr/bin/env bash
#
# H5 — H4의 Fisher 충돌 측정을 실제 CL 궤적 위에서 다시 재기 (libero_spatial)
#
# H4는 고정된 앵커 θ*(사전학습 체크포인트) 한 곳에서만 잰다. 그래서
# "CL을 시작하기 전에 이미 충돌이 예정돼 있다"까지만 말할 수 있고,
# "태스크 0을 배우고 나면 파라미터가 옮겨가서 안 겹칠 수도 있지 않나"는 반론이 남는다.
#
# H5는 stage k마다 앵커를 **E0가 실제로 쓴 그 파라미터**(태스크 k-1 학습 종료 시점)로
# 옮기고, 그 자리에서 태스크별 Fisher를 다시 재서 H4와 동일한 지표를 낸다.
# 지표 함수는 H4.py에서 import하므로 두 결과가 같은 축 위에 놓인다.
#
# 추가로 남기는 CL 전용 값
#   anchor_drift         앵커가 사전학습 지점에서 실제로 얼마나 움직였는가
#   stored_fresh_cosine  EWC가 들고 다닌 누적 Fisher(과거 앵커에서 잰 값) 대
#                        지금 앵커에서 새로 잰 F_old 의 cosine (낮을수록 낡음)
#
# 전제: E0가 만든 CL 체크포인트가 있어야 한다. 학습은 하지 않는다.
#       gym_libero도 필요 없다.
#
# 읽는 법 (H4와 동일)
#   pareto_gain ≈ 0.5 → 좋은 λ가 존재하지 않음(퇴화)   1.0 → λ로 분리 가능
#   H4와 H5가 비슷하면, 충돌은 앵커 위치가 아니라 태스크 구조에서 온다는 뜻이다.
#
# 사용법
#   bash bash/E0/H5.sh
#   CL_ARM=lam1000 bash bash/E0/H5.sh            # 어느 λ 팔의 궤적을 볼지
#   NUM_TASKS=6 bash bash/E0/H5.sh               # 태스크 개수 조절
#   MEASURE_STEPS=200 bash bash/E0/H5.sh         # [C] 실측 켜기 (기본 꺼짐)
#   RECOMPUTE=1 bash bash/E0/H5.sh               # Fisher 캐시 무시하고 다시 재기
#   PLOT_ONLY=1 bash bash/E0/H5.sh               # 이미 쌓인 JSONL로 그림만

set -uo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-1}

# HF_LEROBOT_HOME / HF_HUB_CACHE / PRETRAIN_PATH 를 세팅한다.
source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

H5_PY=./lerobot_lsy/src/lerobot/scripts/H5.py
PYTHON=${PYTHON:-python}   # conda clare 환경이 활성화돼 있다고 가정. 아니면 PYTHON=... 로 지정.

# ── 조절할 것들 ───────────────────────────────────────────────────────────────
NUM_TASKS=${NUM_TASKS:-4}                # 태스크 0..NUM_TASKS-1
SEED=${SEED:-42}
HOLDOUT_EP=${HOLDOUT_EP:-5}              # E0/H4와 같은 분할

# 어느 CL 궤적 위에서 잴 것인가. E0가 만든 팔 이름이다 (lam0 / lam10 / lam100 / lam1000 / laminf).
# lam0은 EWC를 안 쓴 순차 파인튜닝이라 ewc_state.pt가 없다 -> stored_fresh_cosine은 생략된다.
CL_ARM=${CL_ARM:-lam100}
E0_ROOT=${E0_ROOT:-./outputs/E0/libero_spatial/seed_${SEED}}
CL_ROOT=${CL_ROOT:-${E0_ROOT}/${CL_ARM}}

# [A] Fisher / 그래디언트 추정 (H4와 같은 기본값이어야 비교가 된다)
FISHER_BATCHES=${FISHER_BATCHES:-100}
FISHER_BATCH_SIZE=${FISHER_BATCH_SIZE:-8}

# [B] 분석 (H4와 동일해야 한다)
LAMBDAS=${LAMBDAS:-"0,1e-4,1e-3,1e-2,0.03,0.1,0.3,1,3,10,30,100,300,1000,3000,1e4,1e5,1e6,1e7,1e8,inf"}
TOP_P=${TOP_P:-"0.0001,0.001,0.01,0.05,0.1,0.25"}
CURV_DAMPING=${CURV_DAMPING:-1e-3}
LAYER_REPORT=${LAYER_REPORT:-12}

# [C] 실측 (0이면 생략. H4와 달리 기본으로 끈다 — H5는 [A]가 stage마다 도는 만큼 이미 무겁다)
MEASURE_STEPS=${MEASURE_STEPS:-0}
MEASURE_LAMBDAS=${MEASURE_LAMBDAS:-"0,10,100,1000"}
MEASURE_TOP_P=${MEASURE_TOP_P:-0.01}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
LOG_FREQ=${LOG_FREQ:-50}

OUT_ROOT=${OUT_ROOT:-./outputs/H5/libero_spatial/seed_${SEED}/${CL_ARM}}
RESULTS=${RESULTS:-${OUT_ROOT}/h5_results.jsonl}
FIGURE=${FIGURE:-${OUT_ROOT}/H5_fisher_conflict_cl.png}
# H4 결과가 있으면 그림에 고정 앵커 곡선을 같이 그린다 (없으면 무시).
H4_RESULTS=${H4_RESULTS:-./outputs/H4/libero_spatial/seed_${SEED}/h4_results.jsonl}
mkdir -p "${OUT_ROOT}"

DATASET_PREFIX=continuallearning/libero_spatial_image_task_

if [ "${PLOT_ONLY:-0}" != "1" ]; then
    if [ ! -d "${CL_ROOT}" ]; then
        echo "[H5] CL 체크포인트가 없다: ${CL_ROOT}"
        echo "     먼저 E0를 돌려야 한다:  bash bash/E0/E0.sh"
        echo "     또는 CL_ARM= 으로 다른 팔을 지정해라 (lam0/lam10/lam100/lam1000/laminf)."
        exit 1
    fi

    # 같은 JSONL에 두 번 쌓이면 그림에 곡선이 겹친다. Fisher 캐시(비싼 쪽)는 그대로 둔다.
    [ -s "${RESULTS}" ] && mv "${RESULTS}" "${RESULTS}.bak" && echo "[H5] previous results -> ${RESULTS}.bak"

    extra=()
    [ "${RECOMPUTE:-0}" = "1" ] && extra+=("--recompute=true")

    echo ""
    echo "══ [H5] tasks 0..$((NUM_TASKS - 1))  CL anchors from ${CL_ROOT}"

    "${PYTHON}" "${H5_PY}" \
        --seed="${SEED}" \
        --job_name="H5_seed_${SEED}_${CL_ARM}" \
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
        --cl_root="${CL_ROOT}" \
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
        --run_tag="h5_${CL_ARM}" \
        --results_path="${RESULTS}" \
        "${extra[@]}" || { echo "[H5] FAILED"; exit 1; }
fi

if [ -s "${RESULTS}" ]; then
    "${PYTHON}" "${H5_PY}" --plot_only \
        --results="${RESULTS}" --out="${FIGURE}" --h4_results="${H4_RESULTS}"
fi

echo ""
echo "[H5] done.  raw=${RESULTS}  figure=${FIGURE}  table=${FIGURE%.png}.csv"
echo "[H5] Fisher cache=${OUT_ROOT}/stats  (지우지 않으면 다음 실행은 [A]를 건너뛴다)"
