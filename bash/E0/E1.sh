#!/usr/bin/env bash
#
# E1 — 저장된 Fisher가 실제 손실 증가를 예측하는가 (libero_spatial)
#
# EWC가 task 0..stage 를 지나며 들고 온 누적 Fisher F = F₀+..+F_stage 는, 각 태스크를
# 끝낸 "그 시점의" 파라미터에서 잰 값이다. 앵커는 계속 이동했으므로 F₀, F₁이 재던
# loss landscape은 이미 그 자리에 없다. 그렇다면 EWC의 페널티가 말하는 손상과
# 실제 손상이 어긋나 있을 것이다 — 그걸 직접 확인한다.
#
# 방법 (학습 없음, 시뮬레이터 없음)
#   방향 u를 N개 뽑고 거리 r마다  θ = θ* + r‖θ*‖u  로 흔든 뒤 두 값을 잰다.
#     예측  Ω  = ½ Σ F_i (θ_i − θ*_i)²        EWC가 "이만큼 나빠진다"고 말하는 값
#     실제  ΔL = L_old(θ) − L_old(θ*)          held-out에서 실제로 나빠진 양
#   각 축을 최댓값으로 스케일해 산점도를 그린다.
#     y=x 위에 몰림 → landscape이 바뀌어도 Fisher가 예측을 해낸다
#     구름처럼 퍼짐 → 저장된 Fisher 정보가 불확실하다
#
# 전제: E0가 만든 체크포인트 + ewc_state.pt (λ>0 팔만 ewc_state.pt가 있다).
#
# 사용법
#   bash bash/E0/E1.sh
#   CL_ARM=lam1000 bash bash/E0/E1.sh          # 다른 λ 팔
#   STAGE=3 bash bash/E0/E1.sh                 # task 0..3 을 끝낸 지점에서
#   N_DIRECTIONS=30 bash bash/E0/E1.sh         # 빠르게 훑기
#   PLOT_ONLY=1 bash bash/E0/E1.sh             # 이미 쌓인 JSONL로 그림만

set -uo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-1}

# HF_LEROBOT_HOME / HF_HUB_CACHE / PRETRAIN_PATH 를 세팅한다.
source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

E1_PY=./lerobot_lsy/src/lerobot/scripts/E1.py
PYTHON=${PYTHON:-python}   # conda clare 환경이 활성화돼 있다고 가정. 아니면 PYTHON=... 로 지정.

# ── 조절할 것들 ───────────────────────────────────────────────────────────────
SEED=${SEED:-42}
STAGE=${STAGE:-2}                        # task_{STAGE} 를 끝낸 지점이 θ*
CL_ARM=${CL_ARM:-lam100}                 # λ=0 팔은 ewc_state.pt가 없어 못 쓴다
E0_ROOT=${E0_ROOT:-./outputs/E0/libero_spatial/seed_${SEED}}
CL_ROOT=${CL_ROOT:-${E0_ROOT}/${CL_ARM}}

N_DIRECTIONS=${N_DIRECTIONS:-100}
RADII=${RADII:-"0.01,0.02,0.05,0.1,0.2,0.3"}
DIRECTION_SEED=${DIRECTION_SEED:-777}

# ΔL은 짝지은 차이여야 한다 — 배치와 ε/t를 전부 고정한다(E1.py가 처리).
EVAL_BATCHES=${EVAL_BATCHES:-6}          # 태스크당 캐시할 배치 수 (GPU 메모리와 직결)
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
EVAL_SEED=${EVAL_SEED:-12345}

HOLDOUT_EP=${HOLDOUT_EP:-5}
DATASET_PREFIX=continuallearning/libero_spatial_image_task_

OUT_ROOT=${OUT_ROOT:-./outputs/E1/libero_spatial/seed_${SEED}/${CL_ARM}_stage${STAGE}}
RESULTS=${RESULTS:-${OUT_ROOT}/e1_results.jsonl}
FIGURE=${FIGURE:-${OUT_ROOT}/E1_fisher_calibration.png}
mkdir -p "${OUT_ROOT}"

if [ "${PLOT_ONLY:-0}" != "1" ]; then
    if [ ! -f "${CL_ROOT}/task_${STAGE}/ewc_state.pt" ]; then
        echo "[E1] ewc_state.pt 가 없다: ${CL_ROOT}/task_${STAGE}/ewc_state.pt"
        echo "     λ=0 팔은 EWC를 안 써서 Fisher를 저장하지 않는다. CL_ARM= 으로 λ>0 팔을 지정해라."
        exit 1
    fi

    # JSONL은 append 전용이다. 두 번 쌓이면 그림에 옛 점과 새 점이 섞인다.
    [ -s "${RESULTS}" ] && mv "${RESULTS}" "${RESULTS}.bak" && echo "[E1] previous -> ${RESULTS}.bak"

    echo ""
    echo "══ [E1] anchor=${CL_ROOT}/task_${STAGE}  ${N_DIRECTIONS} directions x radii ${RADII}"

    "${PYTHON}" "${E1_PY}" \
        --seed="${SEED}" \
        --job_name="E1_${CL_ARM}_stage${STAGE}" \
        --output_dir="${OUT_ROOT}/run" \
        --dataset.repo_id="${DATASET_PREFIX}0" \
        --policy.path="${PRETRAIN_PATH}" \
        --policy.push_to_hub=false \
        --eval_freq=0 \
        --wandb.enable=false \
        --cl_root="${CL_ROOT}" \
        --stage="${STAGE}" \
        --dataset_prefix="${DATASET_PREFIX}" \
        --holdout_episodes="${HOLDOUT_EP}" \
        --n_directions="${N_DIRECTIONS}" \
        --radii="${RADII}" \
        --direction_seed="${DIRECTION_SEED}" \
        --eval_batches="${EVAL_BATCHES}" \
        --eval_batch_size="${EVAL_BATCH_SIZE}" \
        --eval_seed="${EVAL_SEED}" \
        --run_tag="${CL_ARM}_stage${STAGE}" \
        --results_path="${RESULTS}" \
        || { echo "[E1] FAILED"; exit 1; }
fi

if [ -s "${RESULTS}" ]; then
    "${PYTHON}" "${E1_PY}" --plot_only --results="${RESULTS}" --out="${FIGURE}"
fi

echo ""
echo "[E1] done.  raw=${RESULTS}  figure=${FIGURE}  table=${FIGURE%.png}.csv"
