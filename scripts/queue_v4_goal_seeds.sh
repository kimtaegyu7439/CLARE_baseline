#!/usr/bin/env bash
# GPU 0 — v4 libero_goal 을 train seed 7 -> 1000 순서로 이어 돌린다.
# 두 실행이 한 스크립트 안에서 직렬이라 GPU 폴링이 필요 없다(앞이 끝나야 뒤가 시작).
#   bash scripts/queue_v4_goal_seeds.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=${GPU:-0}
LOG=logs/queue_v4_goal_seeds.log; mkdir -p logs results
ts(){ echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

export CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_EGL_DEVICE_ID=${GPU} MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6

run_seed() {                     # run_seed <train_seed>
    local SEED=$1
    local NAME=l2_codebook_v4_goal_seed${SEED}
    local OUT=results/${NAME}
    # 락 — 다른 큐가 같은 이름을 이미 가져갔으면 건너뛴다
    if ! mkdir "${OUT}.lock" 2>/dev/null; then
        ts "  ${NAME} 은 다른 큐가 가져갔다 — 건너뜀"; return 0
    fi
    mkdir -p "${OUT}"
    ts "▶ ${NAME} 시작 (suite goal, train seed ${SEED}, eval seed 7, gpu ${GPU})"
    python -u Lotus_l2cb_cl.py --suite goal --steps 10000 \
        --seed "${SEED}" --eval_seed 7 --eval_episodes 20 --eval_batch_size 20 \
        --num_workers 6 --sampler_mode hard_linear --anchor_weight speedinv \
        --out "${OUT}" > "logs/${NAME}.log" 2>&1 &
    local PID=$!; echo "${PID}" > "${OUT}/run.pid"
    { echo "# ${NAME}"; echo
      echo "- 시작: $(date '+%F %T')  PID ${PID}  GPU ${GPU}"
      echo "- v4 = hard_linear 샘플러 + 앵커 속도 정규화"
      echo "- suite goal   train seed ${SEED}   eval seed 7   20 rollout"
      echo "- base: Models/Lotus_clare_libero_goal_pretrain"; } > "${OUT}/RUN_INFO.md"
    sleep 120
    if ! kill -0 "${PID}" 2>/dev/null; then
        ts "★ ${NAME} 이 120초 안에 죽었다 — 뒤 실행도 중단"
        tail -30 "logs/${NAME}.log" >> "${LOG}"
        return 1
    fi
    ts "  ${NAME} pid ${PID} 생존 확인"
    wait "${PID}"; local r=$?
    ts "◀ ${NAME} 종료 rc=${r}"
    echo "- 종료: $(date '+%F %T') rc=${r}" >> "${OUT}/RUN_INFO.md"
    python -u lotus_v4_report.py >> "${LOG}" 2>&1 || true
    return "${r}"
}

run_seed 7 && run_seed 1000
ts "큐 완료"
