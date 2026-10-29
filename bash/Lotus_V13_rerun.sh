#!/usr/bin/env bash
# 이미 완주한 V13 LOTUS 런을 **같은 조건으로 다시** 돌려 별도 폴더에 기록한다.
# 목적: 같은 seed 로 다시 돌렸을 때 성능이 얼마나 흔들리는지(GPU 비결정성) 재기.
#
#   bash bash/Lotus_V13_rerun.sh                 # β=1 λ=3 goal seed7 -> GPU 2, _rerun1
#   BETA=1.0 LAM=3.0 SUITE=goal SEED=7 GPU=2 N=1 bash bash/Lotus_V13_rerun.sh
#
# 출력: results/lotus_v13_b<β>_lam<λ>_k25_scopeLL_<suite>_seed<seed>_rerun<N>
#   원본 폴더는 건드리지 않는다. LOTUS_act_V5_report.py 의 "재현 대조" 절이
#   _rerun* 폴더를 찾아 원본 옆에 놓는다. 인자는 bash/Lotus_V13_run.sh 와 같다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh
    conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
BETA=${BETA:-1.0}; LAM=${LAM:-3.0}; SUITE=${SUITE:-goal}; SEED=${SEED:-7}; GPU=${GPU:-2}; N=${N:-1}
LIFELONG_K=${LIFELONG_K:-25}; EPS_W=${EPS_W:-1e-3}; STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-20}; EVAL_SEED=${EVAL_SEED:-7}; NW=${NW:-6}; OMP_THREADS=${OMP_THREADS:-6}
ORIG="lotus_v13_b${BETA}_lam${LAM}_k${LIFELONG_K}_scopeLL_${SUITE}_seed${SEED}"
TAG="${ORIG}_rerun${N}"
[ -d "results/${ORIG}" ] || { echo "✗ 원본이 없다: results/${ORIG}"; exit 1; }
pgrep -f "L2_V13.py .*--out results/${TAG}( |$)" >/dev/null && { echo "⏭ ${TAG} 이미 돌고 있다"; exit 0; }
mkdir -p "results/${TAG}" logs
LOG="logs/v13_rerun_${SUITE}_b${BETA}_lam${LAM}_seed${SEED}.log"
echo "[$(date '+%F %T')] ▶ ${TAG}  (원본 ${ORIG})  GPU${GPU}" | tee -a "${LOG}"
setsid nohup env CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=egl \
    OMP_NUM_THREADS="${OMP_THREADS}" MKL_NUM_THREADS="${OMP_THREADS}" \
    OPENBLAS_NUM_THREADS="${OMP_THREADS}" NUMEXPR_NUM_THREADS="${OMP_THREADS}" \
    python -u L2_V13.py --setting lotus \
        --beta "${BETA}" --eps_w "${EPS_W}" --dir_space act \
        --lambda_level "${LAM}" --lifelong_k "${LIFELONG_K}" --anchor_scope lifelong \
        --suite "${SUITE}" --steps "${STEPS}" --sampler_mode hard_linear \
        --skip_base_eval --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
        --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
        --num_workers "${NW}" --out "results/${TAG}" \
        >> "results/${TAG}/train.log" 2>&1 < /dev/null &
echo "  로그 results/${TAG}/train.log"
