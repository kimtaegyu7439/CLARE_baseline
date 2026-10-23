#!/usr/bin/env bash
# LOTUS 셋팅 lifelong, V7 dea 앵커 (Lotus_V7_dea.py).
#
#   bash Lotus_V7_dea.sh <suite: goal|object|50> <gpu> [steps]
#   BETA=3.0 bash Lotus_V7_dea.sh 50 0
#   FORM=sqerr bash Lotus_V7_dea.sh 50 0            # 기준선 (기존 mse 와 항등)
#   SMOKE=1 bash Lotus_V7_dea.sh 50 2               # 6 step x 1 stage, 롤아웃 1회
#
# Lotus_l2cb_cl.sh 와 같은 구조다. 다른 점은 ENTRY 가 Lotus_V7_dea.py 이고
# 출력 폴더에 팔(form)과 β 를 적어 기존 v3 앵커 런과 섞이지 않게 하는 것.
#
#   기존  results/lotus_l2cb_<suite>
#   여기  results/lotus_v7_<form>_b<β>_<suite>
#
# suite=50 은 base 25 task + 5 stage x 5 task = 50 task.
# ★ base 가 25 태스크라 stage 1 부터 앵커의 과거가 25 개다. dea 는 2-pass 라
#   student forward 가 2 배 -> 스텝당 50 회. 처음 돌릴 때 train.log 의
#   ms/step 을 꼭 확인해라. 감당이 안 되면 FORM=unifmd (1-pass) 가 대안이다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

SUITE=${1:?suite (goal|object|50)}; GPU=${2:-0}; STEPS=${3:-10000}
FORM=${FORM:-dea}
BETA=${BETA:-2.0}
EPS_W=${EPS_W:-1e-3}
SPACE=${SPACE:-act}          # V7 = 액션 공간
REDUCE=${REDUCE:-mean}
SEED=${SEED:-42}
N_EVAL=${N_EVAL:-20}
EVAL_SEED=${EVAL_SEED:-7}
NW=${NW:-6}
EXTRA=${EXTRA:-}

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS} NUMEXPR_NUM_THREADS=${OMP_NUM_THREADS}

TAG="lotus_v7_${FORM}_b${BETA}_${SUITE}"
if [ -n "${SMOKE:-}" ]; then
    # 스모크: 1 stage, 6 step, 롤아웃 1회. 본 런 폴더를 더럽히지 않게 따로 둔다.
    STEPS=6; N_EVAL=1; TAG="smoke_${TAG}"
    EXTRA="--n_stages 1 --skip_base_eval --log_every 1 ${EXTRA}"
fi
OUT=${OUT:-results/${TAG}}; mkdir -p "${OUT}"

echo "[$(date '+%F %T')] LOTUS V7 ${FORM}  suite=${SUITE}  gpu=${GPU}  steps=${STEPS}"
echo "    β=${BETA} ε_w=${EPS_W} space=${SPACE} reduce=${REDUCE} seed=${SEED}  -> ${OUT}"
nohup python -u Lotus_V7_dea.py --suite "${SUITE}" --steps "${STEPS}" \
    --anchor_form "${FORM}" --beta "${BETA}" --eps_w "${EPS_W}" \
    --dir_space "${SPACE}" --anchor_reduce "${REDUCE}" \
    --sampler_mode hard_linear \
    --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
    --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
    --num_workers "${NW}" --out "${OUT}" ${EXTRA} \
    > "${OUT}/train.log" 2>&1 &
echo $! > "${OUT}/run.pid"
echo "  pid $(cat "${OUT}/run.pid")   로그 ${OUT}/train.log"
echo "  진행   tail -f ${OUT}/train.log"
