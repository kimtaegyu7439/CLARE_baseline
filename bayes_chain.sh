#!/usr/bin/env bash
# GPU 2: v3(grad) 완주 대기 -> 런 A(bayes 대각) -> 런 B(bayes + grad) 순차.
#   bash bayes_chain.sh <기다릴PID>
set -uo pipefail
cd /home/sa090180/clare
source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
source bash/clare/env.sh
export CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
WAIT_PID=${1:?}
ts(){ echo "[$(date '+%F %T')] $*"; }

ts "대기 — pid ${WAIT_PID} (v3 grad, stage 7/10) 완주를 기다린다"
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
ts "v3 종료 확인"
python -u l2_report.py results/l2_codebook_k96_grad > logs/l2_codebook_k96_grad_report.log 2>&1 || true
ts "v3 리포트 저장"

run() {  # <이름> <추가인자...>
    local NAME=$1; shift
    sleep 30
    mkdir -p "results/${NAME}"
    ts "▶ ${NAME} 시작"
    python -u L2_codebook_bayes.py --out "results/${NAME}" --chunk_backward \
        --codebook_k 96 --n_pairs 8000 --bayes_temp 1.0 "$@" \
        --passthru --num_tasks 10 --suite libero_spatial --num_workers 6 \
        > "logs/${NAME}.log" 2>&1
    ts "◀ ${NAME} 종료 rc=$?"
    python -u l2_report.py "results/${NAME}" > "logs/${NAME}_report.log" 2>&1 || true
}

run l2_codebook_k96_bayes                     # 런 A — 대각 + 베이즈 가중
run l2_codebook_k96_grad_bayes --grad_enable  # 런 B — 기울기 + 베이즈 가중
ts "체인 완료"
