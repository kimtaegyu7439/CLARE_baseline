#!/usr/bin/env bash
# GPU 3: p(s|j) 완전 공분산 버전 두 개를 순차 실행.
#   ① l2cb_fullcov          대각 아님, 기울기 없음   (v1 대응)
#   ② l2cb_fullcov_grad     대각 아님 + 기울기       (v3 대응)
set -uo pipefail
cd /home/sa090180/clare
source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
source bash/clare/env.sh
export CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 MUJOCO_GL=egl
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
ts(){ echo "[$(date '+%F %T')] $*"; }
run() {
    local NAME=$1; shift
    mkdir -p "results/${NAME}"
    ts "▶ ${NAME} 시작"
    python -u l2_codebook.py --out "results/${NAME}" --chunk_backward \
        --codebook_k 96 --n_pairs 8000 --full_cov_s --cov_ridge 0.05 "$@" \
        --passthru --num_tasks 10 --suite libero_spatial --num_workers 6 \
        > "logs/${NAME}.log" 2>&1
    ts "◀ ${NAME} 종료 rc=$?"
    python -u l2_family_report.py > /dev/null 2>&1 || true
    sleep 30
}
run l2cb_fullcov
run l2cb_fullcov_grad --grad_enable
ts "체인 완료"
