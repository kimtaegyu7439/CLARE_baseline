#!/usr/bin/env bash
# GPU 2 의 v3(grad) 런이 끝나면 이어서 런 A(l2_codebook_k96_bayes) 를 띄운다.
#   bash bayes_queue.sh <기다릴PID>
set -uo pipefail
cd /home/sa090180/clare
source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
source bash/clare/env.sh
WAIT_PID=${1:?기다릴 PID 를 줘라}
NAME=l2_codebook_k96_bayes
ts(){ echo "[$(date '+%F %T')] $*"; }
ts "대기 시작 — pid ${WAIT_PID} (v3 grad) 종료를 기다린다"
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
ts "v3 종료 확인. GPU 2 정리 대기 30초"
sleep 30
export CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
mkdir -p "results/${NAME}"
ts "런 A 시작 — ${NAME}"
nohup python -u L2_codebook_bayes.py --out "results/${NAME}" --chunk_backward \
    --codebook_k 96 --n_pairs 8000 --bayes_temp 1.0 \
    --passthru --num_tasks 10 --suite libero_spatial --num_workers 6 \
    > "logs/${NAME}.log" 2>&1 &
echo $! > "results/${NAME}/run.pid"
ts "  pid $(cat results/${NAME}/run.pid)  로그 logs/${NAME}.log"
