#!/usr/bin/env bash
# GPU 1 이 비면 v4 libero_object 를 train seed 1000 으로 이어 돌린다.
#   bash scripts/queue_v4_object_seed1000.sh <기다릴PID>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=1; WAIT_PID=${1:-}; NAME=l2_codebook_v4_object_seed1000; OUT=results/${NAME}
LOG=logs/queue_v4_object_seed1000.log; mkdir -p logs "${OUT}"
ts(){ echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

ts "대기 — GPU ${GPU} (mem<1000MiB & util<5% 5회 연속)${WAIT_PID:+, pid ${WAIT_PID} 종료도 조건}"
ok=0
while true; do
    read -r mem util < <(nvidia-smi -i "${GPU}" --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr ',' ' ')
    alive=0; [ -n "${WAIT_PID}" ] && kill -0 "${WAIT_PID}" 2>/dev/null && alive=1
    if [ "${mem:-9999}" -lt 1000 ] && [ "${util:-100}" -lt 5 ] && [ "${alive}" -eq 0 ]; then
        ok=$((ok+1)); else ok=0; fi
    [ "${ok}" -ge 5 ] && break
    sleep 60
done
ts "GPU ${GPU} 비었음 확인"
export CUDA_VISIBLE_DEVICES=${GPU} MUJOCO_EGL_DEVICE_ID=${GPU} MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
ts "▶ ${NAME} 시작 (train seed 1000, eval seed 7)"
python -u Lotus_l2cb_cl.py --suite object --steps 10000 \
    --seed 1000 --eval_seed 7 --eval_episodes 20 --eval_batch_size 20 \
    --num_workers 6 --sampler_mode hard_linear --anchor_weight speedinv \
    --out "${OUT}" > "logs/${NAME}.log" 2>&1 &
PID=$!; echo "${PID}" > "${OUT}/run.pid"
{ echo "# ${NAME}"; echo; echo "- 시작: $(date '+%F %T')  PID ${PID}  GPU ${GPU}"
  echo "- v4 = hard_linear 샘플러 + 앵커 속도 정규화"
  echo "- suite object   train seed 1000   eval seed 7   20 rollout"; } > "${OUT}/RUN_INFO.md"
sleep 120
kill -0 "${PID}" 2>/dev/null || { ts "★ 120초 안에 죽었다"; tail -30 "logs/${NAME}.log" >> "${LOG}"; exit 1; }
ts "  pid ${PID} 생존 확인"
wait "${PID}"; ts "◀ ${NAME} 종료 rc=$?"
python -u lotus_v4_report.py >> "${LOG}" 2>&1 || true
