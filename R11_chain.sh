#!/usr/bin/env bash
# R10 체인이 끝나면 R11 을 이어서 돌린다. 사용법: bash R11_chain.sh <대기 pid> [GPU]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
WAIT_PID=${1:?pid}; GPU=${2:-0}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p results/R11 logs/mod0
echo "[$(date '+%F %T')] R10(pid ${WAIT_PID}) 종료 대기"
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] 대기 종료"

echo "[$(date '+%F %T')] === R11 연기시험"
rm -rf /tmp/r11_dry /tmp/r11_dry_ckpt
if python -u R11.py --smoke --out /tmp/r11_dry \
        --passthru --num_tasks 3 --steps_per_task 40 \
        --eval_episodes 1 --eval_batch_size 1 --log_every 20 \
        --ckpt_root /tmp/r11_dry_ckpt > logs/mod0/R11_smoke.log 2>&1; then
    echo "[$(date '+%F %T')] 연기시험 통과"
else
    echo "[$(date '+%F %T')] ★ 연기시험 실패 — logs/mod0/R11_smoke.log. 본 실행은 진행한다."
fi
rm -rf /tmp/r11_dry /tmp/r11_dry_ckpt

echo "[$(date '+%F %T')] === R11 본 실행 (gpu ${GPU})"
rm -rf results/R11 outputs/R11; mkdir -p results/R11
python -u R11.py 2>&1 | tee -a results/R11/train.log
echo "[$(date '+%F %T')] === R11 종료"
