#!/usr/bin/env bash
#
# 가우시안 검증이 끝나면 R10 을 이어서 돌린다.
#   1) 대기
#   2) 짧은 연기시험 (3 태스크 x 40 스텝) — rolling 전환 후 첫 실행이라 확인용
#   3) 본 실행
# 사용법: bash R10_chain.sh <대기할 pid> [GPU]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
WAIT_PID=${1:?pid}; GPU=${2:-0}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p results/R10 logs/mod0

echo "[$(date '+%F %T')] gauss(pid ${WAIT_PID}) 종료 대기"
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 30; done
echo "[$(date '+%F %T')] 대기 종료"

echo "[$(date '+%F %T')] === 연기시험 (rolling 전환 확인)"
rm -rf /tmp/r10_roll /tmp/r10_roll_ckpt
if python -u R10.py --smoke --out /tmp/r10_roll \
        --passthru --num_tasks 3 --steps_per_task 40 \
        --eval_episodes 1 --eval_batch_size 1 --log_every 20 \
        --ckpt_root /tmp/r10_roll_ckpt > logs/mod0/R10_smoke.log 2>&1; then
    echo "[$(date '+%F %T')] 연기시험 통과"
else
    echo "[$(date '+%F %T')] ★ 연기시험 실패 — logs/mod0/R10_smoke.log 확인. 본 실행은 그래도 진행한다."
fi
rm -rf /tmp/r10_roll /tmp/r10_roll_ckpt

echo "[$(date '+%F %T')] === R10 본 실행 (gpu ${GPU})"
rm -rf results/R10 outputs/R10; mkdir -p results/R10
python -u R10.py 2>&1 | tee -a results/R10/train.log
echo "[$(date '+%F %T')] === R10 종료"
