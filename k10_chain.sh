#!/usr/bin/env bash
# K10 체인 — 게이트(GPU1·2·3) -> report/selected -> 학습 3팔(GPU1·2·3) -> 최종 표.
# 팔 하나가 죽어도 나머지는 계속된다(개별 nohup + PID 목록으로 wait).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}

mkdir -p results/K10/gpu1 results/K10/gpu2 results/K10/gpu3 \
         results/K10L results/K7b results/K10LB
LOG=results/K10/chain.log
ts() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

ts "PHASE 1 게이트 시작 (GPU 1·2·3)"
PIDS=()
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 nohup python -u k10_gate.py --arm wit  \
    > results/K10/gpu1/log 2>&1 & PIDS+=($!)
CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 nohup python -u k10_gate.py --arm U    \
    > results/K10/gpu2/log 2>&1 & PIDS+=($!)
CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 nohup python -u k10_gate.py --arm prod \
    > results/K10/gpu3/log 2>&1 & PIDS+=($!)
ts "게이트 pid: ${PIDS[*]}"
for p in "${PIDS[@]}"; do wait "$p" || ts "게이트 pid $p 비정상 종료 (계속 진행)"; done
ts "PHASE 1 게이트 종료"

ts "리포트/선택 시작"
python -u k10_report.py >> "${LOG}" 2>&1 || ts "k10_report 실패 — default 로 계속"
ts "리포트/선택 종료  selected=$(cat results/K10/selected.json 2>/dev/null | tr -d '\n')"

ts "PHASE 2 학습 3팔 시작"
PIDS=()
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 nohup python -u k10_train.py --arm K10L  \
    > results/K10L/train.log 2>&1 & PIDS+=($!)
sleep 20
CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 nohup python -u k10_train.py --arm K7b   \
    > results/K7b/train.log 2>&1 & PIDS+=($!)
sleep 20
CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 nohup python -u k10_train.py --arm K10LB \
    > results/K10LB/train.log 2>&1 & PIDS+=($!)
ts "학습 pid: ${PIDS[*]}"
for p in "${PIDS[@]}"; do wait "$p" || ts "학습 pid $p 비정상 종료 (나머지 계속)"; done
ts "PHASE 2 학습 종료"

ts "최종 표 생성"
python -u k10_final_tables.py >> "${LOG}" 2>&1 || ts "k10_final_tables 실패"
ts "체인 완료"
