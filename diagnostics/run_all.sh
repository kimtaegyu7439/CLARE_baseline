#!/usr/bin/env bash
# GPU 0 이 비면 D1 -> D3 -> D2 순서로 진단을 돌린다 (예약 실행).
#
#   bash diagnostics/run_all.sh [gpu]
#
# 각 단계가 끝날 때마다 REPORT.md 를 다시 쓴다. 사양대로 D1 결과가 나오면
# 먼저 2절을 채운 채 계속 진행한다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"
GPU=${1:-0}
PY=/home/sa090180/miniconda3/envs/clare/bin/python
OUTD=/home/sa090180/clare/results/diag_l2_codebook_k96_grad
LOG="${OUTD}/run.log"
mkdir -p "${OUTD}"

say() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

# ── 1. GPU 가 빌 때까지 대기 ────────────────────────────────────────────────
say "GPU ${GPU} 가 비기를 기다린다 (60초마다 확인)"
free_streak=0
while true; do
    pids=$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    if [ -z "${pids}" ]; then
        free_streak=$((free_streak + 1))
        # 두 번 연속 비어 있어야 진짜로 끝난 것으로 본다 (프로세스 교체 순간 오탐 방지)
        [ "${free_streak}" -ge 2 ] && break
    else
        free_streak=0
    fi
    sleep 60
done
say "GPU ${GPU} 확보 — 진단 시작"

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_EGL_DEVICE_ID="${GPU}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}

# ── 2. 스모크 잔재 치우기 ───────────────────────────────────────────────────
# 검증 때 만든 소표본 CSV 가 남아 있으면 본 결과와 섞인다.
STASH="${OUTD}/_smoke_$(date '+%H%M%S')"
if ls "${OUTD}"/d[123]_*.csv >/dev/null 2>&1; then
    mkdir -p "${STASH}"
    mv "${OUTD}"/d[123]_*.csv "${OUTD}"/d[123]_*.json "${OUTD}"/*.png "${STASH}/" 2>/dev/null
    say "스모크 산출물을 ${STASH} 로 옮겼다"
fi

step() {   # step <이름> <명령...>
    local name="$1"; shift
    say "── ${name} 시작"
    if "$@" >>"${LOG}" 2>&1; then
        say "── ${name} 완료"
    else
        say "── ${name} ★실패★ (종료코드 $?) — 다음 단계로 넘어간다"
    fi
    ${PY} make_report.py >>"${LOG}" 2>&1 && say "   REPORT.md 갱신"
}

step "D1 (앵커 잔차, forward only)"  ${PY} -u d1_anchor_residual.py --n 2000 --batch 250
step "D3 (교란 rollout)"             ${PY} -u d3_perturb_sr.py
step "D2 (실패 단계 귀속 rollout)"    ${PY} -u d2_failure_stage.py

say "전체 완료 -> ${OUTD}/REPORT.md"
