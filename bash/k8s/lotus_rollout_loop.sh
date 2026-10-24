#!/usr/bin/env bash
# 파드 **안에서** 도는 롤아웃 전담 드라이버. 학습 파드가 남긴 스테이지
# 체크포인트를 기다리며 SR 행렬을 채운다.
#
#   SUITE=50 SEEDS="7 42 1000" BETA=3.0 bash bash/k8s/lotus_rollout_loop.sh
#
# 왜 분리하는가: LIBERO-50 은 스테이지마다 최대 50 칸을 돌아서, 학습 루프 안에서
# 롤아웃을 하면 학습이 아주 오래 멈춘다. 학습 파드는 --skip_eval 로 체크포인트만
# 남기고(GPU 1개), 이 파드가 다른 GPU 에서 롤아웃만 돈다.
#
# 시드는 학습 쪽과 같은 순서로 돈다. 학습이 seed7 -> 42 -> 1000 순이라
# 여기도 같은 순서로 기다리면 항상 "이미 나온 체크포인트"를 쫓아간다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${HERE}"

SUITE=${SUITE:?SUITE (goal|object|50)}
SEEDS=${SEEDS:-"7 42 1000"}
BETA=${BETA:-3.0}
LIFELONG_K=${LIFELONG_K:-25}
FORM=${FORM:-dea}
ENTRY=${ENTRY:-Lotus_L3_V7dea.py}
N_EVAL=${N_EVAL:-0}            # 0 = 학습 config.json 값을 그대로
POLL=${POLL:-300}
BASE_EVAL=${BASE_EVAL:-0}
VENV=${VENV:-/home/asdl/tether-venv}

# shellcheck disable=SC1091
. "${VENV}/bin/activate"
source "${HERE}/bash/clare/env.sh"
export MUJOCO_GL=${MUJOCO_GL:-egl} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
T=${OMP_THREADS:-8}
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T

ts() { echo "[$(date '+%F %T')] $*"; }
ts "════ 롤아웃 전담 — suite=${SUITE} β${BETA} K${LIFELONG_K}, 시드 ${SEEDS} ════"
python -c "import torch;print(f'    GPU {torch.cuda.device_count()}개  {torch.cuda.get_device_name(0)}')" 2>/dev/null

for SEED in ${SEEDS}; do
    RUN="results/lotus_l3_${FORM}_b${BETA}_k${LIFELONG_K}_${SUITE}_seed${SEED}"
    # 학습이 아직 config.json 조차 안 썼으면 생길 때까지 기다린다
    while [ ! -f "${RUN}/config.json" ]; do
        ts "⏳ ${RUN}/config.json 대기 (${POLL}s)"; sleep "${POLL}"
    done
    # 롤아웃은 L3 파일 안에 있다(--rollout_only) — 별도 파일이 필요 없다.
    ARGS=(--rollout_only --out "${RUN}" --watch --poll "${POLL}")
    [ "${N_EVAL}" != "0" ] && ARGS+=(--eval_episodes "${N_EVAL}")
    [ "${BASE_EVAL}" = "1" ] && ARGS+=(--base_eval)
    ts "▶ ${RUN} 롤아웃 시작"
    python -u "${ENTRY}" "${ARGS[@]}" >> "${RUN}/rollout.log" 2>&1
    ts "◀ ${RUN} 롤아웃 exit=$?"
    tail -3 "${RUN}/rollout.log" 2>/dev/null | sed 's/^/     /'
done
ts "════ 롤아웃 전부 종료 ════"
