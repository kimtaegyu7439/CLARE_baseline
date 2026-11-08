#!/usr/bin/env bash
# H200 (GPU 1장) — TSM 구성요소 분해 ablation, LOTUS 셋팅.
#   4 팔 (time_bin / state_mean / state_map / tether) × 3 시드 (7/42/1000) = 12 런.
#   MAXJOBS 개씩 병렬. 시드 바깥 / 팔 안쪽으로 돌려 중간에 끊겨도 네 팔이 고르게 찬다.
#
#   (venv 를 켠 상태에서, 레포 루트에서)
#   bash bash/h200_lotus_object_assets.sh                    # 처음 한 번: base ckpt + object 스플릿
#   DRY=1 bash bash/Lotus_TSM_h200_queue.sh                   # 계획만
#   nohup bash bash/Lotus_TSM_h200_queue.sh > logs/tsm_queue.out 2>&1 &
#   tail -f logs/tsm_queue_object.log                         # 진행
#   cat results/LOTUS_TSM_ablation.txt                        # 5 분마다 갱신되는 집계
#
#   SUITE=object  SEEDS="7 42 1000"  ARMS="time_bin state_mean state_map tether"
#   LAM=0.7  (네 팔 공통 λ)   BETA=0.6  C=0.1  (tether 의 MAR 계수 — object 최적, LOTUS_MAR_sweep)
#   MAXJOBS=4 (런당 VRAM ~5G, 병목은 CPU — 롤아웃·데이터로더)   GPU=0
#   WAIT=1    지금 GPU 가 비기를 기다린 뒤 시작 (기본 0 = 바로 시작)
#
# 결과 폴더  results/lotus_tsm_<팔>_<suite>_seed<N>   (LOTUS_TSM_ablation_report.py 가 집계)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"
source "${HERE}/bash/clare/env.sh"

PY=${PY:-$(command -v python || command -v python3)}
"${PY}" -c "import torch, lerobot" 2>/dev/null \
    || { echo "✗ ${PY} 에 torch/lerobot 이 없다 — venv 를 켜고 다시 실행하라 (예: source ~/tether/bin/activate)"; exit 1; }
# H200 venv 의 lerobot 이 옛 버전이면 레포의 lerobot_lsy/src 를 앞에 둔다 (Lotus_V5_h200_queue.sh 와 같다)
_chk='from lerobot.envs.configs import LIBERO_TASK_PREFIX_TO_BENCHMARK; from lerobot.datasets.factory import resolve_delta_timestamps'
if ! "${PY}" -c "${_chk}" 2>/dev/null; then
    if [ -d "${HERE}/lerobot_lsy/src/lerobot" ]; then
        export PYTHONPATH="${HERE}/lerobot_lsy/src${PYTHONPATH:+:${PYTHONPATH}}"
        "${PY}" -c "${_chk}" 2>/dev/null \
            || { echo "✗ 레포의 lerobot_lsy 를 PYTHONPATH 에 넣어도 import 가 안 된다:"; "${PY}" -c "${_chk}" 2>&1 | tail -3; exit 1; }
        echo "  lerobot: 설치본이 옛 버전이라 레포의 lerobot_lsy/src 를 PYTHONPATH 로 쓴다"
    else
        echo "✗ 설치된 lerobot 이 옛 버전이고 레포에 lerobot_lsy/ 도 없다"; exit 1
    fi
fi

SUITE=${SUITE:-object}
SEEDS=${SEEDS:-"7 42 1000"}
ARMS=${ARMS:-"time_bin state_mean state_map tether"}
LAM=${LAM:-0.7}; BETA=${BETA:-0.6}; C=${C:-0.1}
LIFELONG_K=${LIFELONG_K:-25}
STEPS=${STEPS:-10000}; N_EVAL=${N_EVAL:-20}; EVAL_SEED=${EVAL_SEED:-7}
MAXJOBS=${MAXJOBS:-4}; GPU=${GPU:-0}
WAIT=${WAIT:-0}; MEM_IDLE=${MEM_IDLE:-1500}; IDLE_CHECKS=${IDLE_CHECKS:-5}
_per=$(( $(nproc) / MAXJOBS )); [ "${_per}" -lt 2 ] && _per=2; [ "${_per}" -gt 6 ] && _per=6
NW=${NW:-${_per}}; OMP_THREADS=${OMP_THREADS:-${_per}}
ENTRY=Lotus_TSM_ablation.py

# ── 자산 확인 ──────────────────────────────────────────────────────────────
SUITE_UP=$(echo "${SUITE}" | tr '[:lower:]' '[:upper:]')
BASE_CKPT="${CLARE_MODEL_ROOT}/Lotus_clare_libero_${SUITE}_pretrain/model.safetensors"
SPLIT_DIR="${HF_LEROBOT_HOME}/LOTUS_setting/LIBERO-${SUITE_UP}/stage1"
[ -f "${BASE_CKPT}" ] || { echo "✗ base 체크포인트가 없다: ${BASE_CKPT}   → bash bash/h200_lotus_${SUITE}_assets.sh"; exit 1; }
[ -d "${SPLIT_DIR}" ] || { echo "✗ LOTUS 스플릿이 없다: ${SPLIT_DIR}   → bash bash/h200_lotus_${SUITE}_assets.sh"; exit 1; }
[ -f "${HOME}/Models/dit_flow_mt_libero_90_pretrain/config.json" ] || [ -d "${PRETRAIN_PATH}" ] \
    || echo "  (주의) ${PRETRAIN_PATH} 가 없다 — 드라이버가 필요로 하면 실패한다"

mkdir -p logs results
QLOG="logs/tsm_queue_${SUITE}.log"

tag_of() { echo "lotus_tsm_$1_${SUITE}_seed$2"; }          # 팔 seed
train_done() {
    [ -f "results/$1/metrics_train.json" ] && return 0
    grep -q '\[LOTUS\] 완료' "results/$1/train.log" 2>/dev/null
}
already_running() { pgrep -f "${ENTRY} .*--out results/$1( |$)" >/dev/null 2>&1; }
gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null | head -1 | tr -d ' '; }

PLAN=()
for SEED in ${SEEDS}; do for ARM in ${ARMS}; do PLAN+=("${ARM}:${SEED}"); done; done

if [ -n "${DRY:-}" ]; then
    echo "TSM ablation 큐  ${SUITE}  팔∈{${ARMS}} seed∈{${SEEDS}}  λ=${LAM} (tether β=${BETA} c=${C})  K=${LIFELONG_K}"
    echo "  → ${#PLAN[@]} 런, ${MAXJOBS} 병렬, GPU${GPU}, 런당 NW=${NW} OMP=${OMP_THREADS}   python ${PY}"
    echo "  GPU 메모리 지금: $(gpu_mem) MiB"
    for p in "${PLAN[@]}"; do
        IFS=: read -r A S <<< "${p}"; t=$(tag_of "${A}" "${S}")
        s=$(train_done "${t}" && echo 학습완료 || (already_running "${t}" && echo 진행중 || echo 대기))
        printf '  %-11s seed %-5s %-40s %s\n' "${A}" "${S}" "${t}" "${s}"
    done
    exit 0
fi

{
echo "[$(date '+%F %T')] ════ TSM ablation 큐 시작  ${SUITE}  팔∈{${ARMS}} seed∈{${SEEDS}}  ${#PLAN[@]} 런 / ${MAXJOBS} 병렬 ════"
echo "  python ${PY}   NW=${NW} OMP=${OMP_THREADS}   λ=${LAM}  tether: β=${BETA} c=${C}"

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date '+%F %T')] pid ${WAIT_PID} 가 끝날 때까지 대기"
    while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
fi
if [ "${WAIT}" = "1" ]; then
    idle=0
    while :; do
        m=$(gpu_mem); m=${m:-0}
        if [ "${m}" -lt "${MEM_IDLE}" ]; then idle=$((idle+1)); else idle=0; fi
        [ "${idle}" -ge "${IDLE_CHECKS}" ] && break
        sleep 60
    done
    echo "[$(date '+%F %T')] GPU 비었다 (${m} MiB) — 시작"
fi

# 리포트 루프 (읽기 전용, 5 분마다 results/LOTUS_TSM_ablation.txt)
REPORT=LOTUS_TSM_ablation_report.py INTERVAL=300 bash bash/report_loop.sh start 2>&1 | head -1

run_one() {   # 팔 seed
    local TAG; TAG=$(tag_of "$1" "$2")
    mkdir -p "results/${TAG}"
    echo "[$(date '+%F %T')] ▶ ${TAG}"
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=egl \
    OMP_NUM_THREADS="${OMP_THREADS}" MKL_NUM_THREADS="${OMP_THREADS}" \
    OPENBLAS_NUM_THREADS="${OMP_THREADS}" NUMEXPR_NUM_THREADS="${OMP_THREADS}" \
    "${PY}" -u "${ENTRY}" --tsm_mode "$1" \
        --lambda_level "${LAM}" --beta "${BETA}" --anchor_c "${C}" \
        --suite "${SUITE}" --steps "${STEPS}" --lifelong_k "${LIFELONG_K}" \
        --anchor_scope lifelong --skip_base_eval \
        --seed "$2" --eval_seed "${EVAL_SEED}" \
        --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
        --num_workers "${NW}" --out "results/${TAG}" \
        >> "results/${TAG}/train.log" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] ◀ ${TAG} exit=${rc}"
    "${PY}" LOTUS_TSM_ablation_report.py > /dev/null 2>&1 || true
}
for p in "${PLAN[@]}"; do
    IFS=: read -r A S <<< "${p}"; t=$(tag_of "${A}" "${S}")
    if train_done "${t}"; then echo "[$(date '+%F %T')] ⏭ ${t} 이미 완료"; continue; fi
    if already_running "${t}"; then echo "[$(date '+%F %T')] ⏭ ${t} 이미 돌고 있다"; continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "${MAXJOBS}" ]; do wait -n; done
    run_one "${A}" "${S}" &
    sleep 30      # 코드북/환경 생성이 겹치지 않게
done
wait
echo "[$(date '+%F %T')] ════ TSM ablation 큐 완료 ════"
"${PY}" LOTUS_TSM_ablation_report.py 2>&1 | sed -n '1,60p'
} >> "${QLOG}" 2>&1
