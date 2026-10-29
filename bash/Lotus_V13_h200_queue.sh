#!/usr/bin/env bash
# H200 (GPU 1장) 에서 L2_V13 LOTUS **object** 스윕 — β × λ × seed 를 MAXJOBS 개씩 병렬로 돌린다.
# Lotus_V5_h200_queue.sh 의 V13 판. 태그·인자는 bash/Lotus_V13_run.sh 와 같아
# LOTUS_act_V5_report.py 의 V13 절이 그대로 집계한다.
#
#   (venv 를 켠 상태에서)
#   DRY=1 bash bash/Lotus_V13_h200_queue.sh                       # 계획만
#   nohup bash bash/Lotus_V13_h200_queue.sh > logs/h200_v13_queue.out 2>&1 &
#
#   SUITE=object   기본. goal 도 된다
#   BETAS="1.0 1.5 2.0" LAMS="1.5 2.0" SEEDS="7 42 1000"   조합 (기본 3×2×3 = 18 런)
#     ★ 기본 격자는 다른 서버와 겹치지 않게 골랐다. object 는 서버1(λ=3, β=1,2) 과
#       4090_2/서버1 GPU (λ 0.7~1.2, β 1~2) 가 이미 돌고 있어, 그 사이 λ∈{1.5, 2} 를 메운다.
#       다른 조합이 필요하면 BETAS/LAMS 를 넘기면 된다. 리포트는 폴더명을 파싱하므로 격자에 무관하다.
#   MAXJOBS=4  병렬 수 (런당 VRAM ~5G. CPU 가 병목이라 nproc/MAXJOBS 로 스레드를 나눈다)
#   WAIT=0     GPU 가 비기를 기다리지 않고 바로 시작
#   WAIT_PID=<pid>  그 프로세스가 끝난 뒤 시작
#
# 결과 폴더: results/lotus_v13_b<β>_lam<λ>_k25_scopeLL_<suite>_seed<N>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"
source "${HERE}/bash/clare/env.sh"
PY=${PY:-$(command -v python || command -v python3)}
"${PY}" -c "import torch, lerobot" 2>/dev/null \
    || { echo "✗ ${PY} 에 torch/lerobot 이 없다 — venv 를 켜고 다시 실행하라 (예: source ~/tether/bin/activate)"; exit 1; }
# ★ H200 의 venv 에 깔린 lerobot 은 옛 lerobot_lsy 라 LIBERO_TASK_PREFIX_TO_BENCHMARK,
#   resolve_delta_timestamps 가 없다. 레포 안 lerobot_lsy/src 를 PYTHONPATH 맨 앞에 둔다.
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
BETAS=${BETAS:-"1.0 1.5 2.0"}
LAMS=${LAMS:-"1.5 2.0"}
SEEDS=${SEEDS:-"7 42 1000"}
MAXJOBS=${MAXJOBS:-4}
GPU=${GPU:-0}
WAIT=${WAIT:-1}
MEM_IDLE=${MEM_IDLE:-1500}
IDLE_CHECKS=${IDLE_CHECKS:-5}
SPACE=act; SCOPE=lifelong; SCOPE_TAG=LL; LIFELONG_K=${LIFELONG_K:-25}; EPS_W=${EPS_W:-1e-3}
STEPS=${STEPS:-10000}; N_EVAL=${N_EVAL:-20}; EVAL_SEED=${EVAL_SEED:-7}
ENTRY=${ENTRY:-L2_V13.py}
_per=$(( $(nproc) / MAXJOBS )); [ "${_per}" -lt 2 ] && _per=2; [ "${_per}" -gt 6 ] && _per=6
NW=${NW:-${_per}}; OMP_THREADS=${OMP_THREADS:-${_per}}

# 자산 확인 — 없으면 bash/h200_lotus_object_assets.sh 를 먼저
CK="${CLARE_MODEL_ROOT}/Lotus_clare_libero_${SUITE}_pretrain/model.safetensors"
SPLIT="${HF_LEROBOT_HOME}/LOTUS_setting/LIBERO-$(echo "${SUITE}" | tr a-z A-Z)/stage4/meta/info.json"
[ -s "${CK}" ]    || { echo "✗ 사전학습 체크포인트가 없다: ${CK}   → bash bash/h200_lotus_${SUITE}_assets.sh"; exit 1; }
[ -f "${SPLIT}" ] || { echo "✗ LOTUS 분할이 없다: ${SPLIT}   → bash bash/h200_lotus_${SUITE}_assets.sh"; exit 1; }

mkdir -p logs results
QLOG="logs/h200_v13_queue_${SUITE}.log"

tag_of() {  # β λ seed
    echo "lotus_v13_b$1_lam$2_k${LIFELONG_K}_scope${SCOPE_TAG}_${SUITE}_seed$3"
}
train_done() {
    [ -f "results/$1/metrics_train.json" ] && return 0
    grep -q '\[LOTUS\] 완료' "results/$1/train.log" 2>/dev/null
}
already_running() { pgrep -f "${ENTRY} .*--out results/$1( |$)" >/dev/null 2>&1; }
gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null | head -1 | tr -d ' '; }

PLAN=()
for SEED in ${SEEDS}; do for LAM in ${LAMS}; do for BETA in ${BETAS}; do
    PLAN+=("${BETA}:${LAM}:${SEED}")
done; done; done

if [ -n "${DRY:-}" ]; then
    echo "H200 V13 큐  ${SUITE}  β∈{${BETAS}} λ∈{${LAMS}} seed∈{${SEEDS}}  → ${#PLAN[@]} 런, ${MAXJOBS} 병렬, GPU${GPU}, 런당 NW=${NW} OMP=${OMP_THREADS}"
    echo "python: ${PY}";  echo "GPU 메모리 지금: $(gpu_mem) MiB (유휴 기준 < ${MEM_IDLE})"
    for p in "${PLAN[@]}"; do
        IFS=: read -r B L S <<< "${p}"; t=$(tag_of "${B}" "${L}" "${S}")
        st=$(train_done "${t}" && echo 학습완료 || (already_running "${t}" && echo 진행중 || echo 대기))
        printf '  β=%-4s λ=%-4s seed %-5s %-56s %s\n' "${B}" "${L}" "${S}" "${t}" "${st}"
    done
    exit 0
fi

{
echo "[$(date '+%F %T')] ════ H200 V13 큐 시작  ${SUITE} β∈{${BETAS}} λ∈{${LAMS}} seed∈{${SEEDS}}  ${#PLAN[@]} 런 / ${MAXJOBS} 병렬 ════"
echo "  python ${PY}   NW=${NW} OMP=${OMP_THREADS}"

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date '+%F %T')] pid ${WAIT_PID} 가 끝날 때까지 대기"
    while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
    echo "[$(date '+%F %T')] pid ${WAIT_PID} 종료"
fi
if [ "${WAIT}" = "1" ]; then
    idle=0
    while :; do
        m=$(gpu_mem); m=${m:-0}
        if [ "${m}" -lt "${MEM_IDLE}" ]; then idle=$((idle+1)); else idle=0; fi
        [ "${idle}" -ge "${IDLE_CHECKS}" ] && break
        [ $(( $(date +%s) % 1800 )) -lt 60 ] && echo "[$(date '+%F %T')] 대기 중 — GPU 메모리 ${m} MiB (기준 < ${MEM_IDLE}, 연속 ${idle}/${IDLE_CHECKS})"
        sleep 60
    done
    echo "[$(date '+%F %T')] GPU 비었다 (${m} MiB) — 시작"
fi

# 리포트 루프 (읽기 전용, 5 분마다 results/LOTUS_V13_sweep.txt)
#   FAMILY=v13 LOTUS_MAR_sweep_report.py 는 lotus_v13_* 폴더명을 파싱해 (λ,β)×시드를 집계한다
#   (4090_2 동기화 루프와 같은 리포트). 루프 프로세스가 FAMILY 를 물려받는다.
FAMILY=v13 REPORT=LOTUS_MAR_sweep_report.py INTERVAL=300 bash bash/report_loop.sh start 2>&1 | head -1

run_one() {   # β λ seed
    local TAG; TAG=$(tag_of "$1" "$2" "$3")
    mkdir -p "results/${TAG}"
    echo "[$(date '+%F %T')] ▶ ${TAG}"
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=egl \
    OMP_NUM_THREADS="${OMP_THREADS}" MKL_NUM_THREADS="${OMP_THREADS}" \
    OPENBLAS_NUM_THREADS="${OMP_THREADS}" NUMEXPR_NUM_THREADS="${OMP_THREADS}" \
    "${PY}" -u "${ENTRY}" --setting lotus \
        --beta "$1" --eps_w "${EPS_W}" --dir_space "${SPACE}" \
        --lambda_level "$2" --lifelong_k "${LIFELONG_K}" \
        --anchor_scope "${SCOPE}" \
        --suite "${SUITE}" --steps "${STEPS}" --sampler_mode hard_linear \
        --skip_base_eval --seed "$3" --eval_seed "${EVAL_SEED}" \
        --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
        --num_workers "${NW}" --out "results/${TAG}" \
        >> "results/${TAG}/train.log" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] ◀ ${TAG} exit=${rc}"
}
for p in "${PLAN[@]}"; do
    IFS=: read -r B L S <<< "${p}"; t=$(tag_of "${B}" "${L}" "${S}")
    if train_done "${t}"; then echo "[$(date '+%F %T')] ⏭ ${t} 이미 완료"; continue; fi
    if already_running "${t}"; then echo "[$(date '+%F %T')] ⏭ ${t} 이미 돌고 있다"; continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "${MAXJOBS}" ]; do wait -n; done
    run_one "${B}" "${L}" "${S}" &
    sleep 30      # 코드북/환경 생성이 겹치지 않게
done
wait
echo "[$(date '+%F %T')] ════ H200 V13 큐 완료 ════"
} >> "${QLOG}" 2>&1
echo "큐를 띄웠다 — 진행: tail -f ${QLOG}   결과: results/LOTUS_V13_sweep.txt"
