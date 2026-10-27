#!/usr/bin/env bash
# H200 (GPU 1장) 에서 L2_V5 LOTUS goal 스윕 — λ∈{2,3} × β∈{1,2,3} × seed∈{7,42,1000} = 18 런 —
# 을 MAXJOBS 개씩 병렬로 돌린다. 지금 GPU 에서 돌고 있는 작업이 전부 끝난 뒤에 시작한다.
#
#   (venv 를 켠 상태에서)
#   DRY=1 bash bash/Lotus_V5_h200_queue.sh                       # 계획만
#   nohup bash bash/Lotus_V5_h200_queue.sh > logs/h200_queue.out 2>&1 &
#
#   WAIT=0     기다리지 말고 바로 시작 (끊긴 뒤 이어 돌릴 때 — 완주/진행 중인 런은 건너뛴다)
#   MAXJOBS=3  병렬 수 (기본 4; 런당 VRAM ~5G, H200 141G 라 여유는 많고 CPU 가 병목)
#   LAMS="2.0 3.0" BETAS="1.0 2.0 3.0" SEEDS="7 42 1000" 로 조합을 바꾼다
#
# "지금 돌고 있는 작업" 판정: nvidia-smi 의 GPU 메모리 사용량이 MEM_IDLE(MiB) 아래로
# IDLE_CHECKS 번(60 초 간격) 연속 떨어지면 비었다고 본다. 컨테이너에서는 nvidia-smi 가
# 다른 프로세스의 PID 를 못 보여주므로 PID 가 아니라 메모리로 본다.
#
# 태그/로그/인자는 bash/Lotus_V5_run.sh 와 같다 — 결과 폴더가
#   results/lotus_v5act_c0.1_lam<λ>[_b<β>]_k25_scopeLL_goal_seed<N>
# 이라 LOTUS_MAR_sweep_report.py 가 그대로 집계한다. β=1.0 이면 _b 를 생략한다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"
source "${HERE}/bash/clare/env.sh"        # HOME 기준 Models/Datasets 경로

# 켜져 있는 환경(venv/conda)의 python 을 절대경로로 잡는다 — 떼어낸 프로세스도 같은 걸 쓴다.
PY=${PY:-$(command -v python || command -v python3)}
"${PY}" -c "import torch, lerobot" 2>/dev/null \
    || { echo "✗ ${PY} 에 torch/lerobot 이 없다 — venv 를 켜고 다시 실행하라 (예: source ~/tether/bin/activate)"; exit 1; }

SUITE=${SUITE:-goal}
C=${C:-0.1}
LAMS=${LAMS:-"2.0 3.0"}
BETAS=${BETAS:-"1.0 2.0 3.0"}
SEEDS=${SEEDS:-"7 42 1000"}
MAXJOBS=${MAXJOBS:-4}
GPU=${GPU:-0}
WAIT=${WAIT:-1}
MEM_IDLE=${MEM_IDLE:-1500}
IDLE_CHECKS=${IDLE_CHECKS:-3}
SPACE=act; SCOPE=lifelong; SCOPE_TAG=LL; LIFELONG_K=${LIFELONG_K:-25}; WARM=0; REDUCE=mean
STEPS=${STEPS:-10000}; N_EVAL=${N_EVAL:-20}; EVAL_SEED=${EVAL_SEED:-7}
ENTRY=${ENTRY:-L2_V5.py}
# 런당 스레드: CPU 를 MAXJOBS 로 나눠 쓴다 (2..6)
_per=$(( $(nproc) / MAXJOBS )); [ "${_per}" -lt 2 ] && _per=2; [ "${_per}" -gt 6 ] && _per=6
NW=${NW:-${_per}}; OMP_THREADS=${OMP_THREADS:-${_per}}

mkdir -p logs results
QLOG="logs/h200_queue_${SUITE}.log"

tag_of() {  # λ β seed
    local lt bt
    lt=$([ "$1" = "1.0" ] && echo "" || echo "_lam$1")
    bt=$([ "$2" = "1.0" ] && echo "" || echo "_b$2")
    echo "lotus_v5${SPACE}_c${C}${lt}${bt}_k${LIFELONG_K}_scope${SCOPE_TAG}_${SUITE}_seed$3"
}
train_done() {
    [ -f "results/$1/metrics_train.json" ] && return 0
    grep -q '\[LOTUS\] 완료' "results/$1/train.log" 2>/dev/null
}
# 이 스크립트는 파일로 실행되므로(cmdline 에 패턴이 없다) pgrep -f 가 자기 자신을 잡지 않는다.
already_running() { pgrep -f "${ENTRY} .*--out results/$1( |$)" >/dev/null 2>&1; }
gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null | head -1 | tr -d ' '; }

PLAN=()
for SEED in ${SEEDS}; do for LAM in ${LAMS}; do for BETA in ${BETAS}; do
    PLAN+=("${LAM}:${BETA}:${SEED}")
done; done; done

if [ -n "${DRY:-}" ]; then
    echo "H200 큐  ${SUITE} c=${C}  λ∈{${LAMS}} β∈{${BETAS}} seed∈{${SEEDS}}  → ${#PLAN[@]} 런, ${MAXJOBS} 병렬, GPU${GPU}, 런당 NW=${NW} OMP=${OMP_THREADS}"
    echo "python: ${PY}";  echo "GPU 메모리 지금: $(gpu_mem) MiB (유휴 기준 < ${MEM_IDLE})"
    for p in "${PLAN[@]}"; do
        IFS=: read -r L B S <<< "${p}"; t=$(tag_of "${L}" "${B}" "${S}")
        st=$(train_done "${t}" && echo 학습완료 || (already_running "${t}" && echo 진행중 || echo 대기))
        printf '  λ=%-4s β=%-4s seed %-5s %-52s %s\n' "${L}" "${B}" "${S}" "${t}" "${st}"
    done
    exit 0
fi

{
echo "[$(date '+%F %T')] ════ H200 큐 시작  ${SUITE} c=${C} λ∈{${LAMS}} β∈{${BETAS}} seed∈{${SEEDS}}  ${#PLAN[@]} 런 / ${MAXJOBS} 병렬 ════"
echo "  python ${PY}   NW=${NW} OMP=${OMP_THREADS}"

# 1) 지금 돌고 있는 작업이 끝날 때까지
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

# 2) 리포트 루프 (읽기 전용, 5 분마다 results/LOTUS_MAR_sweep.txt)
REPORT=LOTUS_MAR_sweep_report.py INTERVAL=300 bash bash/report_loop.sh start 2>&1 | head -1

# 3) 큐
run_one() {   # λ β seed
    local TAG; TAG=$(tag_of "$1" "$2" "$3")
    mkdir -p "results/${TAG}"
    echo "[$(date '+%F %T')] ▶ ${TAG}"
    CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=egl \
    OMP_NUM_THREADS="${OMP_THREADS}" MKL_NUM_THREADS="${OMP_THREADS}" \
    OPENBLAS_NUM_THREADS="${OMP_THREADS}" NUMEXPR_NUM_THREADS="${OMP_THREADS}" \
    "${PY}" -u "${ENTRY}" --setting lotus \
        --anchor_c "${C}" --dir_space "${SPACE}" --anchor_reduce "${REDUCE}" \
        --lambda_level "$1" --lifelong_k "${LIFELONG_K}" \
        --anchor_scope "${SCOPE}" --anchor_warmup "${WARM}" --beta "$2" \
        --suite "${SUITE}" --steps "${STEPS}" --sampler_mode hard_linear \
        --skip_base_eval --seed "$3" --eval_seed "${EVAL_SEED}" \
        --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
        --num_workers "${NW}" --out "results/${TAG}" \
        >> "results/${TAG}/train.log" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] ◀ ${TAG} exit=${rc}"
}
for p in "${PLAN[@]}"; do
    IFS=: read -r L B S <<< "${p}"; t=$(tag_of "${L}" "${B}" "${S}")
    if train_done "${t}"; then echo "[$(date '+%F %T')] ⏭ ${t} 이미 완료"; continue; fi
    if already_running "${t}"; then echo "[$(date '+%F %T')] ⏭ ${t} 이미 돌고 있다"; continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "${MAXJOBS}" ]; do wait -n; done
    run_one "${L}" "${B}" "${S}" &
    sleep 30      # 코드북/환경 생성이 겹치지 않게
done
wait
echo "[$(date '+%F %T')] ════ H200 큐 완료 ════"
"${PY}" LOTUS_MAR_sweep_report.py > /dev/null 2>&1 || true
} >> "${QLOG}" 2>&1
