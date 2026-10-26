#!/usr/bin/env bash
# L2_V13 (dea W_e 를 크기+방향에 공유) 를 LOTUS 셋팅으로 돌린다.
#
#   bash bash/Lotus_V5_run.sh            # c0.1: goal→0, object→1 / c0.01: goal→2, object→3
#   DRY=1 bash bash/Lotus_V5_run.sh
#   JOBS="goal:0.1:0 object:0.1:1" bash bash/Lotus_V5_run.sh
#
# JOBS 는 "<suite>:<c>:<gpu>" 목록이다. 한 GPU 에 한 런이고, 그 GPU 에서
# 학습이 끝나면 이어서 롤아웃까지 한다(goal/object 는 4 stage x 1 task 라
# 롤아웃이 LIBERO-50 만큼 오래 걸리지 않는다).
#
# 태그: results/lotus_v5act_c<c>_k25_<suite>_seed<seed>
#   c 와 K 를 태그에 넣는다 — dea 런(lotus_l3_dea_*)과 섞이면 안 된다.
#
# 드라이버는 L2_V5.py --setting lotus 이고, 그 안에서 Lotus_L3_V7dea 를 쓴다
# (lifelong 코드북 K 를 태스크당 고정하는 --lifelong_k 가 거기에만 있다).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"

# ★ 떼어낸 서브셸에는 conda 가 없다. 여기서 직접 켠다 — 안 켜면
#   "ModuleNotFoundError: No module named 'torch'" 로 즉시 죽는다.
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh
    conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

JOBS=${JOBS-"goal:3.0:0 object:3.0:1"}   # <suite>:<β>:<gpu>
SEEDS=${SEEDS:-"7"}
SPACE=${SPACE:-act}
SCOPE=${SCOPE:-lifelong}   # 앵커가 보존할 과거의 범위 (base 는 세지 않는다)
SCOPE_TAG=$([ "${SCOPE}" = "all" ] && echo ALL || echo LL)
LIFELONG_K=${LIFELONG_K:-25}
LAM=${LAM:-3.0}            # dea 드라이버 기본과 같은 3.0
EPS_W=${EPS_W:-1e-3}
REDUCE=${REDUCE:-mean}
STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-20}
EVAL_SEED=${EVAL_SEED:-7}
NW=${NW:-6}
OMP_THREADS=${OMP_THREADS:-6}
ENTRY=${ENTRY:-L2_V13.py}

mkdir -p logs results
# ★ 태그에 scope 와 λ 를 넣는다.
#   scope: base 를 과거로 세던 예전 런(INVALID.md)과 섞이면 안 된다.
#   λ    : 앵커 세기가 다르면 다른 팔이다. λ=1.0 은 L2_V5 의 기본값이고 기존
#          런 10 개가 전부 그 값이라, **1.0 일 때만 생략**해 이름을 유지한다
#          (이미 돌고 있는 런의 --out 경로를 바꾸면 그 프로세스가 깨진다).
tag_of() { echo "lotus_v13_b${2}_lam${LAM}_k${LIFELONG_K}_scope${SCOPE_TAG}_${1}_seed${3}"; }

# 학습이 끝났나 — 보통 런은 LC.main 이 맨 끝에만 찍는 "[LOTUS] 완료".
# metrics.json 으로 보면 안 된다(롤아웃 워커가 첫 칸에서 만든다).
train_done() {
    [ -f "results/$1/metrics_train.json" ] && return 0
    grep -q '\[LOTUS\] 완료' "results/$1/train.log" 2>/dev/null
}
already_running() { pgrep -f "L2_V13.py .*--out results/$1( |$)" >/dev/null 2>&1; }

if [ -n "${DRY:-}" ]; then
    echo "L2_V13 (${SPACE} 공간) LOTUS 셋팅  λ_lvl=${LAM}  ε_w=${EPS_W}  lifelong K=${LIFELONG_K}  steps ${STEPS}"
    for j in ${JOBS}; do
        IFS=: read -r SU C G <<< "${j}"
        for s in ${SEEDS}; do
            t=$(tag_of "${SU}" "${C}" "${s}")
            printf '  GPU %-2s %-8s β=%-6s seed %-5s %-52s %s\n' \
                "${G}" "${SU}" "${C}" "${s}" "${t}" \
                "$(train_done "${t}" && echo 학습완료 || echo 미완)"
        done
    done
    exit 0
fi

run_one() {
    local SUITE=$1 C=$2 GPU=$3
    local LOG="logs/v13_${SUITE}_b${C}.log"
    {
        echo "[$(date '+%F %T')] ════ V13 ${SUITE} β=${C} GPU${GPU} (${SPACE} 공간) ════"
        for SEED in ${SEEDS}; do
            TAG=$(tag_of "${SUITE}" "${C}" "${SEED}")
            if train_done "${TAG}"; then
                echo "[$(date '+%F %T')] ⏭ ${TAG} 이미 완료"; continue
            fi
            mkdir -p "results/${TAG}"
            echo "[$(date '+%F %T')] ▶ ${TAG}"
            CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=egl \
            OMP_NUM_THREADS="${OMP_THREADS}" MKL_NUM_THREADS="${OMP_THREADS}" \
            OPENBLAS_NUM_THREADS="${OMP_THREADS}" NUMEXPR_NUM_THREADS="${OMP_THREADS}" \
            python -u "${ENTRY}" --setting lotus \
                --beta "${C}" --eps_w "${EPS_W}" --dir_space "${SPACE}" \
                --lambda_level "${LAM}" --lifelong_k "${LIFELONG_K}" \
                --anchor_scope "${SCOPE}" \
                --suite "${SUITE}" --steps "${STEPS}" --sampler_mode hard_linear \
                --skip_base_eval --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
                --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
                --num_workers "${NW}" --out "results/${TAG}" \
                >> "results/${TAG}/train.log" 2>&1
            rc=$?    # ★ 먼저 잡는다. echo 안의 $(date) 가 $? 를 0 으로 덮어쓴다
            echo "[$(date '+%F %T')] ◀ ${TAG} exit=${rc}"
        done
        echo "[$(date '+%F %T')] ════ ${SUITE} c=${C} 종료 ════"
    } >> "${LOG}" 2>&1
}

echo "════ L2_V13 (${SPACE}) LOTUS — β, λ_lvl=${LAM}, K=${LIFELONG_K}, scope=${SCOPE}, 시드 ${SEEDS} ════"
for j in ${JOBS}; do
    IFS=: read -r SUITE C GPU <<< "${j}"
    t=$(tag_of "${SUITE}" "${C}" "$(echo ${SEEDS} | awk '{print $1}')")
    if already_running "${t}"; then
        echo "  ⏭ ${t} 이미 돌고 있다"; continue
    fi
    setsid nohup bash -c "$(declare -f tag_of train_done run_one); \
        SEEDS='${SEEDS}' SPACE='${SPACE}' SCOPE='${SCOPE}' SCOPE_TAG='${SCOPE_TAG}' \
        EPS_W='${EPS_W}' \
        LIFELONG_K='${LIFELONG_K}' LAM='${LAM}' \
        REDUCE='${REDUCE}' STEPS='${STEPS}' N_EVAL='${N_EVAL}' \
        EVAL_SEED='${EVAL_SEED}' NW='${NW}' OMP_THREADS='${OMP_THREADS}' \
        ENTRY='${ENTRY}'; cd '${HERE}'; run_one '${SUITE}' '${C}' '${GPU}'" \
        > /dev/null 2>&1 &
    echo "  ▶ GPU ${GPU}  ${SUITE}  β=${C}   로그 logs/v13_${SUITE}_b${C}.log"
    sleep 3
done
echo
echo "상태:  nvidia-smi;  tail -f logs/v5_*.log"
echo "리포트: V13 은 LOTUS_act_V5_report.py 의 V13 절에 들어간다 (--v13_betas)"
