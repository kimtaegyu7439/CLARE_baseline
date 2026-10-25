#!/usr/bin/env bash
# Lotus_L3_V7dea 를 **맨 GPU 서버**(쿠버네티스 아님)에서 돌린다.
#
#   bash bash/Lotus_L3_run.sh                 # goal→0, object→1, 50 학습→2/롤아웃→3
#   DRY=1 bash bash/Lotus_L3_run.sh
#   SUITES_GPU="goal:0" SPLIT="" bash bash/Lotus_L3_run.sh
#   BETA=3.0 LIFELONG_K=20 bash bash/Lotus_L3_run.sh
#
# 배치
#   SUITES_GPU  "<suite>:<gpu>" 목록 — 그 GPU 에서 학습+롤아웃을 같이 돈다
#   SPLIT       "<suite>:<학습GPU>:<롤아웃GPU>" — 학습과 롤아웃을 쪼갠다.
#               LIBERO-50 은 스테이지마다 최대 50 칸을 돌아 롤아웃이 학습을
#               아주 오래 막으므로 카드 두 장으로 나눈다.
#   어느 쪽이든 시드는 **GPU 안에서 순차**다 (한 GPU 에 한 런).
#
# 태그: results/lotus_l3_dea_b<β>_k<K>_<suite>_seed<N>
#   ★ K 를 태그에 넣는다 — K=20 으로 돌린 기존 V7dea 런
#     (lotus_v7_dea_b3.0_*) 과 섞이면 안 된다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"

# ★ 떼어낸 서브셸에는 conda 가 없다. 여기서 직접 켠다 — 안 켜면
#   "ModuleNotFoundError: No module named 'torch'" 로 즉시 죽는다.
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh
    conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

# ★ := 가 아니라 - 를 쓴다. ${X:-기본} 은 X="" (빈 문자열)도 기본값으로 바꿔
#   버려서, SPLIT="" 로 "50 은 안 띄운다" 를 표현할 수 없다. 실제로 그 때문에
#   object 만 추가하려고 SPLIT="" 를 넘겼더니 50 학습/롤아웃이 **두 번째로**
#   떠서 같은 출력 폴더에 두 프로세스가 쓰는 사고가 났다.
SUITES_GPU=${SUITES_GPU-"goal:0 object:1"}
SPLIT=${SPLIT-"50:2:3"}
SEEDS=${SEEDS:-"7 42 1000"}
BETA=${BETA:-2.0}
LIFELONG_K=${LIFELONG_K:-25}
FORM=${FORM:-dea}
SPACE=${SPACE:-act}
REDUCE=${REDUCE:-mean}
EPS_W=${EPS_W:-1e-3}
STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-20}
EVAL_SEED=${EVAL_SEED:-7}
NW=${NW:-6}
OMP_THREADS=${OMP_THREADS:-6}
POLL=${POLL:-300}
ENTRY=${ENTRY:-Lotus_L3_V7dea.py}

mkdir -p logs results
tag_of() { echo "lotus_l3_${FORM}_b${BETA}_k${LIFELONG_K}_${1}_seed${2}"; }

# 학습이 끝났나 — skip_eval 런은 metrics_train.json, 보통 런은 metrics.json.
# ★ metrics.json 만 보면 안 된다: 롤아웃 워커가 **첫 칸을 잴 때** 이미 그 파일을
#   만들므로, 학습이 안 끝난 런을 완주로 오판한다.
train_done() {
    [ -f "results/$1/metrics_train.json" ] && return 0
    grep -q '\[LOTUS\] 완료' "results/$1/train.log" 2>/dev/null
}

if [ -n "${DRY:-}" ]; then
    echo "L3 ${FORM} β${BETA} lifelong K=${LIFELONG_K}  steps ${STEPS}  시드 ${SEEDS}"
    for sg in ${SUITES_GPU}; do
        echo "  GPU ${sg#*:}  ${sg%%:*}  (학습+롤아웃)"
        for s in ${SEEDS}; do
            t=$(tag_of "${sg%%:*}" "${s}")
            printf '     seed %-5s %-52s %s\n' "${s}" "${t}" \
                "$(train_done "${t}" && echo 학습완료 || echo 미완)"
        done
    done
    for sp in ${SPLIT}; do
        IFS=: read -r su g1 g2 <<< "${sp}"
        echo "  GPU ${g1} ${su} 학습 / GPU ${g2} ${su} 롤아웃"
        for s in ${SEEDS}; do
            t=$(tag_of "${su}" "${s}")
            printf '     seed %-5s %-52s %s\n' "${s}" "${t}" \
                "$(train_done "${t}" && echo 학습완료 || echo 미완)"
        done
    done
    exit 0
fi

# ── 한 GPU 에서 한 스위트의 시드를 순차로 (학습[+롤아웃]) ────────────────────
train_loop() {
    local SUITE=$1 GPU=$2 SKIP=$3
    local LOG="logs/l3_${SUITE}_train.log"
    {
        echo "[$(date '+%F %T')] ════ L3 ${SUITE} 학습 GPU${GPU} β${BETA} K${LIFELONG_K} ════"
        for SEED in ${SEEDS}; do
            TAG=$(tag_of "${SUITE}" "${SEED}")
            if train_done "${TAG}"; then
                echo "[$(date '+%F %T')] ⏭ ${TAG} 학습 이미 완료"; continue
            fi
            mkdir -p "results/${TAG}"
            echo "[$(date '+%F %T')] ▶ ${TAG}"
            CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
            MUJOCO_GL=egl \
            OMP_NUM_THREADS="${OMP_THREADS}" MKL_NUM_THREADS="${OMP_THREADS}" \
            OPENBLAS_NUM_THREADS="${OMP_THREADS}" NUMEXPR_NUM_THREADS="${OMP_THREADS}" \
            python -u "${ENTRY}" --suite "${SUITE}" --steps "${STEPS}" \
                --lifelong_k "${LIFELONG_K}" \
                --anchor_form "${FORM}" --beta "${BETA}" --eps_w "${EPS_W}" \
                --dir_space "${SPACE}" --anchor_reduce "${REDUCE}" \
                --sampler_mode hard_linear --skip_base_eval \
                --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
                --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
                --num_workers "${NW}" --out "results/${TAG}" ${SKIP} \
                >> "results/${TAG}/train.log" 2>&1
            rc=$?    # ★ 먼저 잡는다. echo 안의 $(date) 가 $? 를 0 으로 덮어쓴다
            echo "[$(date '+%F %T')] ◀ ${TAG} exit=${rc}"
        done
        echo "[$(date '+%F %T')] ════ ${SUITE} 학습 종료 ════"
    } >> "${LOG}" 2>&1
}

# ── 롤아웃 전담: 학습과 같은 시드 순서로 쫓아간다 ───────────────────────────
rollout_loop() {
    local SUITE=$1 GPU=$2
    local LOG="logs/l3_${SUITE}_rollout.log"
    {
        echo "[$(date '+%F %T')] ════ L3 ${SUITE} 롤아웃 전담 GPU${GPU} ════"
        for SEED in ${SEEDS}; do
            TAG=$(tag_of "${SUITE}" "${SEED}")
            while [ ! -f "results/${TAG}/config.json" ]; do
                echo "[$(date '+%F %T')] ⏳ ${TAG}/config.json 대기"; sleep "${POLL}"
            done
            echo "[$(date '+%F %T')] ▶ ${TAG} 롤아웃"
            CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" \
            MUJOCO_GL=egl \
            OMP_NUM_THREADS="${OMP_THREADS}" MKL_NUM_THREADS="${OMP_THREADS}" \
            OPENBLAS_NUM_THREADS="${OMP_THREADS}" NUMEXPR_NUM_THREADS="${OMP_THREADS}" \
            python -u "${ENTRY}" --rollout_only --out "results/${TAG}" \
                --watch --poll "${POLL}" \
                >> "results/${TAG}/rollout.log" 2>&1
            rc=$?
            echo "[$(date '+%F %T')] ◀ ${TAG} 롤아웃 exit=${rc}"
        done
        echo "[$(date '+%F %T')] ════ ${SUITE} 롤아웃 종료 ════"
    } >> "${LOG}" 2>&1
}

# ★ 두 번째 방어선: 같은 스위트가 이미 돌고 있으면 띄우지 않는다.
#   한 출력 폴더에 두 학습이 쓰면 체크포인트가 깨진다.
already_running() {
    pgrep -f "Lotus_L3_V7dea.py --suite $1 " >/dev/null 2>&1
}

echo "════ L3 ${FORM} β${BETA} lifelong K=${LIFELONG_K} — 시드 ${SEEDS} ════"
for sg in ${SUITES_GPU}; do
    SUITE="${sg%%:*}"; GPU="${sg#*:}"
    if already_running "${SUITE}"; then
        echo "  ⏭ ${SUITE} 는 이미 돌고 있다 — 띄우지 않는다"; continue
    fi
    setsid nohup bash -c "$(declare -f tag_of train_done train_loop); \
        SUITES_GPU='${SUITES_GPU}' SEEDS='${SEEDS}' BETA='${BETA}' \
        LIFELONG_K='${LIFELONG_K}' FORM='${FORM}' SPACE='${SPACE}' \
        REDUCE='${REDUCE}' EPS_W='${EPS_W}' STEPS='${STEPS}' N_EVAL='${N_EVAL}' \
        EVAL_SEED='${EVAL_SEED}' NW='${NW}' OMP_THREADS='${OMP_THREADS}' \
        ENTRY='${ENTRY}'; cd '${HERE}'; train_loop '${SUITE}' '${GPU}' ''" \
        > /dev/null 2>&1 &
    echo "  ▶ GPU ${GPU}  ${SUITE} 학습+롤아웃   로그 logs/l3_${SUITE}_train.log"
    sleep 3
done
for sp in ${SPLIT}; do
    IFS=: read -r SUITE G1 G2 <<< "${sp}"
    if already_running "${SUITE}"; then
        echo "  ⏭ ${SUITE} 는 이미 돌고 있다 — 띄우지 않는다"; continue
    fi
    setsid nohup bash -c "$(declare -f tag_of train_done train_loop); \
        SEEDS='${SEEDS}' BETA='${BETA}' LIFELONG_K='${LIFELONG_K}' FORM='${FORM}' \
        SPACE='${SPACE}' REDUCE='${REDUCE}' EPS_W='${EPS_W}' STEPS='${STEPS}' \
        N_EVAL='${N_EVAL}' EVAL_SEED='${EVAL_SEED}' NW='${NW}' \
        OMP_THREADS='${OMP_THREADS}' ENTRY='${ENTRY}'; cd '${HERE}'; \
        train_loop '${SUITE}' '${G1}' '--skip_eval'" > /dev/null 2>&1 &
    echo "  ▶ GPU ${G1}  ${SUITE} 학습만      로그 logs/l3_${SUITE}_train.log"
    sleep 3
    setsid nohup bash -c "$(declare -f tag_of rollout_loop); \
        SEEDS='${SEEDS}' BETA='${BETA}' LIFELONG_K='${LIFELONG_K}' FORM='${FORM}' \
        POLL='${POLL}' OMP_THREADS='${OMP_THREADS}' ENTRY='${ENTRY}'; \
        cd '${HERE}'; rollout_loop '${SUITE}' '${G2}'" > /dev/null 2>&1 &
    echo "  ▶ GPU ${G2}  ${SUITE} 롤아웃 전담  로그 logs/l3_${SUITE}_rollout.log"
    sleep 3
done
echo
echo "상태:  nvidia-smi;  tail -f logs/l3_*_train.log"
