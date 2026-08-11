#!/usr/bin/env bash
#
# R3 — end-effector 절대 궤적을 방법 × CL 스테이지로 그린다 (libero_spatial)
#
# 2 x 2 패널(seq / EWC λ=100 / ER / PackNet), 패널 안에서 CL 스테이지마다 선 색이
# 다른 3D 궤적. 축은 EE 절대 좌표 (x, y, z) [m].
#
# 네 방법 × 네 스테이지의 체크포인트를 **모두 같은 태스크**(PROBE_TASK, 기본 0)에서
# 굴린다. 초기 상태는 인덱스 0..N-1로 지정하므로 16개 체크포인트가 완전히 같은 초기
# 조건을 본다 — 궤적을 겹쳐 읽을 수 있다. 학습은 하지 않는다.
#
# 읽는 법
#   stage 0 (밝은 파랑) = probe_task를 방금 배운 상태. 이게 기준선이다.
#   stage 1~3 (점점 어두워짐) = 그 뒤로 다른 태스크를 배운 상태.
#   색이 갈라지는 정도가 곧 망각의 크기. 겹치면 안 잊은 것이다.
#
# PackNet은 추론 시 마스크를 적용한다(체크포인트의 mask.safetensors). task j를 평가할 때
# j보다 뒤 태스크가 소유한 슬롯을 0으로 만든다 — 그래야 PackNet이 PackNet으로 동작한다.
# 이론상 네 스테이지가 거의 겹쳐야 하고, 안 겹치면 그 자체가 발견이다.
# NO_PACKNET_MASK=1 로 끄면 저장된 가중치를 그대로 쓴다(다른 세 방법과 같은 추론 경로).
#
# 사용법
#   bash bash/E0/R3.sh
#   PROBE_TASK=1 bash bash/E0/R3.sh                 # 태스크 1에서 본다
#   NUM_ROLLOUTS=3 MAX_STEPS=200 bash bash/E0/R3.sh # 빠른 예비 실행
#   PLOT_ONLY=1 bash bash/E0/R3.sh                  # 캐시로 그림만 다시
#   REDO=1 bash bash/E0/R3.sh                       # 캐시 무시하고 다시 굴리기
#
# 비용 감각: 체크포인트 16개 × 롤아웃 5개 × 최대 300스텝. 스텝마다 물리 상태를 읽고
# (공짜) 8스텝마다 flow matching 적분이 한 번 돈다. 20~30분 규모다.
# 캐시가 <RUN_DIR>/<method>_stage<k>.npz 에 남으므로 중간에 죽어도 끝난 것은 안 굴린다.

set -uo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}

# HF_LEROBOT_HOME / HF_HUB_CACHE / PRETRAIN_PATH 를 세팅한다.
source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

R3_PY=./lerobot_lsy/src/lerobot/scripts/R3.py
PYTHON=${PYTHON:-python}   # conda clare 환경이 활성화돼 있다고 가정. 아니면 PYTHON=... 로 지정.

# ── 무엇을 볼 것인가 ──────────────────────────────────────────────────────────
SEED=${SEED:-42}
PROBE_TASK=${PROBE_TASK:-0}               # 모든 체크포인트를 굴릴 태스크
NUM_STAGES=${NUM_STAGES:-4}               # 방법당 task_0..task_{n-1}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-5}           # 스테이지당 궤적 수 (패널당 이것의 NUM_STAGES배)
MAX_STEPS=${MAX_STEPS:-300}               # 0 -> env.episode_length(500)
SETTLE=${SETTLE:-5}                       # 초기 상태 정착 스텝 (R1과 같은 값)

# ★ E0는 seed_42(밑줄 있음), ER/PACKNET은 seed42(밑줄 없음)로 트리를 만든다.
E0_ROOT=${E0_ROOT:-./outputs/E0/libero_spatial/seed_${SEED}}
SEQ_ROOT=${SEQ_ROOT:-${E0_ROOT}/lam0}                              # 순차 파인튜닝
EWC_ROOT=${EWC_ROOT:-${E0_ROOT}/lam100}                            # EWC λ=100
ER_ROOT=${ER_ROOT:-./outputs/ER/libero_spatial/seed${SEED}}        # Experience Replay
PACKNET_ROOT=${PACKNET_ROOT:-./outputs/PACKNET/libero_spatial/seed${SEED}}

# 팔을 더 붙이고 싶을 때. "name=root,name=root" — 각각 패널 하나가 된다.
# 예: EXTRA_ARMS="frozen=${E0_ROOT}/laminf"
EXTRA_ARMS=${EXTRA_ARMS:-""}

# ── 출력 ──────────────────────────────────────────────────────────────────────
RUN_TAG=${RUN_TAG:-libero_spatial_seed${SEED}_probe${PROBE_TASK}}
OUT_ROOT=${OUT_ROOT:-./outputs/R3}
RUN_DIR=${RUN_DIR:-${OUT_ROOT}/${RUN_TAG}}
mkdir -p "${RUN_DIR}"

DATASET_PREFIX=continuallearning/libero_spatial_image_task_
ENV_TASK_PREFIX=Libero_Spatial_Task_

# 정책 골격을 만들 때 쓸 아무 체크포인트. R3는 이걸로 학습하지 않는다
# (실제 파라미터는 --ckpt_roots의 각 스테이지에서 매번 다시 읽는다).
SEED_CKPT=${SEED_CKPT:-${SEQ_ROOT}/task_${PROBE_TASK}/checkpoints/last/pretrained_model}

# ── 존재하는 팔만 모은다 ──────────────────────────────────────────────────────
# 스테이지가 덜 찬 팔도 받아 주되 몇 개인지는 말해 준다 — 패널에서 색 하나가 조용히
# 비면 "그 스테이지는 안 움직였다"로 오독되기 쉽다.
ROOTS=""
add_arm() {   # add_arm <name> <root>
    local n=0 k
    for k in $(seq 0 $((NUM_STAGES - 1))); do
        [ -d "$2/task_${k}/checkpoints/last/pretrained_model" ] && n=$((n + 1))
    done
    if [ "${n}" -eq 0 ]; then
        echo "[R3] skip arm '$1' (체크포인트 없음: $2)"
        return
    fi
    [ "${n}" -lt "${NUM_STAGES}" ] && \
        echo "[R3] WARN arm '$1': 스테이지 ${n}/${NUM_STAGES}개만 있다 ($2) — 나머지 색은 빈다"
    [ -n "${ROOTS}" ] && ROOTS="${ROOTS},"
    ROOTS="${ROOTS}$1=$2"
}
add_arm seq     "${SEQ_ROOT}"
add_arm ewc     "${EWC_ROOT}"
add_arm er      "${ER_ROOT}"
add_arm packnet "${PACKNET_ROOT}"
if [ -n "${EXTRA_ARMS}" ]; then
    IFS=',' read -ra _arms <<< "${EXTRA_ARMS}"
    for _a in "${_arms[@]}"; do
        add_arm "${_a%%=*}" "${_a#*=}"
    done
fi

# PackNet 마스크: 팔이 실제로 잡혔을 때만 켠다(없는 이름을 넘기면 R3가 거부한다).
PACKNET_METHODS=""
if [ "${NO_PACKNET_MASK:-0}" != "1" ] && [[ ",${ROOTS}," == *",packnet="* ]]; then
    PACKNET_METHODS="packnet"
fi

if [ "${PLOT_ONLY:-0}" != "1" ]; then
    if [ -z "${ROOTS}" ]; then
        echo "[R3] 볼 체크포인트가 하나도 없다."
        echo "     먼저 E0를 돌려야 한다:  bash bash/E0/E0.sh"
        exit 1
    fi
    if [ ! -d "${SEED_CKPT}" ]; then
        echo "[R3] 정책 골격용 체크포인트가 없다: ${SEED_CKPT}"
        echo "     SEED_CKPT= 로 직접 지정해라."
        exit 1
    fi

    extra=()
    [ "${REDO:-0}" = "1" ] && extra+=("--recompute=true")

    echo ""
    echo "══ [R3] probe_task=${PROBE_TASK}  arms=${ROOTS}"
    echo "        stages=${NUM_STAGES}  rollouts/stage=${NUM_ROLLOUTS}  max_steps=${MAX_STEPS}"
    echo "        packnet mask: ${PACKNET_METHODS:-(off)}"

    "${PYTHON}" "${R3_PY}" \
        --seed="${SEED}" \
        --job_name="R3_seed_${SEED}_probe${PROBE_TASK}" \
        --output_dir="${RUN_DIR}/run" \
        --dataset.repo_id="${DATASET_PREFIX}${PROBE_TASK}" \
        --policy.path="${SEED_CKPT}" \
        --policy.push_to_hub=false \
        --eval_freq=0 \
        --wandb.enable=false \
        --env.type=libero \
        --env.benchmark=libero_spatial \
        --env.task="${ENV_TASK_PREFIX}${PROBE_TASK}" \
        --ckpt_roots="${ROOTS}" \
        --num_stages="${NUM_STAGES}" \
        --probe_task="${PROBE_TASK}" \
        --num_rollouts="${NUM_ROLLOUTS}" \
        --max_steps="${MAX_STEPS}" \
        --settle_steps="${SETTLE}" \
        --packnet_methods="${PACKNET_METHODS}" \
        --env_task_prefix="${ENV_TASK_PREFIX}" \
        --out_root="${OUT_ROOT}" \
        --run_tag="${RUN_TAG}" \
        --no_plot=true \
        "${extra[@]}" || { echo "[R3] FAILED"; exit 1; }
fi

# 그림은 항상 마지막에 한 번만 그린다(본 실행에서는 --no_plot=true로 꺼 두었다).
# PLOT_ONLY=1 경로와 정확히 같은 코드가 돌아 그림이 갈라지지 않는다.
if compgen -G "${RUN_DIR}/*_stage*.npz" > /dev/null; then
    "${PYTHON}" "${R3_PY}" --plot_only --run_dir="${RUN_DIR}"
else
    echo "[R3] npz 캐시가 없다: ${RUN_DIR}"
fi

echo ""
echo "[R3] done."
echo "[R3] figure = ${RUN_DIR}/R3_ee_trajectory.png  (+ 같은 이름의 .csv)"
echo "[R3] cache  = ${RUN_DIR}/<method>_stage<k>.npz"
echo "[R3]   캐시를 지우지 않으면 다음 실행은 롤아웃을 건너뛴다 (REDO=1로 무시)."
