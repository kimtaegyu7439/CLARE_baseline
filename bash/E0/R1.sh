#!/usr/bin/env bash
#
# R1 — 채점 장소를 자기 롤아웃으로 옮긴다 (libero_spatial)
#
# E0는 held-out demo loss가 완만한데 SR은 절벽처럼 무너지는 해리를 보여줬다.
# 그런데 그 loss는 **전문가가 밟았던 상태**에서 잰 값이다. SR이 무너지는 곳은
# 정책이 스스로 흘러들어간 낯선 상태다. R1은 채점 장소를 자기 롤아웃으로 옮기고
# 해리가 풀리는지 본다. 학습은 하지 않는다 — 저장된 체크포인트로 롤아웃만 돈다.
#
# 재는 것 두 개
#   d(t)   자기 롤아웃 상태가 데모 튜브에서 얼마나 벗어났나
#          (시뮬레이터 물리 상태 공간에서. 정책 잠재공간을 쓰면 인코더가 망각으로
#           변할 때 자(尺)가 같이 휜다.)
#   Δa(t)  그 롤아웃에서 마주친 바로 그 관측 위에서, 현재 정책이 "망각 전의 자기
#          자신"(θ*₁)과 얼마나 다른 행동을 내나
#
# 그림 3장
#   R1_A_tube_departure.png   튜브 이탈 곡선 (방법별 패널 × 스테이지별 곡선)
#   R1_B_state_vs_action.png  d와 Δa의 선후 — 표류가 먼저면 폐루프 증폭, Δa가 먼저면
#                             행동 자체의 손상이 선행. 선후가 곧 진단이다.
#   R1_C_predictor.png        무엇이 SR을 예측하는가 (loss vs dAUC)
#
# 전제
#   E0(그리고 있으면 ER)가 만든 체크포인트 트리 + gym_libero + LIBERO 원본 데모 hdf5.
#   (물체 pose는 LeRobot 데이터셋에 없고 hdf5의 flattened sim state에만 있다.)
#
# 사용법
#   bash bash/E0/R1.sh
#   PROBE_TASK=1 bash bash/E0/R1.sh                 # 태스크 1의 망각을 본다
#   NUM_ROLLOUTS=10 MAX_STEPS=200 bash bash/E0/R1.sh   # 빠른 예비 실행
#   ER_ROOT=./outputs/ER/libero_spatial/seed42 bash bash/E0/R1.sh   # ER 팔 추가
#   STRIDE=1 bash bash/E0/R1.sh                     # Δa를 매 스텝 (비용 8배)
#   PLOT_ONLY=1 bash bash/E0/R1.sh                  # 캐시로 그림만 다시
#   REDO=1 bash bash/E0/R1.sh                       # 롤아웃 캐시 무시하고 다시 굴리기
#   FRESH=1 bash bash/E0/R1.sh                      # JSONL을 치우고 처음부터
#
# 비용 감각: 체크포인트 12개 × 롤아웃 30개 × 최대 500스텝. 스텝마다 물리 상태를 읽고
# (싸다) 8스텝마다 두 정책의 flow matching 샘플 K개를 뽑는다(비싼 쪽). 한나절 규모다.
# 캐시가 cache/rollouts_*.npz에 남으므로 중간에 죽어도 끝난 체크포인트는 다시 굴리지 않는다.

set -uo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-1}

# HF_LEROBOT_HOME / HF_HUB_CACHE / PRETRAIN_PATH 를 세팅한다.
source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

R1_PY=./lerobot_lsy/src/lerobot/scripts/R1.py
PYTHON=${PYTHON:-python}   # conda clare 환경이 활성화돼 있다고 가정. 아니면 PYTHON=... 로 지정.

# ── 무엇을 볼 것인가 ──────────────────────────────────────────────────────────
SEED=${SEED:-42}
PROBE_TASK=${PROBE_TASK:-0}               # 어느 태스크의 망각을 볼 것인가 (0 또는 1)
NUM_STAGES=${NUM_STAGES:-4}               # 방법당 task_0..task_{n-1}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-30}
MAX_STEPS=${MAX_STEPS:-0}                 # 0 -> env.episode_length(500)

E0_ROOT=${E0_ROOT:-./outputs/E0/libero_spatial/seed_${SEED}}
SEQ_ROOT=${SEQ_ROOT:-${E0_ROOT}/lam0}     # 순차 파인튜닝 팔
EWC_ROOT=${EWC_ROOT:-${E0_ROOT}/lam100}   # EWC 팔
ER_ROOT=${ER_ROOT:-./outputs/ER/libero_spatial/seed${SEED}}   # 없으면 자동으로 빠진다

# 팔을 더 붙이고 싶을 때. "name=root,name=root" — 각각 그림 A의 패널 하나가 된다.
# 예: EXTRA_ARMS="frozen=${E0_ROOT}/laminf"     동결(λ=inf) = 망각이 불가능한 상한 기준점
#     EXTRA_ARMS="ewc10=${E0_ROOT}/lam10"       다른 EWC 세기
EXTRA_ARMS=${EXTRA_ARMS:-""}

# 그림 C에만 점을 더 찍고 싶을 때 (λ 스윕 등). "label@stage=path,..." 형식.
# EXTRA_ARMS와 달리 롤아웃은 하되 그림 A의 패널은 만들지 않는다.
EXTRA_CKPTS=${EXTRA_CKPTS:-""}

# ── 롤아웃 설정 ───────────────────────────────────────────────────────────────
K=${K:-4}                                 # ā를 만드는 flow matching 샘플 수
STRIDE=${STRIDE:-0}                       # Δa 측정 간격. 0 -> n_action_steps(=8)
SETTLE=${SETTLE:-5}                       # 초기 상태 정착 스텝 (pruned_init은 물체를
                                          # 테이블 위 7cm에 띄워 둔다 — R1.py 주석 참조)
SAVE_OBS=${SAVE_OBS:-false}               # true면 앞 5개 rollout의 이미지를 디버그 저장

# ── [C] held-out loss ─────────────────────────────────────────────────────────
# 기본은 R1이 직접 재는 것이다(고정 (τ_fm, a0) 격자). 그림 C의 x축은 체크포인트 간
# 작은 차이를 읽어야 하는데, 팔마다 출처가 다르면(E0의 랜덤 τ_fm 평균 vs R1의 고정 격자)
# 같은 축 위의 점들이 서로 다른 잡음을 달고 앉는다. 전부 같은 자로 재는 쪽을 기본으로 둔다.
#
# USE_E0_LOSS=1 이면 E0의 mse를 대신 읽는다(재계산을 아낀다). 이때 팔 -> E0 run_tag
# 대응은 트리 이름에서 자동으로 만든다(lam100 -> "100", laminf -> "inf"). ★ 이 대응을
# 손으로 쓰다 틀리면 λ=10 체크포인트에 λ=100의 loss가 조용히 붙는다 — 그래서 자동이다.
USE_E0_LOSS=${USE_E0_LOSS:-0}
E0_RESULTS=${E0_RESULTS:-${E0_ROOT}/e0_results.jsonl}
E0_RUN_TAGS=${E0_RUN_TAGS:-""}            # 비우면 트리 이름에서 자동 생성
LOSS_BATCHES=${LOSS_BATCHES:-16}
LOSS_BATCH_SIZE=${LOSS_BATCH_SIZE:-16}

# ── 출력 ──────────────────────────────────────────────────────────────────────
RUN_TAG=${RUN_TAG:-libero_spatial_seed${SEED}_probe${PROBE_TASK}}
OUT_ROOT=${OUT_ROOT:-./outputs/R1}
RUN_DIR=${RUN_DIR:-${OUT_ROOT}/${RUN_TAG}}
RESULTS=${RUN_DIR}/r1_results.jsonl
FIGB=${FIGB:-"ewc@1,ewc@2,ewc@3"}         # 그림 B의 열 (method@stage, stage는 0-based)
SPLIT_BY_SUCCESS=${SPLIT_BY_SUCCESS:-true}
mkdir -p "${RUN_DIR}"

DATASET_PREFIX=continuallearning/libero_spatial_image_task_
ENV_TASK_PREFIX=Libero_Spatial_Task_

# reference 정책 θ*₁ = probe_task 학습 완료 직후 체크포인트. "망각 전의 자기 자신"이
# 행동 기준이므로, 순차 팔(아직 아무것도 망각하지 않은 상태)의 task_${PROBE_TASK}를 쓴다.
REF_CKPT=${REF_CKPT:-${SEQ_ROOT}/task_${PROBE_TASK}/checkpoints/last/pretrained_model}

# ── 존재하는 팔만 모은다 ──────────────────────────────────────────────────────
# ER 트리는 아직 없을 수 있다. 없는 경로를 그대로 넘기면 R1이 체크포인트마다 경고를
# 12번 뱉으므로 여기서 미리 걸러 낸다(그림의 패널 수도 실제 팔 수에 맞는다).
# 스테이지가 덜 찬 팔도 받아 주되(있는 스테이지만 그려진다) 몇 개인지는 말해 준다 —
# 그림 A에서 곡선 하나가 조용히 비면 "그 스테이지에 이탈이 없었다"로 오독되기 쉽다.
ROOTS=""
AUTO_TAGS=""
add_arm() {   # add_arm <name> <root>
    local n=0 k base
    for k in $(seq 0 $((NUM_STAGES - 1))); do
        [ -d "$2/task_${k}/checkpoints/last/pretrained_model" ] && n=$((n + 1))
    done
    if [ "${n}" -eq 0 ]; then
        echo "[R1] skip arm '$1' (체크포인트 없음: $2)"
        return
    fi
    [ "${n}" -lt "${NUM_STAGES}" ] && \
        echo "[R1] WARN arm '$1': 스테이지 ${n}/${NUM_STAGES}개만 있다 ($2) — 나머지는 그림에서 빠진다"
    [ -n "${ROOTS}" ] && ROOTS="${ROOTS},"
    ROOTS="${ROOTS}$1=$2"
    # E0 트리(lam*)에서 온 팔만 run_tag를 만들 수 있다. ER 등 다른 출처는 대응이 없어
    # 자동으로 빠지고, 그 팔의 loss는 R1이 직접 잰다.
    base=$(basename "$2")
    if [ "${base#lam}" != "${base}" ]; then
        [ -n "${AUTO_TAGS}" ] && AUTO_TAGS="${AUTO_TAGS},"
        AUTO_TAGS="${AUTO_TAGS}$1=${base#lam}"
    fi
}
add_arm seq "${SEQ_ROOT}"
add_arm ewc "${EWC_ROOT}"
add_arm er  "${ER_ROOT}"
# 추가 팔 ("name=root,name=root")
if [ -n "${EXTRA_ARMS}" ]; then
    IFS=',' read -ra _arms <<< "${EXTRA_ARMS}"
    for _a in "${_arms[@]}"; do
        add_arm "${_a%%=*}" "${_a#*=}"
    done
fi

if [ "${PLOT_ONLY:-0}" != "1" ]; then
    if [ -z "${ROOTS}" ]; then
        echo "[R1] 볼 체크포인트가 하나도 없다."
        echo "     먼저 E0를 돌려야 한다:  bash bash/E0/E0.sh"
        exit 1
    fi
    if [ ! -d "${REF_CKPT}" ]; then
        echo "[R1] reference 체크포인트가 없다: ${REF_CKPT}"
        echo "     REF_CKPT= 로 직접 지정하거나 E0의 task_${PROBE_TASK}를 먼저 끝내라."
        exit 1
    fi

    # JSONL은 append-only지만 그림 쪽에서 (method, stage)마다 최신 행만 남기므로
    # E0/H5와 달리 매번 치울 필요가 없다(부분 재실행이 정상 동작이다).
    # 그래도 처음부터 다시 쌓고 싶으면 FRESH=1.
    if [ "${FRESH:-0}" = "1" ] && [ -s "${RESULTS}" ]; then
        mv "${RESULTS}" "${RESULTS}.bak" && echo "[R1] previous results -> ${RESULTS}.bak"
    fi

    extra=()
    [ "${REDO:-0}" = "1" ] && extra+=("--recompute_rollouts=true")
    [ "${REDO_DEMO_REF:-0}" = "1" ] && extra+=("--recompute_demo_ref=true")
    [ -n "${EXTRA_CKPTS}" ] && extra+=("--extra_ckpts=${EXTRA_CKPTS}")
    [ -n "${PHI_OBJECTS:-}" ] && extra+=("--phi_objects=${PHI_OBJECTS}")
    # USE_E0_LOSS=1 일 때만 E0의 mse로 대체한다. 기본은 R1이 전부 같은 격자로 잰다.
    if [ "${USE_E0_LOSS}" = "1" ] && [ -s "${E0_RESULTS}" ]; then
        tags=${E0_RUN_TAGS:-${AUTO_TAGS}}
        echo "[R1] held-out loss는 E0 결과를 읽는다: ${E0_RESULTS}  (run_tag 대응: ${tags})"
        echo "[R1]   대응에 없는 팔은 R1이 직접 잰다 — 그림 C의 x축에 출처가 섞인다는 뜻이다."
        extra+=("--e0_results=${E0_RESULTS}" "--e0_run_tags=${tags}")
    fi

    echo ""
    echo "══ [R1] probe_task=${PROBE_TASK}  arms=${ROOTS}"
    echo "        rollouts=${NUM_ROLLOUTS}  ref=${REF_CKPT}"

    "${PYTHON}" "${R1_PY}" \
        --seed="${SEED}" \
        --job_name="R1_seed_${SEED}_probe${PROBE_TASK}" \
        --output_dir="${RUN_DIR}/run" \
        --dataset.repo_id="${DATASET_PREFIX}${PROBE_TASK}" \
        --policy.path="${REF_CKPT}" \
        --policy.push_to_hub=false \
        --eval_freq=0 \
        --wandb.enable=false \
        --env.type=libero \
        --env.benchmark=libero_spatial \
        --env.task="${ENV_TASK_PREFIX}${PROBE_TASK}" \
        --ckpt_roots="${ROOTS}" \
        --num_stages="${NUM_STAGES}" \
        --probe_task="${PROBE_TASK}" \
        --ref_ckpt="${REF_CKPT}" \
        --num_rollouts="${NUM_ROLLOUTS}" \
        --max_steps="${MAX_STEPS}" \
        --num_samples="${K}" \
        --action_eval_stride="${STRIDE}" \
        --settle_steps="${SETTLE}" \
        --save_obs="${SAVE_OBS}" \
        --dataset_prefix="${DATASET_PREFIX}" \
        --env_task_prefix="${ENV_TASK_PREFIX}" \
        --loss_batches="${LOSS_BATCHES}" \
        --loss_batch_size="${LOSS_BATCH_SIZE}" \
        --out_root="${OUT_ROOT}" \
        --run_tag="${RUN_TAG}" \
        --figb_ckpts="${FIGB}" \
        --split_by_success="${SPLIT_BY_SUCCESS}" \
        --no_plot=true \
        "${extra[@]}" || { echo "[R1] FAILED"; exit 1; }
fi

# 그림은 항상 마지막에 한 번만 그린다(본 실행에서는 --no_plot=true로 꺼 두었다).
# 이렇게 두면 PLOT_ONLY=1 경로와 정확히 같은 코드가 돌아 그림이 갈라지지 않는다.
if [ -s "${RESULTS}" ]; then
    plot_extra=()
    [ "${SPLIT_BY_SUCCESS}" = "true" ] && plot_extra+=("--split_by_success")
    "${PYTHON}" "${R1_PY}" --plot_only --run_dir="${RUN_DIR}" --figb="${FIGB}" "${plot_extra[@]}"
else
    echo "[R1] 결과가 비어 있다: ${RESULTS}"
fi

echo ""
echo "[R1] done.  raw=${RESULTS}"
echo "[R1] figures=${RUN_DIR}/R1_A_*.png  R1_B_*.png  R1_C_*.png  (+ 같은 이름의 .csv)"
echo "[R1] cache=${RUN_DIR}/cache  demo_ref=${RUN_DIR}/demo_ref.npz"
echo "[R1]   캐시를 지우지 않으면 다음 실행은 롤아웃을 건너뛴다. τ와 z-정규화 통계는"
echo "[R1]   demo_ref.npz에서만 읽는다 — 체크포인트마다 다시 재면 자가 휘어 비교가 깨진다."
