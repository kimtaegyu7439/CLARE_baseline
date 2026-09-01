#!/usr/bin/env bash
#
# R9_A — R9와 같은 측정을 "더 많이 잊은" CL 체크포인트(0..3 순차)에서 다시
#
# R8의 자매 실험이다. R8은 블록 출력 활성 h_ℓ의 조건 대비(CR) 크기를 쟀고, R9는 각
# 서브블록이 residual stream에 더하는 증분 Δ의 **방향** 분리를 잰다.
#   Δ_attn(ℓ,c) = α_attn(c) ⊙ Attn(AdaLN_attn(h; c))
#   Δ_mlp (ℓ,c) = α_mlp (c) ⊙ MLP (AdaLN_mlp (h'; c))
#   cos = ⟨Δ(c₀), Δ(c₁)⟩ / (‖·‖‖·‖)   ·   M = ½(‖Δ(c₀)‖ + ‖Δ(c₁)‖)
# shuffle 기준선(같은 조건, 다른 프로브)이 영점이고, cos이 그보다 낮으면 라우팅이 살아 있다.
#
# 이 스크립트가 하는 일
#   [0] 지금 가장 빨리 비는 GPU를 골라 거기에 붙는다 (여유 메모리를 폴링해 대기)
#   [1] 없는 체크포인트를 **직접 학습해서** 만든다
#   [2] 프로브 (학습 없음, forward만)
#   [3] 그림
#
# 학습 레시피 — E0의 lam0 팔과 동일하게 맞춘다 (그게 이 프로젝트의 실험 세팅이다)
#   5000 스텝 · batch 32 · 태스크당 50 에피소드 중 앞 45개로 학습(뒤 5개 held-out)
#   plain dit_flow_mt 백본 (CLARE 어댑터·PEFT 안 씀) · LIBERO-90 사전학습에서 출발
#
#   cl    = E0.py lam0 (EWC λ=0 = 순차 파인튜닝). task A -> task B 순서로 두 번.
#           outputs/E0/<bench>/seed_<S>/lam0/task_{k}/... 에 쌓인다 (R9가 읽는 트리).
#   joint = R7.py --mode=train_joint. 두 태스크 로더를 스텝마다 **번갈아** 먹인다.
#           holdout 분할·옵티마이저·배치가 순차 팔과 같아서 다른 점이 "데이터를 섞었는가"
#           하나뿐인 통제군이 된다. 스텝 수는 순차 팔 총량과 맞춘다 (5000 × 2 = 10000).
#           R8도 이 체크포인트를 쓰므로 두 실험의 joint가 갈라지지 않는다.
#   pretrain = env.sh의 LIBERO-90 사전학습 체크포인트 (학습 안 함, 하한 기준)
#
# ★ 프로브 지점·조건 구성이 R8과 **같아야** 두 그림을 나란히 읽을 수 있다.
#   NUM_PROBE / PROBE_SEED / T_STEPS / NUM_OBS 를 R8.sh와 같은 값으로 두고, 끝나면
#   R9A_full.method.md의 고정물 해시(x₀ · a_tgt · obs)를 R8 로그와 대조해라.
#
# 물리시간: 조건 관측 s를 rollout step 0, OBS_STRIDE, ..., ROLLOUT_STEPS 에서 뜬다.
#   라우팅 붕괴가 에피소드 초기 상태에서만인지 궤적 내내인지는 다른 주장이라서다.
#   상태열은 전문가 데모 재생으로 만든다 — 세 모델이 같은 s를 봐야 하고 데모는 모델에
#   의존하지 않는다. 롤아웃이 끝난 뒤의 캡처 지점은 정지한 장면이므로 평균에서 뺀다.
#   ▶ 끝나면 콘솔의 "usable pairs per rollout step" 줄을 봐라. 뒤가 0으로 떨어지면
#     ROLLOUT_STEPS를 줄여라. 그림 패널 q가 에피소드별 롤아웃 길이를 그대로 보여 준다.
#
# 시간 감각 (RTX 4090 한 장 기준, 이 서버의 20000스텝 학습이 78분이었던 것에서 환산)
#   cl task_0   5000 스텝  ≈ 20분   (+ E0 프로브: held-out MSE + SR 롤아웃)
#   cl task_1   5000 스텝  ≈ 20분
#   joint      10000 스텝  ≈ 45분   (배치가 16+16이라 스텝당 비용은 비슷)
#   R9 프로브              ≈ 20~40분  (모델 3 × 관측 45 × t 20 × 조건 2 forward)
#   총 2~3시간. 끝난 단계는 .done / 체크포인트 존재로 건너뛰므로 중간에 죽어도 이어진다.
#
# 사용법
#   bash bash/E0/R9.sh                                # 전부 (없는 것만 학습 -> 프로브 -> 그림)
#   PLOT_ONLY=1 bash bash/E0/R9.sh                    # 캐시로 그림만 다시
#   REDO=1 bash bash/E0/R9.sh                         # 프로브 캐시 무시하고 다시 잰다
#   AUTO_TRAIN=0 bash bash/E0/R9.sh                   # 학습 금지 (없으면 그냥 멈춘다)
#   GPU=1 bash bash/E0/R9.sh                          # GPU 자동 선택 대신 직접 지정
#   NEED_MB=16000 bash bash/E0/R9.sh                  # 이만큼 비어야 시작 (기본 11000)
#   MODELS=pretrain,ft_a,cl bash bash/E0/R9.sh        # joint 학습을 건너뛰고 FT0를 통제군으로
#   PER_STEP=0 bash bash/E0/R9.sh                     # 스텝별 판 없이 요약 한 장만
#   TASK_A=0 TASK_B=3 bash bash/E0/R9.sh              # 다른 태스크 쌍 (cl은 0->1->2->3 순차)

set -uo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}

# HF_LEROBOT_HOME / HF_HUB_CACHE / PRETRAIN_PATH 를 세팅한다.
source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

R9_PY=./lerobot_lsy/src/lerobot/scripts/R9_A.py
R7_PY=./lerobot_lsy/src/lerobot/scripts/R7.py
E0_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/E0.sh"
# conda clare 환경의 python. 자식 스크립트(E0.sh)도 이걸 쓰도록 export한다 —
# base의 python에는 torch가 없어서 그냥 두면 학습 단계에서 죽는다.
PYTHON=${PYTHON:-/home/sa090180/miniconda3/envs/clare/bin/python}
export PYTHON

# ★ 설치된 lerobot 패키지는 다른 체크아웃(~/src/lerobot)을 가리킨다. 라이브러리 자체는
#   이 리포와 바이트 단위로 같지만 scripts/ 만 다르고, 거기엔 R3/R7/R8/R9가 없다.
#   R9는 `from lerobot.scripts.R7 import ...` 로 형제 스크립트를 읽으므로 이 체크아웃을
#   앞세워야 한다. 라이브러리가 동일하므로 다른 잡과 같은 코드가 돈다.
export PYTHONPATH="$(pwd)/lerobot_lsy/src${PYTHONPATH:+:${PYTHONPATH}}"

# ═════════════════════════════════════════════════════════════════════════════
#  [0] GPU 고르기 — 지금 가장 여유 있는(= 가장 빨리 비는) 카드에 붙는다
# ═════════════════════════════════════════════════════════════════════════════
# "언제 끝나는가"를 예측하지 않는다. 예측은 틀리고, 필요한 건 "지금 쓸 수 있는가"다.
# 여유 메모리가 NEED_MB 이상인 카드가 나올 때까지 폴링하면 그게 곧 가장 빨리 비는 카드다.
# 점수 = 여유MiB − 사용률×100. 남은 메모리가 비슷하면 놀고 있는 카드를 고른다
# (100% 물린 카드에 얹으면 우리 잡도 느려지고 남의 잡도 느려진다).
NEED_MB=${NEED_MB:-11000}
POLL_SEC=${POLL_SEC:-60}

pick_gpu() {
    local best="" best_score=-999999 best_free=0 line idx used total util free score
    while IFS=, read -r idx used total util; do
        idx=$(echo "${idx}" | tr -d ' '); used=$(echo "${used}" | tr -d ' ')
        total=$(echo "${total}" | tr -d ' '); util=$(echo "${util}" | tr -d ' ')
        [ -z "${idx}" ] && continue
        free=$((total - used))
        score=$((free - util * 100))
        if [ "${score}" -gt "${best_score}" ]; then
            best_score=${score}; best=${idx}; best_free=${free}
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader,nounits 2>/dev/null)
    echo "${best} ${best_free}"
}

if [ -n "${GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU}"
    echo "[R9_A] GPU ${GPU} (직접 지정)"
else
    waited=0
    while :; do
        read -r g free <<< "$(pick_gpu)"
        if [ -z "${g}" ]; then
            echo "[R9_A] nvidia-smi를 못 읽었다. GPU=<idx> 로 직접 지정해라."; exit 1
        fi
        if [ "${free}" -ge "${NEED_MB}" ]; then
            export CUDA_VISIBLE_DEVICES="${g}"
            echo "[R9_A] GPU ${g} 선택 (여유 ${free} MiB / 필요 ${NEED_MB} MiB, 대기 ${waited}초)"
            break
        fi
        [ "${waited}" = "0" ] && nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
            --format=csv 2>/dev/null | sed 's/^/[R9_A]   /'
        echo "[R9_A] 여유 ${free} MiB < ${NEED_MB} MiB — ${POLL_SEC}초 뒤 다시 본다 (누적 ${waited}초)"
        sleep "${POLL_SEC}"; waited=$((waited + POLL_SEC))
    done
fi
# 이후 모든 자식 프로세스(E0.py / er.py / R9.py)가 이 카드만 본다.
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES}

# ═════════════════════════════════════════════════════════════════════════════
#  설정
# ═════════════════════════════════════════════════════════════════════════════
SEED=${SEED:-42}
BENCH=${BENCH:-libero_spatial}
TASK_A=${TASK_A:-0}                       # c₀ = task A 장면 + A 지시문
TASK_B=${TASK_B:-1}                       # c₁ = task B 장면 + B 지시문
COND_MODE=${COND_MODE:-full}              # full = 장면+지시문 함께 (본 실험) / language (부록)
MODELS=${MODELS:-pretrain,joint,cl}       # 1×3 규약. 순서가 곧 그림의 열 순서다.

# ── 학습 레시피 (E0.sh와 같은 값. 바꾸면 기존 E0 결과와 비교할 수 없다) ───────
AUTO_TRAIN=${AUTO_TRAIN:-1}
TRAIN_STEPS=${TRAIN_STEPS:-5000}          # 태스크당 스텝 (E0.sh 기본값)
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
HOLDOUT_EP=${HOLDOUT_EP:-5}               # 50 에피소드 중 뒤 5개는 학습에서 제외
NUM_TASKS=$((TASK_B + 1))                 # cl은 0..TASK_B 를 순서대로 배운다
# joint: 순차 팔의 총 업데이트와 맞춘다 (태스크당 TRAIN_STEPS × 태스크 수).
# R7 train_joint는 두 로더를 번갈아 먹이므로 task당 JOINT_STEPS/태스크수 스텝이 된다.
# 순차 팔이 태스크당 TRAIN_STEPS이므로 여기에 태스크 수를 곱하면 정확히 맞는다.
JOINT_STEPS=${JOINT_STEPS:-$((TRAIN_STEPS * NUM_TASKS))}
WANDB=${WANDB:-false}                     # R9용 보조 학습이라 기본은 끈다

# ── 체크포인트 경로 ───────────────────────────────────────────────────────────
E0_ROOT=${E0_ROOT:-./outputs/E0/${BENCH}/seed_${SEED}}
ARM=${ARM:-lam0}                          # lam0 = EWC λ=0 = 순차 파인튜닝
CKPT_ROOT=${CKPT_ROOT:-${E0_ROOT}/${ARM}} # 아래에 task_{k}/checkpoints/last/pretrained_model
CL_CKPT=${CKPT_ROOT}/task_${CL_STAGE}/checkpoints/last/pretrained_model
PRETRAIN_CKPT=${PRETRAIN_CKPT:-${PRETRAIN_PATH}}
JOINT_DIR=${JOINT_DIR:-./outputs/R9_A_joint/${BENCH}/seed_${SEED}/task${TASK_A}_${TASK_B}}
JOINT_CKPT=${JOINT_CKPT:-${JOINT_DIR}/checkpoints/last/pretrained_model}
ANY_CKPT=${ANY_CKPT:-${PRETRAIN_CKPT}}    # --policy.path 는 설정만 읽는다

# ── 프로브 지점 (R8과 한 글자도 다르면 안 된다) ───────────────────────────────
NUM_PROBE=${NUM_PROBE:-100}
PROBE_SEED=${PROBE_SEED:-20260813}
T_STEPS=${T_STEPS:-20}
T_MAX=${T_MAX:-0.95}
NUM_OBS=${NUM_OBS:-5}                     # 초기 상태(에피소드) 개수
DEMO_EPISODES=${DEMO_EPISODES:-10}        # a_tgt를 뽑을 데모 에피소드 수
EXEC_SLICE=${EXEC_SLICE:-auto}

# ── 물리시간 축 (R9 전용) ─────────────────────────────────────────────────────
ROLLOUT_STEPS=${ROLLOUT_STEPS:-200}
OBS_STRIDE=${OBS_STRIDE:-25}
OBS_DRIVER=${OBS_DRIVER:-demo}
EXCLUDE_DEAD=${EXCLUDE_DEAD:-true}

# ── shuffle 영점 / 판정 규칙 ──────────────────────────────────────────────────
SHUFFLE_SEED=${SHUFFLE_SEED:-20260901}
NORM_FLOOR=${NORM_FLOOR:-1e-8}
ROUTE_GAP=${ROUTE_GAP:-0.05}
ROUTE_MAG=${ROUTE_MAG:-0.05}
PER_STEP=${PER_STEP:-1}

# ── 출력 ──────────────────────────────────────────────────────────────────────
OUT_ROOT=${OUT_ROOT:-./outputs/R9_A}
CL_STAGE=${CL_STAGE:-3}                    # CL이 몇 단계까지 배웠는가 (0..CL_STAGE)
NUM_TASKS=$((CL_STAGE + 1))
RUN_TAG=${RUN_TAG:-${BENCH}_seed${SEED}_${ARM}_cl${CL_STAGE}_cond${TASK_A}v${TASK_B}}
RUN_DIR=${OUT_ROOT}/${RUN_TAG}
mkdir -p "${RUN_DIR}"

DATASET_PREFIX=continuallearning/${BENCH}_image_task_
ENV_TASK_PREFIX=${ENV_TASK_PREFIX:-Libero_Spatial_Task_}
EPISODE_LENGTH=${EPISODE_LENGTH:-300}     # R9는 자체적으로 ROLLOUT_STEPS+settle로 다시 잡는다

if [ "${PLOT_ONLY:-0}" = "1" ]; then
    "${PYTHON}" "${R9_PY}" --plot_only --run_dir="${RUN_DIR}" --per_step="${PER_STEP}"
    exit $?
fi

want_model() { [[ ",${MODELS}," == *",$1,"* ]]; }

# ═════════════════════════════════════════════════════════════════════════════
#  [1] 없는 체크포인트를 만든다
# ═════════════════════════════════════════════════════════════════════════════
if [ ! -d "${PRETRAIN_CKPT}" ]; then
    echo "[R9_A] 사전학습 체크포인트가 없다: ${PRETRAIN_CKPT}"
    echo "     bash/clare/download_assets.sh 로 받거나 PRETRAIN_CKPT= 로 지정해라."
    exit 1
fi

# ── cl: E0.py lam0 팔 (순차 파인튜닝). E0.sh를 그대로 부른다 ─────────────────
# 두 번 구현하지 않는 이유: .done 표시, 미완료 스테이지 정리, 프로브까지 E0.sh가 이미
# 한다. 여기서 다시 쓰면 "E0와 같은 레시피"라는 보장이 코드 두 벌로 갈라진다.
if want_model cl && [ ! -d "${CL_CKPT}" ]; then
    if [ "${AUTO_TRAIN}" != "1" ]; then
        echo "[R9_A] cl 체크포인트가 없다: ${CL_CKPT}   (AUTO_TRAIN=0 이라 멈춘다)"; exit 1
    fi
    echo ""
    echo "══ [R9_A] cl 학습: E0 lam0 (순차 파인튜닝) task 0..${TASK_B}"
    echo "        ${TRAIN_STEPS} 스텝 × batch ${BATCH_SIZE} × 태스크 ${NUM_TASKS}개  ->  ${CKPT_ROOT}"
    LAMBDAS="0" \
    NUM_TASKS="${NUM_TASKS}" \
    SEED="${SEED}" \
    STEPS="${TRAIN_STEPS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    HOLDOUT_EP="${HOLDOUT_EP}" \
    OUT_ROOT="${E0_ROOT}" \
    PROBE_SR="${E0_PROBE_SR:-true}" \
    PYTHON="${PYTHON}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        bash "${E0_SH}" || { echo "[R9_A] E0(lam0) 학습 실패"; exit 1; }
    [ -d "${CL_CKPT}" ] || { echo "[R9_A] E0가 끝났는데 체크포인트가 없다: ${CL_CKPT}"; exit 1; }
fi

# ── joint: 두 태스크를 균형 배치로 섞어 한 번에 학습 (통제군) ────────────────
if want_model joint && [ ! -d "${JOINT_CKPT}" ]; then
    if [ "${AUTO_TRAIN}" != "1" ]; then
        echo "[R9_A] joint 체크포인트가 없다: ${JOINT_CKPT}   (AUTO_TRAIN=0 이라 멈춘다)"
        echo "     joint 없이 보려면 MODELS=pretrain,ft_a,cl 로 FT${TASK_A}를 통제군으로 써라."
        exit 1
    fi
    # 미완료 잔해가 있으면 train 계열이 FileExistsError로 죽는다 (E0.sh와 같은 처리).
    if [ -d "${JOINT_DIR}" ]; then
        if [ "${REDO_INCOMPLETE:-1}" = "1" ]; then
            echo "[R9_A] 미완료 joint 잔해 제거: ${JOINT_DIR}"; rm -rf "${JOINT_DIR}"
        else
            echo "[R9_A] 미완료 joint 잔해가 있다 (REDO_INCOMPLETE=0): ${JOINT_DIR}"; exit 1
        fi
    fi
    echo ""
    echo "══ [R9_A] joint 학습: task ${TASK_A} + task ${TASK_B} 동시 (통제군)"
    echo "        R7.py --mode=train_joint · ${JOINT_STEPS} 스텝 (task당 $((JOINT_STEPS / NUM_TASKS)))"
    echo "        -> ${JOINT_DIR}"
    # ★ er.py로 배치를 섞지 않고 R7의 train_joint를 쓴다. 그쪽이 두 로더를 스텝마다
    #   **번갈아** 먹이면서 E0와 같은 holdout 분할·옵티마이저·배치를 그대로 쓰므로,
    #   순차 팔과 다른 점이 "데이터를 섞었는가" 하나뿐인 진짜 통제군이 된다.
    #   R8도 같은 체크포인트를 쓰므로 두 실험의 joint가 갈라지지 않는다.
    "${PYTHON}" "${R7_PY}" \
        --mode=train_joint \
        --seed="${SEED}" \
        --job_name="R9_joint_${BENCH}_seed${SEED}_task${TASK_A}_${TASK_B}" \
        --output_dir="${JOINT_DIR}" \
        --dataset.repo_id="${DATASET_PREFIX}${TASK_A}" \
        --policy.path="${PRETRAIN_CKPT}" \
        --policy.push_to_hub=false \
        --batch_size="${BATCH_SIZE}" \
        --num_workers="${NUM_WORKERS}" \
        --joint_steps="${JOINT_STEPS}" \
        --holdout_episodes="${HOLDOUT_EP}" \
        --log_freq=100 \
        --task_a="${TASK_A}" \
        --task_b="${TASK_B}" \
        --cl_stage="${CL_STAGE}" \
        --dataset_prefix="${DATASET_PREFIX}" \
        --env.type=libero \
        --env.benchmark="${BENCH}" \
        --env.task="${ENV_TASK_PREFIX}${TASK_A}" \
        --eval_freq=0 \
        --wandb.enable="${WANDB}" \
        || { echo "[R9_A] joint 학습 실패"; exit 1; }
    [ -d "${JOINT_CKPT}" ] || { echo "[R9_A] joint가 끝났는데 체크포인트가 없다: ${JOINT_CKPT}"; exit 1; }
fi

# ── 최종 확인 ─────────────────────────────────────────────────────────────────
for spec in "pretrain:${PRETRAIN_CKPT}" "joint:${JOINT_CKPT}" "cl:${CL_CKPT}"; do
    key=${spec%%:*}; path=${spec#*:}
    want_model "${key}" || continue
    [ -d "${path}" ] || { echo "[R9_A] ${key} 체크포인트가 없다: ${path}"; exit 1; }
done

# ═════════════════════════════════════════════════════════════════════════════
#  [2] 프로브 (학습 없음)
# ═════════════════════════════════════════════════════════════════════════════
extra=()
[ "${REDO:-0}" = "1" ] && extra+=("--recompute=true")

echo ""
echo "══ [R9_A] ${BENCH} seed ${SEED} ${ARM}  ·  c₀ = task ${TASK_A} vs c₁ = task ${TASK_B}"
echo "        GPU ${CUDA_VISIBLE_DEVICES}  ·  models=${MODELS}"
echo "        pretrain = ${PRETRAIN_CKPT}"
echo "        joint    = ${JOINT_CKPT}"
echo "        cl       = ${CL_CKPT}"
echo "        물리시간 0..${ROLLOUT_STEPS} / ${OBS_STRIDE} (driver=${OBS_DRIVER})  ·  "\
"프로브 ${NUM_PROBE} × t ${T_STEPS} × 초기상태 ${NUM_OBS}"

"${PYTHON}" "${R9_PY}" \
    --seed="${SEED}" \
    --job_name="R9_${RUN_TAG}" \
    --output_dir="${RUN_DIR}/run" \
    --dataset.repo_id="${DATASET_PREFIX}${TASK_A}" \
    --policy.path="${ANY_CKPT}" \
    --policy.push_to_hub=false \
    --eval_freq=0 \
    --wandb.enable=false \
    --env.type=libero \
    --env.benchmark="${BENCH}" \
    --env.task="${ENV_TASK_PREFIX}${TASK_A}" \
    --env.episode_length="${EPISODE_LENGTH}" \
    --ckpt_root="${CKPT_ROOT}" \
    --pretrain_ckpt="${PRETRAIN_CKPT}" \
    --joint_ckpt="${JOINT_CKPT}" \
    --models="${MODELS}" \
    --task_a="${TASK_A}" \
    --task_b="${TASK_B}" \
    --cl_stage="${CL_STAGE}" \
    --cond_mode="${COND_MODE}" \
    --exec_slice="${EXEC_SLICE}" \
    --num_probe="${NUM_PROBE}" \
    --probe_seed="${PROBE_SEED}" \
    --t_steps="${T_STEPS}" \
    --t_max="${T_MAX}" \
    --num_obs="${NUM_OBS}" \
    --demo_episodes="${DEMO_EPISODES}" \
    --rollout_steps="${ROLLOUT_STEPS}" \
    --obs_stride="${OBS_STRIDE}" \
    --obs_driver="${OBS_DRIVER}" \
    --exclude_dead_obs="${EXCLUDE_DEAD}" \
    --shuffle_seed="${SHUFFLE_SEED}" \
    --norm_floor="${NORM_FLOOR}" \
    --route_gap_thresh="${ROUTE_GAP}" \
    --route_mag_frac="${ROUTE_MAG}" \
    --dataset_prefix="${DATASET_PREFIX}" \
    --env_task_prefix="${ENV_TASK_PREFIX}" \
    --out_root="${OUT_ROOT}" \
    --run_tag="${RUN_TAG}" \
    --no_plot=true \
    "${extra[@]}" || { echo "[R9_A] 프로브 실패"; exit 1; }

# ═════════════════════════════════════════════════════════════════════════════
#  [3] 그림 — PLOT_ONLY=1 경로와 정확히 같은 코드로 그린다 (R3.sh와 같은 이유)
# ═════════════════════════════════════════════════════════════════════════════
if compgen -G "${RUN_DIR}/R9A_*.npz" > /dev/null; then
    "${PYTHON}" "${R9_PY}" --plot_only --run_dir="${RUN_DIR}" --per_step="${PER_STEP}"
else
    echo "[R9_A] npz 캐시가 없다: ${RUN_DIR}"; exit 1
fi

echo ""
echo "→ ${RUN_DIR}/R9A_full.png / .pdf        요약: 물리시간 평균 히트맵(a~f) + 단면 g~p"
echo "                                        + 패널 q = 에피소드별 롤아웃 길이 타임라인"
echo "→ ${RUN_DIR}/R9A_full_step0000.png ...  rollout step마다 한 장 (최대 9장)"
echo "→ ${RUN_DIR}/R9A_full.npz              cos · M · cos_shuffle · 표본 수, (S, L, T)"
echo "→ ${RUN_DIR}/R9A_full.summary.json     모델별 집계 + 스텝별 routing 블록 수 + 에피소드 길이"
echo "→ ${RUN_DIR}/R9A_full.method.md        Δ 추출 방식 · hook 지점 · 시드 · 고정물 해시"
echo ""
echo "   ※ method.md의 고정물 해시를 R8 로그와 대조해라. 다르면 두 그림이 서로 다른"
echo "     프로브 지점을 본 것이다."
