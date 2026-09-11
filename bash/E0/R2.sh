#!/usr/bin/env bash
#
# R2 — 모드 결어긋남(mode decoherence) 검증 (libero_spatial)
#
# R1은 "채점 장소"를 자기 롤아웃으로 옮겨 loss-SR 해리를 풀었다. 그런데 남은 수수께끼가
# 있었다: Seq stage3/4는 dwell 0.82~0.89(=데모 튜브 안에 잘 있다)인데 SR=0이고,
# Δa의 크기가 SR을 못 가른다. R2의 가설은 이렇다 —
#
#   망각은 모드의 **중심**보다 **경계**를 먼저 침식한다. 중심은 데이터가 두꺼워 안정적이지만
#   모드 사이 경계는 데이터가 희박하고 손실이 평평해 작은 파라미터 변화에도 크게 밀린다.
#   경계가 밀리면 같은 a₀가 다른 모드로 배정되고, 재계획마다 다른 계획이 실행되어
#   **개별 행동은 유효한데 시간적으로 일관되지 않게** 된다.
#
# 실험 두 개
#   A 끈끈한 노이즈  재계획 시 flow matching 초기 노이즈 a₀를 고정(sticky)하거나 시간상관
#                    (ou)으로 바꾼다. 학습도 파라미터도 환경도 안 건드린다. SR이 오르면
#                    결어긋남이 SR 붕괴의 **인과적 원인**이고, 동시에 "학습 없이 망각을
#                    완화하는 추론 시점 개입"이 된다.
#   B 모드 센서스    고정 관측 × 고정 a₀ 격자에서 Φ_θ(o,a₀)를 직접 샘플링해 stage1의 클러스터
#                    구조 기준으로 center_shift(중심 이동) 대 assign_change(경계 이동)를 잰다.
#                    시뮬레이터가 필요 없다.
#
# 그림 3장
#   R2_A_sticky.png              노이즈 고정이 SR을 올리는가 + 많이 흔들리던 팔일수록 효과 큰가
#   R2_B_census.png              중심은 그대로인데 배정만 바뀌는가 (가설의 핵심 비대칭)
#   R2_C_predictor_*.png         R1-C의 확장. loss vs 모드 지표 중 무엇이 SR을 예측하는가
#
# 전제
#   R1이 먼저 끝나 있어야 한다. demo_ref.npz(φ 정규화 통계와 τ)와 r1_results.jsonl을
#   **그대로 읽는다** — 자를 다시 만들면 R1과 R2의 d(t)를 나란히 놓을 수 없다.
#
# 사용법
#   bash bash/E0/R2.sh                                   # 전체 (12 ckpt × 3 모드)
#   TARGETS="ewc@2,ewc@3,seq@2,seq@3" bash bash/E0/R2.sh  # 먼저 이것만 (권장 첫 실행)
#   SKIP_STICKY=1 bash bash/E0/R2.sh                     # 실험 B만 (시뮬레이터 불필요, 빠름)
#   SKIP_CENSUS=1 bash bash/E0/R2.sh                     # 실험 A만
#   NOISE_MODES="fresh,sticky" bash bash/E0/R2.sh        # OU 빼고
#   PLOT_ONLY=1 bash bash/E0/R2.sh                       # 캐시로 그림만 다시
#   REDO=1 bash bash/E0/R2.sh                            # 롤아웃 캐시 무시하고 다시
#
# ★ TARGETS를 줘도 reference 체크포인트(seq@PROBE_TASK)는 항상 함께 돈다.
#   switch 임계, 클러스터 구조, demo_match 임계가 전부 거기서 나오는 기준점이기 때문이다.
#
# 비용 감각: 실험 A는 R1과 같은 롤아웃을 노이즈 모드 수만큼 반복한다(= R1 × 3).
# 실험 B는 롤아웃이 없어 훨씬 싸다(체크포인트당 200관측 × 64샘플 × Euler 100스텝).
# 캐시가 cache/에 남으므로 중간에 죽어도 끝난 것은 다시 돌지 않는다.

set -uo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-1}

# HF_LEROBOT_HOME / HF_HUB_CACHE / PRETRAIN_PATH 를 세팅한다.
source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

R2_PY=./lerobot_lsy/src/lerobot/scripts/R2.py
PYTHON=${PYTHON:-python}   # conda clare 환경이 활성화돼 있다고 가정. 아니면 PYTHON=... 로 지정.

# ── 무엇을 볼 것인가 (R1과 같아야 짝지은 비교가 성립한다) ─────────────────────
SEED=${SEED:-42}
PROBE_TASK=${PROBE_TASK:-0}
NUM_STAGES=${NUM_STAGES:-4}
NUM_ROLLOUTS=${NUM_ROLLOUTS:-30}
MAX_STEPS=${MAX_STEPS:-0}                 # 0 -> env.episode_length(300)

E0_ROOT=${E0_ROOT:-./outputs/E0/libero_spatial/seed_${SEED}}
SEQ_ROOT=${SEQ_ROOT:-${E0_ROOT}/lam0}
EWC_ROOT=${EWC_ROOT:-${E0_ROOT}/lam100}
FROZEN_ROOT=${FROZEN_ROOT:-${E0_ROOT}/laminf}
EXTRA_ARMS=${EXTRA_ARMS:-""}

# ── R2 고유 ───────────────────────────────────────────────────────────────────
# "" -> 전부. "ewc@2,ewc@3" 처럼 부분 지정 (stage는 R1과 같은 0-based.
# 그림 라벨의 stage3/stage4가 여기서는 @2,@3이다.)
TARGETS=${TARGETS:-""}
NOISE_MODES=${NOISE_MODES:-"fresh,sticky,ou"}
OU_RHO=${OU_RHO:-0.9}
SWITCH_THRESH=${SWITCH_THRESH:-0}         # 0 -> reference(fresh)의 90퍼센타일로 자동 보정
SWITCH_PCT=${SWITCH_PCT:-90}

CENSUS_OBS=${CENSUS_OBS:-200}
CENSUS_K=${CENSUS_K:-64}
CENSUS_KMAX=${CENSUS_KMAX:-8}
CENSUS_SIL_MIN=${CENSUS_SIL_MIN:-0.5}
DEMO_MATCH_PCT=${DEMO_MATCH_PCT:-50}

RIGHT_KEY=${RIGHT_KEY:-demo_match_mass}   # 그림 C 오른쪽 축
K=${K:-4}                                 # ā를 만드는 샘플 수 (R1과 같아야 한다)
SETTLE=${SETTLE:-5}

# ── 출력 ──────────────────────────────────────────────────────────────────────
RUN_TAG=${RUN_TAG:-libero_spatial_seed${SEED}_probe${PROBE_TASK}}
OUT_ROOT=${OUT_ROOT:-./outputs/R2}
RUN_DIR=${RUN_DIR:-${OUT_ROOT}/${RUN_TAG}}
R1_RUN_DIR=${R1_RUN_DIR:-./outputs/R1/${RUN_TAG}}
RESULTS=${RUN_DIR}/r2_results.jsonl
mkdir -p "${RUN_DIR}"

DATASET_PREFIX=continuallearning/libero_spatial_image_task_
ENV_TASK_PREFIX=Libero_Spatial_Task_
REF_CKPT=${REF_CKPT:-${SEQ_ROOT}/task_${PROBE_TASK}/checkpoints/last/pretrained_model}

# ── R1 산출물 확인 — R2는 자를 새로 만들지 않는다 ─────────────────────────────
if [ "${PLOT_ONLY:-0}" != "1" ]; then
    if [ ! -f "${R1_RUN_DIR}/demo_ref.npz" ]; then
        echo "[R2] R1의 demo_ref.npz가 없다: ${R1_RUN_DIR}/demo_ref.npz"
        echo "     먼저 R1을 돌려라:  bash bash/E0/R1.sh"
        echo "     (R2는 φ 정규화 통계와 τ를 절대 재계산하지 않는다 — 자가 달라지면"
        echo "      R1과 R2의 d(t)를 나란히 놓을 수 없기 때문이다.)"
        exit 1
    fi
    if [ ! -f "${R1_RUN_DIR}/r1_results.jsonl" ]; then
        echo "[R2] R1 결과가 없다: ${R1_RUN_DIR}/r1_results.jsonl  (fresh SR 검산에 필요하다)"
        exit 1
    fi
fi

# ── 존재하는 팔만 모은다 (R1.sh의 add_arm과 같은 규칙) ────────────────────────
ROOTS=""
add_arm() {   # add_arm <name> <root>
    local n=0 k
    for k in $(seq 0 $((NUM_STAGES - 1))); do
        [ -d "$2/task_${k}/checkpoints/last/pretrained_model" ] && n=$((n + 1))
    done
    if [ "${n}" -eq 0 ]; then
        echo "[R2] skip arm '$1' (체크포인트 없음: $2)"
        return
    fi
    [ "${n}" -lt "${NUM_STAGES}" ] && \
        echo "[R2] WARN arm '$1': 스테이지 ${n}/${NUM_STAGES}개만 있다 ($2)"
    [ -n "${ROOTS}" ] && ROOTS="${ROOTS},"
    ROOTS="${ROOTS}$1=$2"
}
# ★ 순서가 중요하다: 첫 팔의 task_${PROBE_TASK}가 reference θ*₁이자 모든 상대 지표의
#   기준점(stage1)이 된다. R1과 같은 순서(seq 먼저)여야 두 실험이 같은 기준을 쓴다.
add_arm seq    "${SEQ_ROOT}"
add_arm ewc    "${EWC_ROOT}"
add_arm frozen "${FROZEN_ROOT}"
if [ -n "${EXTRA_ARMS}" ]; then
    IFS=',' read -ra _arms <<< "${EXTRA_ARMS}"
    for _a in "${_arms[@]}"; do
        add_arm "${_a%%=*}" "${_a#*=}"
    done
fi

if [ "${PLOT_ONLY:-0}" != "1" ]; then
    if [ -z "${ROOTS}" ]; then
        echo "[R2] 볼 체크포인트가 하나도 없다. 먼저 E0를 돌려야 한다: bash bash/E0/E0.sh"
        exit 1
    fi
    if [ ! -d "${REF_CKPT}" ]; then
        echo "[R2] reference 체크포인트가 없다: ${REF_CKPT}"
        exit 1
    fi
    # JSONL은 append-only지만 그림 쪽에서 최신 행만 남기므로 매번 치울 필요는 없다.
    if [ "${FRESH:-0}" = "1" ] && [ -s "${RESULTS}" ]; then
        mv "${RESULTS}" "${RESULTS}.bak" && echo "[R2] previous results -> ${RESULTS}.bak"
    fi

    extra=()
    [ "${REDO:-0}" = "1" ] && extra+=("--recompute_rollouts=true")
    [ "${REDO_CENSUS:-0}" = "1" ] && extra+=("--recompute_census=true")
    [ -n "${TARGETS}" ] && extra+=("--targets=${TARGETS}")
    [ "${SKIP_STICKY:-0}" = "1" ] && extra+=("--skip_sticky=true")
    [ "${SKIP_CENSUS:-0}" = "1" ] && extra+=("--skip_census=true")
    extra+=("--fresh_sr_tol=${FRESH_SR_TOL:-0.2}")
    # ★ R1을 그대로 흉내내고 싶을 때만 0. 기본은 env.reset() 앞에서 전역 numpy RNG를
    #   고정한다 — 안 하면 같은 rollout_id도 매번 다른 물리 상태에서 출발해서
    #   fresh 대 sticky 비교가 "초기 상태 차이"를 재게 된다 (R2.py 주석 참조).
    [ "${DETERMINISTIC_RESET:-1}" = "0" ] && extra+=("--deterministic_reset=false")

    echo ""
    echo "══ [R2] probe_task=${PROBE_TASK}  arms=${ROOTS}"
    echo "        modes=${NOISE_MODES}  targets=${TARGETS:-<all>}  rollouts=${NUM_ROLLOUTS}"
    echo "        R1 산출물: ${R1_RUN_DIR}"

    "${PYTHON}" "${R2_PY}" \
        --seed="${SEED}" \
        --job_name="R2_seed_${SEED}_probe${PROBE_TASK}" \
        --output_dir="${RUN_DIR}/run" \
        --dataset.repo_id="${DATASET_PREFIX}${PROBE_TASK}" \
        --policy.path="${REF_CKPT}" \
        --policy.push_to_hub=false \
        --eval_freq=0 \
        --wandb.enable=false \
        --env.type=libero \
        --env.benchmark=libero_spatial \
        --env.task="${ENV_TASK_PREFIX}${PROBE_TASK}" \
        --r1_run_dir="${R1_RUN_DIR}" \
        --ckpt_roots="${ROOTS}" \
        --ref_ckpt="${REF_CKPT}" \
        --num_stages="${NUM_STAGES}" \
        --probe_task="${PROBE_TASK}" \
        --num_rollouts="${NUM_ROLLOUTS}" \
        --max_steps="${MAX_STEPS}" \
        --num_samples="${K}" \
        --settle_steps="${SETTLE}" \
        --noise_modes="${NOISE_MODES}" \
        --ou_rho="${OU_RHO}" \
        --switch_thresh="${SWITCH_THRESH}" \
        --switch_pct="${SWITCH_PCT}" \
        --census_obs="${CENSUS_OBS}" \
        --census_k="${CENSUS_K}" \
        --census_kmax="${CENSUS_KMAX}" \
        --census_sil_min="${CENSUS_SIL_MIN}" \
        --demo_match_pct="${DEMO_MATCH_PCT}" \
        --dataset_prefix="${DATASET_PREFIX}" \
        --env_task_prefix="${ENV_TASK_PREFIX}" \
        --out_root="${OUT_ROOT}" \
        --run_tag="${RUN_TAG}" \
        --right_key="${RIGHT_KEY}" \
        "${extra[@]}" || { echo "[R2] FAILED"; exit 1; }
else
    "${PYTHON}" "${R2_PY}" --plot_only \
        --run_dir="${RUN_DIR}" --r1_run_dir="${R1_RUN_DIR}" --right_key="${RIGHT_KEY}"
fi

echo ""
echo "[R2] done.  raw=${RESULTS}"
ls -1 "${RUN_DIR}"/R2_*.png 2>/dev/null
