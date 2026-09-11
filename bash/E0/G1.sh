#!/usr/bin/env bash
#
# G1 — 망각이 denoising timestep의 어디에 몰려 있는가 (libero_spatial)
#
# 가설: flow matching 디코더에서 초기 노이즈 근처의 적분 구간이 최종 behavioral mode를
# 고른다. 그렇다면 이전 태스크의 held-out MSE가 거의 안 오르는데 SR이 무너지는 현상은,
# 망각이 전 구간에 고르게 퍼진 게 아니라 **이른 timestep의 작은 속도장 변화**가
# 불균형하게 SR을 떨어뜨리기 때문일 수 있다.
#
# 학습은 하지 않는다. 저장된 체크포인트 두 개만 읽는다.
#   θ_old = 이전 태스크를 막 끝낸 체크포인트   (기본: E0 seq(lam0) task_0)
#   θ_new = 다음 태스크까지 학습한 체크포인트  (기본: E0 seq(lam0) task_1)
#   평가 태스크 = 이전 태스크 (기본: task 0)
# 기본값이 seq 팔인 이유는 R1에서 이 구간의 SR이 100% -> 13.3%로 가장 크게 떨어져
# 신호가 제일 선명하기 때문이다.
#
# 재는 것
#   [A] 얼린 궤적 위의 속도 드리프트. θ_old로 궤적을 한 번 만들고 **같은 x_t**를 두
#       모델에 넣는다. 각자 롤아웃해서 비교하면 상태가 갈라져 교란이 생긴다.
#   [B] held-out MSE (E0와 같은 고정 τ 격자)
#   [C] 개입 롤아웃. 특정 구간만 θ_new의 속도를 쓰고 나머지는 θ_old를 쓴 뒤 SR을 잰다.
#       가설의 핵심 예측은 "앞 구간만 바꿔도 SR이 크게 떨어진다"이다.
#   [D] 에피소드 단위 상관. old 롤아웃 중 매 재계획마다 D_t를 기록해 성공/실패와 잇는다.
#
# 비용: [A][B]는 시뮬레이터가 없어 몇 분. [C]가 대부분이다 —
#   조건 수 = 2(old/new) + 1(독립 노이즈) + N_GRID(단일 스텝) + 구간 6개 ≈ 19개,
#   각 조건이 SR_EPISODES개 환경을 한 배치로 도는 롤아웃 하나. 서너 시간 규모다.
#
# 사용법
#   bash bash/E0/G1.sh
#   SR_EPISODES=10 N_GRID=6 bash bash/E0/G1.sh        # 빠른 예비 실행
#   SKIP_SR=true bash bash/E0/G1.sh                   # [A][B]만 (시뮬레이터 불필요, 몇 분)
#   OLD_TASK=0 NEW_TASK=2 bash bash/E0/G1.sh          # 다른 스테이지 쌍
#   ARM=lam100 bash bash/E0/G1.sh                     # EWC 팔로
#   PLOT_ONLY=1 bash bash/E0/G1.sh                    # 저장된 raw로 그림만

set -uo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}

# HF_LEROBOT_HOME / HF_HUB_CACHE / PRETRAIN_PATH 를 세팅한다.
source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

G1_PY=./lerobot_lsy/src/lerobot/scripts/G1.py
PYTHON=${PYTHON:-python}   # conda clare 환경이 활성화돼 있다고 가정.

# ── 무엇을 비교하는가 ─────────────────────────────────────────────────────────
SEED=${SEED:-42}
ARM=${ARM:-lam0}                          # E0 트리의 어느 팔인가 (lam0=순차 파인튜닝)
E0_ROOT=${E0_ROOT:-./outputs/E0/libero_spatial/seed_${SEED}}
OLD_TASK=${OLD_TASK:-0}                   # θ_old = 이 태스크를 막 끝낸 체크포인트
NEW_TASK=${NEW_TASK:-1}                   # θ_new = 이 태스크까지 학습한 체크포인트
PROBE_TASK=${PROBE_TASK:-${OLD_TASK}}     # 평가 태스크 = 이전 태스크

OLD_CKPT=${OLD_CKPT:-${E0_ROOT}/${ARM}/task_${OLD_TASK}/checkpoints/last/pretrained_model}
NEW_CKPT=${NEW_CKPT:-${E0_ROOT}/${ARM}/task_${NEW_TASK}/checkpoints/last/pretrained_model}

# ── [A] 얼린 궤적 드리프트 ────────────────────────────────────────────────────
DRIFT_BATCHES=${DRIFT_BATCHES:-4}
DRIFT_BATCH_SIZE=${DRIFT_BATCH_SIZE:-8}
DRIFT_N_NOISE=${DRIFT_N_NOISE:-4}         # 관측당 초기 노이즈 개수

# ── [B] held-out MSE ──────────────────────────────────────────────────────────
PROBE_BATCHES=${PROBE_BATCHES:-16}
PROBE_BATCH_SIZE=${PROBE_BATCH_SIZE:-16}

# ── [C] 개입 롤아웃 ───────────────────────────────────────────────────────────
SKIP_SR=${SKIP_SR:-false}
SR_EPISODES=${SR_EPISODES:-20}            # = 동시에 굴리는 환경 수. 한 배치로 돈다.
N_GRID=${N_GRID:-10}                      # 단일 스텝 개입을 잴 timestep 격자 수
WINDOWS=${WINDOWS:-"early10:0.0-0.1,early20:0.0-0.2,early30:0.0-0.3,mid20:0.4-0.6,late20:0.8-1.0,late30:0.7-1.0"}
# R1과 같은 에피소드 상한. 여기서 바꾸면 G1 안에서는 일관되지만 R1의 SR과는 못 비교한다.
EPISODE_LENGTH=${EPISODE_LENGTH:-300}

# ── 출력 ──────────────────────────────────────────────────────────────────────
OUT_ROOT=${OUT_ROOT:-./outputs/G1}
RUN_TAG=${RUN_TAG:-libero_spatial_seed${SEED}_${ARM}_old${OLD_TASK}_new${NEW_TASK}_probe${PROBE_TASK}}
RUN_DIR=${OUT_ROOT}/${RUN_TAG}
mkdir -p "${RUN_DIR}"

DATASET_PREFIX=continuallearning/libero_spatial_image_task_
ENV_TASK_PREFIX=Libero_Spatial_Task_

if [ "${PLOT_ONLY:-0}" = "1" ]; then
    "${PYTHON}" "${G1_PY}" --plot_only --run_dir="${RUN_DIR}"
    exit $?
fi

for p in "${OLD_CKPT}" "${NEW_CKPT}"; do
    [ -d "${p}" ] || { echo "[G1] 체크포인트가 없다: ${p}"; exit 1; }
done

echo ""
echo "══ [G1] old=${OLD_CKPT}"
echo "        new=${NEW_CKPT}"
echo "        probe_task=${PROBE_TASK}  skip_sr=${SKIP_SR}  sr_episodes=${SR_EPISODES}"

"${PYTHON}" "${G1_PY}" \
    --seed="${SEED}" \
    --job_name="G1_${RUN_TAG}" \
    --output_dir="${RUN_DIR}/run" \
    --dataset.repo_id="${DATASET_PREFIX}${PROBE_TASK}" \
    --policy.path="${OLD_CKPT}" \
    --policy.push_to_hub=false \
    --eval_freq=0 \
    --wandb.enable=false \
    --env.type=libero \
    --env.benchmark=libero_spatial \
    --env.task="${ENV_TASK_PREFIX}${PROBE_TASK}" \
    --env.episode_length="${EPISODE_LENGTH}" \
    --old_ckpt="${OLD_CKPT}" \
    --new_ckpt="${NEW_CKPT}" \
    --probe_task="${PROBE_TASK}" \
    --dataset_prefix="${DATASET_PREFIX}" \
    --env_task_prefix="${ENV_TASK_PREFIX}" \
    --drift_batches="${DRIFT_BATCHES}" \
    --drift_batch_size="${DRIFT_BATCH_SIZE}" \
    --drift_n_noise="${DRIFT_N_NOISE}" \
    --probe_batches="${PROBE_BATCHES}" \
    --probe_batch_size="${PROBE_BATCH_SIZE}" \
    --skip_sr="${SKIP_SR}" \
    --sr_episodes="${SR_EPISODES}" \
    --n_grid="${N_GRID}" \
    --windows="${WINDOWS}" \
    --out_root="${OUT_ROOT}" \
    --run_tag="${RUN_TAG}" || { echo "[G1] FAILED"; exit 1; }

echo ""
echo "[G1] done.  raw=${RUN_DIR}/g1_raw.npz"
echo "[G1] figures=${RUN_DIR}/G1_F1..F5*.png   table=${RUN_DIR}/G1_summary.csv"
echo "[G1] summary=${RUN_DIR}/g1_summary.json"
