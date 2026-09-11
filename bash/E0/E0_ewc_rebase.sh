#!/usr/bin/env bash
#
# E0-EWC-rebase — EWC 팔을 seq(lam0)의 task_0 위에 다시 세운다.
#
# 왜
#   task 0 에서는 앵커가 없어 EWC 페널티가 0이다(E0.py의 update_policy 분기). 즉 λ=100 과
#   λ=0 의 목적함수가 완전히 같으므로 두 팔의 task_0 은 같은 모델이어야 한다.
#   그런데 실제로는 다르다:
#       lam0 / laminf / lam*_fisherbug (8/3~8/4 실행)  md5 3efcdc23  ← 5개 런이 전부 동일
#       lam10 / lam100 / lam1000       (8/5 재실행)    md5 20f86491  ← 3개 런이 전부 동일
#   공통 키 원소의 33.6%가 다르고 텐서별 상대 L2 차이 median 9.2%다. 부동소수점 잡음이
#   아니다. 같은 날 실행끼리는 비트 단위로 같으니 학습 자체는 결정론적이고, 8/5 재실행이
#   커밋되지 않은 다른 코드 상태에서 돌았다고 보는 게 맞다(그 diff는 복원 불가).
#
#   결과적으로 R2-B에서 EWC의 CL stage 1이 0이 아니다. 아직 아무것도 잊지 않은 시점인데도
#   center_shift 0.25 / assign_change 0.20 이 찍힌다. 이건 망각이 아니라 런 차이다.
#   그 상태로는 EWC 곡선의 상승분을 읽을 수 없다.
#
# 무엇을
#   lam0/task_0 을 EWC 팔의 stage 0 으로 그대로 쓰고, task_1..3 을 거기서 다시 학습한다.
#   lam0/task_0 은 seq·frozen·er(심볼릭 링크) 세 팔이 이미 공유하는 기준점이라 여기에
#   맞추는 게 최소 변경이다.
#
#   걸림돌 하나: E0.py 는 lambda>0 인 팔에서만 ewc_state.pt 를 남기므로 lam0/task_0 에는
#   Fisher/anchor 가 없다. 학습 없이 그 체크포인트에서 Fisher 만 새로 만들어 채운다
#   (util/make_ewc_state.py, E0.build_ewc_state 를 그대로 호출한다).
#
# 기존 lam100 은 건드리지 않는다. 새 팔은 lam100_rebased 로 따로 만든다.
#
# 사용법
#   bash bash/E0/E0_ewc_rebase.sh
#   LAMBDA=1000 ARM_OUT=lam1000_rebased bash bash/E0/E0_ewc_rebase.sh
#   PROBE_SR=false bash bash/E0/E0_ewc_rebase.sh      # SR 프로브 생략(빠름)

set -uo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-1}

source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

E0_PY=./lerobot_lsy/src/lerobot/scripts/E0.py
FISHER_PY=./lerobot_lsy/src/lerobot/scripts/util/make_ewc_state.py
PYTHON=${PYTHON:-python}

SEED=${SEED:-42}
NUM_TASKS=${NUM_TASKS:-4}
LAMBDA=${LAMBDA:-100}
STEPS=${STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
LOG_FREQ=${LOG_FREQ:-100}

HOLDOUT_EP=${HOLDOUT_EP:-5}
PROBE_BATCHES=${PROBE_BATCHES:-16}
PROBE_SR=${PROBE_SR:-true}
PROBE_N_EP=${PROBE_N_EP:-20}
PROBE_EVAL_BS=${PROBE_EVAL_BS:-20}

OUT_ROOT=${OUT_ROOT:-./outputs/E0/libero_spatial/seed_${SEED}}
BASE_ARM=${BASE_ARM:-lam0}                       # stage 0 을 빌려올 팔
ARM_OUT=${ARM_OUT:-lam${LAMBDA}_rebased}
RESULTS=${RESULTS:-${OUT_ROOT}/e0_results.jsonl}
RUN_TAG=${RUN_TAG:-${LAMBDA}_rebased}            # 그림 범례에서 기존 lam100 과 구분된다

DATASET_PREFIX=continuallearning/libero_spatial_image_task_
ENV_TASK_PREFIX=Libero_Spatial_Task_

BASE_CKPT=${OUT_ROOT}/${BASE_ARM}/task_0/checkpoints/last/pretrained_model
[ -d "${BASE_CKPT}" ] || { echo "[rebase] 기준 체크포인트가 없다: ${BASE_CKPT}"; exit 1; }

# ── stage 0: lam0/task_0 을 그대로 쓰고 Fisher 만 새로 만든다 ─────────────────
S0=${OUT_ROOT}/${ARM_OUT}/task_0
mkdir -p "${S0}"
if [ ! -e "${S0}/checkpoints" ]; then
    ln -s "$(cd ${OUT_ROOT}/${BASE_ARM}/task_0 && pwd)/checkpoints" "${S0}/checkpoints"
    echo "[rebase] stage 0 체크포인트 -> ${BASE_ARM}/task_0 심볼릭 링크"
fi
if [ ! -f "${S0}/ewc_state.pt" ]; then
    echo ""
    echo "══ [rebase] stage 0 의 Fisher/anchor 생성 (학습 없음)"
    "${PYTHON}" "${FISHER_PY}" \
        --ckpt="${BASE_CKPT}" \
        --repo_id="${DATASET_PREFIX}0" \
        --out="${S0}/ewc_state.pt" \
        --holdout_episodes="${HOLDOUT_EP}" \
        --seed="${SEED}" || { echo "[rebase] Fisher 생성 실패"; exit 1; }
else
    echo "[rebase] stage 0 ewc_state.pt 이미 있음"
fi
touch "${S0}/.done"

# ── stage 1..N-1: 거기서부터 순차 학습 ───────────────────────────────────────
for k in $(seq 1 $((NUM_TASKS - 1))); do
    out_dir="${OUT_ROOT}/${ARM_OUT}/task_${k}"
    prev_dir="${OUT_ROOT}/${ARM_OUT}/task_$((k - 1))"

    if [ -f "${out_dir}/.done" ]; then
        echo "[rebase] skip (done) ${out_dir}"
        continue
    fi
    if [ -d "${out_dir}" ]; then
        if [ "${REDO_INCOMPLETE:-1}" = "1" ]; then
            echo "[rebase] incomplete stage -> removing: ${out_dir}"
            rm -rf "${out_dir}"
        else
            echo "[rebase] incomplete stage left as-is: ${out_dir}"; exit 1
        fi
    fi

    echo ""
    echo "══ [rebase] λ=${LAMBDA}  task=${k}  init=${prev_dir}/checkpoints/last/pretrained_model"

    "${PYTHON}" "${E0_PY}" \
        --seed="${SEED}" \
        --job_name="E0_${ARM_OUT}_task_${k}" \
        --output_dir="${out_dir}" \
        --dataset.repo_id="${DATASET_PREFIX}${k}" \
        --policy.path="${prev_dir}/checkpoints/last/pretrained_model" \
        --policy.push_to_hub=false \
        --batch_size="${BATCH_SIZE}" \
        --num_workers="${NUM_WORKERS}" \
        --steps="${STEPS}" \
        --log_freq="${LOG_FREQ}" \
        --save_freq="${STEPS}" \
        --eval_freq=0 \
        --env.type=libero \
        --env.benchmark=libero_spatial \
        --env.task="${ENV_TASK_PREFIX}${k}" \
        --ewc_lambda="${LAMBDA}" \
        --ewc_state_path="${prev_dir}/ewc_state.pt" \
        --run_tag="${RUN_TAG}" \
        --current_task="${k}" \
        --task_ids="$(seq -s, 0 "${k}")" \
        --dataset_prefix="${DATASET_PREFIX}" \
        --env_task_prefix="${ENV_TASK_PREFIX}" \
        --results_path="${RESULTS}" \
        --holdout_episodes="${HOLDOUT_EP}" \
        --probe_batches="${PROBE_BATCHES}" \
        --probe_sr="${PROBE_SR}" \
        --probe_n_episodes="${PROBE_N_EP}" \
        --probe_eval_batch_size="${PROBE_EVAL_BS}" \
        --wandb.enable=false || { echo "[rebase] FAILED task=${k}"; exit 1; }
done

echo ""
echo "[rebase] done.  tree=${OUT_ROOT}/${ARM_OUT}/task_{0..$((NUM_TASKS - 1))}"
echo "[rebase] stage 0 은 ${BASE_ARM}/task_0 과 같은 체크포인트다 -> R2-B 의 CL stage 1 이 정확히 0 이 된다."
echo "[rebase] R1/R2 에 붙이려면:  EWC_ROOT=${OUT_ROOT}/${ARM_OUT}"
