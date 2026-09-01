#!/usr/bin/env bash
#
# R11 — prefix handover: joint이 앞 K스텝을 몰고 seq CL이 이어받으면 SR은 어떻게 되는가
#
# R9/R9_A는 forward 프로브라 "조건 라우팅이 joint와 CL에서 거의 같다"까지만 말했다.
# 그것만으로는 행동이 왜 무너지는지 알 수 없다. R11은 시뮬레이터에서 직접 개입한다.
#
#   step 0 … K-1   joint (tasks 0–3 mixed)
#   step K … 끝    seq CL (task 0 → … → 3)
#
# K를 키우며 SR을 재고, 태스크마다 그림 한 장을 낸다.
#   K=0 은 순수 CL, K가 에피소드 길이를 넘으면 순수 joint —— 이 두 끝점이 따로 잰
#   joint/CL SR과 맞아야 한다(맞지 않으면 인계 장치가 틀린 것이다. summary에 기록된다).
#
# 읽는 법
#   K를 조금만 줘도 SR이 오른다   CL의 고장이 궤적 **초반**에 몰려 있다
#   끝까지 줘야 오른다            CL이 전 구간에서 고장나 있다 (joint이 사실상 다 한 것)
#   아무리 줘도 안 오른다         좋은 상태에서도 CL이 태스크를 못 끝낸다
#
# 비교 대상 체크포인트 — R9_A와 **같은 것**을 쓴다. 두 실험이 갈라지면 안 된다.
#   joint = R9A_joint.py 로 pretrain에서 새로 20000스텝, 태스크 0,1,2,3을 번갈아
#           (태스크당 5000스텝 = 순차 팔과 총 업데이트·노출량이 정확히 같다)
#   cl    = E0.py lam0 (EWC λ=0 = 순차 파인튜닝) 를 task 0→1→2→3 으로 네 번
#
# 시간 감각: 태스크당 K 11지점 × 에피소드 50개 한 배치 ≈ 50~70분. 4 태스크를 GPU
#   4장에 하나씩 붙여 동시에 돌리므로 전체도 ≈ 1시간. K마다 결과를 파일에 쌓으므로
#   중간에 죽어도 다시 돌리면 이어서 간다(--redo 로 다시 재게 할 수 있다).
#
# 사용법
#   bash bash/E0/R11.sh                         # 전부 (측정 -> 그림)
#   PLOT_ONLY=1 bash bash/E0/R11.sh             # 이미 있는 json으로 그림만
#   REDO=1 bash bash/E0/R11.sh                  # 이미 잰 K도 다시
#   TASKS=0,3 bash bash/E0/R11.sh               # 일부 태스크만
#   SWITCH_STEPS=0,20,60,150,500 bash ...       # 다른 K 격자
#   SR_EPISODES=25 bash bash/E0/R11.sh          # 에피소드 수(=동시 env 수)를 줄인다
#   WARM_FOLLOWER=false bash bash/E0/R11.sh     # 인계받는 정책의 관측 큐를 데우지 않는다

set -uo pipefail
export MUJOCO_GL=egl

source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$(pwd)/lerobot_lsy/src${PYTHONPATH:+:${PYTHONPATH}}"

R11_PY=./lerobot_lsy/src/lerobot/scripts/R11.py
PYTHON=${PYTHON:-/home/sa090180/miniconda3/envs/clare/bin/python}

SEED=${SEED:-42}
ARM=${ARM:-lam0}
CL_STAGE=${CL_STAGE:-3}                      # 순차 CL을 몇 번째 태스크까지 학습했는가
TASKS=${TASKS:-0,1,2,3}
SWITCH_STEPS=${SWITCH_STEPS:-0,10,20,30,45,60,80,110,150,200,500}
SR_EPISODES=${SR_EPISODES:-50}
ENV_SEED=${ENV_SEED:-100000}
WARM_FOLLOWER=${WARM_FOLLOWER:-true}
NEED_MB=${NEED_MB:-9000}                     # 태스크 하나를 붙이기 전에 이만큼은 비어야 한다

JOINT_CKPT=${JOINT_CKPT:-./outputs/R9_joint/libero_spatial/seed_${SEED}/task0123/checkpoints/last/pretrained_model}
CL_CKPT=${CL_CKPT:-./outputs/E0/libero_spatial/seed_${SEED}/${ARM}/task_${CL_STAGE}/checkpoints/last/pretrained_model}
SR_DIR=${SR_DIR:-outputs/R9_A/_sr}           # 그림 아래 "단독 SR" 표의 출처

OUT_ROOT=${OUT_ROOT:-outputs/R11}
RUN_TAG=${RUN_TAG:-libero_spatial_seed${SEED}_${ARM}_cl${CL_STAGE}}
RUN_DIR=${OUT_ROOT}/${RUN_TAG}
LOG_DIR=${OUT_ROOT}/_logs
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

DATASET_PREFIX=continuallearning/libero_spatial_image_task_
ENV_TASK_PREFIX=Libero_Spatial_Task_

plot () {
    "${PYTHON}" "${R11_PY}" --plot_only=true \
        --policy.path="${JOINT_CKPT}" --policy.push_to_hub=false \
        --dataset.repo_id="${DATASET_PREFIX}0" \
        --env.type=libero --env.benchmark=libero_spatial \
        --out_root="${OUT_ROOT}" --run_tag="${RUN_TAG}" --sr_dir="${SR_DIR}" \
        --output_dir="${RUN_DIR}/_cfg_plot" --wandb.enable=false
}

if [ "${PLOT_ONLY:-0}" = "1" ]; then plot; exit $?; fi

for p in "${JOINT_CKPT}" "${CL_CKPT}"; do
    [ -d "${p}" ] || { echo "[R11] 체크포인트가 없다: ${p}"; exit 1; }
done

# ── GPU 하나가 비기를 기다린다 ────────────────────────────────────────────────
# ★ 태스크마다 GPU를 하나씩 잡는다. 다른 실험이 돌고 있으면 자리가 날 때까지 기다린다.
#   (여유 메모리 기준. 무한 대기하지 않도록 상한을 둔다.)
wait_for_gpu () {
    local want=$1 waited=0
    while [ "${waited}" -lt 7200 ]; do
        local free
        free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits \
               | awk -F', ' -v i="${want}" 'NR==i+1 {print $1-$2}')
        [ "${free:-0}" -ge "${NEED_MB}" ] && return 0
        [ $((waited % 300)) -eq 0 ] && echo "[R11] GPU ${want} 대기 중 (여유 ${free}MB < ${NEED_MB}MB)"
        sleep 30; waited=$((waited + 30))
    done
    echo "[R11] GPU ${want} 가 2시간 동안 비지 않았다"; return 1
}

N_GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "══ [R11] joint=${JOINT_CKPT}"
echo "        cl   =${CL_CKPT}"
echo "        tasks=${TASKS}  K=${SWITCH_STEPS}  episodes=${SR_EPISODES}  GPU ${N_GPU}장"

i=0; pids=(); names=()
IFS=',' read -ra TASK_ARR <<< "${TASKS}"
for T in "${TASK_ARR[@]}"; do
    GPU=$(( i % N_GPU )); i=$((i + 1))
    wait_for_gpu "${GPU}" || exit 1
    LOG="${LOG_DIR}/R11_task${T}_$(date +%m%d_%H%M).log"
    echo "── task ${T} -> GPU ${GPU}   log=${LOG}"
    (
      export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}"
      "${PYTHON}" "${R11_PY}" \
        --seed="${SEED}" --job_name="R11_task${T}" \
        --task="${T}" \
        --joint_ckpt="${JOINT_CKPT}" --cl_ckpt="${CL_CKPT}" \
        --policy.path="${JOINT_CKPT}" --policy.device=cuda --policy.push_to_hub=false \
        --dataset.repo_id="${DATASET_PREFIX}${T}" \
        --env.type=libero --env.benchmark=libero_spatial \
        --env_task_prefix="${ENV_TASK_PREFIX}" \
        --switch_steps="${SWITCH_STEPS}" \
        --sr_episodes="${SR_EPISODES}" --env_seed="${ENV_SEED}" \
        --warm_follower="${WARM_FOLLOWER}" \
        --redo="$([ "${REDO:-0}" = "1" ] && echo true || echo false)" \
        --out_root="${OUT_ROOT}" --run_tag="${RUN_TAG}" --sr_dir="${SR_DIR}" \
        --output_dir="${RUN_DIR}/_cfg_task${T}" --wandb.enable=false
    ) > "${LOG}" 2>&1 &
    pids+=($!); names+=("task${T}")
    sleep 20      # env 생성이 겹치면 EGL 초기화가 서로를 방해한다
done

fail=0
for k in "${!pids[@]}"; do
    if wait "${pids[$k]}"; then echo "[R11] ${names[$k]} 완료"
    else echo "[R11] ${names[$k]} FAILED — ${LOG_DIR} 의 로그를 봐라"; fail=1; fi
done

echo ""
echo "══ [R11] 그림"
plot || { echo "[R11] 그림 실패"; exit 1; }

echo ""
echo "[R11] done.  결과=${RUN_DIR}/R11_task*.json"
echo "[R11] 그림=${RUN_DIR}/R11_task{0,1,2,3}.png   설명=${RUN_DIR}/R11.method.md"
[ "${fail}" = "1" ] && echo "[R11] ★ 일부 태스크가 실패했다. 위 로그 확인." && exit 1
exit 0
