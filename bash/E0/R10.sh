#!/usr/bin/env bash
#
# R10 — 블록별 "조건 기여의 크기 대 task 쏠림" 지도
#
# R9는 조건 기여 Δ의 방향 분리가 joint와 CL에서 거의 같다는 것을 보였다(라우팅은 안 죽는다).
# R10은 그 다음을 묻는다: 조건 기여가 **크기에서** 눌리는가, 그리고 남은 성분이 어느 task를
# 닮았는가.
#   vbase = ½(Δ(c₀)+Δ(c₁))   조건 평균 기여   (무조건 성분이 아니다 — 캡션 참조)
#   delta = ½(Δ(c₀)−Δ(c₁))   조건 대비 성분
#   밝기 ρ = ‖vbase‖/‖delta‖        색 lean = cos(vbase,u₁) − cos(vbase,u₀)
#   u₀ = FT0에 task0 조건을 넣은 기여, u₁ = FT1에 task1 조건을 넣은 기여 (조건부 기여!)
#
# 필요한 체크포인트 5개: pretrain · joint · cl · FT0 · FT1
#   FT0 = ckpt_root/task_{A} (task A만 배운 상태) — 순차 트리 0단계와 같다
#   FT1 = task B만 사전학습에서 바로 배운 것 — 별도 학습이 필요하다(--ft1_ckpt)
#
# 사용법
#   bash bash/E0/R10.sh
#   PLOT_ONLY=1 bash bash/E0/R10.sh
#   REDO=1 bash bash/E0/R10.sh
#   PER_STEP=0 bash bash/E0/R10.sh

set -uo pipefail
export MUJOCO_GL=${MUJOCO_GL:-egl}
source "$(dirname "${BASH_SOURCE[0]}")/../clare/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$(pwd)/lerobot_lsy/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON=${PYTHON:-/home/sa090180/miniconda3/envs/clare/bin/python}
R10_PY=./lerobot_lsy/src/lerobot/scripts/R10.py

# GPU 자동 선택 (R9.sh와 같은 규칙: 여유 메모리 − 사용률×100)
NEED_MB=${NEED_MB:-11000}; POLL_SEC=${POLL_SEC:-60}
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES="${GPU}"; else
  waited=0
  while :; do
    best=""; bs=-999999; bf=0
    while IFS=, read -r i u tot ut; do
      i=${i// /}; u=${u// /}; tot=${tot// /}; ut=${ut// /}; [ -z "$i" ] && continue
      fr=$((tot-u)); sc=$((fr - ut*100))
      if [ "$sc" -gt "$bs" ]; then bs=$sc; best=$i; bf=$fr; fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
    if [ "$bf" -ge "$NEED_MB" ]; then export CUDA_VISIBLE_DEVICES="$best"
      echo "[R10] GPU $best (여유 ${bf} MiB, 대기 ${waited}초)"; break; fi
    echo "[R10] 여유 ${bf} MiB < ${NEED_MB} — ${POLL_SEC}초 대기"; sleep "$POLL_SEC"; waited=$((waited+POLL_SEC))
  done
fi
export MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES}

SEED=${SEED:-42}; BENCH=${BENCH:-libero_spatial}
TASK_A=${TASK_A:-0}; TASK_B=${TASK_B:-1}
COND_MODE=${COND_MODE:-full}
E0_ROOT=${E0_ROOT:-./outputs/E0/${BENCH}/seed_${SEED}}
CKPT_ROOT=${CKPT_ROOT:-${E0_ROOT}/lam0}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-${PRETRAIN_PATH}}
JOINT_CKPT=${JOINT_CKPT:-./outputs/R9_joint/${BENCH}/seed_${SEED}/task${TASK_A}_${TASK_B}/checkpoints/last/pretrained_model}
FT0_CKPT=${FT0_CKPT:-${CKPT_ROOT}/task_${TASK_A}/checkpoints/last/pretrained_model}
FT1_CKPT=${FT1_CKPT:-${E0_ROOT}/ft1/task_${TASK_B}/checkpoints/last/pretrained_model}

ROLLOUT_STEPS=${ROLLOUT_STEPS:-75}; OBS_STRIDE=${OBS_STRIDE:-10}
NUM_PROBE=${NUM_PROBE:-100}; PROBE_SEED=${PROBE_SEED:-20260813}
T_STEPS=${T_STEPS:-20}; NUM_OBS=${NUM_OBS:-5}; DEMO_EPISODES=${DEMO_EPISODES:-10}
PER_STEP=${PER_STEP:-1}
OUT_ROOT=${OUT_ROOT:-./outputs/R10}
RUN_TAG=${RUN_TAG:-${BENCH}_seed${SEED}_task${TASK_A}v${TASK_B}}
RUN_DIR=${OUT_ROOT}/${RUN_TAG}; mkdir -p "${RUN_DIR}"
DATASET_PREFIX=continuallearning/${BENCH}_image_task_
ENV_TASK_PREFIX=${ENV_TASK_PREFIX:-Libero_Spatial_Task_}

if [ "${PLOT_ONLY:-0}" = "1" ]; then
  "${PYTHON}" "${R10_PY}" --plot_only --run_dir="${RUN_DIR}" --per_step="${PER_STEP}"; exit $?
fi
for p in "${PRETRAIN_CKPT}" "${JOINT_CKPT}" "${CKPT_ROOT}/task_${TASK_B}/checkpoints/last/pretrained_model" "${FT0_CKPT}" "${FT1_CKPT}"; do
  [ -d "$p" ] || { echo "[R10] 체크포인트 없음: $p"; exit 1; }
done
extra=(); [ "${REDO:-0}" = "1" ] && extra+=("--recompute=true")

echo "══ [R10] ${BENCH} seed ${SEED} · c₀=task ${TASK_A} vs c₁=task ${TASK_B} · GPU ${CUDA_VISIBLE_DEVICES}"
echo "        FT0 = ${FT0_CKPT}"
echo "        FT1 = ${FT1_CKPT}"

"${PYTHON}" "${R10_PY}" \
  --seed="${SEED}" --job_name="R10_${RUN_TAG}" --output_dir="${RUN_DIR}/run" \
  --dataset.repo_id="${DATASET_PREFIX}${TASK_A}" \
  --policy.path="${PRETRAIN_CKPT}" --policy.push_to_hub=false \
  --eval_freq=0 --wandb.enable=false \
  --env.type=libero --env.benchmark="${BENCH}" \
  --env.task="${ENV_TASK_PREFIX}${TASK_A}" --env.episode_length=300 \
  --ckpt_root="${CKPT_ROOT}" --pretrain_ckpt="${PRETRAIN_CKPT}" \
  --joint_ckpt="${JOINT_CKPT}" --ft0_ckpt="${FT0_CKPT}" --ft1_ckpt="${FT1_CKPT}" \
  --models="pretrain,joint,cl" --task_a="${TASK_A}" --task_b="${TASK_B}" \
  --cond_mode="${COND_MODE}" --exec_slice=auto \
  --num_probe="${NUM_PROBE}" --probe_seed="${PROBE_SEED}" \
  --t_steps="${T_STEPS}" --t_max=0.95 \
  --num_obs="${NUM_OBS}" --demo_episodes="${DEMO_EPISODES}" \
  --rollout_steps="${ROLLOUT_STEPS}" --obs_stride="${OBS_STRIDE}" --obs_driver=demo \
  --dataset_prefix="${DATASET_PREFIX}" --env_task_prefix="${ENV_TASK_PREFIX}" \
  --out_root="${OUT_ROOT}" --run_tag="${RUN_TAG}" --no_plot=true \
  "${extra[@]}" || { echo "[R10] FAILED"; exit 1; }

if compgen -G "${RUN_DIR}/R10_*.npz" > /dev/null; then
  "${PYTHON}" "${R10_PY}" --plot_only --run_dir="${RUN_DIR}" --per_step="${PER_STEP}"
else echo "[R10] npz 없음"; exit 1; fi
echo ""
echo "→ ${RUN_DIR}/R10_full.png / .pdf        결합 지도 (색=lean, 밝기=ρ)"
echo "→ ${RUN_DIR}/R10_full_rho.png           ρ 단독 + 분자/분모"
echo "→ ${RUN_DIR}/R10_full.summary.json      모델별 ρ·lean·hot 블록 + cos(u₀,u₁)"
