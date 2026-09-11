#!/usr/bin/env bash
#
# CLARE 체크포인트 성능 평가 — libero_object (10태스크 CL 시퀀스)
#
# 스테이지 k 체크포인트로 그때까지 배운 태스크 0..k를 평가한다. 총 10×11/2 = 55회.
# 행=체크포인트, 열=태스크인 하삼각 행렬이 나온다.
#
# eval.py가 아니라 eval_peft.py --peft_weight_path 를 쓴다. 어댑터 선택은 롤아웃 중
# 관측을 보고 CLARE 판별기가 하므로 태스크 ID를 따로 넘기지 않는다.
#
# ★ --policy.path 는 스테이지 체크포인트의 pretrained_model/ 이 아니라 PRETRAIN_PATH다.
#   CLARE는 베이스를 얼리고 어댑터만 학습하므로 학습 스크립트도 스테이지마다
#   --policy.path=$PRETRAIN_PATH 로 돌았다. 반면 체크포인트의 pretrained_model/은
#   **PEFT로 감싼 뒤의** state dict라(cond_proj.weight -> cond_proj.base_layer.original_layer.weight)
#   이걸 맨 정책에 로드하면 cond_proj가 랜덤 초기화인 채 남아 성공률이 전부 0으로 나온다.
#   에러 없이 조용히 틀리므로 특히 조심할 것.
#
# 사용법
#   bash bash/clare/eval_libero_object_clare.sh
#   CUDA_VISIBLE_DEVICES=1 STAGES="0 1 2 3 4" bash bash/clare/eval_libero_object_clare.sh
#   N_EVAL=20 bash bash/clare/eval_libero_object_clare.sh      # 빠른 예비 확인
#   REDO=1    bash bash/clare/eval_libero_object_clare.sh      # 끝난 조합도 다시
#
# 이미 끝난 칸은 eval_info.json 존재로 건너뛴다. 중단했다 다시 돌려도 안전하다.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-$CUDA_VISIBLE_DEVICES}

SEED=${SEED:-42}
NUM_TASKS=${NUM_TASKS:-10}
# 다른 단일 스위트 표(libero_10 / goal / spatial)가 100 에피소드라 기본을 맞춘다.
N_EVAL=${N_EVAL:-100}
BS_EVAL=${BS_EVAL:-10}
RENDER=${RENDER:-0}                 # 저장할 비디오 수. 0이면 인코딩 비용이 사라진다.
REDO=${REDO:-0}

BASE_POLICY=${BASE_POLICY:-${PRETRAIN_PATH}}   # 얼려 둔 베이스. 위 ★ 주석 참고.
SUFFIX=${SUFFIX:-encoder_mlp_adapter_threshold_1_0}
CKPT_BASE=${CKPT_BASE:-./outputs/libero_object/clare}
OUT_ROOT=${OUT_ROOT:-./outputs/CLARE_eval/libero_object/seed${SEED}}
PYTHON=${PYTHON:-python}
EVAL_PY=./lerobot_lsy/src/lerobot/scripts/eval_peft.py

declare -a STAGE_CKPT TASK_HANDLE TASK_REPO
for i in $(seq 0 $((NUM_TASKS - 1))); do
    STAGE_CKPT[$i]="${CKPT_BASE}/dit_flow_mt_cl_seed_${SEED}_libero_object_task_${i}_${SUFFIX}/checkpoints/last"
    TASK_HANDLE[$i]="Libero_Object_Task_${i}"
    TASK_REPO[$i]="continuallearning/libero_object_image_task_${i}"
done

STAGES=${STAGES:-$(seq 0 $((NUM_TASKS - 1)))}
TASKS=${TASKS:-""}

mkdir -p "${OUT_ROOT}"

echo "══ CLARE eval  libero_object  seed=${SEED}  gpu=${CUDA_VISIBLE_DEVICES}"
echo "   stages   : $(echo ${STAGES} | tr '\n' ' ')"
echo "   episodes : ${N_EVAL}  (환경 ${BS_EVAL}개 동시)"
echo "   out      : ${OUT_ROOT}"
echo ""

n_run=0; n_skip=0; n_fail=0
for k in ${STAGES}; do
    ckpt="${STAGE_CKPT[$k]}"
    if [ ! -d "${ckpt}/adapter" ]; then
        echo "[eval] SKIP stage ${k}: 어댑터 없음 (${ckpt}/adapter)"
        continue
    fi
    task_list=${TASKS:-$(seq 0 "${k}")}
    for t in ${task_list}; do
        [ "${t}" -le "${k}" ] || continue
        out="${OUT_ROOT}/stage${k}/task${t}"
        if [ -f "${out}/eval_info.json" ] && [ "${REDO}" != "1" ]; then
            n_skip=$((n_skip + 1)); continue
        fi
        mkdir -p "${out}"
        echo "── $(date +%H:%M:%S) stage ${k} × task ${t} (${TASK_HANDLE[$t]})"
        "${PYTHON}" "${EVAL_PY}" \
            --policy.path="${BASE_POLICY}" \
            --peft_weight_path="${ckpt}/adapter" \
            --policy.device=cuda \
            --dataset.repo_id="${TASK_REPO[$t]}" \
            --env.type=libero \
            --env.benchmark=libero_object \
            --env.task="${TASK_HANDLE[$t]}" \
            --eval.batch_size="${BS_EVAL}" \
            --eval.n_episodes="${N_EVAL}" \
            --eval.max_episodes_rendered="${RENDER}" \
            --output_dir="${out}" \
            --seed="${SEED}" \
            >"${out}/eval.log" 2>&1 \
            || { echo "[eval] FAILED stage ${k} task ${t}  (로그: ${out}/eval.log)"; n_fail=$((n_fail + 1)); continue; }
        n_run=$((n_run + 1))
    done
done

echo ""
echo "[eval] 실행 ${n_run}  건너뜀 ${n_skip}  실패 ${n_fail}"

# 여러 샤드를 동시에 돌릴 때는 COLLECT=0으로 끄고, 끝난 뒤 한 번만 모은다.
if [ "${COLLECT:-1}" = "1" ]; then
    OUT_ROOT="${OUT_ROOT}" SEED="${SEED}" NUM_TASKS="${NUM_TASKS}" BENCH="libero_object" \
        bash "$(dirname "${BASH_SOURCE[0]}")/collect_clare_sr.sh"
fi
