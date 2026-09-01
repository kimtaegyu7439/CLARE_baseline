#!/usr/bin/env bash
#
# CLARE 체크포인트 성능 재평가 — libero_40 (40스테이지 CL 시퀀스)
#
# ── 왜 이 스크립트가 생겼는가 ────────────────────────────────────────────────
# 기존 libero_40 SR 매트릭스(outputs/libero_40/libero_40_SR.txt)는 틀렸다.
# clare.py는 --env.task의 쉼표 목록을 쪼개 태스크마다 env 설정을 만들면서 task 핸들만
# 바꾸고 benchmark는 스테이지 하나짜리 값(--env.benchmark)을 그대로 뒀다. 그런데
# gym.make(handle, **gym_kwargs)는 benchmark만 덮어쓰고 task_id는 핸들 등록값을 남기므로,
# (benchmark, task_id) 짝이 깨져 과거 태스크가 전부 "현재 suite의 같은 인덱스 태스크"로
# 리매핑돼 평가됐다. 그래서 매트릭스에 SR(k,j)≈0 ⟺ (j%10) > (k%10) 라는 규칙이 생겼다.
#
# 원인은 envs/configs.py의 LiberoEnv.resolved_benchmark로 고쳤다(핸들에서 benchmark를
# 되찾는다). 학습 자체는 영향을 받지 않았으므로 -- env는 평가에만 쓰이고 학습 데이터는
# 스테이지별 --dataset.repo_id로 올바르게 들어갔다 -- 재학습 없이 이 스크립트로 다시 재면 된다.
#
# ── 무엇을 재는가 ─────────────────────────────────────────────────────────────
# 스테이지 k 체크포인트로 그때까지 배운 태스크 0..k를 평가한다. 총 40×41/2 = 820회.
# 행=체크포인트, 열=태스크인 하삼각 행렬이 나온다.
#
#   스테이지  0..9   libero_10       (Libero_10_Task_0..9)
#            10..19  libero_goal     (Libero_Goal_Task_0..9)
#            20..29  libero_spatial  (Libero_Spatial_Task_0..9)
#            30..39  libero_object   (Libero_Object_Task_0..9)
#
# eval.py가 아니라 eval_peft.py --peft_weight_path 를 쓴다. 어댑터 선택은 롤아웃 중
# 관측을 보고 CLARE 판별기가 하므로 태스크 ID를 따로 넘기지 않는다.
#
# ★ --policy.path 는 스테이지 체크포인트의 pretrained_model/ 이 아니라 PRETRAIN_PATH다.
#   CLARE는 베이스를 얼리고 어댑터만 학습하므로 학습 스크립트도 스테이지마다
#   --policy.path=$PRETRAIN_PATH 로 돌았다. 반면 체크포인트의 pretrained_model/은
#   **PEFT로 감싼 뒤의** state dict라(cond_proj.weight -> cond_proj.base_layer.original_layer.weight,
#   여기에 어댑터/판별기 텐서 426개가 추가) 이걸 맨 정책에 로드하면 cond_proj가
#   랜덤 초기화인 채 남아 성공률이 전부 0으로 나온다. 조용히 틀리므로 특히 조심할 것.
#   (ER 쪽 eval_libero_40.sh가 pretrained_model/을 쓰는 건 ER은 PEFT가 없어서 맞다.)
#
# 사용법
#   bash bash/clare/eval_libero_40_clare.sh
#   CUDA_VISIBLE_DEVICES=1 STAGES="$(seq 0 13)"  bash bash/clare/eval_libero_40_clare.sh
#   REDO=1 bash bash/clare/eval_libero_40_clare.sh        # 이미 끝난 칸도 다시
#
# 이미 끝난 칸은 eval_info.json 존재로 건너뛴다. 중단했다 다시 돌려도 안전하다.

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-$CUDA_VISIBLE_DEVICES}

SEED=${SEED:-42}
NUM_TASKS=${NUM_TASKS:-40}
N_EVAL=${N_EVAL:-20}
BS_EVAL=${BS_EVAL:-20}
RENDER=${RENDER:-0}                 # 저장할 비디오 수. 0이면 인코딩 비용이 사라진다.
REDO=${REDO:-0}

BASE_POLICY=${BASE_POLICY:-${PRETRAIN_PATH}}   # 얼려 둔 베이스. 위 ★ 주석 참고.
SUFFIX=${SUFFIX:-encoder_mlp_adapter_threshold_1_0}
CKPT_BASE=${CKPT_BASE:-./outputs/libero_40}
OUT_ROOT=${OUT_ROOT:-./outputs/CLARE_eval/libero_40/seed${SEED}}
PYTHON=${PYTHON:-python}
EVAL_PY=./lerobot_lsy/src/lerobot/scripts/eval_peft.py

# 벤치마크 블록: 이름 / gym 핸들 접두사 / 각 10태스크.
BLOCK_BENCH=(libero_10       libero_goal        libero_spatial       libero_object)
BLOCK_HANDLE=(Libero_10_Task Libero_Goal_Task   Libero_Spatial_Task  Libero_Object_Task)

declare -a STAGE_CKPT TASK_BENCH TASK_HANDLE TASK_REPO
for i in $(seq 0 $((NUM_TASKS - 1))); do
    b=$((i / 10)); j=$((i % 10))
    bench=${BLOCK_BENCH[$b]}
    STAGE_CKPT[$i]="${CKPT_BASE}/dit_flow_mt_cl_seed_${SEED}_libero_40_${bench}_task_${j}_${SUFFIX}/checkpoints/last"
    TASK_BENCH[$i]="${bench}"
    TASK_HANDLE[$i]="${BLOCK_HANDLE[$b]}_${j}"
    TASK_REPO[$i]="continuallearning/${bench}_image_task_${j}"
done

STAGES=${STAGES:-$(seq 0 $((NUM_TASKS - 1)))}
TASKS=${TASKS:-""}

mkdir -p "${OUT_ROOT}"

echo "══ CLARE eval  libero_40  seed=${SEED}  gpu=${CUDA_VISIBLE_DEVICES}"
echo "   stages   : $(echo ${STAGES} | tr '\n' ' ')"
echo "   tasks    : ${TASKS:-<스테이지마다 0..k>}"
echo "   episodes : ${N_EVAL}  (환경 ${BS_EVAL}개 동시)"
echo "   out      : ${OUT_ROOT}"
echo ""

n_run=0; n_skip=0; n_fail=0
for k in ${STAGES}; do
    ckpt="${STAGE_CKPT[$k]}"
    if [ ! -d "${ckpt}/adapter" ]; then
        echo "[eval] SKIP stage ${k}: 체크포인트 없음 (${ckpt})"
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
        echo "── $(date +%H:%M:%S) stage ${k} × task ${t} (${TASK_HANDLE[$t]} @ ${TASK_BENCH[$t]})"
        "${PYTHON}" "${EVAL_PY}" \
            --policy.path="${BASE_POLICY}" \
            --peft_weight_path="${ckpt}/adapter" \
            --policy.device=cuda \
            --dataset.repo_id="${TASK_REPO[$t]}" \
            --env.type=libero \
            --env.benchmark="${TASK_BENCH[$t]}" \
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
    OUT_ROOT="${OUT_ROOT}" SEED="${SEED}" NUM_TASKS="${NUM_TASKS}" \
        bash "$(dirname "${BASH_SOURCE[0]}")/collect_libero_40_clare.sh"
fi
