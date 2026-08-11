#!/usr/bin/env bash
#
# PackNet — libero_spatial 태스크 0..3 순차 학습 (R1에 packnet 팔을 붙이기 위한 실행)
#
# E0가 만든 seq(λ=0) / ewc(λ=100) / frozen(λ=inf) 팔과 **같은 조건**에서 PackNet을 돌린다.
# R1이 읽는 트리 모양을 그대로 맞춘다:  <root>/task_{k}/checkpoints/last/pretrained_model
#
# PackNet (Mallya & Lazebnik, CVPR 2018)
#   태스크마다  (1) 남은 자리에서 학습 -> (2) 크기 하위 PRUNE_RATIO 가지치기
#   -> (3) 살아남은 자리만 다시 학습.  이후 태스크는 가지친 빈자리만 쓰고, 옛 태스크가
#   차지한 자리는 gradient를 0으로 막아 글자 그대로 보존한다.
#   scripts/packnet.py가 그 방식이며, 한 번 실행이 (1)+(3) 두 구간을 이어서 돈다:
#       step 0..STEPS-1            현재 태스크 학습 (pre-prune)
#       step == STEPS              가지치기 + 옵티마이저 교체
#       step STEPS..STEPS+POST-1   가지친 뒤 재학습 (post-prune)
#   checkpoints/last 는 재학습까지 끝난 마지막 체크포인트다.
#
# E0와 맞춰 둔 것
#   STEPS=5000(pre-prune), 배치 32, seed=42, 태스크당 뒤 HOLDOUT_EP개는 학습에서 제외.
#   POST_PRUNE_STEPS는 PackNet에만 있는 구간이라 E0 팔보다 총 스텝이 많다 —
#   가지친 직후의 모델은 쓸 수 없으므로 이 구간을 빼면 방법 자체가 성립하지 않는다.
#
# ★ packnet.py는 이 실행을 위해 두 군데 고쳤다. 둘 다 "옛 태스크는 안 움직인다"는
#   PackNet의 전제가 실제로는 깨져 있던 지점이다.
#
#   (1) 마스크 키를 모듈 이름 -> 파라미터 이름으로.
#       named_modules로 잡으면 module.weight 하나만 가리킨다. 가중치를 raw Parameter로
#       들고 있는 nn.MultiheadAttention의 in_proj_weight(6층 × 786K = 4.7M)와
#       velocity_net.dec_pos(8K)가 마스크 밖에 남아 태스크가 바뀌어도 계속 갱신됐다.
#       지금은 학습 가능한 파라미터가 전부 셋 중 하나에 들어간다:
#           마스크(ndim>=2 가중치) / 동결(bias·LayerNorm) / ignore_modules(얼린 백본).
#       로그의 "PackNet coverage: N masked / M frozen"이 그 분할을 찍는다.
#
#   (2) 옵티마이저 스텝 뒤에 보호 가중치를 되돌린다(snapshot/restore_protected).
#       gradient를 0으로 만드는 것만으로는 가중치가 멈추지 않는다. 이 정책의 프리셋은
#       torch Adam + weight_decay=1e-6인데, (AdamW와 달리) 평범한 Adam은 wd를 gradient에
#       더한 뒤 m/sqrt(v)로 정규화한다. 그래서 grad=0인 파라미터도 매 스텝 약 lr·sign(w)
#       만큼 움직인다 — 실측 10스텝에 4e-6, |w|와 무관. 10k 스텝이면 반올림 오차가 아니다.
#
#   검증(각 20+20스텝, 태스크 0->1->2): 태스크0 소유 슬롯 10,950,412개와 태스크1 소유
#   8,212,806개가 이후 태스크 학습 뒤에도 max|Δ|=0, bias/norm 74,894개도 변화 0.
#   가지치기로 비운 자리의 정확히 25%(=1-PRUNE_RATIO)를 다음 태스크가 채운다.
#
# 사용법
#   bash bash/clare/packnet/packnet_task0123.sh
#   PRUNE_RATIO=0.5 bash bash/clare/packnet/packnet_task0123.sh
#   REDO_INCOMPLETE=0 bash bash/clare/packnet/packnet_task0123.sh
#
# 끝난 스테이지는 out_dir/.done 으로 표시된다. 이 파일이 있는 스테이지만 건너뛴다.

set -uo pipefail

export MUJOCO_GL=${MUJOCO_GL:-egl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}

# HF_LEROBOT_HOME / HF_HUB_CACHE / PRETRAIN_PATH 를 세팅한다.
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

PACKNET_PY=./lerobot_lsy/src/lerobot/scripts/packnet.py
PYTHON=${PYTHON:-python}   # conda clare 환경이 활성화돼 있다고 가정. 아니면 PYTHON=... 로 지정.

# ── 조절할 것들 ───────────────────────────────────────────────────────────────
SEED=${SEED:-42}
NUM_TASKS=${NUM_TASKS:-4}                # 태스크 0..NUM_TASKS-1
STEPS=${STEPS:-5000}                     # 가지치기 전 학습 스텝 (E0와 동일)
POST_PRUNE_STEPS=${POST_PRUNE_STEPS:-5000}   # 가지친 뒤 재학습 스텝
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
LOG_FREQ=${LOG_FREQ:-100}

PRUNE_RATIO=${PRUNE_RATIO:-0.75}         # 현재 태스크 몫에서 잘라내 다음 태스크에 넘길 비율
# 이 정책에서 얼어 있는 백본. mask를 만들 필요도, gradient를 막을 필요도 없다.
IGNORE_MODULES=${IGNORE_MODULES:-language_encoder,pretrained_rgb_encoder}

HOLDOUT_EP=${HOLDOUT_EP:-5}              # 태스크당 50 에피소드 중 뒤 5개는 학습에서 제외

OUT_ROOT=${OUT_ROOT:-./outputs/PACKNET/libero_spatial/seed${SEED}}
mkdir -p "${OUT_ROOT}"

DATASET_PREFIX=continuallearning/libero_spatial_image_task_
ENV_TASK_PREFIX=Libero_Spatial_Task_

# 학습 대상 에피소드 목록 "[0,1,...,44]". E0는 샘플러로 갈랐지만 packnet.py에는 그
# 통로가 없으므로 --dataset.episodes로 준다. 0에서 시작하는 연속 구간이라
# LeRobotDataset의 재인덱싱 문제(E0.py episode_sampler 주석)에 걸리지 않는다.
TOTAL_EP=${TOTAL_EP:-50}
EPISODES=$(${PYTHON} -c "print('[' + ','.join(str(i) for i in range(${TOTAL_EP} - ${HOLDOUT_EP})) + ']')")

# ── 태스크 0..NUM_TASKS-1 순차 학습 ───────────────────────────────────────────
for k in $(seq 0 $((NUM_TASKS - 1))); do
    out_dir="${OUT_ROOT}/task_${k}"
    prev_dir="${OUT_ROOT}/task_$((k - 1))"

    # 끝까지 간 스테이지만 건너뛴다. 디렉터리 존재만 보면 중간에 죽은 스테이지가
    # 영원히 재실행되지 않는다(E0.sh와 같은 이유).
    if [ -f "${out_dir}/.done" ]; then
        echo "[PackNet] skip (done) ${out_dir}"
        continue
    fi
    if [ -d "${out_dir}" ]; then
        if [ "${REDO_INCOMPLETE:-1}" = "1" ]; then
            echo "[PackNet] incomplete stage -> removing and redoing: ${out_dir}"
            rm -rf "${out_dir}"
        else
            echo "[PackNet] incomplete stage left as-is (REDO_INCOMPLETE=0): ${out_dir}"
            exit 1
        fi
    fi

    # 첫 태스크는 사전학습 체크포인트(마스크를 새로 만든다), 이후는 직전 태스크
    # 체크포인트. 직전 체크포인트의 pretrained_model 안에 mask.safetensors가 같이
    # 저장돼 있고 packnet.py가 거기서 읽는다 — 이 한 줄이 스테이지를 잇는 연결고리다.
    policy_path="${PRETRAIN_PATH}"
    [ "${k}" -gt 0 ] && policy_path="${prev_dir}/checkpoints/last/pretrained_model"

    echo ""
    echo "══ [PackNet] task=${k}  init=${policy_path}"
    echo "            pre-prune ${STEPS} + prune(${PRUNE_RATIO}) + post-prune ${POST_PRUNE_STEPS}"

    "${PYTHON}" "${PACKNET_PY}" \
        --seed="${SEED}" \
        --job_name="PackNet_task_${k}" \
        --output_dir="${out_dir}" \
        --dataset.repo_id="${DATASET_PREFIX}${k}" \
        --dataset.episodes="${EPISODES}" \
        --policy.path="${policy_path}" \
        --policy.push_to_hub=false \
        --batch_size="${BATCH_SIZE}" \
        --num_workers="${NUM_WORKERS}" \
        --steps="${STEPS}" \
        --post_prune_steps="${POST_PRUNE_STEPS}" \
        --log_freq="${LOG_FREQ}" \
        --save_freq="${STEPS}" \
        --eval_freq=0 \
        --env.type=libero \
        --env.benchmark=libero_spatial \
        --env.task="${ENV_TASK_PREFIX}${k}" \
        --current_task="${k}" \
        --prune_ratio="${PRUNE_RATIO}" \
        --ignore_modules="${IGNORE_MODULES}" \
        --wandb.enable=false || { echo "[PackNet] FAILED task=${k}"; exit 1; }

    touch "${out_dir}/.done"
done

echo ""
echo "[PackNet] done.  tree=${OUT_ROOT}/task_{0..$((NUM_TASKS - 1))}/checkpoints/last/pretrained_model"
echo "[PackNet] R1에 붙이려면:  EXTRA_ARMS=\"packnet=${OUT_ROOT}\" bash bash/E0/R1.sh"
