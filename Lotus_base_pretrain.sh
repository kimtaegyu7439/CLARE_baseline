# LOTUS setting 의 base phase — CLARE 정책(DiT-Flow MT)을 스위트별 base 데이터로
# from scratch 사전학습한다. bash/clare/pretrain.sh 와 같은 형식이고, 데이터 경로와
# 저장 위치만 다르다.
#
#   bash Lotus_base_pretrain.sh <suite> <gpu> [seed]
#   예)  bash Lotus_base_pretrain.sh goal   0 7
#        bash Lotus_base_pretrain.sh object 1 7
#        bash Lotus_base_pretrain.sh 50     2 7
#        bash Lotus_base_pretrain.sh 90     3 7
#
# base 데이터는 lotus_build_setting.py 가 만든 LeRobot v2.1 데이터셋이다.
#   LIBERO-OBJECT/base   6 task x 50 demo =  300 ep,  44,411 frame
#   LIBERO-GOAL/base     6 task x 50 demo =  300 ep,  40,442 frame
#   LIBERO-50/base      25 task x 50 demo = 1250 ep, 194,098 frame
# suite=90 만 예외다 — LOTUS 분할이 아니라 libero_90 전체(yzembodied, 4500 ep)를
# 그대로 쓴다. LOTUS 실험용이 아니라 비교용 범용 백본이라 저장 이름도 다르다.
SUITE=${1:?suite 를 주세요 (goal|object|50)};
GPU=${2:-0};
SEED=${3:-7};

export MUJOCO_GL=egl;
export CUDA_VISIBLE_DEVICES="${GPU}";

# Sets HF_LEROBOT_HOME (dataset download dir) and HF_HUB_CACHE (model download dir).
source "$(dirname "${BASH_SOURCE[0]}")/bash/clare/env.sh";

case "${SUITE}" in
    object) DIR=LIBERO-OBJECT;;
    goal)   DIR=LIBERO-GOAL;;
    50)     DIR=LIBERO-50;;
    90)     DIR=;;
    *) echo "알 수 없는 suite: ${SUITE} (goal|object|50|90)"; exit 1;;
esac

# repo_id 는 이름표일 뿐이다 — Hub 에 없고 실제 적재는 dataset.root 로 한다.
if [ "${SUITE}" = "90" ]; then
    DATASET_REPO_ID=${DATASET_REPO_ID:-local/yzembodied_libero_90};
    DATASET_ROOT=${DATASET_ROOT:-/home/sa090180/Datasets/lerobot/yzembodied_libero_90_dataset};
    JOB_NAME=Lotus_clare_libero90_pretrain;
else
    DATASET_REPO_ID=${DATASET_REPO_ID:-local/lotus_libero_${SUITE}_base};
    DATASET_ROOT=${DATASET_ROOT:-/home/sa090180/Datasets/lerobot/LOTUS_setting/${DIR}/base};
    JOB_NAME=Lotus_clare_libero_${SUITE}_pretrain;
fi

STEPS=200000;
LOG_STEPS=500;
SAVE_STEPS=50000;
BS=32;

JOB=${JOB_NAME};
OUT_DIR=./outputs/${JOB};
FINAL=/home/sa090180/Models/${JOB};

python ./lerobot_lsy/src/lerobot/scripts/train.py \
    --seed=$SEED \
    --job_name=$JOB \
    --output_dir=$OUT_DIR \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT \
    --policy.type=ditflow_mt \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=16 \
    --steps=$STEPS \
    --eval_freq=0 \
    --save_freq=$SAVE_STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=lotus_experiments \
    ${WANDB_ENTITY_ARG} \
&& \
mkdir -p "${FINAL}" && cp -aL "${OUT_DIR}/checkpoints/last/pretrained_model/." "${FINAL}/" && \
echo "[$(date '+%F %T')] ${JOB} 최종 가중치 -> ${FINAL}";
