export CUDA_VISIBLE_DEVICES=0;
SEED=${1:-1000};

# Sets HF_LEROBOT_HOME (dataset download dir) and HF_HUB_CACHE (model download dir).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh";

DATASET_REPO_ID=${DATASET_REPO_ID:-continuallearning/libero_90_image};
# Leave empty to let LeRobot resolve it to $HF_LEROBOT_HOME/$DATASET_REPO_ID.
DATASET_ROOT=${DATASET_ROOT:-${HF_LEROBOT_HOME}/${DATASET_REPO_ID}};

STEPS=200000;
LOG_STEPS=500;
SAVE_STEPS=50000;
BS=32;

python ./lerobot_lsy/src/lerobot/scripts/train.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_pretrain \
    --output_dir=./outputs/dit_flow_mt_pretrain \
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
    --wandb.project=clare_experiments \
    ${WANDB_ENTITY_ARG};
