#!/usr/bin/env bash
# libero_goal seed1000 재개 — 08:48 디스크 풀 크래시로 task_8 저장이 깨졌다.
# task_0..7 은 완주(36칸)했고 task_7 의 checkpoints/last/adapter 가 온전하므로
# task_8 부터 이어 돌린다. 로그는 **append** 로 붙여야 앞의 36칸이 보존된다.
export MUJOCO_GL=egl;
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2};
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES}};
SEED=${1:-1000};
# 원본 suite 스크립트는 aicp_queue.sh 가 conda 를 켜 준다고 가정한다.
# 이 재개 스크립트는 단독 실행되므로 직접 켠다.
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare;
fi
source "$(dirname "${BASH_SOURCE[0]}")/aicp_env.sh";
HF_ORG=continuallearning;
STEPS=20000;
LOG_STEPS=100;
N_EVAL=50;
BS_EVAL=50;

python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_goal_task_8_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/aicp/libero_goal/clare/dit_flow_mt_cl_seed_${SEED}_libero_goal_task_8_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_goal_image_task_8 \
    --policy.path=${PRETRAIN_PATH} \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_goal \
    --env.task=Libero_Goal_Task_0,Libero_Goal_Task_1,Libero_Goal_Task_2,Libero_Goal_Task_3,Libero_Goal_Task_4,Libero_Goal_Task_5,Libero_Goal_Task_6,Libero_Goal_Task_7,Libero_Goal_Task_8 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/aicp/libero_goal/clare/dit_flow_mt_cl_seed_${SEED}_libero_goal_task_7_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
    --expand_threshold=1.0 \
    --detect_distribution_shift_steps=200 \
    --detect_distribution_shift_batch_size=32 \
    --detect_distribution_shift_num_workers=16 \
    --detect_distribution_shift_log_freq=10 \
    --train_discriminators_steps=2000 \
    --train_discriminators_batch_size=32 \
    --train_discriminators_num_workers=16 \
    --train_discriminators_log_freq=50 \
    --train_discriminators_eval_freq=2000 \
    --train_discriminators_save_freq=2000 \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=clare_libero_goal \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_goal_task_9_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/aicp/libero_goal/clare/dit_flow_mt_cl_seed_${SEED}_libero_goal_task_9_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_goal_image_task_9 \
    --policy.path=${PRETRAIN_PATH} \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_goal \
    --env.task=Libero_Goal_Task_0,Libero_Goal_Task_1,Libero_Goal_Task_2,Libero_Goal_Task_3,Libero_Goal_Task_4,Libero_Goal_Task_5,Libero_Goal_Task_6,Libero_Goal_Task_7,Libero_Goal_Task_8,Libero_Goal_Task_9 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/aicp/libero_goal/clare/dit_flow_mt_cl_seed_${SEED}_libero_goal_task_8_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
    --expand_threshold=1.0 \
    --detect_distribution_shift_steps=200 \
    --detect_distribution_shift_batch_size=32 \
    --detect_distribution_shift_num_workers=16 \
    --detect_distribution_shift_log_freq=10 \
    --train_discriminators_steps=2000 \
    --train_discriminators_batch_size=32 \
    --train_discriminators_num_workers=16 \
    --train_discriminators_log_freq=50 \
    --train_discriminators_eval_freq=2000 \
    --train_discriminators_save_freq=2000 \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=clare_libero_goal \
    ${WANDB_ENTITY_ARG};
