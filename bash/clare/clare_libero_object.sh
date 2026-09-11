export MUJOCO_GL=egl;
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3};
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-$CUDA_VISIBLE_DEVICES};
SEED=${1:-42};

# Sets HF_LEROBOT_HOME (dataset download dir), HF_HUB_CACHE (model download dir)
# and PRETRAIN_PATH. Edit bash/clare/env.sh to change those locations.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh";

# CLARE -- libero_object 10태스크 CL 시퀀스.
# bash/clare/clare_libero_spatial.sh 와 같은 구조/하이퍼파라미터다(스위트만 다름):
# 베이스 정책은 매 스테이지 PRETRAIN_PATH로 얼려 두고 어댑터만 이어받는다
# (task_0은 --peft_cfg_path로 빈 설계도, task_1부터는 앞 스테이지의 adapter/).
#
# 성공률은 학습이 끝난 뒤 bash/clare/eval_libero_object_clare.sh 로 따로 잰다.
# (학습 중 평가는 eval_freq=200000 > steps 라 사실상 마지막에만 걸린다.)

HF_ORG=continuallearning;

STEPS=20000;
LOG_STEPS=100;
N_EVAL=100;
BS_EVAL=50;

python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_0_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_0_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_0 \
    --policy.path=${PRETRAIN_PATH} \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_cfg_path=./peft_lsy/peft_config/clare_dit_flow_encoder_adapter \
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
    --wandb.project=clare_libero_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_1_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_1_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_1 \
    --policy.path=${PRETRAIN_PATH} \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_0_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare_libero_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_2_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_2_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_2 \
    --policy.path=${PRETRAIN_PATH} \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_1_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare_libero_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_3_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_3_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_3 \
    --policy.path=${PRETRAIN_PATH} \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_2_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare_libero_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_4_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_4_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_4 \
    --policy.path=${PRETRAIN_PATH} \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_3_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare_libero_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_5_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_5_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_5 \
    --policy.path=${PRETRAIN_PATH} \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_4_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare_libero_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_6_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_6_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_6 \
    --policy.path=${PRETRAIN_PATH} \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5,Libero_Object_Task_6 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_5_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare_libero_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_7_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_7_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_7 \
    --policy.path=${PRETRAIN_PATH} \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5,Libero_Object_Task_6,Libero_Object_Task_7 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_6_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare_libero_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_8_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_8_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_8 \
    --policy.path=${PRETRAIN_PATH} \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5,Libero_Object_Task_6,Libero_Object_Task_7,Libero_Object_Task_8 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_7_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare_libero_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_9_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_9_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_9 \
    --policy.path=${PRETRAIN_PATH} \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5,Libero_Object_Task_6,Libero_Object_Task_7,Libero_Object_Task_8,Libero_Object_Task_9 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_object/clare/dit_flow_mt_cl_seed_${SEED}_libero_object_task_8_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare_libero_object \
    ${WANDB_ENTITY_ARG} \
;
