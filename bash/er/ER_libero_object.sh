export MUJOCO_GL=egl;
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2};
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-$CUDA_VISIBLE_DEVICES};
SEED=${1:-42};

# Sets HF_LEROBOT_HOME (dataset download dir), HF_HUB_CACHE (model download dir)
# and PRETRAIN_PATH. Edit bash/clare/env.sh to change those locations.
source "$(dirname "${BASH_SOURCE[0]}")/er_env.sh";

# Experience Replay (Chaudhry et al., "On Tiny Episodic Memories in Continual
# Learning") -- libero_object 10태스크 CL 시퀀스. bash/er/ER_libero_spatial.sh 와
# 같은 구조/하이퍼파라미터다(스위트만 다름): 매 스텝 현재 태스크 배치와 메모리 버퍼
# 배치를 함께 뽑아 하나의 그래디언트 스텝을 밟는다. 어댑터가 없는 전체 파인튜닝이라
# 각 스테이지는 PRETRAIN_PATH가 아니라 **앞 스테이지 체크포인트**에서 이어받는다.
#
# 버퍼는 bash/er/make_object_er_buffers.sh 가 만든
# continuallearning/libero_object_image_task_0_{er,1..8}_new 로, 과거 태스크당
# 5에피소드를 담는다. spatial/goal과 같은 규칙이다.

HF_ORG=continuallearning;

STEPS=20000;
LOG_STEPS=100;
N_EVAL=100;
BS_EVAL=50;

# Task 0 has an empty buffer, so it is plain fine-tuning at the full batch size.
# From task 1 on the batch is split BS (current task) + REPLAY_BS (buffer);
# BS + REPLAY_BS == BS_FIRST keeps the memory footprint of every stage equal.
BS_FIRST=32;
BS=16;
REPLAY_BS=16;
NUM_WORKERS=12;
REPLAY_NUM_WORKERS=4;

# ★ 학습 중 시뮬레이터 평가는 기본으로 꺼 둔다 (EVAL_FREQ=0).
#   train.py:371 의 조건이 `cfg.eval_freq > 0` 이라, 평가가 한 번도 실행되지 않아도
#   시작 시점에 make_env(n_envs=eval.batch_size)가 호출되어 GPU를 다 먹는다.
#   성공률은 학습이 끝난 뒤 bash/er/eval_libero_object.sh 로 따로 잰다.
EVAL_FREQ=${EVAL_FREQ:-0};

python ./lerobot_lsy/src/lerobot/scripts/train.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_0_er \
    --output_dir=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_0_er \
    --dataset.repo_id=continuallearning/libero_object_image_task_0 \
    --policy.path=${PRETRAIN_PATH} \
    --policy.push_to_hub=false \
    --batch_size=$BS_FIRST \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_1_er \
    --output_dir=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_1_er \
    --dataset.repo_id=continuallearning/libero_object_image_task_1 \
    --replay_dataset.repo_id=continuallearning/libero_object_image_task_0_er_new \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --policy.path=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_0_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_2_er \
    --output_dir=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_2_er \
    --dataset.repo_id=continuallearning/libero_object_image_task_2 \
    --replay_dataset.repo_id=continuallearning/libero_object_image_task_0_1_new \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --policy.path=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_1_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_3_er \
    --output_dir=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_3_er \
    --dataset.repo_id=continuallearning/libero_object_image_task_3 \
    --replay_dataset.repo_id=continuallearning/libero_object_image_task_0_2_new \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --policy.path=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_2_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_4_er \
    --output_dir=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_4_er \
    --dataset.repo_id=continuallearning/libero_object_image_task_4 \
    --replay_dataset.repo_id=continuallearning/libero_object_image_task_0_3_new \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --policy.path=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_3_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_5_er \
    --output_dir=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_5_er \
    --dataset.repo_id=continuallearning/libero_object_image_task_5 \
    --replay_dataset.repo_id=continuallearning/libero_object_image_task_0_4_new \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --policy.path=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_4_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_6_er \
    --output_dir=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_6_er \
    --dataset.repo_id=continuallearning/libero_object_image_task_6 \
    --replay_dataset.repo_id=continuallearning/libero_object_image_task_0_5_new \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --policy.path=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_5_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5,Libero_Object_Task_6 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_7_er \
    --output_dir=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_7_er \
    --dataset.repo_id=continuallearning/libero_object_image_task_7 \
    --replay_dataset.repo_id=continuallearning/libero_object_image_task_0_6_new \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --policy.path=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_6_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5,Libero_Object_Task_6,Libero_Object_Task_7 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_8_er \
    --output_dir=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_8_er \
    --dataset.repo_id=continuallearning/libero_object_image_task_8 \
    --replay_dataset.repo_id=continuallearning/libero_object_image_task_0_7_new \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --policy.path=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_7_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5,Libero_Object_Task_6,Libero_Object_Task_7,Libero_Object_Task_8 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_object \
    ${WANDB_ENTITY_ARG} \
&& \

python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_object_task_9_er \
    --output_dir=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_9_er \
    --dataset.repo_id=continuallearning/libero_object_image_task_9 \
    --replay_dataset.repo_id=continuallearning/libero_object_image_task_0_8_new \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --policy.path=./outputs/libero_object/er/dit_flow_mt_cl_seed_${SEED}_libero_object_task_8_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5,Libero_Object_Task_6,Libero_Object_Task_7,Libero_Object_Task_8,Libero_Object_Task_9 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_object \
    ${WANDB_ENTITY_ARG} \
;
