export MUJOCO_GL=egl;
export CUDA_VISIBLE_DEVICES=0;
export MUJOCO_EGL_DEVICE_ID=0;
SEED=${1:-42};

# Sets HF_LEROBOT_HOME (dataset download dir), HF_HUB_CACHE (model download dir)
# and PRETRAIN_PATH. Edit bash/clare/env.sh to change those locations.
source "$(dirname "${BASH_SOURCE[0]}")/er_env.sh";

# Experience Replay (Chaudhry et al., "On Tiny Episodic Memories in Continual
# Learning"): every step draws a batch from the current task AND a batch from a
# memory buffer holding the tasks seen so far, and takes one joint gradient step
# on the concatenation. Full fine-tuning -- no adapters, so each stage starts
# from the previous stage's checkpoint instead of from PRETRAIN_PATH.
#
# The buffers are the pre-built continuallearning/libero_40_image_task_0_{er,1..8}
# datasets, which hold 5 episodes per past task (5 x number of past tasks episodes in total).
# LIBERO-10 is the first block of the 40-task order, so libero_40_image_task_0_N
# is exactly LIBERO-10 tasks 0..N -- there is no separate libero_10_* buffer.

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
#
# 왜 0이어야 하는가 — 200000 같은 큰 값으로는 부족하다.
#   train.py:371 의 조건이 `cfg.eval_freq > 0` 이라, 평가가 한 번도 실행되지 않아도
#   시작 시점에 make_env(n_envs=eval.batch_size)가 호출된다. LIBERO 환경 하나가
#   MuJoCo + EGL 렌더링 컨텍스트를 GPU에 잡으므로 batch_size=50이면 정책이 올라갈
#   자리가 남지 않는다 (실측: "Creating env" 70초 뒤 LanguageEncoder.to(cuda)에서
#   CUDA out of memory). eval_freq=0 이면 make_env 호출 자체를 건너뛴다.
#
#   성공률은 학습이 끝난 뒤 체크포인트로 따로 잰다 -> bash/er/eval_*.sh
#
# 학습 중 평가를 굳이 켜려면 EVAL_FREQ=$STEPS 로 주되 BS_EVAL도 같이 낮춰라(10 이하).
EVAL_FREQ=${EVAL_FREQ:-0};

python ./lerobot_lsy/src/lerobot/scripts/train.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_10_task_0_er \
    --output_dir=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_0_er \
    --dataset.repo_id=continuallearning/libero_10_image_task_0 \
    --policy.path=${PRETRAIN_PATH} \
    --policy.push_to_hub=false \
    --batch_size=$BS_FIRST \
    --num_workers=$NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_10 \
    --env.task=Libero_10_Task_0 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_10 \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_10_task_1_er \
    --output_dir=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_1_er \
    --dataset.repo_id=continuallearning/libero_10_image_task_1 \
    --replay_dataset.repo_id=continuallearning/libero_40_image_task_0_er \
    --policy.path=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_0_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_10 \
    --env.task=Libero_10_Task_0,Libero_10_Task_1 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_10 \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_10_task_2_er \
    --output_dir=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_2_er \
    --dataset.repo_id=continuallearning/libero_10_image_task_2 \
    --replay_dataset.repo_id=continuallearning/libero_40_image_task_0_1 \
    --policy.path=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_1_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_10 \
    --env.task=Libero_10_Task_0,Libero_10_Task_1,Libero_10_Task_2 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_10 \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_10_task_3_er \
    --output_dir=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_3_er \
    --dataset.repo_id=continuallearning/libero_10_image_task_3 \
    --replay_dataset.repo_id=continuallearning/libero_40_image_task_0_2 \
    --policy.path=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_2_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_10 \
    --env.task=Libero_10_Task_0,Libero_10_Task_1,Libero_10_Task_2,Libero_10_Task_3 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_10 \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_10_task_4_er \
    --output_dir=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_4_er \
    --dataset.repo_id=continuallearning/libero_10_image_task_4 \
    --replay_dataset.repo_id=continuallearning/libero_40_image_task_0_3 \
    --policy.path=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_3_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_10 \
    --env.task=Libero_10_Task_0,Libero_10_Task_1,Libero_10_Task_2,Libero_10_Task_3,Libero_10_Task_4 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_10 \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_10_task_5_er \
    --output_dir=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_5_er \
    --dataset.repo_id=continuallearning/libero_10_image_task_5 \
    --replay_dataset.repo_id=continuallearning/libero_40_image_task_0_4 \
    --policy.path=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_4_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_10 \
    --env.task=Libero_10_Task_0,Libero_10_Task_1,Libero_10_Task_2,Libero_10_Task_3,Libero_10_Task_4,Libero_10_Task_5 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_10 \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_10_task_6_er \
    --output_dir=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_6_er \
    --dataset.repo_id=continuallearning/libero_10_image_task_6 \
    --replay_dataset.repo_id=continuallearning/libero_40_image_task_0_5 \
    --policy.path=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_5_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_10 \
    --env.task=Libero_10_Task_0,Libero_10_Task_1,Libero_10_Task_2,Libero_10_Task_3,Libero_10_Task_4,Libero_10_Task_5,Libero_10_Task_6 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_10 \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_10_task_7_er \
    --output_dir=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_7_er \
    --dataset.repo_id=continuallearning/libero_10_image_task_7 \
    --replay_dataset.repo_id=continuallearning/libero_40_image_task_0_6 \
    --policy.path=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_6_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_10 \
    --env.task=Libero_10_Task_0,Libero_10_Task_1,Libero_10_Task_2,Libero_10_Task_3,Libero_10_Task_4,Libero_10_Task_5,Libero_10_Task_6,Libero_10_Task_7 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_10 \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_10_task_8_er \
    --output_dir=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_8_er \
    --dataset.repo_id=continuallearning/libero_10_image_task_8 \
    --replay_dataset.repo_id=continuallearning/libero_40_image_task_0_7 \
    --policy.path=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_7_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_10 \
    --env.task=Libero_10_Task_0,Libero_10_Task_1,Libero_10_Task_2,Libero_10_Task_3,Libero_10_Task_4,Libero_10_Task_5,Libero_10_Task_6,Libero_10_Task_7,Libero_10_Task_8 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_10 \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/er.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_10_task_9_er \
    --output_dir=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_9_er \
    --dataset.repo_id=continuallearning/libero_10_image_task_9 \
    --replay_dataset.repo_id=continuallearning/libero_40_image_task_0_8 \
    --policy.path=./outputs/libero_10/er/dit_flow_mt_cl_seed_${SEED}_libero_10_task_8_er/checkpoints/last/pretrained_model \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=$NUM_WORKERS \
    --replay_batch_size=$REPLAY_BS \
    --replay_num_workers=$REPLAY_NUM_WORKERS \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_10 \
    --env.task=Libero_10_Task_0,Libero_10_Task_1,Libero_10_Task_2,Libero_10_Task_3,Libero_10_Task_4,Libero_10_Task_5,Libero_10_Task_6,Libero_10_Task_7,Libero_10_Task_8,Libero_10_Task_9 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --max_episodes_rendered=4 \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=ER_clare_10 \
    ${WANDB_ENTITY_ARG};
