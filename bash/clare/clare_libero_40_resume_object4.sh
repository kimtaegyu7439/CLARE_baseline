export MUJOCO_GL=egl;
export CUDA_VISIBLE_DEVICES=2;
export MUJOCO_EGL_DEVICE_ID=2;
SEED=${1:-42};

# Sets HF_LEROBOT_HOME (dataset download dir), HF_HUB_CACHE (model download dir)
# and PRETRAIN_PATH. Edit bash/clare/env.sh to change those locations.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh";

HF_ORG=continuallearning;

STEPS=20000;
LOG_STEPS=100;
N_EVAL=10;
BS_EVAL=10;

# ── 이어서 실행용 (object task 4 ~ 9) ────────────────────────────────────────
# clare_libero_40_10_goal_spatial_object.sh 가 object task 4의 eval 도중 죽어서
# 거기서부터 다시 도는 스크립트다. 아래 블록 6개는 원본 1241~1455행을 그대로 복사한
# 것이라 인자가 한 글자도 다르지 않다.
#
# 왜 원본을 주석 처리하지 않았나
#   블록들이 `\`로 이어지는데, 그 사이에 `#`을 넣으면 주석이 앞줄과 합쳐져 뒤따르는
#   인자를 통째로 삼킨다. bash -n 은 통과하고 동작만 조용히 달라진다.
#
# 왜 task 4를 건너뛰지 않고 다시 도나
#   한 스테이지는 어댑터 20000스텝 + 판별기 2000스텝 = 22000스텝이다. 판별기 구간이
#   running_mean/std 를 갱신하고, 그 통계가 다음 태스크의 detect_distribution_shift
#   z-score 기준이 된다. 게다가 루프 안에서 eval 이 save 보다 먼저라(clare.py 954행 ->
#   1064행), 22000스텝 eval 에서 죽었으면 그 체크포인트는 저장되기 전이다. 즉
#   checkpoints/last 는 판별기가 안 붙은 20000스텝을 가리킨다 -> 다시 돌려야 한다.
#
# 실행 전 확인
#   1. object task 3 의 checkpoints/last/adapter 가 있어야 한다 (사슬의 유일한 고리)
#   2. object task 4 의 output_dir 은 지워야 한다
#      (clare.py 는 validate 를 오버라이드하지 않아 디렉터리가 있으면 FileExistsError)
#   아래 두 줄이 그걸 대신 확인해 준다. 파이썬까지 가서 죽지 않고 여기서 멈춘다.

OUT=./outputs/libero_40;
PREV=${OUT}/dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_3_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter;
HERE=${OUT}/dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_4_encoder_mlp_adapter_threshold_1_0;
[ -e "${PREV}" ] || { echo "[resume] 이전 스테이지 어댑터가 없다: ${PREV}"; exit 1; };
[ -e "${HERE}" ] && { echo "[resume] object task 4 출력이 남아 있다. 지우고 다시 실행하라: ${HERE}"; exit 1; };
echo "[resume] object task 4 부터 시작한다 (task 4..9, 6 스테이지)";

python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_4_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_40/dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_4_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_4 \
    --policy.path=$PRETRAIN_PATH \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_10_Task_0,Libero_10_Task_1,Libero_10_Task_2,Libero_10_Task_3,Libero_10_Task_4,Libero_10_Task_5,Libero_10_Task_6,Libero_10_Task_7,Libero_10_Task_8,Libero_10_Task_9,Libero_Goal_Task_0,Libero_Goal_Task_1,Libero_Goal_Task_2,Libero_Goal_Task_3,Libero_Goal_Task_4,Libero_Goal_Task_5,Libero_Goal_Task_6,Libero_Goal_Task_7,Libero_Goal_Task_8,Libero_Goal_Task_9,Libero_Spatial_Task_0,Libero_Spatial_Task_1,Libero_Spatial_Task_2,Libero_Spatial_Task_3,Libero_Spatial_Task_4,Libero_Spatial_Task_5,Libero_Spatial_Task_6,Libero_Spatial_Task_7,Libero_Spatial_Task_8,Libero_Spatial_Task_9,Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_40/dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_3_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare40 \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_5_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_40/dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_5_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_5 \
    --policy.path=$PRETRAIN_PATH \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_10_Task_0,Libero_10_Task_1,Libero_10_Task_2,Libero_10_Task_3,Libero_10_Task_4,Libero_10_Task_5,Libero_10_Task_6,Libero_10_Task_7,Libero_10_Task_8,Libero_10_Task_9,Libero_Goal_Task_0,Libero_Goal_Task_1,Libero_Goal_Task_2,Libero_Goal_Task_3,Libero_Goal_Task_4,Libero_Goal_Task_5,Libero_Goal_Task_6,Libero_Goal_Task_7,Libero_Goal_Task_8,Libero_Goal_Task_9,Libero_Spatial_Task_0,Libero_Spatial_Task_1,Libero_Spatial_Task_2,Libero_Spatial_Task_3,Libero_Spatial_Task_4,Libero_Spatial_Task_5,Libero_Spatial_Task_6,Libero_Spatial_Task_7,Libero_Spatial_Task_8,Libero_Spatial_Task_9,Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_40/dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_4_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare40 \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_6_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_40/dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_6_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_6 \
    --policy.path=$PRETRAIN_PATH \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_10_Task_0,Libero_10_Task_1,Libero_10_Task_2,Libero_10_Task_3,Libero_10_Task_4,Libero_10_Task_5,Libero_10_Task_6,Libero_10_Task_7,Libero_10_Task_8,Libero_10_Task_9,Libero_Goal_Task_0,Libero_Goal_Task_1,Libero_Goal_Task_2,Libero_Goal_Task_3,Libero_Goal_Task_4,Libero_Goal_Task_5,Libero_Goal_Task_6,Libero_Goal_Task_7,Libero_Goal_Task_8,Libero_Goal_Task_9,Libero_Spatial_Task_0,Libero_Spatial_Task_1,Libero_Spatial_Task_2,Libero_Spatial_Task_3,Libero_Spatial_Task_4,Libero_Spatial_Task_5,Libero_Spatial_Task_6,Libero_Spatial_Task_7,Libero_Spatial_Task_8,Libero_Spatial_Task_9,Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5,Libero_Object_Task_6 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_40/dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_5_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare40 \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_7_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_40/dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_7_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_7 \
    --policy.path=$PRETRAIN_PATH \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_10_Task_0,Libero_10_Task_1,Libero_10_Task_2,Libero_10_Task_3,Libero_10_Task_4,Libero_10_Task_5,Libero_10_Task_6,Libero_10_Task_7,Libero_10_Task_8,Libero_10_Task_9,Libero_Goal_Task_0,Libero_Goal_Task_1,Libero_Goal_Task_2,Libero_Goal_Task_3,Libero_Goal_Task_4,Libero_Goal_Task_5,Libero_Goal_Task_6,Libero_Goal_Task_7,Libero_Goal_Task_8,Libero_Goal_Task_9,Libero_Spatial_Task_0,Libero_Spatial_Task_1,Libero_Spatial_Task_2,Libero_Spatial_Task_3,Libero_Spatial_Task_4,Libero_Spatial_Task_5,Libero_Spatial_Task_6,Libero_Spatial_Task_7,Libero_Spatial_Task_8,Libero_Spatial_Task_9,Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5,Libero_Object_Task_6,Libero_Object_Task_7 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_40/dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_6_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare40 \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_8_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_40/dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_8_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_8 \
    --policy.path=$PRETRAIN_PATH \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_10_Task_0,Libero_10_Task_1,Libero_10_Task_2,Libero_10_Task_3,Libero_10_Task_4,Libero_10_Task_5,Libero_10_Task_6,Libero_10_Task_7,Libero_10_Task_8,Libero_10_Task_9,Libero_Goal_Task_0,Libero_Goal_Task_1,Libero_Goal_Task_2,Libero_Goal_Task_3,Libero_Goal_Task_4,Libero_Goal_Task_5,Libero_Goal_Task_6,Libero_Goal_Task_7,Libero_Goal_Task_8,Libero_Goal_Task_9,Libero_Spatial_Task_0,Libero_Spatial_Task_1,Libero_Spatial_Task_2,Libero_Spatial_Task_3,Libero_Spatial_Task_4,Libero_Spatial_Task_5,Libero_Spatial_Task_6,Libero_Spatial_Task_7,Libero_Spatial_Task_8,Libero_Spatial_Task_9,Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5,Libero_Object_Task_6,Libero_Object_Task_7,Libero_Object_Task_8 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_40/dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_7_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare40 \
    ${WANDB_ENTITY_ARG} \
&& \
python ./lerobot_lsy/src/lerobot/scripts/clare.py \
    --seed=$SEED \
    --job_name=dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_9_encoder_mlp_adapter_threshold_1_0 \
    --output_dir=./outputs/libero_40/dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_9_encoder_mlp_adapter_threshold_1_0 \
    --dataset.repo_id=continuallearning/libero_object_image_task_9 \
    --policy.path=$PRETRAIN_PATH \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --num_workers=16 \
    --steps=$STEPS \
    --env.type=libero \
    --env.benchmark=libero_object \
    --env.task=Libero_10_Task_0,Libero_10_Task_1,Libero_10_Task_2,Libero_10_Task_3,Libero_10_Task_4,Libero_10_Task_5,Libero_10_Task_6,Libero_10_Task_7,Libero_10_Task_8,Libero_10_Task_9,Libero_Goal_Task_0,Libero_Goal_Task_1,Libero_Goal_Task_2,Libero_Goal_Task_3,Libero_Goal_Task_4,Libero_Goal_Task_5,Libero_Goal_Task_6,Libero_Goal_Task_7,Libero_Goal_Task_8,Libero_Goal_Task_9,Libero_Spatial_Task_0,Libero_Spatial_Task_1,Libero_Spatial_Task_2,Libero_Spatial_Task_3,Libero_Spatial_Task_4,Libero_Spatial_Task_5,Libero_Spatial_Task_6,Libero_Spatial_Task_7,Libero_Spatial_Task_8,Libero_Spatial_Task_9,Libero_Object_Task_0,Libero_Object_Task_1,Libero_Object_Task_2,Libero_Object_Task_3,Libero_Object_Task_4,Libero_Object_Task_5,Libero_Object_Task_6,Libero_Object_Task_7,Libero_Object_Task_8,Libero_Object_Task_9 \
    --eval.batch_size=$BS_EVAL \
    --eval.n_episodes=$N_EVAL \
    --eval.max_episodes_rendered=4 \
    --eval_freq=200000 \
    --save_freq=$STEPS \
    --log_freq=$LOG_STEPS \
    --peft_weight_path=./outputs/libero_40/dit_flow_mt_cl_seed_${SEED}_libero_40_libero_object_task_8_encoder_mlp_adapter_threshold_1_0/checkpoints/last/adapter \
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
    --wandb.project=clare40 \
    ${WANDB_ENTITY_ARG};
