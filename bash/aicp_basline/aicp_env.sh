# Central path configuration for CLARE.
#
# Every training script sources this file, so datasets and model weights are
# downloaded into the directories below instead of the default ~/.cache location.
# Override any of these by exporting them before launching a script.

export EVAL_SEED=${EVAL_SEED:-7};

# --- User-facing roots -------------------------------------------------------
export CLARE_DATA_ROOT=${CLARE_DATA_ROOT:-/home/sa090180/Datasets};
export CLARE_MODEL_ROOT=${CLARE_MODEL_ROOT:-/home/sa090180/Models};

# --- Datasets ----------------------------------------------------------------
# LeRobotDataset resolves its storage dir to $HF_LEROBOT_HOME/<repo_id> whenever
# --dataset.root is not passed (see lerobot_lsy/src/lerobot/constants.py), so the
# LIBERO task datasets land in $CLARE_DATA_ROOT/lerobot/continuallearning/...
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-${CLARE_DATA_ROOT}/lerobot};

# --- Models ------------------------------------------------------------------
# Hugging Face Hub cache: the CLIP text encoder (openai/clip-vit-base-patch32)
# and the ViT backbone (facebook/dinov2-base) are pulled with from_pretrained()
# at policy construction time and would otherwise go to ~/.cache/huggingface/hub.
export HF_HUB_CACHE=${HF_HUB_CACHE:-${CLARE_MODEL_ROOT}/hf_hub};
export TORCH_HOME=${TORCH_HOME:-${CLARE_MODEL_ROOT}/torch};

# --policy.path 로 넘어가는 사전학습 체크포인트.
#
# ★ PRETRAIN_REPO 는 **HF Hub 의 org/repo 이름**이지 로컬 경로가 아니다.
#   download_assets.sh 의 snapshot_download 에서만 쓰인다. 그래서 로컬에
#   continuallearning/ 폴더가 없는 것이 정상이다.
# ★ AICP 모델은 aicp_pretrain.py 가 만든 **로컬 산출물**이라 받을 것이 없다.
#   따라서 PRETRAIN_REPO 는 손대지 않고 PRETRAIN_PATH 만 직접 지정한다.
#   (밖에서 PRETRAIN_PATH=... 로 실행하면 그 값이 우선한다 — ${VAR:-기본값})
export PRETRAIN_REPO=${PRETRAIN_REPO:-continuallearning/dit_flow_mt_libero_90_pretrain};
export PRETRAIN_PATH=${PRETRAIN_PATH:-/home/sa090180/Models/aicp_clare_pretrain}; # base phase 산출물

# --- Training outputs --------------------------------------------------------
# Checkpoints written during continual learning. Kept inside the repo by default.
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-./outputs};

# --- W&B ---------------------------------------------------------------------
# Set this to your W&B entity, or leave it empty to fall back to your default one.
export WANDB_ENTITY=${WANDB_ENTITY:-sa090180};
if [ -n "${WANDB_ENTITY}" ]; then
    export WANDB_ENTITY_ARG="--wandb.entity=${WANDB_ENTITY}";
else
    export WANDB_ENTITY_ARG="";
fi

# CHECKPOINT_ROOT is left to the training script, which creates it relative to
# the working directory it was launched from.
mkdir -p "${HF_LEROBOT_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" "${CLARE_MODEL_ROOT}";
