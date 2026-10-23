# Central path configuration for CLARE.
#
# Every training script sources this file, so datasets and model weights are
# downloaded into the directories below instead of the default ~/.cache location.
# Override any of these by exporting them before launching a script.

# --- User-facing roots -------------------------------------------------------
export CLARE_DATA_ROOT=${CLARE_DATA_ROOT:-${HOME}/Datasets};
export CLARE_MODEL_ROOT=${CLARE_MODEL_ROOT:-${HOME}/Models};

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

# Pretrained VLA checkpoint, downloaded to PRETRAIN_PATH and passed as --policy.path.
# The repo the README names, dit_flow_mt_libero_90_pretrain_new, is not publicly
# readable (HF answers 401); dit_flow_mt_libero_90_pretrain is the published
# LIBERO-90 checkpoint of the same policy type (ditflow_mt).
export PRETRAIN_REPO=${PRETRAIN_REPO:-continuallearning/dit_flow_mt_libero_90_pretrain};
export PRETRAIN_PATH=${PRETRAIN_PATH:-${CLARE_MODEL_ROOT}/$(basename "${PRETRAIN_REPO}")};


# --- NVIDIA EGL (컨테이너에 그래픽 드라이버가 없을 때) -----------------------
# 일부 GPU 컨테이너는 NVIDIA_DRIVER_CAPABILITIES 가 compute,utility 뿐이라
# libEGL_nvidia.so 가 들어오지 않는다. 그러면 MuJoCo 오프스크린 렌더링이
# Mesa llvmpipe(CPU 소프트웨어 래스터라이저)로 조용히 떨어져 롤아웃이
# 7 배 가까이 느려진다 (실측 0.15ms -> 0.97ms per render, 1.0 -> 9.4 s/it).
#
# 커널 모듈과 **같은 버전**의 드라이버 .run 을 --extract-only 로 풀어
# 유저스페이스 .so 만 ~/nvegl/lib 에 두면 루트 없이 쓸 수 있다.
# dlopen 되는 libnvidia-gpucomp / libnvidia-nvvm / libnvidia-egl-gbm 까지
# 같이 있어야 한다 — ldd 로는 안 잡혀서 빠뜨리기 쉽다.
#
# 그 디렉토리가 있을 때만 적용되므로, 드라이버가 제대로 들어있는 서버에는
# 아무 영향이 없다.
if [ -d "${HOME}/nvegl/lib" ] && [ -f "${HOME}/nvegl/10_nvidia.json" ]; then
    export LD_LIBRARY_PATH="${HOME}/nvegl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}";
    export __EGL_VENDOR_LIBRARY_FILENAMES="${HOME}/nvegl/10_nvidia.json";
fi

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
