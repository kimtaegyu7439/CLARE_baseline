# Download the pretrained CLARE checkpoint and the LIBERO datasets into the
# directories configured in env.sh.
#
# Usage:
#   bash bash/clare/download_assets.sh                 # checkpoint + libero_spatial
#   bash bash/clare/download_assets.sh libero_10       # checkpoint + libero_10
#   bash bash/clare/download_assets.sh libero_goal libero_object
#
# Run from the repository root.

set -e;

source "$(dirname "${BASH_SOURCE[0]}")/env.sh";

SUITES=("$@");
if [ ${#SUITES[@]} -eq 0 ]; then
    SUITES=(libero_spatial);
fi

HF_ORG=continuallearning;
# PRETRAIN_REPO comes from env.sh; override it there or inline to use another checkpoint.

# Fail early and clearly if this is not the environment the project is installed in,
# rather than surfacing a ModuleNotFoundError from inside a download step.
python -c "import huggingface_hub, lerobot" 2>/dev/null || {
    echo "[download] '$(command -v python)' cannot import huggingface_hub and lerobot.";
    echo "  Activate the project environment first:  conda activate clare";
    exit 1;
}

echo "[download] model root   : ${CLARE_MODEL_ROOT}";
echo "[download] dataset root : ${HF_LEROBOT_HOME}";

# --- 1. Pretrained VLA checkpoint -> $PRETRAIN_PATH --------------------------
if [ -d "${PRETRAIN_PATH}" ] && [ -n "$(ls -A "${PRETRAIN_PATH}" 2>/dev/null)" ]; then
    echo "[download] checkpoint already present at ${PRETRAIN_PATH}, skipping";
else
    echo "[download] fetching ${PRETRAIN_REPO} -> ${PRETRAIN_PATH}";
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('${PRETRAIN_REPO}', local_dir='${PRETRAIN_PATH}')
" || {
        echo "";
        echo "[download] FAILED to fetch ${PRETRAIN_REPO}. See the traceback above.";
        echo "  On a 401: Hugging Face returns it both for private and for missing";
        echo "  repos. If you were granted access, run 'hf auth login' first, or";
        echo "  point PRETRAIN_REPO at another checkpoint.";
        exit 1;
    }
fi

# --- 2. LIBERO datasets -> $HF_LEROBOT_HOME/<repo_id> ------------------------
# Downloaded through LeRobotDataset so the on-disk layout matches what training
# expects (metadata + episode chunks under the repo_id directory).
for SUITE in "${SUITES[@]}"; do
    for TASK in 0 1 2 3 4 5 6 7 8 9; do
        REPO_ID="${HF_ORG}/${SUITE}_image_task_${TASK}";
        echo "[download] dataset ${REPO_ID}";
        python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
LeRobotDataset('${REPO_ID}')
";
    done
done

echo "[download] done";
echo "  datasets : ${HF_LEROBOT_HOME}/${HF_ORG}/";
echo "  model    : ${PRETRAIN_PATH}";
echo "  hf cache : ${HF_HUB_CACHE}";
