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
    echo "  Activate the project environment first:";
    echo "    venv  : source <venv>/bin/activate";
    echo "    conda : conda activate clare";
    exit 1;
}

echo "[download] model root   : ${CLARE_MODEL_ROOT}";
echo "[download] dataset root : ${HF_LEROBOT_HOME}";

# --- 1. Pretrained VLA checkpoint -> $PRETRAIN_PATH --------------------------
# ★ 존재 판정은 **model.safetensors** 로 한다. 전에 "폴더가 비지 않았으면 skip"
#   이었는데, 큰 파일 다운로드가 중간에 죽으면 config.json 만 남은 폴더를 보고
#   건너뛰어 버린다. 그러면 한참 뒤 학습이 체크포인트를 못 찾고 죽는다.
_pretrain_ok() { [ -s "${PRETRAIN_PATH}/model.safetensors" ]; }
_fetch_pretrain() {
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('${PRETRAIN_REPO}', local_dir='${PRETRAIN_PATH}')
";
}

if _pretrain_ok; then
    echo "[download] checkpoint already present at ${PRETRAIN_PATH}, skipping";
else
    if [ -d "${PRETRAIN_PATH}" ]; then
        echo "[download] ${PRETRAIN_PATH} 가 있지만 model.safetensors 가 없다 — 이어서 받는다";
    fi
    echo "[download] fetching ${PRETRAIN_REPO} -> ${PRETRAIN_PATH}";
    if ! _fetch_pretrain; then
        # Xet 전송(cas-server.xethub.hf.co)이 막힌 네트워크가 있다. 작은 파일은
        # 일반 HTTP 로 받아지고 LFS 대용량만 Xet 를 타므로 거기서만 멈춘다.
        # 끄면 예전 CDN 경로로 받는다. 이후 데이터셋 단계에도 계속 적용된다.
        echo "";
        echo "[download] 1차 실패 — Xet 전송을 끄고 일반 HTTP 로 재시도한다";
        export HF_HUB_DISABLE_XET=1;
        _fetch_pretrain || {
            echo "";
            echo "[download] FAILED to fetch ${PRETRAIN_REPO}. See the traceback above.";
            echo "  · ConnectionError (cas-server.xethub.hf.co): Xet 가 막힌 망이다.";
            echo "    HF_HUB_DISABLE_XET=1 을 미리 주고 다시 돌려라.";
            echo "  · 401: Hugging Face 는 비공개와 없는 repo 에 모두 401 을 준다.";
            echo "    접근 권한이 있으면 'hf auth login', 아니면 PRETRAIN_REPO 를 바꿔라.";
            exit 1;
        };
    fi
    _pretrain_ok || {
        echo "[download] model.safetensors 가 끝까지 안 받아졌다 — 다시 돌려라";
        exit 1;
    };
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
