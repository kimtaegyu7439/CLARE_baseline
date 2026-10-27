#!/usr/bin/env bash
# H200 에서 LOTUS goal 실험에 필요한 것을 갖춘다 (egress 가 막힌 망이라 GitHub Release 로 받는다).
#   bash bash/h200_lotus_assets.sh
#
# 1) base 사전학습 체크포인트  $HOME/Models/Lotus_clare_libero_goal_pretrain  (release 에서 739M)
#    — base phase 6 태스크를 다시 학습하지 않으려고 서버1 에서 학습한 것을 그대로 가져온다.
# 2) LOTUS 분할 데이터셋  $HOME/Datasets/lerobot/LOTUS_setting/LIBERO-GOAL/{base,stage1..4}
#    — 별도 다운로드가 아니다. 이미 있는 libero_goal_image_task_0..9 에서
#      lotus_build_setting.py 가 에피소드를 복사해 만든다 (몇 분, 4.9G).
#      매니페스트 lotus_setting.json 은 레포의 configs/ 에 있다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"
source bash/clare/env.sh                      # HF_LEROBOT_HOME, CLARE_MODEL_ROOT ($HOME 기준)
B=${B:-https://github.com/kimtaegyu7439/CLARE_baseline/releases/download/assets-lotus-goal}
W=${W:-$HOME/_lotus_assets}
CK="${CLARE_MODEL_ROOT}/Lotus_clare_libero_goal_pretrain"
LOTUS="${HF_LEROBOT_HOME}/LOTUS_setting"

echo "══ 1) base 체크포인트 ══"
if [ -s "${CK}/model.safetensors" ]; then
    echo "  이미 있다: ${CK}"
else
    mkdir -p "${W}" && cd "${W}"
    curl -fsSL -o SHA256SUMS_lotus_goal_pretrain.txt "${B}/SHA256SUMS_lotus_goal_pretrain.txt"
    echo "  [받기] Lotus_clare_libero_goal_pretrain.tar (739M)"
    curl -fL -C - -o Lotus_clare_libero_goal_pretrain.tar "${B}/Lotus_clare_libero_goal_pretrain.tar"
    sha256sum -c SHA256SUMS_lotus_goal_pretrain.txt
    # tar 안 경로는 Models/Lotus_clare_libero_goal_pretrain/... 이다 → CLARE_MODEL_ROOT 의 부모에 푼다
    tar xf Lotus_clare_libero_goal_pretrain.tar -C "$(dirname "${CLARE_MODEL_ROOT}")"
    [ -s "${CK}/model.safetensors" ] || { echo "✗ 풀었는데 체크포인트가 없다: ${CK}"; exit 1; }
    rm -f Lotus_clare_libero_goal_pretrain.tar
    cd "${HERE}"
    echo "  완료: ${CK}"
fi

echo "══ 2) LOTUS 분할 데이터셋 ══"
if [ -f "${LOTUS}/LIBERO-GOAL/stage4/meta/info.json" ] && [ -f "${LOTUS}/LIBERO-GOAL/base/meta/info.json" ] && [ -z "${FORCE:-}" ]; then
    echo "  이미 있다: ${LOTUS}/LIBERO-GOAL  (FORCE=1 로 다시)"
else
    miss=0
    for i in 0 1 2 3 4 5 6 7 8 9; do
        [ -f "${HF_LEROBOT_HOME}/continuallearning/libero_goal_image_task_${i}/meta/info.json" ] \
            || { echo "  ✗ 소스가 없다: ${HF_LEROBOT_HOME}/continuallearning/libero_goal_image_task_${i}"; miss=1; }
    done
    [ "${miss}" = 0 ] || { echo "✗ libero_goal_image_task_0..9 가 전부 있어야 한다"; exit 1; }
    mkdir -p "${LOTUS}"
    [ -f "${LOTUS}/lotus_setting.json" ] || cp configs/lotus_setting.json "${LOTUS}/lotus_setting.json"
    python lotus_build_setting.py --only LIBERO-GOAL
    [ -f "${LOTUS}/LIBERO-GOAL/stage4/meta/info.json" ] || { echo "✗ 분할 생성 실패"; exit 1; }
    echo "  완료: ${LOTUS}/LIBERO-GOAL"
fi
du -sh "${CK}" "${LOTUS}/LIBERO-GOAL"
echo "준비 끝"
