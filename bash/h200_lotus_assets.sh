#!/usr/bin/env bash
# H200 에서 LOTUS 실험에 필요한 것을 갖춘다 (egress 가 막힌 망이라 GitHub Release 로 받는다).
#   bash bash/h200_lotus_assets.sh                 # goal   (release assets-lotus-goal)
#   SUITE=object bash bash/h200_lotus_assets.sh    # object (release assets-lotus-object)
#
# goal   1) 사전학습 체크포인트 $HOME/Models/Lotus_clare_libero_goal_pretrain (739M)
#        2) LOTUS 분할 $HOME/Datasets/lerobot/LOTUS_setting/LIBERO-GOAL — 이미 있는
#           libero_goal_image_task_0..9 에서 lotus_build_setting.py 가 만든다 (4.9G)
# object 1) 사전학습 체크포인트 $HOME/Models/Lotus_clare_libero_object_pretrain (739M)
#        2) LOTUS 분할 tar (920M) 를 그대로 푼다: LIBERO-OBJECT/{base/meta, stage1..4}.
#           base 의 data 는 안 보낸다 — scope=lifelong + --skip_base_eval 이면 드라이버가
#           base 에서 meta 만 읽는다 (make_policy 의 ds_meta). 서버1 에서 로드 검증함.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"
source bash/clare/env.sh                      # HF_LEROBOT_HOME, CLARE_MODEL_ROOT ($HOME 기준)
SUITE=${SUITE:-goal}
B=${B:-https://github.com/kimtaegyu7439/CLARE_baseline/releases/download/assets-lotus-${SUITE}}
W=${W:-$HOME/_lotus_assets}
CK="${CLARE_MODEL_ROOT}/Lotus_clare_libero_${SUITE}_pretrain"
LOTUS="${HF_LEROBOT_HOME}/LOTUS_setting"
SU=$(echo "${SUITE}" | tr a-z A-Z)            # LIBERO-GOAL / LIBERO-OBJECT

echo "══ 1) base 체크포인트 (${SUITE}) ══"
if [ -s "${CK}/model.safetensors" ]; then
    echo "  이미 있다: ${CK}"
else
    mkdir -p "${W}" && cd "${W}"
    curl -fsSL -o "SHA256SUMS_lotus_${SUITE}_pretrain.txt" "${B}/SHA256SUMS_lotus_${SUITE}_pretrain.txt"
    echo "  [받기] Lotus_clare_libero_${SUITE}_pretrain.tar (739M)"
    curl -fL -C - -o "Lotus_clare_libero_${SUITE}_pretrain.tar" "${B}/Lotus_clare_libero_${SUITE}_pretrain.tar"
    sha256sum -c "SHA256SUMS_lotus_${SUITE}_pretrain.txt"
    # tar 안 경로는 Models/Lotus_clare_libero_<suite>_pretrain/... 이다 → CLARE_MODEL_ROOT 의 부모에 푼다
    tar xf "Lotus_clare_libero_${SUITE}_pretrain.tar" -C "$(dirname "${CLARE_MODEL_ROOT}")"
    [ -s "${CK}/model.safetensors" ] || { echo "✗ 풀었는데 체크포인트가 없다: ${CK}"; exit 1; }
    rm -f "Lotus_clare_libero_${SUITE}_pretrain.tar"
    cd "${HERE}"
    echo "  완료: ${CK}"
fi

echo "══ 2) LOTUS 분할 데이터셋 (${SU}) ══"
mkdir -p "${LOTUS}"
[ -f "${LOTUS}/lotus_setting.json" ] || cp configs/lotus_setting.json "${LOTUS}/lotus_setting.json"
if [ -f "${LOTUS}/${SU}/stage4/meta/info.json" ] && [ -f "${LOTUS}/${SU}/base/meta/info.json" ] && [ -z "${FORCE:-}" ]; then
    echo "  이미 있다: ${LOTUS}/${SU}  (FORCE=1 로 다시)"
elif [ "${SUITE}" = "object" ]; then
    mkdir -p "${W}" && cd "${W}"
    curl -fsSL -o SHA256SUMS_lotus_object_split.txt "${B}/SHA256SUMS_lotus_object_split.txt"
    echo "  [받기] lotus_object_split.tar (920M)"
    curl -fL -C - -o lotus_object_split.tar "${B}/lotus_object_split.tar"
    sha256sum -c SHA256SUMS_lotus_object_split.txt
    # tar 안 경로는 LOTUS_setting/LIBERO-OBJECT/... 이다 → HF_LEROBOT_HOME 에 푼다
    tar xf lotus_object_split.tar -C "${HF_LEROBOT_HOME}"
    [ -f "${LOTUS}/${SU}/stage4/meta/info.json" ] || { echo "✗ 풀었는데 분할이 없다: ${LOTUS}/${SU}"; exit 1; }
    rm -f lotus_object_split.tar
    cd "${HERE}"
    echo "  완료: ${LOTUS}/${SU}"
else
    miss=0
    for i in 0 1 2 3 4 5 6 7 8 9; do
        [ -f "${HF_LEROBOT_HOME}/continuallearning/libero_${SUITE}_image_task_${i}/meta/info.json" ] \
            || { echo "  ✗ 소스가 없다: ${HF_LEROBOT_HOME}/continuallearning/libero_${SUITE}_image_task_${i}"; miss=1; }
    done
    [ "${miss}" = 0 ] || { echo "✗ libero_${SUITE}_image_task_0..9 가 전부 있어야 한다"; exit 1; }
    python lotus_build_setting.py --only "${SU}"
    [ -f "${LOTUS}/${SU}/stage4/meta/info.json" ] || { echo "✗ 분할 생성 실패"; exit 1; }
    echo "  완료: ${LOTUS}/${SU}"
fi
du -sh "${CK}" "${LOTUS}/${SU}"
echo "준비 끝"
