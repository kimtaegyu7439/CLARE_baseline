#!/usr/bin/env bash
# H200 에서 LOTUS **object** 실험에 필요한 것을 갖춘다 (GitHub Release 로 받는다).
#   bash bash/h200_lotus_object_assets.sh
#
# goal 때(h200_lotus_assets.sh)와 다른 점: H200 에 libero_object_image_task_0..9 가 없어
# 분할을 현장에서 만들 수 없다. 그래서 서버1 에서 만든 **LOTUS 분할 자체**를 보낸다.
#   1) base 사전학습 체크포인트  $HOME/Models/Lotus_clare_libero_object_pretrain   (739M)
#   2) LOTUS 분할 데이터셋      $HOME/Datasets/lerobot/LOTUS_setting/LIBERO-OBJECT/{base,stage1..4}
#      (7.7G — release 자산 한도 2 GiB 때문에 1.9G 조각 tar.part00.. 으로 나눠 올렸다.
#       받아서 이어붙이지 않고 cat | tar 로 바로 푼다)
#   3) 매니페스트 lotus_setting.json 은 레포 configs/ 에 있다 (학습이 experiments 를 읽는다)
#
# 학습(Lotus_L3_V7dea / L2_V13 --setting lotus)은 위 셋만 읽는다. 원본 libero_object 는 필요 없다.
# 롤아웃은 LIBERO 시뮬레이터(goal 때 이미 깔린 것)를 쓴다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"
source bash/clare/env.sh                      # HF_LEROBOT_HOME, CLARE_MODEL_ROOT ($HOME 기준)
B=${B:-https://github.com/kimtaegyu7439/CLARE_baseline/releases/download/assets-lotus-object}
W=${W:-$HOME/_lotus_assets}
CK="${CLARE_MODEL_ROOT}/Lotus_clare_libero_object_pretrain"
LOTUS="${HF_LEROBOT_HOME}/LOTUS_setting"
SUMS="SHA256SUMS_lotus_object.txt"

mkdir -p "${W}" "${LOTUS}"
cd "${W}"
curl -fsSL -o "${SUMS}" "${B}/${SUMS}"
PARTS=$(awk '{print $2}' "${SUMS}" | grep '^LOTUS_LIBERO-OBJECT.tar.part' | sort)

echo "══ 1) base 체크포인트 ══"
if [ -s "${CK}/model.safetensors" ] && [ -z "${FORCE:-}" ]; then
    echo "  이미 있다: ${CK}  (FORCE=1 로 다시)"
else
    echo "  [받기] Lotus_clare_libero_object_pretrain.tar (739M)"
    curl -fL -C - -o Lotus_clare_libero_object_pretrain.tar "${B}/Lotus_clare_libero_object_pretrain.tar"
    grep ' Lotus_clare_libero_object_pretrain.tar$' "${SUMS}" | sha256sum -c -
    # tar 안 경로는 Models/Lotus_clare_libero_object_pretrain/... → CLARE_MODEL_ROOT 의 부모에 푼다
    tar xf Lotus_clare_libero_object_pretrain.tar -C "$(dirname "${CLARE_MODEL_ROOT}")"
    [ -s "${CK}/model.safetensors" ] || { echo "✗ 풀었는데 체크포인트가 없다: ${CK}"; exit 1; }
    rm -f Lotus_clare_libero_object_pretrain.tar
    echo "  완료: ${CK}"
fi

echo "══ 2) LOTUS 분할 데이터셋 (LIBERO-OBJECT) ══"
if [ -f "${LOTUS}/LIBERO-OBJECT/stage4/meta/info.json" ] && [ -f "${LOTUS}/LIBERO-OBJECT/base/meta/info.json" ] && [ -z "${FORCE:-}" ]; then
    echo "  이미 있다: ${LOTUS}/LIBERO-OBJECT  (FORCE=1 로 다시)"
else
    for p in ${PARTS}; do
        echo "  [받기] ${p}"
        curl -fL -C - -o "${p}" "${B}/${p}"        # -C - : 끊기면 이어받는다
    done
    grep ' LOTUS_LIBERO-OBJECT.tar.part' "${SUMS}" | sha256sum -c -
    # tar 안 경로는 LOTUS_setting/LIBERO-OBJECT/... → HF_LEROBOT_HOME 에 푼다
    cat ${PARTS} | tar xf - -C "${HF_LEROBOT_HOME}"
    [ -f "${LOTUS}/LIBERO-OBJECT/stage4/meta/info.json" ] || { echo "✗ 풀었는데 분할이 없다"; exit 1; }
    rm -f ${PARTS}
    echo "  완료: ${LOTUS}/LIBERO-OBJECT"
fi

echo "══ 3) 매니페스트 ══"
[ -f "${LOTUS}/lotus_setting.json" ] || cp "${HERE}/configs/lotus_setting.json" "${LOTUS}/lotus_setting.json"
cd "${HERE}"
du -sh "${CK}" "${LOTUS}/LIBERO-OBJECT"
for s in base stage1 stage2 stage3 stage4; do
    python3 -c "import json,sys;d=json.load(open('${LOTUS}/LIBERO-OBJECT/${s}/meta/info.json'));print('  ${s}', d['total_episodes'],'ep', d['total_frames'],'frame')"
done
echo "준비 끝 — 기대값: base 300 ep 44411 frame, stage1..4 각 10 ep"
