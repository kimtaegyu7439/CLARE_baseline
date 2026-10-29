#!/usr/bin/env bash
# H200 에서 LOTUS **object** 실험에 필요한 것을 갖춘다 — bash/h200_lotus_assets.sh 의 SUITE=object 래퍼.
#   bash bash/h200_lotus_object_assets.sh
#
# release assets-lotus-object 에 올라간 것 (서버1 에서 로드 검증함):
#   Lotus_clare_libero_object_pretrain.tar  (739M)  → $HOME/Models/Lotus_clare_libero_object_pretrain
#   lotus_object_split.tar                  (920M)  → $HOME/Datasets/lerobot/LOTUS_setting/LIBERO-OBJECT/{base/meta, stage1..4}
# base 의 data(6.8G)는 보내지 않는다 — scope=lifelong + --skip_base_eval 이면 드라이버가 base 에서
# meta 만 읽는다(make_policy 의 ds_meta). 매니페스트 lotus_setting.json 은 레포 configs/ 에서 복사한다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE=object exec bash "${HERE}/bash/h200_lotus_assets.sh"
