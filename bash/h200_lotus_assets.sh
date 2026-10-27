#!/usr/bin/env bash
# LOTUS goal 자산(사전학습 체크포인트 + LIBERO-GOAL 데이터 + lotus_setting.json)을
# GitHub Release 에서 받아 $HOME 아래에 푼다 (egress 가 막힌 H200 용).
#   bash bash/h200_lotus_assets.sh
# 결과: $HOME/Models/Lotus_clare_libero_goal_pretrain/model.safetensors
#       $HOME/Datasets/lerobot/LOTUS_setting/{LIBERO-GOAL,lotus_setting.json}
set -euo pipefail
B=${B:-https://github.com/kimtaegyu7439/CLARE_baseline/releases/download/assets-lotus-goal}
W=${W:-$HOME/_lotus_assets}
if [ -s "$HOME/Models/Lotus_clare_libero_goal_pretrain/model.safetensors" ] \
   && [ -d "$HOME/Datasets/lerobot/LOTUS_setting/LIBERO-GOAL" ] && [ -z "${FORCE:-}" ]; then
    echo "이미 있다 — 건너뜀 (FORCE=1 로 다시)"; exit 0
fi
mkdir -p "$W" && cd "$W"
curl -fsSL -o SHA256SUMS_lotus_goal.txt "$B/SHA256SUMS_lotus_goal.txt"
for f in $(awk '{print $2}' SHA256SUMS_lotus_goal.txt); do
    echo "[받기] $f"; curl -fL -C - -o "$f" "$B/$f"
done
sha256sum -c SHA256SUMS_lotus_goal.txt
echo "[풀기] -> $HOME"; cat lotus_goal_assets.part* | tar xf - -C "$HOME"
[ -s "$HOME/Models/Lotus_clare_libero_goal_pretrain/model.safetensors" ] || { echo "✗ 체크포인트가 없다"; exit 1; }
ls "$HOME/Datasets/lerobot/LOTUS_setting/LIBERO-GOAL"
rm -f lotus_goal_assets.part*
echo "완료"
