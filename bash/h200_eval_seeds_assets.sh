#!/usr/bin/env bash
# H200 에 평가 시드 스윕용 LOTUS 런 체크포인트(stage1..4 pretrained_model 만)를 받는다.
#   bash bash/h200_eval_seeds_assets.sh            # 기본 세 런 (λ0.7 β2 object seed 7/42/1000)
#   RUNS="lotus_v13_..._seed7" bash bash/h200_eval_seeds_assets.sh
# 릴리스 assets-lotus-evalseeds 의 <run>_stage<k>.tar 를 results/ 에 푼다 (run/config.json + checkpoints/stageK/.../pretrained_model).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"
REPO=${REPO:-kimtaegyu7439/CLARE_baseline}; TAG=${TAG:-assets-lotus-evalseeds}
RUNS=${RUNS:-"lotus_v13_b2.0_lam0.7_k25_scopeLL_object_seed7 lotus_v13_b2.0_lam0.7_k25_scopeLL_object_seed42 lotus_v13_b2.0_lam0.7_k25_scopeLL_object_seed1000"}
BASE="https://github.com/${REPO}/releases/download/${TAG}"
DL=${DL:-"${HOME}/_dl_evalseeds"}; mkdir -p "${DL}" results
curl -fsSL --retry 5 -o "${DL}/SHA256SUMS_evalseeds.txt" "${BASE}/SHA256SUMS_evalseeds.txt"
for R in ${RUNS}; do for K in 1 2 3 4; do
    F="${R}_stage${K}.tar"
    if [ -s "results/${R}/checkpoints/stage${K}/checkpoints/last/pretrained_model/model.safetensors" ]; then
        echo "⏭ ${R} stage${K} 이미 있음"; continue; fi
    echo "▼ ${F}"
    curl -fL --retry 5 -C - -o "${DL}/${F}" "${BASE}/${F}"
    ( cd "${DL}" && grep " ${F}\$" SHA256SUMS_evalseeds.txt | sha256sum -c - ) || { echo "✗ 체크섬 불일치: ${F}"; exit 1; }
    tar -xf "${DL}/${F}" -C results && rm -f "${DL}/${F}"
done; done
echo "완료 — results/ 아래 런 폴더:"; ls -d results/lotus_v13_b2.0_lam0.7_k25_scopeLL_object_seed*
