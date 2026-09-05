#!/usr/bin/env bash
#
# 코드북 셀 안의 state 분포가 가우시안인지 확인한다.
#
#   bash L2_state_base_gaussian_check.sh                     기본: libero_spatial task 0, K=96
#   bash L2_state_base_gaussian_check.sh libero_spatial "0 1 2"
#   K=48 N_PAIRS=16000 bash L2_state_base_gaussian_check.sh
#   CELL=7 NO_TSNE=1 bash L2_state_base_gaussian_check.sh     특정 셀만, t-SNE 생략(빠름)
#   PICK=worst bash L2_state_base_gaussian_check.sh            예시 셀을 최악의 셀로
#   GRID=9 GRID_SEED=3 bash L2_state_base_gaussian_check.sh    셀 격자를 9개, 다른 추첨으로
#   EMBED=umap bash L2_state_base_gaussian_check.sh            왼쪽 패널을 UMAP 으로
#                                                              (umap-learn 없으면 t-SNE 로 떨어짐)
#
# 예시 셀은 기본이 PICK=median (KS 중앙값 셀) 이다. largest 로 고르면 KS 가 표본 수와
# 음의 상관이라 "가장 가우시안해 보이는 셀"을 고르게 되므로 기본값으로 쓰지 않는다.
# Q1/Q2 결론은 예시 셀이 아니라 전 셀 통계(패널 C/D/F)가 진다.
#
# state 만 읽으므로 정책(DINOv2)을 올리지 않고 GPU 도 쓰지 않는다.
# 결과: results/${NAME}/state_gauss_task<t>_k<K>.{png,txt,json}
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"

SUITE=${1:-libero_spatial}; TASKS=${2:-0}
NAME=${NAME:-L2_state_gauss}
K=${K:-96}; N_PAIRS=${N_PAIRS:-8000}; MIN_N=${MIN_N:-5}
CELL=${CELL:--1}; PICK=${PICK:-median}; SEED=${SEED:-0}
GRID=${GRID:-6}; GRID_SEED=${GRID_SEED:--1}; PLANE=${PLANE:-pca}
PERPLEXITY=${PERPLEXITY:-40}; TSNE_MAX=${TSNE_MAX:-4000}; EMBED=${EMBED:-tsne}
CKPT=${CKPT:-outputs/B2_lam3/libero_spatial_seed42_ours/task_0/checkpoints/005000/pretrained_model}

export CUDA_VISIBLE_DEVICES=""                       # GPU 를 잡지 않는다 (학습 잡과 안 겹치게)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
EXTRA=()
[ "${NO_TSNE:-0}" = "1" ] && EXTRA+=(--no_tsne)

OUT="results/${NAME}"; mkdir -p "${OUT}" logs
for T in ${TASKS}; do
    LOG="logs/${NAME}_task${T}_k${K}.log"
    echo "[$(date '+%F %T')] ▶ ${SUITE} task ${T}  K=${K}  n_pairs=${N_PAIRS}  -> ${OUT}"
    python -u L2_state_base_gaussian_check.py \
        --suite "${SUITE}" --task "${T}" --codebook_k "${K}" --n_pairs "${N_PAIRS}" \
        --min_n "${MIN_N}" --cell "${CELL}" --pick "${PICK}" --seed "${SEED}" \
        --grid "${GRID}" --grid_seed "${GRID_SEED}" --plane "${PLANE}" \
        --perplexity "${PERPLEXITY}" --tsne_max "${TSNE_MAX}" --embed "${EMBED}" \
        --ckpt "${CKPT}" --out "${OUT}" "${EXTRA[@]}" 2>&1 | tee "${LOG}"
    echo "[$(date '+%F %T')] ◀ task ${T} 종료 (로그 ${LOG})"
done
echo "[$(date '+%F %T')] 완료 -> ${OUT}"
