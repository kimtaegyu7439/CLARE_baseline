#!/usr/bin/env bash
#
# 앵커 집계를 "스텝당 과거 하나 추첨" -> "과거 전부의 합" 으로 바꾼 뒤 전 팔 재실행.
#
#   구버전: j = rng.randrange(k)  -> 배치 32개 전부에 같은 ℓ_j. 스텝마다 신호가 80배씩 튐.
#   신버전: j = 0..k-1 전부의 합.  실효 가중치가 k·λ 가 되므로 λ 를 다시 훑는다.
#
# 바뀐 것은 --anchor_agg sum 뿐이고 나머지 하이퍼파라미터는 전부 동일하다.
# 기존 results/ outputs/ 는 건드리지 않는다 -> results/mod/, outputs/mod/ 에만 쓴다.
#
# 사용법:  bash B_mod_run.sh <GPU>     # 0,1,2,3 이 각자 자기 큐를 순서대로 돈다
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=${1:?GPU 번호}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p results/mod logs/mod

# run <이름> <스크립트> <인자...>
#   인자 끝에 --anchor_agg/--out_dir/--ckpt_root 가 덧붙는다. 따라서 호출부는
#   그 셋이 **B1 의 파서로 가도록** 끝나 있어야 한다(팔 스크립트는 --passthru 로 끝낸다).
run() {
    local name=$1 script=$2; shift 2
    local rd="results/mod/${name}" ck="outputs/mod/${name}"
    if grep -q AvgSR_final "${rd}/metrics.json" 2>/dev/null; then
        echo "[skip] ${name} 이미 완료"; return 0
    fi
    echo "[$(date '+%F %T')] === ${name} 시작 (gpu ${GPU})"
    if python -u "${script}" "$@" \
            --anchor_agg sum --out_dir "${rd}" --ckpt_root "${ck}" \
            >"logs/mod/${name}.log" 2>&1; then
        echo "[$(date '+%F %T')] === ${name} 완료"
    else
        echo "[$(date '+%F %T')] === ${name} 실패 -> logs/mod/${name}.log"
    fi
}

case "${GPU}" in
  0) run B1_lam1  B1.py --lambda_anchor 1
     run B1_lam3  B1.py --lambda_anchor 3
     run B1_lam10 B1.py --lambda_anchor 10
     run B1_lam30 B1.py --lambda_anchor 30 ;;
  1) run B2_lam1  B2.py --passthru --lambda_anchor 1
     run B2_lam3  B2.py --passthru --lambda_anchor 3
     run B2_lam10 B2.py --passthru --lambda_anchor 10
     run B2_lam30 B2.py --passthru --lambda_anchor 30 ;;
  2) run B8_lam1  B8.py --passthru --lambda_anchor 1
     run B8_lam3  B8.py --passthru --lambda_anchor 3
     run B8_lam10 B8.py --passthru --lambda_anchor 10
     run B7       B7.py --passthru ;;
  3) run B9_1023 B9.py --order 1,0,2,3 --lambda_anchor 3 --passthru
     run B9_0321 B9.py --order 0,3,2,1 --lambda_anchor 3 --passthru
     run B9_2103 B9.py --order 2,1,0,3 --lambda_anchor 3 --passthru
     run B9_3210 B9.py --order 3,2,1,0 --lambda_anchor 3 --passthru ;;
  *) echo "GPU 는 0..3"; exit 1 ;;
esac
echo "[$(date '+%F %T')] gpu ${GPU} 큐 종료"
