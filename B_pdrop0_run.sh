#!/usr/bin/env bash
#
# p_drop = 0 재실행 — condition dropout 을 완전히 끈 채 전 팔을 다시 잰다.
#
# 왜
#   results/B_default 에서 p_drop=0.1 이 무조건부 필드 v(·,∅) 를 **현재 태스크로**
#   고정한다는 것을 확인했다(스테이지 3에서 d_3<0.05 인 지점이 98.4%). 새 태스크가
#   올 때마다 기본 필드를 통째로 빼앗고, 과거 태스크는 매번 0.4 크기의 오프셋을
#   새 기준선 위에서 다시 세워야 한다. ER 은 이 현상이 없다(d 0.13~0.27).
#   p_drop 은 원래 classifier-free guidance 를 쓰려고 넣었는데, w 스윕에서 w>1 은
#   항상 손해였다(B2λ3: w=1.0 -> 80.0, w=1.25 -> 56.2). 즉 얻는 것이 없다.
#
# 무엇이 다른가
#   B_mod(results/mod) 와 --p_drop 0 하나만 다르다. anchor_agg=sum 은 동일.
#   출력은 results/mod0/, outputs/mod0/ 로만 나간다.
#
# 사용법
#   bash B_pdrop0_run.sh <GPU>              # 1,2,3
#   WAIT_PID=<pid> bash B_pdrop0_run.sh 3   # 그 pid 가 끝난 뒤 시작
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=${1:?GPU 번호}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p results/mod0 logs/mod0

if [ -n "${WAIT_PID:-}" ]; then
    echo "[$(date '+%F %T')] gpu ${GPU}: pid ${WAIT_PID} 종료 대기"
    while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
    echo "[$(date '+%F %T')] gpu ${GPU}: 대기 종료"
fi

# run <이름> <스크립트> <인자...>
#   인자 끝에 --p_drop 0 --anchor_agg sum --out_dir --ckpt_root 가 붙는다.
#   따라서 호출부는 그 넷이 B1 파서로 가도록 끝나 있어야 한다(팔은 --passthru 로 끝냄).
run() {
    local name=$1 script=$2; shift 2
    local rd="results/mod0/${name}" ck="outputs/mod0/${name}"
    if grep -q AvgSR_final "${rd}/metrics.json" 2>/dev/null; then
        echo "[skip] ${name} 이미 완료"; return 0
    fi
    echo "[$(date '+%F %T')] === ${name} 시작 (gpu ${GPU})"
    if python -u "${script}" "$@" \
            --p_drop 0 --anchor_agg sum --out_dir "${rd}" --ckpt_root "${ck}" \
            >"logs/mod0/${name}.log" 2>&1; then
        echo "[$(date '+%F %T')] === ${name} 완료"
    else
        echo "[$(date '+%F %T')] === ${name} 실패 -> logs/mod0/${name}.log"
    fi
}

case "${GPU}" in
  1) run B1_lam1  B1.py --lambda_anchor 1
     run B1_lam3  B1.py --lambda_anchor 3
     run B1_lam10 B1.py --lambda_anchor 10
     run B1_lam30 B1.py --lambda_anchor 30
     run B2_lam1  B2.py --passthru --lambda_anchor 1
     run B2_lam3  B2.py --passthru --lambda_anchor 3 ;;
  2) run B2_lam10 B2.py --passthru --lambda_anchor 10
     run B2_lam30 B2.py --passthru --lambda_anchor 30
     run B8_lam1  B8.py --passthru --lambda_anchor 1
     run B8_lam3  B8.py --passthru --lambda_anchor 3
     run B8_lam10 B8.py --passthru --lambda_anchor 10
     run B7       B7.py --passthru ;;
  3) run B9_1023 B9.py --order 1,0,2,3 --lambda_anchor 3 --passthru
     run B9_0321 B9.py --order 0,3,2,1 --lambda_anchor 3 --passthru
     run B9_2103 B9.py --order 2,1,0,3 --lambda_anchor 3 --passthru
     run B9_3210 B9.py --order 3,2,1,0 --lambda_anchor 3 --passthru ;;
  *) echo "GPU 는 1,2,3 (0번은 비워 둔다)"; exit 1 ;;
esac
echo "[$(date '+%F %T')] gpu ${GPU} p_drop0 큐 종료"
