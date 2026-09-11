#!/usr/bin/env bash
#
# GPU0의 clare.py가 끝나면 E1을 자동으로 시작한다 (일회성 예약 실행용).
#
#   nohup bash bash/E0/E1_watch.sh > /dev/null 2>&1 &
#
# 세션이 끊겨도 계속 돈다. 진행 상황은 LOG 파일로 본다.
#   tail -f outputs/E1/e1_watch_*.log
# 멈추려면:
#   pkill -f E1_watch.sh ; pkill -f scripts/E1.py
#
# 왜 기다리는가
#   E1은 GPU0을 쓴다. GPU1은 E0(λ=1000 잔여)와 R1이 이미 나눠 쓰고 있어 자리가 없다.
#   지금 GPU0에서는 clare.py가 libero_10 task_9(마지막 태스크)를 돌고 있다.
#
# 왜 E0를 기다리지 않는가
#   E1이 읽는 것은 lam100/task_2 의 체크포인트와 ewc_state.pt 두 개뿐이고,
#   그 스테이지는 이미 .done 상태다(Fisher 누적 수정판: cos(stage0,stage2)=0.18).
#   뒤에 돌 λ=1000은 E1과 무관하다. 그래서 CLARE만 기다린다.
#   다만 방어적으로 아래에서 .done과 파일 존재를 한 번 더 확인한다.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

LOG=${LOG:-./outputs/E1/e1_watch_$(date +%Y%m%d_%H%M%S).log}
mkdir -p "$(dirname "${LOG}")"

ARM=${ARM:-lam100}
STAGE=${STAGE:-2}
E0_ROOT=./outputs/E0/libero_spatial/seed_42
NEED_DONE=${E0_ROOT}/${ARM}/task_${STAGE}/.done
NEED_EWC=${E0_ROOT}/${ARM}/task_${STAGE}/ewc_state.pt

free_mb() {   # GPU0의 여유 메모리(MiB)
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0 2>/dev/null \
        | awk -F', ' '{print $1 - $2}'
}

{
    echo "[watch] $(date '+%F %T') clare.py 종료 대기 (60초 간격)"
    while pgrep -f "scripts/clare\.py" > /dev/null; do
        sleep 60
    done
    echo "[watch] $(date '+%F %T') clare.py 종료 확인"

    # GPU0이 실제로 비워질 때까지 잠깐 더 본다(프로세스 종료와 메모리 반환에 시차가 있다).
    for _ in $(seq 20); do
        [ "$(free_mb)" -ge 12000 ] 2>/dev/null && break
        echo "[watch] GPU0 여유 $(free_mb)MiB — 12GB 될 때까지 대기"
        sleep 30
    done
    echo "[watch] GPU0 여유 $(free_mb)MiB"

    # E1이 실제로 읽는 두 파일이 있는지 확인. 없으면 도는 의미가 없으므로 멈춘다.
    if [ ! -f "${NEED_DONE}" ] || [ ! -f "${NEED_EWC}" ]; then
        echo "[watch] ERROR ${ARM}/task_${STAGE} 가 준비되지 않았다"
        echo "        .done      : ${NEED_DONE}"
        echo "        ewc_state  : ${NEED_EWC}"
        echo "        E0를 먼저 끝내고 다시 띄워라."
        exit 1
    fi

    # Fisher 누적 버그판으로 만든 예전 E1 결과는 옆으로 치운다(질문 자체가 달랐다).
    OUT=./outputs/E1/libero_spatial/seed_42/${ARM}_stage${STAGE}
    if [ -d "${OUT}" ]; then
        BK="${OUT}_fisherbug_$(date +%Y%m%d_%H%M%S)"
        mv "${OUT}" "${BK}"
        echo "[watch] 예전 E1 결과 보존 -> ${BK}"
    fi

    echo "[watch] $(date '+%F %T') E1 시작 (GPU0, ${ARM} stage${STAGE})"
    CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
    PYTHON=/home/sa090180/miniconda3/envs/clare/bin/python \
    CL_ARM=${ARM} STAGE=${STAGE} \
        bash bash/E0/E1.sh
    echo "[watch] $(date '+%F %T') E1 종료 (exit=$?)"
} >> "${LOG}" 2>&1
