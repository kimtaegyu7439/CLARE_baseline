#!/usr/bin/env bash
#
# E0가 끝나면 R1을 자동으로 시작한다 (일회성 예약 실행용).
#
#   nohup bash bash/E0/R1_watch.sh > /dev/null 2>&1 &
#
# 세션이 끊겨도 계속 돈다. 진행 상황은 LOG 파일로 본다.
#   tail -f outputs/R1/r1_*.log
# 멈추려면:
#   pkill -f R1_watch.sh ; pkill -f scripts/R1.py
#
# 왜 기다리는가: 지금 도는 E0가 lam100 task_3다. 그게 끝나야 EWC 팔이 4/4가 되어
# 그림 A의 stage4 곡선과 그림 B의 ewc@3 열이 채워진다.
#
# ★ 기다리는 대상은 "E0.sh 전체"가 아니라 "lam100 task_3의 .done"이다.
#   E0.sh는 LAMBDAS="10 100 1000"으로 떠 있어 lam100이 끝나면 λ=1000 4스테이지를
#   이어서 돈다(스테이지당 ~45분, 약 3시간). 그걸 다 기다리면 R1이 3시간 늦게 시작하는데,
#   R1이 필요로 하는 체크포인트는 lam100 task_3가 마지막이다. GPU1은 24GB 중
#   E0가 5.7GB만 쓰므로 R1(약 7GB)과 같이 올라간다 — 서로 조금 느려지는 대신
#   전체는 더 일찍 끝난다.
#   WAIT_FULL_E0=1 로 두면 예전처럼 E0.sh가 완전히 끝날 때까지 기다린다(GPU 독점).

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

LOG=${LOG:-./outputs/R1/r1_$(date +%Y%m%d_%H%M%S).log}
mkdir -p "$(dirname "${LOG}")"

E0_DONE=./outputs/E0/libero_spatial/seed_42/lam100/task_3/.done

free_mb() {   # GPU1의 여유 메모리(MiB)
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 1 2>/dev/null \
        | awk -F', ' '{print $1 - $2}'
}

{
    if [ "${WAIT_FULL_E0:-0}" = "1" ]; then
        echo "[watch] $(date '+%F %T') E0.sh 전체 종료 대기 (60초 간격)"
        # 패턴을 'bash/E0/E0\.sh'로 좁힌 이유: 그냥 'E0.sh'로 두면 이 파일 경로가
        # 자기 자신에 걸려 영원히 기다리게 되는 사고가 난다.
        while pgrep -f "scripts/E0\.py" > /dev/null || pgrep -f "bash/E0/E0\.sh" > /dev/null; do
            sleep 60
        done
        echo "[watch] $(date '+%F %T') E0.sh 종료 확인"
    else
        echo "[watch] $(date '+%F %T') lam100 task_3의 .done 대기 (60초 간격)"
        # E0.py가 아예 죽어 버렸는데 .done도 없으면 영원히 기다리게 된다. 그 경우
        # (프로세스도 없고 .done도 없음)는 실패로 보고 있는 스테이지로 진행한다.
        while [ ! -f "${E0_DONE}" ]; do
            if ! pgrep -f "scripts/E0\.py" > /dev/null && ! pgrep -f "bash/E0/E0\.sh" > /dev/null; then
                echo "[watch] E0가 사라졌는데 .done이 없다 — 실패로 보고 진행한다"
                break
            fi
            sleep 60
        done
        # GPU1에 R1이 올라갈 자리(약 7GB)가 날 때까지 잠깐 더 본다.
        for _ in $(seq 30); do
            [ "$(free_mb)" -ge 10000 ] 2>/dev/null && break
            echo "[watch] GPU1 여유 $(free_mb)MiB — 10GB 될 때까지 대기"
            sleep 60
        done
    fi

    if [ -f "${E0_DONE}" ]; then
        echo "[watch] lam100 task_3 완료 -> EWC 팔 4/4"
    else
        # .done이 없으면 E0가 중간에 죽었다는 뜻. 그래도 R1은 있는 스테이지로 돈다
        # (R1.sh가 "스테이지 3/4개만 있다"고 경고하고 넘어간다).
        echo "[watch] WARN ${E0_DONE} 없음 — E0가 실패했을 수 있다. 있는 스테이지로만 진행한다"
    fi
    echo "[watch] GPU1 여유 $(free_mb)MiB"

    echo "[watch] $(date '+%F %T') R1 시작"
    CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
    PYTHON=/home/sa090180/miniconda3/envs/clare/bin/python \
    EXTRA_ARMS="frozen=./outputs/E0/libero_spatial/seed_42/laminf" \
    PROBE_TASK=0 NUM_ROLLOUTS=30 NUM_STAGES=4 MAX_STEPS=300 K=4 \
        bash bash/E0/R1.sh
    echo "[watch] $(date '+%F %T') R1 종료 (exit=$?)"
} >> "${LOG}" 2>&1
