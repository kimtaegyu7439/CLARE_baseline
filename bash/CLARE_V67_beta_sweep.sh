#!/usr/bin/env bash
# L2_V6 / L2_V7 — dea 앵커 β 스윕. seed 42 / 1000. GPU 0 에서 2 프로세스 병행.
#
#   bash bash/CLARE_V67_beta_sweep.sh                 # 기본: v6,v7 x seed 42,1000 x β 6개 = 24런
#   DRY=1 bash bash/CLARE_V67_beta_sweep.sh           # 실행 계획 + 병렬 예산만 출력
#   MAXJOBS=3 bash bash/CLARE_V67_beta_sweep.sh       # 동시 3개
#   MAXJOBS=4 NW=4 bash bash/CLARE_V67_beta_sweep.sh  # 동시 4개, 런당 워커 4개
#   VERS=v7 SEEDS=42 bash bash/CLARE_V67_beta_sweep.sh
#   BETAS="2.0 2.5" bash bash/CLARE_V67_beta_sweep.sh
#   nohup bash bash/CLARE_V67_beta_sweep.sh > logs/v67_sweep.out 2>&1 &
#
# ── 병렬 관련 변수 (셋 다 환경변수로 조절) ──────────────────────────────
#   MAXJOBS      GPU 하나에 동시에 띄우는 **런 개수**. 이게 "병렬"의 주 손잡이다.
#                기본 2 — H200 1장 기준. 런당 VRAM 약 6G 라 2개면 12G 다.
#                참고로 이 코드를 만든 4090 서버는 GPU당 1개씩 4 GPU 에 분산했다.
#   NW           런 하나의 데이터로더 워커 프로세스 수 -> --num_workers. 기본 6.
#   OMP_THREADS  런 하나의 CPU 스레드 수 -> OMP_NUM_THREADS. 기본 4.
#   MKL_THREADS  같은 용도 -> MKL_NUM_THREADS. 기본 4.
#
#   CPU 총 수요 = MAXJOBS x (NW + OMP_THREADS). 논리코어보다 크면 경고만 내고
#   그대로 진행한다(막지는 않는다 — 워커는 I/O 대기가 많아 과할당이 보통 이득이다).
#
# ★ batch_size 는 일부러 노출하지 않는다. 앵커가 과거 태스크당 n=B 개를 뽑으므로
#   (L2_V6.py:341, L2_V4_seq.py:1261) B 를 바꾸면 앵커 추정량의 분산이 바뀐다 —
#   병렬 설정이 아니라 실험 조건이다. 32 로 고정해 seed7 런과 맞춘다.
#
# V6 = 속도 공간 분해(--dir_space vel), V7 = 액션 공간 분해(L2_V7.py 가 act 로 고정).
# 수식은 같고 분해 공간만 다르다. β 의 적정 스케일이 공간마다 다르므로(V7 주석의
# 2r₀²: 속도 12.4 vs 액션 1.35) 같은 β 격자를 두 공간에 걸어 비교한다.
#
# 각 런은 CLARE_V6_queue.sh 가 담당한다 — 학습(롤아웃 없이) -> 롤아웃 50회 -> 리포트.
# 이 스크립트는 조합을 만들어 슬롯에 넣고, 이미 끝난 런은 건너뛰는 일만 한다.
#
# ★ 재실행 안전. 완주한 런(55/55칸)은 건너뛰고, 중간에 끊긴 런은 queue 스크립트의
#   --resume 이 이어받는다. 12일짜리 스윕이라 끊겨도 같은 명령으로 다시 띄우면 된다.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${HERE}"

GPU=${GPU:-0}
FORM=${FORM:-dea}
# 바깥 -> 안쪽 순서. ver, seed 를 고정한 채 β 를 다 돌려 곡선 하나를 먼저 완성한다.
VERS=${VERS:-"v6 v7"}
SEEDS=${SEEDS:-"42 1000"}
BETAS=${BETAS:-"0.5 1.0 1.5 2.0 2.5 3.0"}
# ── 병렬 손잡이. 따로 안 주면 지금 서버에서 돌린 값 그대로다 ──────────────
MAXJOBS=${MAXJOBS:-2}            # GPU 하나에 동시에 띄우는 런 개수
NW=${NW:-6}                      # 런당 데이터로더 워커 -> --num_workers (현 서버값)
OMP_THREADS=${OMP_THREADS:-4}    # 런당 OMP_NUM_THREADS (현 서버값)
MKL_THREADS=${MKL_THREADS:-4}    # 런당 MKL_NUM_THREADS (현 서버값)

EPS_W=${EPS_W:-1e-3}
REDUCE=${REDUCE:-mean}
NTASK=${NTASK:-10}
N_EVAL=${N_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-7}
# 런당 체크포인트가 11G 다. 남은 용량이 이 밑으로 내려가면 새 런을 띄우지 않는다.
MIN_FREE_GB=${MIN_FREE_GB:-40}
# 런 사이 GPU 메모리 반납 여유
GAP=${GAP:-30}

Q="${HERE}/logs/CLARE_V67_sweep.queue"
mkdir -p logs results
ts() { echo "[$(date '+%F %T')] $*" | tee -a "${Q}"; }

[ -x "$(command -v python3)" ] || { echo "python3 없음"; exit 1; }
[ -f "${HERE}/CLARE_V6_queue.sh" ] || { echo "CLARE_V6_queue.sh 없음"; exit 1; }
for v in ${VERS}; do
    case "${v}" in
        v6) [ -f "${HERE}/L2_V6.py" ] || { echo "L2_V6.py 없음"; exit 1; } ;;
        v7) [ -f "${HERE}/L2_V7.py" ] || { echo "L2_V7.py 없음"; exit 1; } ;;
        *)  echo "VERS 는 v6 / v7 만 된다: ${v}"; exit 1 ;;
    esac
done

entry_of() { [ "$1" = "v7" ] && echo L2_V7.py || echo L2_V6.py; }

# 완주 판정 — queue 스크립트와 같은 기준(sr_matrix.csv 의 채운 칸 수)
need_cells() { echo $(( NTASK * (NTASK + 1) / 2 )); }
filled_cells() {
    python3 - "$1" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "sr_matrix.csv"
if not p.exists():
    print(0); raise SystemExit
b = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
print(sum(1 for l in b[1:] for c in l.split(",")[1:] if c.strip()))
PY
}

free_gb() { df -BG --output=avail "${HERE}" | tail -1 | tr -dc '0-9'; }

# ── 병렬 예산 — 논리코어 대비 CPU 수요를 알려준다 (막지는 않는다) ──────────
NCPU=$(nproc 2>/dev/null || echo 0)
CPU_PER_RUN=$(( NW + OMP_THREADS ))
CPU_WANT=$(( MAXJOBS * CPU_PER_RUN ))
par_summary() {
    echo "병렬: MAXJOBS=${MAXJOBS} x (NW=${NW} + OMP_THREADS=${OMP_THREADS}) = CPU ${CPU_WANT} / 논리코어 ${NCPU}"
    echo "      MKL_THREADS=${MKL_THREADS}   VRAM 예상 약 $(( MAXJOBS * 6 ))G (런당 6G)"
}
cpu_warn() {
    [ "${NCPU}" -gt 0 ] && [ "${CPU_WANT}" -gt "${NCPU}" ] && \
        ts "⚠ CPU 수요 ${CPU_WANT} > 논리코어 ${NCPU} — 과할당이다. 느리면 NW 를 줄여라 (예: NW=$(( NCPU / MAXJOBS - OMP_THREADS > 1 ? NCPU / MAXJOBS - OMP_THREADS : 1 )))"
    return 0
}

# ── 계획 수집 ───────────────────────────────────────────────────────────
PLAN=()
for VER in ${VERS}; do
  for SEED in ${SEEDS}; do
    for BETA in ${BETAS}; do
      PLAN+=("${VER}|${SEED}|${BETA}")
    done
  done
done
NEED=$(need_cells)

if [ -n "${DRY:-}" ]; then
    echo "GPU ${GPU}  form=${FORM}  칸수 ${NEED}"
    par_summary
    echo "총 ${#PLAN[@]} 런 — 런당 약 24h / 11G  =>  약 $(( ${#PLAN[@]} * 24 / MAXJOBS ))h, 약 $(( ${#PLAN[@]} * 11 ))G"
    echo "현재 여유 $(free_gb)G (하한 ${MIN_FREE_GB}G)"
    [ "${NCPU}" -gt 0 ] && [ "${CPU_WANT}" -gt "${NCPU}" ] && \
        echo "⚠ CPU 수요 ${CPU_WANT} > 논리코어 ${NCPU} — 과할당"
    printf '%s\n' "${PLAN[@]}" | while IFS='|' read -r VER SEED BETA; do
        TAG="clare_${VER}_${FORM}_b${BETA}_goal_seed${SEED}"
        f=$(filled_cells "results/${TAG}")
        [ "${f}" -ge "${NEED}" ] && st="건너뜀 (완주 ${f}/${NEED})" || st="실행 (${f}/${NEED})"
        printf '  %-40s %-14s entry=%s\n' "${TAG}" "${st}" "$(entry_of "${VER}")"
    done
    exit 0
fi

ts "════ V6/V7 β 스윕 시작 — GPU ${GPU}, 동시 ${MAXJOBS}개, 총 ${#PLAN[@]} 런 ════"
ts "    VERS='${VERS}'  SEEDS='${SEEDS}'  BETAS='${BETAS}'  form=${FORM}"
ts "    $(par_summary | head -1)"
ts "    $(par_summary | tail -1)"
cpu_warn

running=0
launched=0; skipped=0
for item in "${PLAN[@]}"; do
    IFS='|' read -r VER SEED BETA <<< "${item}"
    TAG="clare_${VER}_${FORM}_b${BETA}_goal_seed${SEED}"

    f=$(filled_cells "results/${TAG}")
    if [ "${f}" -ge "${NEED}" ]; then
        ts "⏭ ${TAG} — 이미 완주 (${f}/${NEED})"
        skipped=$((skipped+1))
        continue
    fi

    # 슬롯이 빌 때까지 대기
    while [ "${running}" -ge "${MAXJOBS}" ]; do
        wait -n 2>/dev/null || true
        running=$((running-1))
    done

    # 디스크 하한 — 11G 런을 띄우기 전에 확인
    FREE=$(free_gb)
    while [ "${FREE}" -lt "${MIN_FREE_GB}" ]; do
        if [ "${running}" -gt 0 ]; then
            ts "⏸ 여유 ${FREE}G < ${MIN_FREE_GB}G — 진행 중인 ${running}개가 끝나기를 기다린다"
            wait -n 2>/dev/null || true
            running=$((running-1))
            FREE=$(free_gb)
        else
            ts "✗ 여유 ${FREE}G < ${MIN_FREE_GB}G 이고 진행 중인 런이 없다 — 중단"
            ts "   용량을 확보한 뒤 같은 명령으로 다시 띄우면 남은 런만 돈다"
            exit 1
        fi
    done

    ENTRY="$(entry_of "${VER}")"
    ts "▶ ${TAG}  (entry=${ENTRY} β=${BETA} seed=${SEED}, 여유 ${FREE}G)"
    (
        GPU="${GPU}" FORM="${FORM}" SEED="${SEED}" BETA="${BETA}" \
        ENTRY="${ENTRY}" VER="${VER}" TAGSUF="_b${BETA}" \
        EPS_W="${EPS_W}" REDUCE="${REDUCE}" \
        NTASK="${NTASK}" N_EVAL="${N_EVAL}" EVAL_SEED="${EVAL_SEED}" \
        MIN_FREE_GB="${MIN_FREE_GB}" \
        NW="${NW}" OMP_NUM_THREADS="${OMP_THREADS}" MKL_NUM_THREADS="${MKL_THREADS}" \
        bash "${HERE}/CLARE_V6_queue.sh"
        rc=$?
        ts "◀ ${TAG} exit=${rc}  ($(filled_cells "results/${TAG}")/${NEED}칸)"
    ) &
    running=$((running+1))
    launched=$((launched+1))
    sleep "${GAP}"            # 두 프로세스가 동시에 데이터로더를 깔지 않게 살짝 띄운다
done

while [ "${running}" -gt 0 ]; do
    wait -n 2>/dev/null || true
    running=$((running-1))
done

ts "════ 스윕 종료 — 띄움 ${launched}, 건너뜀 ${skipped} ════"
for item in "${PLAN[@]}"; do
    IFS='|' read -r VER SEED BETA <<< "${item}"
    TAG="clare_${VER}_${FORM}_b${BETA}_goal_seed${SEED}"
    printf '  %-40s %s/%s칸\n' "${TAG}" "$(filled_cells "results/${TAG}")" "${NEED}" | tee -a "${Q}"
done
python3 -u CLARE_V6_report.py > /dev/null 2>&1 \
    && ts "리포트 갱신 -> results/direction error aware anchor_V6_SR.txt"
