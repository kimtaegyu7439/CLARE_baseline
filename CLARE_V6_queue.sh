#!/usr/bin/env bash
# L2_V6 — Directional-Error-Aware 앵커 3 팔. CLARE 셋팅 libero_goal.
#
#   GPU=3 FORM=dea    bash CLARE_V6_queue.sh
#   GPU=0 FORM=sqerr  bash CLARE_V6_queue.sh
#   GPU=1 FORM=unifmd bash CLARE_V6_queue.sh
#   DRY=1 GPU=3 FORM=dea bash CLARE_V6_queue.sh
#
# 세 팔은 방향 항 e_j 의 계수만 다르다 (method note 2절 표):
#   sqerr   2·r_s·r_t / N   -> ‖v_s−v̄‖² 와 항등. 즉 기존 MSE 앵커
#   unifmd  β / N           -> 크기·방향을 분리하고 균일 배분
#   dea     β·w_j (detach)  -> 방향 예산을 오차 큰 과거 태스크로
#
# 프로토콜은 V5 시드 런과 맞춘다 — 태스크당 10,000 step, 50 demo, 10 task = 55칸,
# 칸당 **50 rollout**(eval seed 7~56), train seed 7, 코드북 K=96.
# 학습과 롤아웃을 분리한다(--eval_after_each_task false -> CLARE_rollout_only.py).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
# ── 파이썬 환경 ─────────────────────────────────────────────────────────
# 세 경우를 다 받는다. 서버마다 홈도, 환경 방식도 다르다.
#   1) 이미 쓸 수 있는 python 이 켜져 있다 (venv 든 conda 든)  -> 아무것도 안 한다
#   2) VENV=<경로>            -> 그 가상환경(venv/virtualenv)을 켠다
#   3) CONDA_ENV=<이름>       -> conda 를 켠다 (CONDA_SH 로 conda.sh 경로 지정 가능)
# activate 스크립트가 구버전이면 set -u 에서 터질 수 있어 잠시 끈다.
_have_torch() { python -c 'import torch' >/dev/null 2>&1; }
_act() { set +u; # shellcheck disable=SC1090
         source "$1" >/dev/null 2>&1; set -u; }

if ! _have_torch; then
    if [ -n "${VENV:-}" ]; then
        _va=""
        for _c in "${VENV}/bin/activate" "${VENV}"; do
            [ -f "${_c}" ] && { _va="${_c}"; break; }
        done
        if [ -n "${_va}" ]; then
            _act "${_va}"
        else
            echo "경고: VENV='${VENV}' 에서 activate 를 못 찾았다"
        fi
    fi
fi

if ! _have_torch; then
    CONDA_ENV=${CONDA_ENV:-clare}
    if [ -z "${CONDA_SH:-}" ]; then
        for _c in "${HOME}/miniconda3/etc/profile.d/conda.sh" \
                  "${HOME}/anaconda3/etc/profile.d/conda.sh" \
                  "${HOME}/mambaforge/etc/profile.d/conda.sh" \
                  "${CONDA_EXE:+$(dirname "$(dirname "${CONDA_EXE:-/x}")")/etc/profile.d/conda.sh}"; do
            [ -n "${_c}" ] && [ -f "${_c}" ] && { CONDA_SH="${_c}"; break; }
        done
    fi
    if [ -n "${CONDA_SH:-}" ] && [ -f "${CONDA_SH}" ]; then
        _act "${CONDA_SH}"
        set +u; conda activate "${CONDA_ENV}" >/dev/null 2>&1 || true; set -u
    fi
fi

if ! _have_torch; then
    echo "✗ 현재 python 에 torch 가 없다 ($(command -v python || echo 'python 없음'))"
    echo "  셋 중 하나로 알려줘라:"
    echo "   · 가상환경을 미리 켜고 실행   source ~/tether/bin/activate && bash ..."
    echo "   · venv 경로를 넘겨서          VENV=~/tether bash ..."
    echo "   · conda 환경 이름을 넘겨서    CONDA_ENV=<이름> bash ..."
    exit 1
fi
source "${HERE}/bash/clare/env.sh"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

GPU=${GPU:?GPU 를 지정해라}
FORM=${FORM:?FORM 을 지정해라 (dea|sqerr|unifmd)}
SEED=${SEED:-7}
SUITE=libero_goal
STEPS=${STEPS:-10000}
N_EVAL=${N_EVAL:-50}
EVAL_SEED=${EVAL_SEED:-7}
EPS=${EPS:-50}
NTASK=${NTASK:-10}
NW=${NW:-6}
CODEBOOK_K=${CODEBOOK_K:-96}
BETA=${BETA:-2.0}
# 학습 배치. 기본 32 는 seed7 기준 런과 맞춘 값이다.
# ★ 이걸 바꾸면 앵커도 같이 바뀐다 — 과거 태스크당 n=B 개를 뽑으므로
#   (L2_V6.py:341) B=24 면 앵커 표본도 24 개가 된다. 새 태스크 표본 수만
#   독립적으로 줄이는 설정이 아니다.
BATCH=${BATCH:-32}
EPS_W=${EPS_W:-1e-3}
# 같은 form 을 다른 하이퍼로 또 돌릴 때 결과가 섞이지 않게 태그에 붙인다
TAGSUF=${TAGSUF:-}
# L2_V6.py(속도 공간) 또는 L2_V7.py(액션 공간)
ENTRY=${ENTRY:-L2_V6.py}
VER=${VER:-v6}
REDUCE=${REDUCE:-mean}
MIN_FREE_GB=${MIN_FREE_GB:-30}
PRETRAIN=${PRETRAIN_PATH:-${HOME}/Models/dit_flow_mt_libero_90_pretrain}
QUEUE="${HERE}/logs/CLARE_V6.queue"

export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p logs results
[ -d "${PRETRAIN}" ] || { echo "사전학습 체크포인트 없음: ${PRETRAIN}"; exit 1; }

# 배치가 기본값이 아니면 태그에 박는다 — 안 그러면 batch 24/32 결과가
# 같은 폴더에 덮어써진다.
[ "${BATCH}" != "32" ] && TAGSUF="${TAGSUF}_b${BATCH}bs"
TAG="clare_${VER}_${FORM}${TAGSUF}_goal_seed${SEED}"
OUT="results/${TAG}"
CKPT="${HERE}/outputs/CLARE_V6/${VER}_${FORM}${TAGSUF}_seed${SEED}"
LOG="logs/${TAG}.log"
NEED=$(( NTASK * (NTASK + 1) / 2 ))

ts() { echo "[$(date '+%F %T')] $*" | tee -a "${QUEUE}"; }

# ── 롤아웃 동시 실행 제한 (VRAM 세마포어) ──────────────────────────────
# 실측: eval_batch_size=50 롤아웃 1 개의 피크 VRAM 이 약 22G 다. 환경 50 개를
# 띄우고 EGL 오프스크린 렌더링까지 GPU 에 올리기 때문이고, 학습(약 5G)보다
# 네 배 넘게 크다. 학습 8 개는 40G 면 되지만 롤아웃 8 개는 175G 라
# H200(143G)도 넘긴다.
#   -> 학습은 그대로 병렬로 두고, **롤아웃만** 슬롯 수만큼으로 묶는다.
# 슬롯 파일 N 개를 flock -n 으로 하나씩 찔러보고, 다 차 있으면 기다린다.
ROLLOUT_SLOTS=${ROLLOUT_SLOTS:-4}
ROLL_FD=""
SLOTDIR="${HERE}/logs/.rollout_slots"
acquire_rollout_slot() {
    [ "${ROLLOUT_SLOTS}" -le 0 ] && return 0
    mkdir -p "${SLOTDIR}"
    local i fd waited=0
    while true; do
        for i in $(seq 1 "${ROLLOUT_SLOTS}"); do
            exec {fd}>"${SLOTDIR}/slot${i}.lock" 2>/dev/null || continue
            if flock -n "${fd}" 2>/dev/null; then
                ROLL_FD="${fd}"
                ts "   롤아웃 슬롯 ${i}/${ROLLOUT_SLOTS} 획득 (대기 ${waited}s)"
                return 0
            fi
            exec {fd}>&- 2>/dev/null
        done
        [ "${waited}" -eq 0 ] && ts "   롤아웃 슬롯 ${ROLLOUT_SLOTS} 개가 다 찼다 — 대기"
        sleep 30; waited=$((waited + 30))
    done
}
release_rollout_slot() {
    [ -n "${ROLL_FD}" ] || return 0
    flock -u "${ROLL_FD}" 2>/dev/null
    exec {ROLL_FD}>&- 2>/dev/null
    ROLL_FD=""
}
# 중간에 죽어도 슬롯을 물고 있지 않게 한다
trap 'release_rollout_slot' EXIT INT TERM
filled() {
    python3 - "$1" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "sr_matrix.csv"
if not p.exists(): print(0); raise SystemExit
b = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
print(sum(1 for l in b[1:] for c in l.split(",")[1:] if c.strip()))
PY
}

if [ -n "${DRY:-}" ]; then
    echo "GPU ${GPU}  form=${FORM} β=${BETA}  seed ${SEED}  학습 -> 롤아웃 ${N_EVAL}회 -> ${OUT}"
    exit 0
fi

FREE=$(df -BG --output=avail "${HERE}" | tail -1 | tr -dc '0-9')
[ "${FREE}" -lt "${MIN_FREE_GB}" ] && { ts "✗ ${TAG} — 디스크 ${FREE}G < ${MIN_FREE_GB}G"; exit 1; }

# ── 1단계: 학습 (롤아웃 없이) ───────────────────────────────────────────
ts "▶ ${TAG} 학습  GPU ${GPU}  form=${FORM} β=${BETA} ε_w=${EPS_W} reduce=${REDUCE}"
echo "════ $(date '+%F %T') 학습 시작 ${FORM} ════" >> "${LOG}"
PRETRAIN_PATH="${PRETRAIN}" python -u "${ENTRY}" \
    --anchor_form "${FORM}" --beta "${BETA}" --eps_w "${EPS_W}" --anchor_reduce "${REDUCE}" \
    --arm v4 --suite "${SUITE}" --num_tasks "${NTASK}" \
    --episodes_per_task "${EPS}" --steps_per_task "${STEPS}" \
    --batch_size "${BATCH}" --num_workers "${NW}" \
    --codebook_k "${CODEBOOK_K}" --lambda_level 3.0 \
    --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
    --eval_after_each_task false \
    --seed "${SEED}" --eval_seed "${EVAL_SEED}" \
    --ckpt_root "${CKPT}" --out "${OUT}" --resume >> "${LOG}" 2>&1
rc=$?
ts "◀ ${TAG} 학습 exit=${rc}"
[ "${rc}" -ne 0 ] && { ts "   -> ${LOG} 확인"; exit "${rc}"; }

# ── 2단계: 롤아웃 50회 ──────────────────────────────────────────────────
# DO_ROLLOUT=0 이면 학습만 하고 끝낸다. 체크포인트가 남으므로 롤아웃은
# 나중에 따로 돌릴 수 있다(bash/rollout_first.sh 의 1 단계).
#   쓰는 경우: 그 서버의 EGL 이 GPU 를 못 써서 롤아웃이 CPU 소프트웨어
#   렌더링으로 떨어질 때. 학습은 렌더링을 안 하므로 영향이 없다.
if [ "${DO_ROLLOUT:-1}" = "0" ]; then
    ts "⏭ ${TAG} 롤아웃 건너뜀 (DO_ROLLOUT=0) — 체크포인트는 ${CKPT} 에 있다"
    exit 0
fi
CKDIR=$(ls -d "${CKPT}"/*_ours 2>/dev/null | head -1)
[ -z "${CKDIR}" ] && { ts "   ★ 체크포인트 폴더 없음"; exit 1; }
ts "▶ ${TAG} 롤아웃 ${N_EVAL}회 — 슬롯 대기"
acquire_rollout_slot
python -u CLARE_rollout_only.py --ckpt_dir "${CKDIR}" --out "${OUT}" \
    --suite "${SUITE}" --rows all \
    --eval_episodes "${N_EVAL}" --eval_batch_size "${N_EVAL}" \
    --eval_seed "${EVAL_SEED}" >> "${LOG}" 2>&1
rc=$?
release_rollout_slot
FILLED=$(filled "${OUT}")
ts "◀ ${TAG} 롤아웃 exit=${rc}  (${FILLED}/${NEED}칸)"
python -u CLARE_V6_report.py > /dev/null 2>&1 && ts "   기록 -> results/direction error aware anchor_V6_SR.txt"

# ★ 롤아웃이 한 칸도 못 채웠으면 실패다. 예전에는 여기서 리포트 생성이 마지막
#   명령이라 그 종료코드(0)가 스크립트 종료코드가 됐고, 55 칸이 전부 None 인
#   런이 exit=0 으로 보고됐다. 40 시간 학습해 놓고 실패를 못 알아챘다.
#   (실제 사례: gym-libero 가 libero_goal 을 거부해 55 칸 전부 0 초 만에 None)
if [ "${rc}" -ne 0 ] || [ "${FILLED}" -eq 0 ]; then
    ts "✗ ${TAG} 롤아웃 실패 — 칸 ${FILLED}/${NEED}, exit=${rc}"
    ts "   ${LOG} 에서 'SR rollout 실패' 를 찾아봐라. 체크포인트는 남아 있으므로"
    ts "   환경을 고친 뒤 롤아웃만 다시 돌리면 된다(재학습 불필요)."
    exit "$([ "${rc}" -ne 0 ] && echo "${rc}" || echo 3)"
fi
exit 0
