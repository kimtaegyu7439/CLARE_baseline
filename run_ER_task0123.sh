#!/usr/bin/env bash
#
# ER — libero_spatial 태스크 0..3 학습 + SR 표 생성
#
# 기존 bash/er/ER_task0123.sh 는 체크포인트만 남기고 SR 을 재지 않는다. 그래서
# 학습 뒤에 E0.py --reprobe 로 스테이지마다 SR 을 재고 4x4 표를 만든다.
# 프로토브 프로토콜은 E0(seq-FT) 와 문자 그대로 같다 — 칸당 20 롤아웃, start_seed 42.
# 따라서 나오는 표는 E0 λ=0 표, B1 4태스크 표와 같은 자로 비교된다.
#
# 기존 저장소 파일은 하나도 수정하지 않는다.
#
# 사용법
#   bash run_ER_task0123.sh          # GPU 1
#   bash run_ER_task0123.sh 3        # GPU 3
#   TRAIN_ONLY=1 bash run_ER_task0123.sh
#   PROBE_ONLY=1 bash run_ER_task0123.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    # shellcheck disable=SC1091
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh
    conda activate clare
fi
# shellcheck disable=SC1091
source "${HERE}/bash/clare/env.sh"

GPU=${1:-1}
export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_EGL_DEVICE_ID="${GPU}"
export MUJOCO_GL=${MUJOCO_GL:-egl}

SEED=${SEED:-42}
NUM_TASKS=${NUM_TASKS:-4}
OUT_ROOT=${OUT_ROOT:-./outputs/ER/libero_spatial/seed${SEED}}
RES_DIR=${RES_DIR:-${HERE}/results/ER_task0123}
RESULTS=${RESULTS:-${RES_DIR}/er_results.jsonl}
LOG=${RES_DIR}/run.log

# E0 와 같은 프로브 설정 (bash/E0/E0.sh:50-53)
HOLDOUT_EP=${HOLDOUT_EP:-5}
PROBE_BATCHES=${PROBE_BATCHES:-16}
PROBE_N_EP=${PROBE_N_EP:-20}
PROBE_EVAL_BS=${PROBE_EVAL_BS:-20}

DATASET_PREFIX=continuallearning/libero_spatial_image_task_
ENV_TASK_PREFIX=Libero_Spatial_Task_
E0_PY=./lerobot_lsy/src/lerobot/scripts/E0.py
PYTHON=${PYTHON:-python}

mkdir -p "${RES_DIR}"
log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

log "ER task0123  gpu=${GPU}  out=${OUT_ROOT}"

# ── 1. 학습 (기존 스크립트를 그대로 호출한다) ─────────────────────────────────
if [ "${PROBE_ONLY:-0}" != "1" ]; then
    log "학습 시작 — bash/er/ER_task0123.sh"
    SEED="${SEED}" NUM_TASKS="${NUM_TASKS}" \
        bash bash/er/ER_task0123.sh >>"${LOG}" 2>&1
    rc=$?
    log "학습 종료 rc=${rc}"
    [ $rc -ne 0 ] && { log "학습 실패 — 중단"; exit 1; }
fi
[ "${TRAIN_ONLY:-0}" = "1" ] && { log "TRAIN_ONLY — 프로브 생략"; exit 0; }

# ── 2. 스테이지마다 SR 프로브 (E0.py --reprobe) ───────────────────────────────
# jsonl 은 append 전용이라 새로 시작할 때 치운다. 안 치우면 옛 행과 섞인다.
[ -s "${RESULTS}" ] && mv "${RESULTS}" "${RESULTS}.bak" && log "이전 결과 -> ${RESULTS}.bak"

for k in $(seq 0 $((NUM_TASKS - 1))); do
    ckpt="${OUT_ROOT}/task_${k}/checkpoints/last/pretrained_model"
    if [ ! -d "${ckpt}" ]; then
        log "SKIP task ${k}: 체크포인트 없음 (${ckpt})"
        continue
    fi
    log "── probe stage ${k}  (태스크 0..${k}, 칸당 ${PROBE_N_EP} 롤아웃)"
    "${PYTHON}" "${E0_PY}" \
        --seed="${SEED}" \
        --job_name="ER_probe_task_${k}" \
        --output_dir="${OUT_ROOT}/task_${k}" \
        --dataset.repo_id="${DATASET_PREFIX}${k}" \
        --policy.path="${ckpt}" \
        --policy.push_to_hub=false \
        --reprobe=true \
        --eval_freq=0 \
        --env.type=libero \
        --env.benchmark=libero_spatial \
        --env.task="${ENV_TASK_PREFIX}${k}" \
        --ewc_lambda=0 \
        --run_tag="er" \
        --current_task="${k}" \
        --task_ids="$(seq -s, 0 "${k}")" \
        --dataset_prefix="${DATASET_PREFIX}" \
        --env_task_prefix="${ENV_TASK_PREFIX}" \
        --results_path="${RESULTS}" \
        --holdout_episodes="${HOLDOUT_EP}" \
        --probe_batches="${PROBE_BATCHES}" \
        --probe_sr=true \
        --probe_n_episodes="${PROBE_N_EP}" \
        --probe_eval_batch_size="${PROBE_EVAL_BS}" \
        --wandb.enable=false \
        >>"${LOG}" 2>&1 || log "FAILED probe stage ${k}"
done

# ── 3. 표 만들기 (ER / seq-FT / B1 나란히) ────────────────────────────────────
"${PYTHON}" - "${RESULTS}" "${NUM_TASKS}" "${RES_DIR}" <<'PYEOF' 2>&1 | tee -a "${LOG}"
import json, sys
from pathlib import Path

res, n, out_dir = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])


def from_jsonl(path, tag):
    cells = {}
    if not path.exists():
        return cells
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if tag is not None and r.get("run_tag") != tag:
            continue
        if r.get("sr") is not None:
            cells[(r["stage"], r["probe_task"])] = float(r["sr"])
    return cells


def from_csv(path):
    cells = {}
    if not path.exists():
        return cells
    for line in path.read_text().splitlines()[1:]:
        f = line.split(",")
        for t, v in enumerate(f[1:]):
            if v.strip():
                cells[(int(f[0]), t)] = float(v)
    return cells


def metrics(c):
    last = [c.get((n - 1, t)) for t in range(n)]
    diag = [c.get((t, t)) for t in range(n)]
    if any(v is None for v in last + diag):
        return None
    return (sum(last) / n,
            sum(last[t] - diag[t] for t in range(n - 1)) / (n - 1),
            sum(diag) / n)


arms = [
    ("ER",     from_jsonl(res, "er")),
    ("seq-FT", from_jsonl(Path("outputs/E0/libero_spatial/seed_42/e0_results.jsonl"), "0")),
    ("B1",     from_csv(Path("results/B1/sr_matrix.csv"))),
]

L = ["=" * 66,
     f"libero_spatial 태스크 0..{n-1}  SR 비교   seed 42, 칸당 20 롤아웃",
     "=" * 66, "",
     "학습 세팅 공통: 5000 steps/task, 45 에피소드(뒤 5개 hold-out), 배치 32",
     "ER: 현재 24 + 버퍼 8,  과거 태스크당 5 에피소드",
     "B1: p_drop 0.1, lambda_anchor 1.0", "",
     f"{'방법':10s}{'AvgSR':>9s}{'BWT':>9s}{'습득':>9s}"]
for name, c in arms:
    m = metrics(c)
    L.append(f"{name:10s}{m[0]:9.1f}{m[1]:+9.1f}{m[2]:9.1f}" if m
             else f"{name:10s}{'-':>9s}{'-':>9s}{'-':>9s}   미완 ({len(c)}/{n*(n+1)//2}칸)")
L.append("")

for name, c in arms:
    L += ["-" * 66, f"{name}   행 = 태스크 k 학습 후, 열 = 평가 태스크", "-" * 66,
          "after\\task " + "".join(f"{t:>7d}" for t in range(n))]
    for k in range(n):
        L.append(f"{k:>10d} " + "".join(
            f"{c[(k, t)]:7.0f}" if (k, t) in c else "      ." for t in range(k + 1)))
    L.append("")

txt = out_dir / "ER_task0123_SR.txt"
with txt.open("w") as f:
    cells = arms[0][1]
    f.write("LIBERO_SPATIAL\t" + "\t".join(str(t) for t in range(n)) + "\n")
    for k in range(n):
        f.write(f"{k}\t" + "\t".join(
            f"{cells[(k, t)]:.0f}" if (k, t) in cells else "" for t in range(k + 1)) + "\n")

report = "\n".join(L)
(out_dir / "comparison.txt").write_text(report)
print(report)
print(f"saved -> {txt}")
print(f"saved -> {out_dir / 'comparison.txt'}")
PYEOF

log "완료"
