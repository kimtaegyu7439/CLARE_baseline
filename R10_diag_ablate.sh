#!/usr/bin/env bash
#
# 진단 함수의 VRAM 기여분 분리 — 짝지은 대조군.
#
#   A. diagon    현행. probe_batch(이미지 포함 약 201MB)를 스테이지 내내 들고 있고
#                100스텝마다 condition_deltas 가 velocity_net forward 를 3회 더 돈다.
#   B. diagoff   --no_diag 로 그 둘을 끈다.
#
# ★ 알고리즘은 완전히 동일하다. 수송에 쓰는 mu/sigma 통계는 양쪽 다 전수 패스로
#   낸다. 빠지는 것은 diagnostics.jsonl 의 delta_cur/delta_prev 열뿐이므로,
#   여기서 확인되면 본 실행에도 --no_diag 를 그대로 쓸 수 있다.
#
# 스테이지 0(앵커 없음)과 1(앵커 첫 등장)만. 롤아웃은 끈다 — VRAM 만 잰다.
#
# 사용법: bash R10_diag_ablate.sh <GPU> [대기할 pid]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "${HERE}"
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-}")" != "clare" ]; then
    source /home/sa090180/miniconda3/etc/profile.d/conda.sh; conda activate clare
fi
source "${HERE}/bash/clare/env.sh"
GPU=${1:-1}; WAIT_PID=${2:-}
export CUDA_VISIBLE_DEVICES="${GPU}" MUJOCO_EGL_DEVICE_ID="${GPU}" MUJOCO_GL=${MUJOCO_GL:-egl}
OUT=results/R10_diag_ablate
mkdir -p "${OUT}" logs/mod0

if [ -n "${WAIT_PID}" ]; then
    echo "[$(date '+%F %T')] pid ${WAIT_PID} 종료 대기 (gpu ${GPU})"
    while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
    echo "[$(date '+%F %T')] 대기 종료"
fi

sample() {   # sample <라벨> <csv> -> 샘플러 pid
    local tag=$1 csv=$2
    ( while true; do
        echo "$(date +%s),${tag},$(nvidia-smi -i ${GPU} --query-gpu=memory.used \
              --format=csv,noheader,nounits)" >> "${csv}"
        sleep 5
      done ) & echo $!
}

run() {   # run <이름> <추가인자>
    local name=$1 extra=${2:-}
    local dir="${OUT}/${name}"
    rm -rf "${dir}" "outputs/R10_diag_${name}"; mkdir -p "${dir}"
    local pid; pid=$(sample "${name}" "${OUT}/vram.csv")
    echo "[$(date '+%F %T')] === ${name} (${extra:-진단 켬}) 시작"
    if python -u R10.py --out "${dir}" --chunk_backward \
            --passthru --num_tasks 2 --eval_after_each_task false ${extra} \
            --ckpt_root "outputs/R10_diag_${name}" > "${dir}/train.log" 2>&1; then
        echo "[$(date '+%F %T')] === ${name} 완료"
    else
        echo "[$(date '+%F %T')] === ${name} 실패 -> ${dir}/train.log"
    fi
    kill "${pid}" 2>/dev/null || true
}

echo "ts,variant,mib" > "${OUT}/vram.csv"
run diagon
run diagoff "--no_diag"

python - "${OUT}" <<'PYEOF' | tee "${OUT}/report.txt"
import csv, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = [r for r in csv.DictReader(open(out / "vram.csv")) if r["mib"].strip().isdigit()]
L = ["=" * 76, "진단 함수의 VRAM 기여분 — 짝지은 대조군", "=" * 76, "",
     "A(diagon)  현행 — probe_batch 상주(약 201MB) + 100스텝마다 δ forward 3회",
     "B(diagoff) --no_diag — 둘 다 끔. 알고리즘(mu/sigma 수송)은 동일하다.",
     "스테이지 0(앵커 없음) + 1(앵커 등장), 롤아웃 없음, --chunk_backward 켬", "",
     f"{'변형':>9}{'표본':>7}{'p10':>9}{'중앙':>9}{'p90':>9}{'최대':>9}", "-" * 76]
med = {}
for v in ("diagon", "diagoff"):
    m = sorted(int(r["mib"]) for r in rows if r["variant"] == v)
    if not m:
        continue
    n = len(m); med[v] = m[n // 2]
    L.append(f"{v:>9}{n:>7}{m[n//10]:>9}{m[n//2]:>9}{m[9*n//10]:>9}{m[-1]:>9}")
if len(med) == 2:
    L += ["", f"  진단 기여분  = {med['diagon'] - med['diagoff']} MiB (중앙값 기준)",
          f"  진단 없는 R10 ≈ {med['diagoff']} MiB",
          "", "  참고  ER 실측 4,479 MiB"]
L += ["", "★ 알고리즘이 동일하므로 --no_diag 를 본 실행에 그대로 써도 된다.",
      "  빠지는 것은 diagnostics.jsonl 의 delta_cur / delta_prev 열뿐이다."]
print("\n".join(L))
PYEOF
echo "[$(date '+%F %T')] 완료 -> ${OUT}/report.txt"
