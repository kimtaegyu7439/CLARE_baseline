#!/usr/bin/env bash
#
# eval_libero_40_clare.sh가 흩뿌려 놓은 eval_info.json들을 하나의 SR 행렬로 모은다.
# 평가가 도는 중에 따로 실행해 진행 상황을 봐도 된다(아직 없는 칸은 빈칸으로 남는다).
#
#   OUT_ROOT=... bash bash/clare/collect_libero_40_clare.sh

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SEED=${SEED:-42}
NUM_TASKS=${NUM_TASKS:-40}
OUT_ROOT=${OUT_ROOT:-./outputs/CLARE_eval/libero_40/seed${SEED}}
PYTHON=${PYTHON:-python}

"${PYTHON}" - "${OUT_ROOT}" "${NUM_TASKS}" <<'PYEOF'
import json, sys
from pathlib import Path

root, n = Path(sys.argv[1]), int(sys.argv[2])
cells, broken = {}, 0
for p in sorted(root.glob("stage*/task*/eval_info.json")):
    k = int(p.parent.parent.name.removeprefix("stage"))
    t = int(p.parent.name.removeprefix("task"))
    try:
        cells[(k, t)] = json.loads(p.read_text())["aggregated"]["pc_success"]
    except Exception as e:                       # 중간에 죽어 잘린 파일은 건너뛴다
        print(f"  WARN 읽기 실패 {p}: {e}"); broken += 1

want = n * (n + 1) // 2
print(f"[collect] {len(cells)}/{want} 칸 완료" + (f" (손상 {broken})" if broken else ""))
if not cells:
    raise SystemExit(0)

# 1) 기존 libero_40_SR.txt와 같은 탭 구분 형식
sr = root / "libero_40_SR.txt"
with open(sr, "w") as f:
    f.write("LIBERO_40\t" + "\t".join(str(t) for t in range(n)) + "\n")
    for k in range(n):
        row = [cells.get((k, t)) for t in range(k + 1)]
        f.write(f"{k}\t" + "\t".join("" if v is None else f"{v:.0f}" for v in row) + "\n")

# 2) 스테이지별 평균(= 그 시점까지 배운 태스크 평균 SR)
csv = root / "summary.csv"
with open(csv, "w") as f:
    f.write("checkpoint," + ",".join(f"task{t}" for t in range(n)) + ",avg_seen\n")
    for k in range(n):
        row = [cells.get((k, t)) for t in range(n)]
        seen = [v for v in row[: k + 1] if v is not None]
        f.write(f"stage{k}," + ",".join("" if v is None else f"{v:.1f}" for v in row)
                + "," + (f"{sum(seen)/len(seen):.1f}" if seen else "") + "\n")

print(f"[collect] saved -> {sr}\n[collect] saved -> {csv}")

# 3) 옛 버그가 되살아났는지 감시한다.
#    버그가 있으면 SR(k,j)≈0 ⟺ (j%10) > (k%10) 라는 규칙이 매트릭스를 지배했다.
lo = [v for (k, t), v in cells.items() if t % 10 > k % 10]
hi = [v for (k, t), v in cells.items() if t % 10 <= k % 10]
if lo and hi:
    print(f"[check] (j%10)>(k%10) 칸 평균 {sum(lo)/len(lo):.1f}  "
          f"(n={len(lo)})   나머지 평균 {sum(hi)/len(hi):.1f} (n={len(hi)})")
    print("        앞쪽이 여전히 ~0이면 benchmark/task_id 짝이 또 깨진 것이다.")

# 화면용 하삼각 행렬
print("\n     SR(%)  " + "".join(f"{'t'+str(t):>5}" for t in range(n)) + f"{'avg':>7}")
for k in range(n):
    row = [cells.get((k, t)) for t in range(n)]
    seen = [v for v in row[: k + 1] if v is not None]
    print(f"  s{k:<3}    " + "".join(f"{'' if v is None else f'{v:.0f}':>5}" for v in row)
          + f"{(f'{sum(seen)/len(seen):.1f}' if seen else ''):>7}")
PYEOF
