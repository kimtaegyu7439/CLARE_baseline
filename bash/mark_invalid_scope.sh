#!/usr/bin/env bash
# anchor_scope=all 로 돌아간 LOTUS 런들을 무효로 표기한다.
# 폴더를 지우거나 옮기지 않는다 — 안에 INVALID.md 를 넣고, 리포트가 집지
# 않도록 새 런은 태그에 _scopeLL 을 붙인다.
set -uo pipefail
cd "${1:?결과 폴더의 부모 경로}"
n=0
for d in lotus_l3_dea_b*_k*_* lotus_v7_dea_b*_* lotus_v5act_c*_k*_*; do
    [ -d "$d" ] || continue
    case "$d" in *_scopeLL_*) continue;; esac     # 새 런은 건드리지 않는다
    cat > "$d/INVALID.md" <<'MD'
# ★ 이 런의 성능은 무효다 — anchor_scope=all

앵커가 **base 태스크까지 과거로 세고** 있었다. 그런데 지표는
`--skip_base_eval` 로 base 를 애초에 재지 않는다. 즉 점수에 들어가지
않는 태스크를 보존하는 데 앵커 예산을 썼다.

dea 는 Σ_j w_j = 1 이라 β 가 **총 예산**이므로, base 가 과거에 끼면
lifelong 태스크당 몫이 그만큼 줄어든다. 실측 w_mean = 1/N 이고 태스크당
방향 계수는 β/N 이다:

    libero_goal  scope=all       N 6→9    태스크당 β/N = 0.50 → 0.33  (β=3)
    libero_goal  scope=lifelong  N 1→3    태스크당 β/N = 3.00 → 1.00
    libero_50    scope=all       N 25→45  태스크당 β/N = 0.12 → 0.07
    libero_50    scope=lifelong  N 5→20   태스크당 β/N = 0.60 → 0.15

CLARE 셋팅은 N 1→9 로 3.00 → 0.33 구간을 쓴다. scope=all 로 돌린 LOTUS 는
그 구간의 아래쪽 1/6 만 밟았고, 그래서 CLARE 에서 잘 나온 설정이 LOTUS 에서
안 나왔다.

낭비된 예산 비율: goal stage1 은 과거 6 개가 전부 base 라 100%, goal stage4 는
9 중 6 (67%), libero_50 stage5 는 45 중 25 (56%).

-> 다시 돌린 런은 태그에 `_scopeLL_` 이 붙어 있다. 그쪽을 봐라.
MD
    n=$((n+1))
    echo "  표기: $d"
done
echo "총 ${n}개 폴더에 INVALID.md 를 넣었다"
