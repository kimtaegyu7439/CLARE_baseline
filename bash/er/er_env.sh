# ER 전용 환경 설정.
#
# 경로 설정(HF_LEROBOT_HOME / HF_HUB_CACHE / PRETRAIN_PATH ...)은 clare/env.sh 와
# 같아야 하므로 그대로 가져다 쓰고, ER 에만 필요한 것을 여기서 덧붙인다.
# clare/env.sh 는 PRETRAIN_PATH 를 무조건 export 하므로, 밖에서 준 값이 있으면
# 미리 붙잡아 두었다가 source 뒤에 되돌린다(그래야 ${VAR:-기본값} 관례가 유지된다).
_ER_PRETRAIN_USER="${PRETRAIN_PATH:-}";
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../clare" && pwd)/env.sh";

# ── 사전학습 체크포인트 ──────────────────────────────────────────────────────
# ER 은 어댑터를 쓰지 않는 full fine-tuning 이라 --policy.path 는 **task 0 에서만**
# 쓰인다(스테이지 1 부터는 직전 스테이지 체크포인트에서 이어받는다).
# 그래서 이 값이 바뀌면 사슬 전체의 출발점이 바뀐다.
#
#   기본값  aicp_pretrain.py 가 만든 base phase 산출물 (libero90, seed 7, 200k step)
#   기존값  clare/env.sh 의 dit_flow_mt_libero_90_pretrain (HF 에서 받은 것)
# results/ER_10task_SR.txt 와 ER_libero_40_SR.txt 는 **기존값**으로 잰 것이다.
export PRETRAIN_PATH="${_ER_PRETRAIN_USER:-/home/sa090180/Models/aicp_clare_pretrain}";
unset _ER_PRETRAIN_USER;

# ── 롤아웃 시드 고정 ─────────────────────────────────────────────────────────
# 원래 lerobot 은 --seed 하나로 학습 RNG 와 롤아웃 에피소드 시드를 **둘 다** 잡는다
# (er.py 의 start_seed=cfg.seed, eval.py 의 start_seed=cfg.seed).
# 그래서 학습 시드를 바꾸면 평가하는 초기 상태 50개까지 같이 바뀌어,
# "학습 변동성"과 "평가 조건 변동"이 섞인다.
#
# EVAL_SEED 를 주면 두 파일이 롤아웃 시드만 이 값으로 덮어쓴다. 학습 시드는
# --seed 가 그대로 잡는다. 즉 학습만 바꿔가며 **같은 초기 상태**로 비교할 수 있다.
#   bash bash/er/ER_libero_spatial.sh 43   ->  학습 43 / 롤아웃 7~(7+N-1)
#
# 밖에서 덮어쓸 수 있게 ${VAR:-기본값} 형태로 둔다.
export EVAL_SEED=${EVAL_SEED:-7};
