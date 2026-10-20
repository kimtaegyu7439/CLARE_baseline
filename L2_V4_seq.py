"""순차(CLARE) 셋팅 드라이버 — **v4 / v3 / Seq-FT** 를 한 파일에서.

l2_codebook_id.py(v3 전용 독립 실행본)를 복사해 v4 두 변경과 Seq-FT 경로를 더한 것이다.
원본은 그대로 둔다. task 0 부터 하나씩 배우고 태스크가 끝날 때마다 코드북을 만드는
**원본 셋팅**이다 (LOTUS 셋팅이 아니다 — 그건 L2_V4.py 다).

    --arm v4      hard_linear 샘플러 + 앵커 속도 정규화   (= 논문의 TETHER)
    --arm v3      soft_kernel + MSE 앵커                  (= l2_codebook_id 기본)
    --arm seq_ft  앵커 없음. 코드북도 만들지 않는다        (베이스라인)

v4 가 v3 에서 바꾸는 것 — 정확히 두 가지. FM 손실은 건드리지 않는다.

  (1) 샘플러  k* ~ Cat(pi);  s~ = mu_s + sig_s (*) eps
              o~ = m_{k*} + A_{k*}(z~ - zbar_{k*})
              커널 가중 w, 대역폭 h, 관측 잡음 sig_o 를 쓰지 않는다.

  (2) 앵커    w = 1/(||v~_T[0:6]||^2 + c),  c = kappa * speed_med2_j
              median(w)*w_clip 에서 자르고 배치 평균 1 로 정규화.
              pose 6 차원에만 곱하고 gripper 는 그대로. mean(w)=1 이라
              lambda_level 을 다시 훑지 않아도 된다.

Seq-FT 는 앵커 항이 아예 없다. 코드북 수집·생성·teacher 스냅샷을 전부 건너뛰므로
학습이 v3/v4 보다 빠르다.

    # Seq-FT, spatial task 0~3, 평가 없이 체크포인트만
    python L2_V4_seq.py --arm seq_ft --suite libero_spatial --num_tasks 4 \
        --steps_per_task 5000 --eval_after_each_task false \
        --ckpt_root outputs/SEQFT --out results/seqft_spatial_t0123

    # v4 (TETHER)
    python L2_V4_seq.py --arm v4 --suite libero_spatial --num_tasks 4 \
        --steps_per_task 5000 --eval_after_each_task false \
        --ckpt_root outputs/V4SEQ --out results/v4_spatial_t0123
"""


from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from termcolor import colored
from torch.amp import GradScaler

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lerobot_lsy" / "src"))

from lerobot.configs.default import DatasetConfig, EvalConfig          # noqa: E402
from lerobot.configs.policies import PreTrainedConfig                  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig                  # noqa: E402
from lerobot.datasets.factory import make_dataset                      # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata    # noqa: E402
from lerobot.datasets.utils import cycle                               # noqa: E402
from lerobot.envs.configs import LiberoEnv                             # noqa: E402
from lerobot.optim.factory import make_optimizer_and_scheduler         # noqa: E402
from lerobot.policies.factory import make_policy                       # noqa: E402
from lerobot.utils.random_utils import set_seed                        # noqa: E402
from lerobot.utils.train_utils import (                                # noqa: E402
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import get_safe_torch_device, init_logging    # noqa: E402

# E0(이전 seq-FT 실행)의 45에피소드 분할 규칙. lerobot 패키지 안에 있어 B1 과 무관하다.
from lerobot.scripts.E0 import episode_sampler, split_episodes, to_device  # noqa: E402



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 아래는 B1.py 에서 그대로 옮겨 온 부분 (학습 루프 · 조건 조립 · 평가 · 지표)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ═════════════════════════════════════════════════════════════════════════════
#  기호
# ═════════════════════════════════════════════════════════════════════════════
#  K        태스크 개수 (= args.num_tasks). 스테이지 개수와 같다.
#  k        **스테이지** 인덱스 0..K-1. "몇 번째로 배웠는가".
#  task_id  **실제** 태스크 번호. order[k] 다. --task_order 를 주지 않으면 k 와 같다.
#  j        과거 스테이지 인덱스 (앵커가 되살릴 대상). 0 <= j < k.
#  i        평가 대상 스테이지 인덱스. SR 행렬의 열.
#
#  B        배치 크기 (코드에서 bsz). 기본 32.
#  H        액션 청크 길이 = policy.config.horizon = 16.
#  A        액션 차원 = 7 (xyz 3 + rpy 3 + gripper 1).
#  o        관측. 이미지 2장 + 상태. 배치 안에서는 observation.* 키들.
#  ℓ_j      태스크 j 의 자연어 명령어 문자열. instructions[f"task{j}"].
#  ∅        NULL 조건 = 빈 문자열(NULL_TEXT).
#
#  ─ 텐서 차원 약어 (이 파일의 shape 주석은 전부 이 기호를 쓴다) ─────────────
#  B        배치 크기 = 32. 앵커 안에서는 n 이라는 이름으로도 쓴다.
#  S        관측 프레임 수 n_obs_steps = 2   ([t-1, t]).
#  M        카메라 수 len(config.image_features) = 2.
#  Ds       프레임당 상태 차원 = 8.   observation.state 는 (B,S,Ds) = (B,2,8).
#  Dc       DINOv2 CLS 차원 = 768.    rgb_cls 출력은 (B*S*M, Dc) = (B*4,768).
#  Do       코드북이 다루는 o 차원 = S*M*Dc = 3072.
#  Dz       코드북이 다루는 s 차원 = S*Ds = 16.
#  N        코드북 표본 수 = 저수지에 모인 (s,o) 쌍의 개수 (최대 n_pairs = 8000).
#  Ke       코드북 **셀 개수** K_eff (병합 후 살아남은 셀. 최대 codebook_k = 96).
#           ★ N 개 표본이 Ke 개 셀로 뭉개진다. 셀 하나에 속한 표본 수는 n_k[j] 이고
#             Σ_j n_k[j] = N 이다. 즉 (Ke,16) 은 "표본 x state" 가 아니라
#             "셀 x state" — 셀마다 대표 벡터 하나씩이다.
#
#  x_t      (B,H,A) flow 경로 위의 점.  x_t = (1-t)·ε + t·a
#  t        (B,)   flow 시각 ~ U(0,1). 0=순수 노이즈, 1=정답 액션.
#  ε        (B,H,A) 표준정규 노이즈 (코드에서 noise).
#  a        (B,H,A) 정답 액션 청크 (batch["action"], MIN_MAX 로 [-1,1] 정규화됨).
#  target   (B,H,A) flow matching 정답 속도 = a - ε.  (코드/수식에서 v* 또는 g)
#  v        (B,H,A) 모델이 낸 속도. velocity_net(x_t, t, cond) 의 출력.
#
#  cond     (B,2576) 조건 벡터 = concat[lang 512, state 16, image 2048].
#  tail     (B,2064) cond 에서 언어를 뺀 나머지. 명령어와 무관해 배치당 1회만 계산.
#  lang_vec (B,512)  cond 의 앞부분. encode_lang() 이 만든다.
#
#  R        SR 행렬. R[k][i] = 스테이지 k 까지 배운 뒤 스테이지 i 태스크의 성공률(%).
#           행/열 모두 **스테이지 기준**이다(순서를 바꿔도 다른 팔과 지표가 같아지도록).
#  δ (delta) ‖v(o,ℓ) - v(o,∅)‖. 명령어를 넣고 뺀 차이 = 조건 민감도.
#           0 에 가까우면 모델이 명령어를 무시하고 있다는 뜻.
#  Ĝ        ‖v_teacher(ℓ_j) - target‖. 과거 정답과 현재 정답의 충돌량. B8 이 쓴다.
# ═════════════════════════════════════════════════════════════════════════════

NULL_TEXT = ""          # NULL 조건(빈 문자열). δ 진단에서만 쓴다.
OUT_DIR = REPO / "results" / "L2_codebook"   # --out 을 안 주면 여기

# 이 팔이 고정으로 쓰는 값 — 원본에서 argv 로 B1 에 넘기던 것들이다.
# 인자로 두면 "안 건드리는 손잡이"가 늘어 읽기만 어려워진다.
MODE = "ours"           # 체크포인트 경로 <suite>_seed<seed>_ours 에 들어간다
P_DROP = 0.0            # condition dropout 안 씀
LAMBDA_ANCHOR = 1.0     # 최종 손실 L = L_FM + λ_a · L_anchor
GUIDANCE_W = 1.0        # 평가 시 표준 조건부 추론 (CFG 없음)


# ═════════════════════════════════════════════════════════════════════════════
#  설정
# ═════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    """이 팔의 인자 전부. 원본은 l2_codebook 파서 + B1 파서 두 개를 sys.argv 를
    갈아 끼워 이어 붙였는데, 어느 인자가 어디로 가는지 읽기 어려워 하나로 합쳤다.
    고정값(mode/p_drop/lambda_anchor/guidance_w)은 인자에서 빼고 모듈 상수로 뒀다.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # ── 데이터 / 학습 예산 ───────────────────────────────────────────────────
    p.add_argument("--suite", default="libero_spatial",
                   help="LIBERO 스위트 이름. 데이터셋 repo 와 env 이름을 여기서 만든다.")
    p.add_argument("--num_tasks", type=int, default=4, help="태스크 0..num_tasks-1 (= K)")
    p.add_argument("--episodes_per_task", type=int, default=45,
                   help="학습에 쓰는 앞쪽 에피소드 수. 나머지 5개는 hold-out (E0 와 동일).")
    p.add_argument("--steps_per_task", type=int, default=5000,
                   help="태스크당 그래디언트 스텝. ★ 코사인 스케줄의 분모이기도 하다 "
                        "— 늘리면 학습률 곡선 전체가 바뀐다(results/B_fill 참조).")
    p.add_argument("--batch_size", type=int, default=32, help="배치 크기 B")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader 워커 수")
    p.add_argument("--task_order", default=None,
                   help="학습 순서. 예 \"1,0,2,3\". 기본은 0,1,..,K-1. "
                        "SR 행렬과 teacher 인덱스는 **스테이지 기준**으로 유지된다.")

    # ── 코드북 (이 팔의 본체) ────────────────────────────────────────────────
    p.add_argument("--codebook_k", type=int, default=96, help="k-means 셀 수 K")
    p.add_argument("--n_pairs", type=int, default=8000, help="저수지 표본 (s,o) 쌍 수")
    p.add_argument("--h_scale", type=float, default=1.0,
                   help="softmax 대역폭 배율. 1.0 = 스펙(최근접거리 중앙값)")
    p.add_argument("--ridge_rho", type=float, default=0.05, help="A_k 리지 회귀의 ρ")
    p.add_argument("--grad_min_frames", type=int, default=24,
                   help="이 수보다 표본이 적은 셀은 A_k = 0 으로 강등한다")
    p.add_argument("--lambda_level", type=float, default=3.0, help="level 앵커 가중 λ_lvl")
    # ── 팔 선택 ─────────────────────────────────────────────────────────────
    p.add_argument("--arm", choices=["v4", "v3", "seq_ft"], default="v3",
                   help="v4 = hard_linear + speedinv (TETHER), v3 = 원본, "
                        "seq_ft = 앵커 없음(코드북도 안 만든다). --sampler_mode/"
                        "--anchor_weight 를 직접 주면 그쪽이 우선한다.")
    p.add_argument("--sampler_mode",
                   choices=["soft_kernel", "hard_linear", "indep_gauss", "shuffled"],
                   default=None,
                   help="조건 좌표 (s̃,õ) 생성 절차. hard_linear = joint reconstruction. "
                        "indep_gauss / shuffled 는 ablation 대조군이다.")
    p.add_argument("--shuffle_group", type=int, default=4,
                   help="sampler_mode=shuffled 에서 짝을 섞는 묶음 크기")
    p.add_argument("--anchor_weight", choices=["mse", "speedinv"], default=None)
    p.add_argument("--kappa", type=float, default=1.0, help="c = kappa x speed_med2")
    p.add_argument("--anchor_wnorm", choices=["mean", "none"], default="mean",
                   help="mean (v4) 가중을 배치 평균 1 로 정규화 — 앵커 크기가 c 와 무관. "
                        "none (v5) w~_h 를 그대로 사용 — c 가 작을수록 앵커가 세진다.")
    p.add_argument("--anchor_c", type=float, default=0.0,
                   help="0 이면 c = kappa x speed_med2 (태스크별 median, 기본). "
                        ">0 이면 **모든 태스크**에 이 값을 c 로 고정한다. "
                        "libero_goal 의 speed_med2 가 0.6~0.8 이므로 0.1/0.01 은 "
                        "그보다 훨씬 작아 가중이 날카로워진다(느린 프레임을 더 키움).")
    p.add_argument("--w_clip", type=float, default=20.0, help="w 를 median(w)x이 값에서 자름")
    p.add_argument("--log_every_anchor", type=int, default=100)
    p.add_argument("--anchor_teacher_chunk", type=int, default=0,
                   help="batched 경로에서 teacher forward 한 번에 넣을 최대 행 수. "
                        "0 = 과거 태스크 전체를 한 번에(kB). ★ no_grad 라도 forward 중 "
                        "attention·MLP 중간 텐서는 배치에 비례해 잡히므로, kB 를 통째로 "
                        "넣으면 태스크별 실행에서는 없던 OOM 이 날 수 있다. student 쪽 "
                        "분할(태스크별 chunk backward)과는 별개로 조절한다.")
    p.add_argument("--anchor_sched", choices=["naive", "batched"], default="naive",
                   help="앵커의 teacher 연산을 어떻게 묶는가. naive = 과거 태스크마다 따로 "
                        "(기존 동작, 기본값). batched = teacher 의 cond_tail·velocity 를 과거 "
                        "태스크 전체에 대해 한 번에 — teacher 는 no_grad 라 배치를 키워도 "
                        "활성값이 안 늘고, student 는 그대로 태스크별 chunk backward 를 지킨다. "
                        "목적함수는 같다. ★ dropout 이 난수를 소비하므로 기존 실행과 비트 "
                        "단위로 같지는 않다(통계적으로 동일).")

    # ── 평가 ─────────────────────────────────────────────────────────────────
    p.add_argument("--eval_episodes", type=int, default=20, help="SR 칸당 롤아웃 수 (E0와 동일)")
    p.add_argument("--eval_batch_size", type=int, default=20,
                   help="동시에 띄우는 시뮬 환경 수. 20 이면 한 배치에 다 들어간다.")
    p.add_argument("--eval_after_each_task", type=lambda s: s.lower() != "false", default=True,
                   help="False 면 학습만 하고 SR 행렬을 비워 둔다.")

    # ── 실행 환경 ────────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42,
                   help="학습/평가 시드이자 코드북 k-means 시드. 체크포인트 경로가 "
                        "<suite>_seed<seed>_ours 라 시드가 다르면 자동으로 갈린다.")
    p.add_argument("--eval_seed", type=int, default=None,
                   help="롤아웃 초기 상태 시드. 생략하면 --seed 를 그대로 쓴다(기존 동작). "
                        "train seed 만 바꿔 변동을 재려면 이걸 고정해야 한다 — 안 그러면 "
                        "초기 상태까지 같이 바뀌어 학습 변동과 평가 변동이 섞인다.")
    p.add_argument("--out", default=None, help="지표/로그 출력 디렉토리")
    p.add_argument("--ckpt_root", default=str(REPO / "outputs" / "L2_codebook"),
                   help="체크포인트 루트. 실제 경로는 <root>/<suite>_seed<seed>_ours/task_<k>")
    p.add_argument("--log_every", type=int, default=100, help="진단 기록 주기(스텝)")
    p.add_argument("--no_diag", action="store_true", help="조건 민감도 δ 진단을 끈다")
    p.add_argument("--smoke", action="store_true", help="2태스크 x 100스텝 x 2에피소드")
    p.add_argument("--device", default="cuda")
    p.add_argument("--teacher_bf16", action="store_true", help="OOM 시 teacher를 bf16으로")
    p.add_argument("--skip_verify", action="store_true", help="시작 시 등가성 검사 생략(권장 안 함)")
    p.add_argument("--resume", action="store_true",
                   help="끊긴 실행을 이어 간다. 마지막으로 끝난 스테이지의 체크포인트·코드북을 "
                        "읽고, 그 행에 빈 SR 칸이 있으면 롤아웃부터, 없으면 다음 태스크 학습부터. "
                        "이어 갈 것이 없으면 처음부터 돈다.")
    args = p.parse_args()

    if args.eval_seed is None:
        args.eval_seed = args.seed

    if args.smoke:
        args.num_tasks = min(args.num_tasks, 2)
        args.steps_per_task = 100
        args.eval_episodes = 2
        args.eval_batch_size = 2
        if args.task_order:
            keep = [int(x) for x in args.task_order.split(",") if int(x) < args.num_tasks]
            args.task_order = ",".join(str(x) for x in keep)
    args.out_dir = str(Path(args.out) if args.out else OUT_DIR)
    # --arm 을 두 플래그로 편다. 개별 플래그를 직접 준 경우는 그쪽을 존중한다.
    preset = {"v4": ("hard_linear", "speedinv"), "v3": ("soft_kernel", "mse"),
              "seq_ft": ("soft_kernel", "mse")}[args.arm]
    if args.sampler_mode is None:
        args.sampler_mode = preset[0]
    if args.anchor_weight is None:
        args.anchor_weight = preset[1]
    return args


def suite_prefixes(suite: str) -> tuple[str, str]:
    """스위트 이름 -> (데이터셋 repo prefix, env task prefix). E0.sh:62-63 과 같은 규칙."""
    # libero_spatial -> "Libero_Spatial_Task_"  (env 이름),
    #                   "continuallearning/libero_spatial_image_task_"  (데이터셋 repo)
    env_prefix = "_".join(w.capitalize() for w in suite.split("_")) + "_Task_" # 첫 글자를 대문자로 바꿔서 재조립 + 데이터 경로 생성
    return f"continuallearning/{suite}_image_task_", env_prefix


def build_cfg(args, task_k: int, policy_path: str, out_dir: Path) -> TrainPipelineConfig:
    """저장소 설정 객체를 직접 조립한다.

    TrainPipelineConfig.validate() 는 sys.argv 를 다시 파싱해 --policy.path 를 찾으므로
    (configs/train.py:100-101) 프로그램적으로 만들 때는 부를 수 없다. 대신 그 안에서
    하던 일을 여기서 그대로 한다: 체크포인트 config.json 을 읽어 policy 설정을 만들고
    pretrained_path 를 박는다.
    """
    ds_prefix, env_prefix = suite_prefixes(args.suite)

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = args.device
    policy_cfg.push_to_hub = False

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=f"{ds_prefix}{task_k}"),
        env=LiberoEnv(benchmark=args.suite, task=f"{env_prefix}{task_k}"),
        policy=policy_cfg,
        output_dir=out_dir,
        job_name=f"L2CB_task_{task_k}",
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps_per_task,
        eval_freq=0,          # 롤아웃은 태스크가 끝난 뒤 따로. E0.sh:112 와 같다.
        log_freq=args.log_every,
        save_freq=args.steps_per_task,
        eval=EvalConfig(
            n_episodes=args.eval_episodes,
            batch_size=args.eval_batch_size,
            max_episodes_rendered=0,
        ),
    )
    # validate() 가 하던 프리셋 적용(configs/train.py:145-147). use_policy_training_preset
    # 기본값이 True 라 옵티마이저/스케줄러는 정책 설정에서 나온다.
    cfg.optimizer = policy_cfg.get_optimizer_preset()
    cfg.scheduler = policy_cfg.get_scheduler_preset()
    cfg.checkpoint_path = None
    return cfg


def task_instruction(repo_id: str) -> str:
    """태스크 데이터셋의 자연어 명령어. batch["task"] 에 실려 오는 바로 그 문자열."""
    tasks = LeRobotDatasetMetadata(repo_id).tasks
    if hasattr(tasks, "index"):          # pandas DataFrame 인 경우
        return str(list(tasks.index)[0])
    return str(next(iter(tasks.values())))


# ═════════════════════════════════════════════════════════════════════════════
#  조건 벡터 — 언어 부분만 갈아 끼우기
# ═════════════════════════════════════════════════════════════════════════════
#  _prepare_global_conditioning (modeling_dit_flow_mt.py:1125-1215) 은
#  concat([lang_proj(clip(task)) (B,512), state (B,16), img (B,2048)]) = (B,2576) 이다.
#  언어 외 2064차원은 명령어와 무관하므로 배치당 한 번만 계산하고, 앞 512만 바꿔 끼운다.
#  한 스텝에 DINOv2 를 3번 돌리지 않기 위한 최적화이며, verify_conditioning() 이
#  실제 함수 출력과 allclose 인지 시작 시 확인한다.
# ═════════════════════════════════════════════════════════════════════════════
def prep_batch(policy, batch: dict) -> dict:
    """DiTFlowMTPolicy.forward (modeling_dit_flow_mt.py:984-996) 의 전처리 3단계."""
    # 들어오는 batch: observation.state       (B,S,Ds)   = (B,2,8)
    #                 observation.images.* 각 (B,S,C,H,W) = (B,2,3,224,224)
    #                 action                  (B,H,A)    = (B,16,7)
    #                 task                    길이 B 의 문자열 리스트
    batch = policy.normalize_inputs(batch)          # 관측을 통계로 정규화 (shape 불변)
    if policy.config.image_features:
        batch = dict(batch)
        # 카메라 여러 대를 한 축으로 쌓는다: (B,S,M,C,H,W) = (B,2,2,3,224,224)
        batch["observation.images"] = torch.stack(
            [batch[k] for k in policy.config.image_features], dim=-4   # 각 (B,S,C,H,W)
        )
    # batch["action"] 을 MIN_MAX 로 [-1,1] 에 넣는다. 이후 모든 수식이 이 스케일이다.
    return policy.normalize_targets(batch)          # action (B,H,A) — shape 은 그대로


def encode_lang(policy, texts: list[str]) -> torch.Tensor:
    """명령어 -> 조건 벡터의 앞 512차원. LanguageEncoder 는 문자열 캐시가 있어 싸다."""
    dit = policy.dit_flow                           # texts: 길이 B 의 문자열 리스트
    with torch.no_grad():
        emb = dit.language_encoder(texts)           # (B,512) CLIP 텍스트 임베딩. 동결.
    # language_embedding_projection 은 학습 대상이므로 no_grad 밖에 둔다.
    return dit.language_embedding_projection(emb)   # (B,512) -> (B,512)  cond 의 앞부분


def rgb_cls(policy, batch: dict) -> torch.Tensor | None:
    """DINOv2 CLS 토큰. cond_tail 에서 가장 비싼 부분이고 **동결**이라 값이 안 변한다.

    teacher 스냅샷은 pretrained_rgb_encoder 를 student 와 **객체째 공유**하므로
    (snapshot 참조) 같은 배치에 대해 CLS 토큰이 문자 그대로 같다. 앵커가 과거
    태스크 여러 개를 도는 동안 이걸 한 번만 계산해 돌려 쓰면 DINOv2 호출이
    스테이지당 k번 -> 1번으로 줄어든다.
    """
    import einops

    dit = policy.dit_flow
    cfg = policy.config
    if not cfg.image_features:
        return None
    bsz, n_obs = batch["observation.state"].shape[:2]         # B, S(=2)
    images = einops.rearrange(
        batch["observation.images"], "b s n ... -> (b s n) ...",   # (B,S,M,C,H,W)
        b=bsz, s=n_obs, n=len(cfg.image_features),
    )                                    # -> (B*S*M, C,H,W) = (B*4, 3,224,224)
    with torch.no_grad():
        return dit.pretrained_rgb_encoder(images)             # (B*S*M, Dc) = (B*4,768)

def cond_tail(policy, batch: dict, cls: torch.Tensor | None = None) -> torch.Tensor:
    """조건 벡터에서 언어를 뺀 나머지 (상태 + 이미지). 명령어와 무관하다.

    cls 를 주면 DINOv2 를 건너뛴다. rgb_embedding_projection 은 **학습 대상**이라
    정책마다 다르므로 투영은 매번 다시 한다.
    """
    import einops

    dit = policy.dit_flow
    cfg = policy.config
    # bsz = 배치 크기 B,  n_obs = 관측 프레임 수 (n_obs_steps=2, 즉 [t-1, t])
    bsz, n_obs = batch["observation.state"].shape[:2]         # B, S(=2)

    # 8차원 상태 x 2프레임 = 16
    parts = [batch["observation.state"].flatten(start_dim=1)]  # (B,S,Ds) -> (B,16) = s
    if cfg.image_features:
        if cls is None:
            # 배치·프레임·카메라를 한 축으로 접어 인코더에 한 번에 넣는다
            images = einops.rearrange(
                batch["observation.images"], "b s n ... -> (b s n) ...",   # (B,S,M,C,H,W)
                b=bsz, s=n_obs, n=len(cfg.image_features),
            )                                                 # -> (B*4, 3,224,224)
            with torch.no_grad():
                cls = dit.pretrained_rgb_encoder(images)      # (B*4,768) CLS 토큰. 동결.
        cls_tokens = cls                                      # (B*S*M, Dc) = (B*4,768)
        emb = dit.rgb_embedding_projection(cls_tokens)        # (B*4,768) -> (B*4,512)
        # 다시 펼친다: 512 x 2프레임 x 2카메라 = 2048
        feats = einops.rearrange(
            emb, "(b s n) ... -> b s (n ...)", b=bsz, s=n_obs, n=len(cfg.image_features)
        )                                     # (B*4,512) -> (B,S,M*512) = (B,2,1024)
        parts.append(feats.flatten(start_dim=1))                        # (B,2048)
    if cfg.env_state_feature:                                # libero 에서는 안 쓴다
        parts.append(batch["observation.environment_state"].flatten(start_dim=1))
    return torch.cat(parts, dim=-1)                  # (B,16) ⊕ (B,2048) -> (B,2064)


def make_cond(lang_vec: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """(B,512) + (B,2064) -> (B,2576). concat 순서가 원본과 같아야 한다(verify 가 확인)."""
    return torch.cat([lang_vec, tail], dim=-1)               # (B,2576)


# ═════════════════════════════════════════════════════════════════════════════
#  Flow matching — modeling_dit_flow_mt.py:1281-1297 의 구성을 그대로 복제
# ═════════════════════════════════════════════════════════════════════════════
def sample_fm(policy, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x_t, t, target) 을 만든다. 앵커 항이 x_t, t 를 직접 잡아야 해서 복제했다."""
    dit = policy.dit_flow
    traj = batch["action"]                        # a: (B,H,A) = (B,16,7) 정답 청크, [-1,1]
    noise = dit.velocity_net.sample_noise(traj.shape[0], traj.device)   # ε: (B,H,A) ~ N(0,I)
    t = dit.noise_distribution.sample((traj.shape[0],)).to(traj.device)  # t: (B,) ~ U(0,1)
    # 직선 경로 위의 점. t=0 이면 순수 노이즈, t=1 이면 정답.
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * traj  # (B,1,1)*(B,H,A)->(B,H,A)
    # 그 직선의 속도. t 와 무관한 상수 벡터라 목표가 "직선을 따라가라"가 된다.
    return x_t, t, traj - noise                   # (B,H,A), (B,), (B,H,A) = a - ε


def fm_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """do_mask_loss_for_padding=False 경로. modeling_dit_flow_mt.py:1298-1320."""
    # 패딩 마스크를 쓰지 않는 경로. (B,H,A) 전 원소 평균이라 청크 길이로도 나뉜다.
    # pred, target: (B,H,A) -> 원소별 제곱오차 (B,H,A) -> 스칼라 ()
    return F.mse_loss(pred, target, reduction="none").mean()


# ═════════════════════════════════════════════════════════════════════════════
#  teacher 스냅샷
# ═════════════════════════════════════════════════════════════════════════════
def snapshot(policy, bf16: bool = False):
    """teacher 스냅샷. 동결 백본(DINOv2/CLIP)은 student 것을 **참조로 공유**한다.

    측정: 전체 193.6M(739 MiB) 중
        pretrained_rgb_encoder (DINOv2)  86.6M  330 MiB   동결
        language_encoder       (CLIP)    63.2M  241 MiB   동결
        ------------------------------------------------ 소계 571 MiB (77%)
        velocity_net                     43.2M  165 MiB   학습 (cond_proj 포함)
        rgb_embedding_projection          0.4M    2 MiB   학습
        language_embedding_projection     0.3M    1 MiB   학습
        ------------------------------------------------ 소계 167 MiB
    동결 백본은 requires_grad=False 이고 학습 내내 값이 변하지 않으므로 teacher 마다
    복사할 이유가 없다. 공유하면 teacher 하나가 739 -> 167 MiB (4.4x) 가 되어
    frozen teachers 가 40태스크에서도 24GB 안에 들어온다(30GB -> 7.8GB).

    ★ 수치는 완전히 동일하다. 동결 모듈이라 student 쪽 값이 변하지 않기 때문이다.
    ★ bf16 과는 함께 쓸 수 없다 — .to(bfloat16) 이 공유된 백본까지 바꿔 student 를
      오염시킨다. bf16 을 요청하면 공유하지 않고 전체를 복사한다.
    """
    SHARED = ("pretrained_rgb_encoder", "language_encoder")   # 동결 백본 = 복사하지 않을 모듈
    if bf16:
        return copy.deepcopy(policy).eval().requires_grad_(False).to(torch.bfloat16)

    d = policy.dit_flow
    saved = {}                     # 잠시 떼어낸 백본 모듈들. finally 에서 되돌린다.
    for name in SHARED:                      # 복사 전에 떼어내 peak 메모리도 아낀다
        if hasattr(d, name):
            saved[name] = getattr(d, name)
            setattr(d, name, None)
    try:
        snap = copy.deepcopy(policy).eval().requires_grad_(False)
    finally:
        for name, m in saved.items():        # student 원상복구
            setattr(d, name, m)
    for name, m in saved.items():            # teacher 는 같은 객체를 가리킨다
        setattr(snap.dit_flow, name, m)
    torch.cuda.empty_cache()
    return snap


# ═════════════════════════════════════════════════════════════════════════════
#  등가성 검사 — "재구현이 아니다"를 코드로 증명한다
# ═════════════════════════════════════════════════════════════════════════════
def verify(policy, batch: dict, device) -> None:
    policy.train()
    raw = {k: (v.clone() if isinstance(v, torch.Tensor) else list(v) if isinstance(v, list) else v)
           for k, v in batch.items()}

    # (1) 조건 벡터 조립이 _prepare_global_conditioning 과 같은가
    prepped = prep_batch(policy, {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                                  for k, v in raw.items()})
    with torch.no_grad():
        ref_cond = policy.dit_flow._prepare_global_conditioning(prepped)   # (B,2576)
        mine = make_cond(encode_lang(policy, list(prepped["task"])),       # (B,512)
                         cond_tail(policy, prepped))                       # (B,2064) -> (B,2576)
    if not torch.allclose(ref_cond, mine, atol=1e-5, rtol=1e-4):
        raise AssertionError(
            f"조건 벡터 조립이 _prepare_global_conditioning 과 다르다 "
            f"(최대차 {(ref_cond - mine).abs().max().item():.3e}). "
            "modeling_dit_flow_mt.py 의 concat 순서가 바뀐 것일 수 있다."
        )
    logging.info(f"[B1][verify] conditioning OK  max|Δ|={(ref_cond - mine).abs().max().item():.2e}")

    # (2) 복제한 FM 손실이 policy.forward 와 같은가.
    #     RNG 를 같은 상태에서 출발시키고 소비 순서를 맞춘다(cond -> noise -> t -> net).
    seed = 20260821                # 두 경로에 같은 RNG 상태를 주기 위한 임의의 고정값
    b1 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in raw.items()}
    torch.manual_seed(seed)
    ref_loss, _ = policy.forward(b1)          # 저장소 원본이 계산한 손실

    b2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in raw.items()}
    torch.manual_seed(seed)
    p2 = prep_batch(policy, b2)
    cond = make_cond(encode_lang(policy, list(p2["task"])), cond_tail(policy, p2))  # (B,2576)
    x_t, t, target = sample_fm(policy, p2)          # (B,H,A), (B,), (B,H,A)
    # velocity_net: (B,H,A) x (B,) x (B,2576) -> (B,H,A);  fm_loss -> 스칼라 ()
    my_loss = fm_loss(policy.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond), target)

    if not torch.allclose(ref_loss.detach(), my_loss.detach(), atol=1e-5, rtol=1e-4):
        raise AssertionError(
            f"복제한 flow matching 손실이 policy.forward 와 다르다 "
            f"(ref {float(ref_loss):.6f} vs mine {float(my_loss):.6f}). "
            "modeling_dit_flow_mt.py:compute_loss 가 바뀌었는지 확인하라."
        )
    logging.info(f"[B1][verify] flow-matching OK  ref={float(ref_loss):.6f} mine={float(my_loss):.6f}")

def rollout_sr(policy, cfg: TrainPipelineConfig, env_task: str, args) -> float | None:
    """시뮬레이터 롤아웃 성공률(%). E0 와 같은 인자로 부른다."""
    from lerobot.envs.factory import make_env
    from lerobot.scripts.eval import eval_policy

    env = None
    try:
        env_cfg = copy.deepcopy(cfg.env)
        env_cfg.task = env_task
        env = make_env(env_cfg, n_envs=args.eval_batch_size, use_async_envs=False)
        policy.eval()
        with torch.no_grad():
            info = eval_policy(env, policy, args.eval_episodes, start_seed=args.eval_seed)
        return float(info["aggregated"]["pc_success"])
    except Exception as e:                       # SR 실패가 학습 전체를 죽이면 안 된다
        logging.warning(f"[B1] SR rollout 실패 ({env_task}): {type(e).__name__}: {e}")
        return None
    finally:
        if env is not None:
            env.close()
        policy.train()


# ═════════════════════════════════════════════════════════════════════════════
#  지표
# ═════════════════════════════════════════════════════════════════════════════
def compute_metrics(R: list[list[float | None]], K: int) -> dict:
    # R: (K,K) 중첩 리스트(텐서 아님). R[k][i] = 스테이지 k 학습 후 스테이지 i 의 SR(%).
    last = [R[K - 1][i] for i in range(K)]   # 길이 K. 마지막 행 = 전부 배운 뒤의 최종 성능
    diag = [R[i][i] for i in range(K)]       # 길이 K. 대각선 = 막 배웠을 때의 성능(습득)
    ok = all(v is not None for v in last) and all(v is not None for v in diag)
    out = {"learning_sr": {f"task{i}": diag[i] for i in range(K)}}
    if not ok:
        out["note"] = "SR 칸에 결측이 있어 집계 지표를 내지 않는다"
        return out
    forgetting = {}
    for i in range(K):
        # 태스크 i 를 배운 뒤(k>=i) 관측된 SR 들. 그 최고점에서 최종값을 뺀 것이 망각량.
        seen = [R[k][i] for k in range(i, K) if R[k][i] is not None]
        forgetting[f"task{i}"] = max(seen) - last[i]
    out.update({
        "AvgSR_final": sum(last) / K,          # 최종 행의 평균 = 대표 성능
        # BWT(backward transfer): 마지막 태스크를 뺀 과거들이 습득 시점 대비 얼마나
        # 변했는가. 음수면 망각, 양수면 뒤 태스크가 과거에 도움이 된 것.
        "BWT": sum(last[i] - diag[i] for i in range(K - 1)) / max(1, K - 1),
        "Forgetting": forgetting,
        "Forgetting_mean": sum(forgetting.values()) / K,
        "final_row": {f"task{i}": last[i] for i in range(K)},
    })
    return out


def summary_table(R, K, metrics) -> str:
    lines = ["", "=" * 62, "B1 SR matrix   행 = 태스크 k 학습 후, 열 = 평가 태스크 i", "=" * 62]
    lines.append("after\\task " + "".join(f"{i:>8d}" for i in range(K)))
    for k in range(K):
        cells = "".join(
            f"{R[k][i]:8.1f}" if R[k][i] is not None else "       ." for i in range(k + 1)
        )
        lines.append(f"{k:>10d} " + cells)
    lines.append("-" * 62)
    if "AvgSR_final" in metrics:
        lines.append(f"AvgSR_final  {metrics['AvgSR_final']:.1f}")
        lines.append(f"BWT          {metrics['BWT']:+.1f}")
        lines.append(f"Forgetting   mean {metrics['Forgetting_mean']:.1f}   " +
                     "  ".join(f"t{i}:{v:.0f}" for i, (_, v) in enumerate(metrics["Forgetting"].items())))
    else:
        lines.append(metrics.get("note", ""))
    lines.append("=" * 62)
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  메인
# ═════════════════════════════════════════════════════════════════════════════
RESUME_KEYS = ("suite", "num_tasks", "episodes_per_task", "steps_per_task", "batch_size",
               "task_order", "codebook_k", "n_pairs", "lambda_level", "arm", "sampler_mode",
               "anchor_weight", "kappa", "anchor_c", "anchor_wnorm", "w_clip", "eval_episodes", "eval_batch_size", "seed",
               "eval_seed")   # 이게 하나라도 다르면 이어 붙이면 안 된다


def _resume_state(args, out_dir: Path, ckpt_root: Path, K: int):
    """--resume: 끊긴 실행에서 (끝난 스테이지 수 n_done, SR 행렬 R) 을 되살린다.

    스테이지 k 가 끝났다 = checkpoints/last 링크가 있고(save_checkpoint 가 끝난 뒤에야
    걸린다) 코드북 파일도 있다(on_task_end 에서 체크포인트 **다음에** 쓴다). seq_ft 는
    코드북을 만들지 않으므로 링크만 본다. 앞 스테이지부터 연속으로 끝난 것만 센다.
    ★ RNG 는 이어지지 않는다(시작 시 set_seed 로 다시 깐다). cudnn.benchmark 때문에
      원래도 비트 단위 재현이 아니므로, 끊기지 않은 실행과는 통계적으로만 같다.
    """
    R: list[list[float | None]] = [[None] * K for _ in range(K)]
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        diff = {kk: (old.get(kk), getattr(args, kk)) for kk in RESUME_KEYS
                if kk in old and old.get(kk) != getattr(args, kk)}
        if diff:
            raise ValueError(f"--resume: 이전 실행과 설정이 다르다 {diff} — 다른 --out 을 쓰라")

    n_done = 0
    for k in range(K):
        link = ckpt_root / f"task_{k}" / "checkpoints" / "last"
        book = out_dir / "codebooks" / f"codebook_task{k}.pt"
        if not (link / "pretrained_model" / "model.safetensors").exists():
            break
        if args.arm != "seq_ft" and not book.exists():
            break
        n_done = k + 1

    csv = out_dir / "sr_matrix.csv"
    if n_done and csv.exists():
        rows = [l for l in csv.read_text().splitlines() if l.strip() and not l.startswith("#")]
        for line in rows[1:]:
            c = line.split(",")
            k = int(c[0])
            if k >= n_done:          # 학습이 안 끝난 스테이지의 칸은 믿지 않는다
                continue
            for i, v in enumerate(c[1:K + 1]):
                if v.strip():
                    R[k][i] = float(v)

    # 학습이 안 끝난 스테이지의 진단 기록은 새로 쓸 것과 겹치므로 잘라낸다
    for name in ("diagnostics.jsonl", "l2cb.jsonl"):
        p = out_dir / name
        if p.exists():
            keep = [l for l in p.read_text().splitlines()
                    if l.strip() and json.loads(l).get("task", 0) < n_done]
            p.write_text("".join(l + "\n" for l in keep))

    missing = [(k, i) for k in range(n_done - 1) for i in range(k + 1) if R[k][i] is None]
    if missing:
        logging.warning(f"[resume] 마지막 이전 행의 빈 칸은 채우지 않는다(그 시점 정책이 없다): {missing}")
    last_row = ([i for i in range(n_done) if R[n_done - 1][i] is None] if n_done else [])
    logging.info(colored(
        f"[resume] 끝난 스테이지 {n_done}/{K}  "
        + (f"-> 스테이지 {n_done-1} 빈 칸 {last_row} 롤아웃부터" if last_row and args.eval_after_each_task
           else f"-> 스테이지 {n_done} 학습부터" if n_done < K else "-> 이미 전부 끝남"),
        "magenta", attrs=["bold"]))
    return n_done, R


def _train_loop(args, anchor) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    init_logging()

    device = get_safe_torch_device(args.device, log=True)
    # ★ benchmark=True 는 알고리즘 선택을 런타임에 정하므로 실행마다 결과가 미세하게
    #   달라진다. 같은 시드로 다시 돌려도 비트 단위로 같지는 않다.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)
    rng = random.Random(args.seed)          # 앵커의 j 추첨 + p_drop 전용. 학습 RNG와 섞지 않는다.

    ds_prefix, env_prefix = suite_prefixes(args.suite)   # 데이터셋/환경 이름 접두사
    K = args.num_tasks
    order = ([int(x) for x in args.task_order.split(",")] if args.task_order
             else list(range(K)))
    if len(order) != K or sorted(order) != list(range(K)):
        raise ValueError(f"task_order={order} 가 0..{K-1} 의 순열이 아니다")
    logging.info(f"[B1] 학습 순서: {' -> '.join(str(t) for t in order)}")
    holdout = None                          # 평가용으로 떼어둘 에피소드 수. 첫 태스크에서 확정.

    instructions: dict[str, str] = {}       # "task{k}" -> ℓ_k. 키가 **스테이지** 인덱스다.
    R: list[list[float | None]] = [[None] * K for _ in range(K)]   # SR 행렬 (아직 안 잰 칸은 None)
    diag_log: list[dict] = []               # 스텝별 손실·δ 기록 (diagnostics.jsonl 로도 나간다)
    ckpt_root = Path(args.ckpt_root) / f"{args.suite}_seed{args.seed}_{MODE}"
    n_done = 0               # --resume 으로 이어 받은 끝난 스테이지 수. 0 = 처음부터.
    if args.resume:
        n_done, R = _resume_state(args, out_dir, ckpt_root, K)

    json.dump(vars(args), (out_dir / "config.json").open("w"), indent=2, ensure_ascii=False)
    logging.info(colored(f"[L2CB] tasks=0..{K-1}  anchor={anchor.describe()}",
                         "green", attrs=["bold"]))

    policy = None            # 학습 중인 정책. 태스크 0 에서 한 번 만들고 계속 이어 쓴다.
    policy_path = None       # 다음 태스크의 초기값 경로. 처음엔 사전학습, 이후엔 직전 체크포인트.
    if n_done:
        policy_path = str(ckpt_root / f"task_{n_done-1}" / "checkpoints" / "last" / "pretrained_model")
    t_start = time.perf_counter()
    step_times: list[float] = []      # 스텝당 소요(초). 전체 소요 추정용.
    rollout_times: list[float] = []   # SR 칸 하나당 소요(초).

    def eval_row(k: int, cfg) -> None:
        """SR 평가: 스테이지 k 학습 후 스테이지 0..k. 이미 채워진 칸(--resume)은 건너뛴다."""
        for i in range(k + 1):
            if R[k][i] is not None:
                continue
            r0 = time.perf_counter()
            sr = rollout_sr(policy, cfg, f"{env_prefix}{order[i]}", args)
            rollout_times.append(time.perf_counter() - r0)
            R[k][i] = sr
            logging.info(colored(
                f"[B1] SR  after stage {k}(task {order[k]})  on stage {i}(task {order[i]}) = {sr}",
                "yellow"))
            write_outputs(out_dir, R, K, args, order)

    for k in range(K):
        task_id = order[k]                 # 실제 태스크. k 는 스테이지 인덱스다.
        repo_id = f"{ds_prefix}{task_id}"
        # instructions 는 **스테이지 기준**으로 담는다. 앵커가 rng.randrange(k) 로
        # 과거 "스테이지"를 뽑으므로, 이렇게 해야 순서를 바꿔도 올바른 명령어가 나온다.
        instructions[f"task{k}"] = task_instruction(repo_id)
        json.dump(instructions, (out_dir / "instructions.json").open("w"), indent=2, ensure_ascii=False)

        # --resume: 학습이 끝난 스테이지는 건너뛴다. 단, 마지막으로 끝난 스테이지의
        # 행에 빈 SR 칸이 남았으면 학습 없이 그 칸만 롤아웃한다(eval_only).
        eval_only = k < n_done
        if eval_only and not (args.eval_after_each_task and k == n_done - 1
                              and any(R[k][i] is None for i in range(k + 1))):
            continue

        # ── 초기화: task 0 은 사전학습, 이후는 직전 태스크 (E0.sh:88-90) ──────────
        if policy_path is None:
            import os
            policy_path = os.environ.get("PRETRAIN_PATH") or str(
                Path(os.environ.get("CLARE_MODEL_ROOT", str(Path.home() / "Models")))
                / "dit_flow_mt_libero_90_pretrain"
            )
        stage_dir = ckpt_root / f"task_{k}"
        cfg = build_cfg(args, task_id, policy_path, stage_dir)

        logging.info(colored(f"\n══ [B1] task {k}  init={policy_path}", "cyan", attrs=["bold"]))
        dataset = make_dataset(cfg)

        if holdout is None:
            total = dataset.meta.total_episodes
            holdout = total - args.episodes_per_task
            if holdout < 0:
                raise ValueError(f"episodes_per_task={args.episodes_per_task} > 전체 {total}")
            logging.info(f"[B1] 에피소드 분할: 학습 {args.episodes_per_task} / hold-out {holdout}")
        if holdout > 0:
            train_eps, _ = split_episodes(cfg.dataset.repo_id, cfg.dataset.root, holdout)
        else:
            # hold-out 0 = 데이터셋 전체로 학습. CLARE/ER 스크립트가 쓰는 방식이다
            # (bash/clare/clare_libero_spatial.sh 에는 hold-out 개념이 없다).
            # split_episodes 는 0 < holdout < total 을 요구하므로(E0.py:189) 우회한다.
            train_eps = list(range(dataset.meta.total_episodes))

        # 태스크 0 에서만 정책을 만들고, 이후는 메모리에 있는 걸 이어 쓴다.
        # E0 는 스테이지마다 프로세스를 새로 띄워 체크포인트를 다시 읽지만 float32
        # 왕복이라 값이 같다. 옵티마이저/스케줄러는 E0 를 따라 태스크마다 새로 만든다.
        if policy is None:
            policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
            if n_done:   # 체크포인트에서 이어 붙였다: 과거 코드북 + teacher 를 복원한다
                anchor.restore(policy, n_done, args, device)
        if eval_only:    # 학습·저장·코드북은 이미 끝났다. 빈 SR 칸만 채운다.
            eval_row(k, cfg)
            del dataset
            torch.cuda.empty_cache()
            continue
        # ★ 태스크마다 새로 만든다. 코사인 스케줄이 cfg.steps(=steps_per_task) 를 분모로
        #   쓰므로, 매 태스크가 lr 1e-4 에서 시작해 0 으로 착지한다.
        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
        grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)   # AMP 손실 스케일러 (현재 비활성화)

        sampler = episode_sampler(cfg, dataset, train_eps)
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            sampler=sampler,
            pin_memory=device.type == "cuda",
            drop_last=False, # epoch돌 때 마지막 batch수보다 적은 데이터 쓸건지
            multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
            persistent_workers=cfg.num_workers > 0,
        )
        dl_iter = cycle(loader)      # 끝나면 처음부터 다시 도는 무한 이터레이터

        probe_batch = None           # 진단 δ 를 항상 같은 데이터에서 재기 위한 고정 배치

        # 태스크 시작 훅. 기본 팔(B1/B2/B7/B8/B9)에는 없으므로 아무 일도 하지 않는다.
        # R10 처럼 태스크 데이터 전체로 통계를 먼저 잡아야 하는 팔이 쓴다.
        anchor.on_task_start(policy, k, args, instructions, device,
                             cfg=cfg, dataset=dataset, train_eps=train_eps,
                             prep=prep_batch)
        policy.train()

        for it in range(args.steps_per_task):
            t0 = time.perf_counter()
            batch = to_device(next(dl_iter), device)
            batch = prep_batch(policy, batch)
            if probe_batch is None:      # 진단용 고정 배치 (현재 태스크)
                # ★ 이미지까지 통째로 복제해 스테이지 내내 상주한다(약 201MB).
                #   --no_diag 면 표식만 남기고 복제하지 않는다.
                probe_batch = True if args.no_diag else {
                    kk: (vv.clone() if isinstance(vv, torch.Tensor) else list(vv))
                    for kk, vv in batch.items()}
                if k == 0 and not args.skip_verify:
                    verify(policy, {kk: (vv.clone() if isinstance(vv, torch.Tensor) else list(vv))
                                    for kk, vv in to_device(next(dl_iter), device).items()}, device)

            bsz = batch["action"].shape[0]              # B   (action: (B,H,A)=(B,16,7))
            # DINOv2 CLS 토큰은 동결이라 값이 안 변한다. 한 번만 뽑아 student 와
            # 모든 teacher 가 돌려 쓴다(앵커가 과거 k개를 돌아도 DINOv2 는 1회).
            cls = rgb_cls(policy, batch)                # (B*S*M, Dc) = (B*4,768)
            anchor.cls = cls
            tail = cond_tail(policy, batch, cls)        # (B,2064) 명령어와 무관한 부분
            x_t, t, target = sample_fm(policy, batch)   # (B,H,A), (B,), (B,H,A)

            # 현재 태스크 명령어로 조건을 만든다. condition dropout 은 쓰지 않는다
            # (이 팔은 p_drop = 0 고정. 명령어를 ∅ 로 바꾸는 경로가 없다).
            cond = make_cond(encode_lang(policy, list(batch["task"])), tail)   # (B,2576)

            with torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
                # (B,H,A) x (B,) x (B,2576) -> (B,H,A)
                pred = policy.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond)
                loss_fm = fm_loss(pred, target)                    # 스칼라 ()

                # ── (2) counterfactual conditional anchoring ─────────────────
                # 코드북 앵커. chunk backward 라 여기서 돌아오는 값은 detach 된
                # 로그용이고, 그래디언트는 anchor.loss() 안에서 이미 흘렀다.
                loss_anchor = anchor.loss(policy, batch, tail, x_t, t, k,   # 스칼라 ()
                                          instructions, rng, args, device)   # (로그용, detach)

                # 최종 손실. FM 항의 가중이 1 이므로 앞선 이론식의 μ=1 에 해당한다.
                loss = loss_fm + LAMBDA_ANCHOR * loss_anchor       # 스칼라 ()

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)     # 클리핑 전에 AMP 스케일을 되돌린다
            grad_norm = torch.nn.utils.clip_grad_norm_(     # 스칼라. 클리핑 **전**의 노름
                policy.parameters(), cfg.optimizer.grad_clip_norm, error_if_nonfinite=False
            )
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
            step_times.append(time.perf_counter() - t0)

            if not torch.isfinite(loss):
                raise RuntimeError(f"loss가 발산했다 (task {k}, step {it}): {loss}")

            # ── 진단 ─────────────────────────────────────────────────────────
            if it % args.log_every == 0 or it == args.steps_per_task - 1:
                row = {
                    "task": k, "step": it,
                    "fm_loss": float(loss_fm.detach()),
                    "anchor_loss": float(loss_anchor.detach()),
                    "grad_norm": float(grad_norm),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if not args.no_diag:
                    row.update(condition_deltas(policy, probe_batch, instructions, k, args))
                diag_log.append(row)
                logging.info(
                    f"[B1] k={k} step={it:5d} fm={row['fm_loss']:.4f} "
                    f"anc={row['anchor_loss']:.4f} "
                    + (f"δcur={row['delta_cur']:.4f} "
                       + (f"δprev={row['delta_prev']:.4f}"
                          if row["delta_prev"] is not None else "δprev=-")
                       if "delta_cur" in row else "(δ 진단 꺼짐)")
                )
                (out_dir / "diagnostics.jsonl").open("a").write(json.dumps(row) + "\n")

        # ── 태스크 종료: 체크포인트 저장 -> teacher 교체 ─────────────────────────
        ckpt = get_step_checkpoint_dir(cfg.output_dir, args.steps_per_task, args.steps_per_task)
        save_checkpoint(ckpt, args.steps_per_task, cfg, policy, optimizer, lr_scheduler)
        update_last_checkpoint(ckpt)
        policy_path = str(ckpt / "pretrained_model")
        logging.info(f"[B1] saved ckpt_task{k} -> {ckpt}")

        anchor.on_task_end(policy, k, args, instructions, device,
                           dl_iter=dl_iter, cfg=cfg, prep=prep_batch)
        torch.cuda.empty_cache()

        # ── SR 평가: 태스크 0..k ────────────────────────────────────────────────
        if args.eval_after_each_task:
            eval_row(k, cfg)

        del loader, dl_iter, dataset
        torch.cuda.empty_cache()

    metrics = write_outputs(out_dir, R, K, args, order)
    table = summary_table(R, K, metrics)
    print(table)
    logging.info(table)

    wall = time.perf_counter() - t_start
    sps = len(step_times) / max(1e-9, sum(step_times))
    logging.info(f"[B1] 완료  wall {wall/60:.1f}분   {sps:.2f} step/s   "
                 f"rollout 평균 {sum(rollout_times)/max(1,len(rollout_times)):.1f}s")
    if args.smoke:
        full_steps = 5000 * 4
        full_rollouts = 4 * 5 // 2 + 4      # 1+2+3+4 = 10 칸
        est = full_steps / max(1e-9, sps) + 10 * (sum(rollout_times) / max(1, len(rollout_times))) * (
            args.eval_episodes and 20 / max(1, args.eval_episodes))
        logging.info(colored(
            f"[B1] full-run 추정: 학습 {full_steps/max(1e-9,sps)/3600:.1f}h + "
            f"롤아웃 10칸 -> 총 약 {est/3600:.1f}h", "magenta"))


def condition_deltas(policy, probe: dict, instructions: dict, k: int, args) -> dict:
    """조건 민감도. δ가 0에 가까우면 명령어를 무시하고 있다는 뜻이다."""
    was_training = policy.training
    policy.eval()
    out = {"delta_prev": None, "delta_cur": None}
    try:
        with torch.no_grad():
            tail = cond_tail(policy, probe)                # (B,2064)
            x_t, t, _ = sample_fm(policy, probe)           # (B,H,A), (B,)
            bsz = x_t.shape[0]                             # B
            net = policy.dit_flow.velocity_net             # (B,H,A)x(B,)x(B,2576)->(B,H,A)

            v_null = net(noisy_actions=x_t, time=t,        # (B,H,A) 명령어를 뺀 출력 v(·,∅)
                         global_cond=make_cond(encode_lang(policy, [NULL_TEXT] * bsz), tail))
            v_cur = net(noisy_actions=x_t, time=t,         # (B,H,A) 현재 태스크 v(·,ℓ_k)
                        global_cond=make_cond(encode_lang(policy, [instructions[f"task{k}"]] * bsz), tail))
            # δ_cur: p_drop 이 켜져 있으면 0 에 수렴한다 — 기본 출력이 곧 현재 태스크라서.
            # (B,H,A) -> flatten (B,H*A) -> 행별 노름 (B,) -> 스칼라
            out["delta_cur"] = float((v_cur - v_null).flatten(1).norm(dim=1).mean())
            if k >= 1:
                v_prev = net(noisy_actions=x_t, time=t,    # (B,H,A) 첫 태스크 v(·,ℓ_0)
                             global_cond=make_cond(encode_lang(policy, [instructions["task0"]] * bsz), tail))
                # δ_prev: 과거 태스크를 기본 출력에서 얼마나 떼어 놓고 있는가.
                # 0 이면 명령어를 무시하는 상태(condition blindness).
                out["delta_prev"] = float((v_prev - v_null).flatten(1).norm(dim=1).mean())
    finally:
        if was_training:
            policy.train()
    return out


def write_outputs(out_dir: Path, R, K: int, args, order=None) -> dict:
    order = order or list(range(K))
    ident = order == list(range(K))     # 학습 순서가 0,1,..,K-1 그대로인가
    # 행/열은 **스테이지 인덱스**다. 순서를 바꾼 실행에서 오독하지 않도록 맨 위에
    # 학습 순서를 적어 둔다(# 로 시작하므로 파서는 건너뛰면 된다).
    with (out_dir / "sr_matrix.csv").open("w") as f:
        f.write(f"# task_order: {','.join(str(t) for t in order)}"
                + ("" if ident else "   (행/열 = 스테이지 인덱스. 열 i = task "
                                    + ",".join(f"{i}->{t}" for i, t in enumerate(order)) + ")")
                + "\n")
        f.write("after_stage," + ",".join(f"stage{i}(task{order[i]})" for i in range(K)) + "\n")
        for k in range(K):
            cells = ["" if R[k][i] is None else f"{R[k][i]:.1f}" for i in range(K)]
            f.write(f"{k}," + ",".join(cells) + "\n")
    # 순서를 바꾼 경우엔 태스크 기준 표도 함께 낸다(다른 팔과 직접 대조용)
    if not ident:
        pos = {t: i for i, t in enumerate(order)}   # 실제 task -> 그 태스크를 배운 스테이지
        with (out_dir / "sr_matrix_bytask.csv").open("w") as f:
            f.write(f"# task_order: {','.join(str(t) for t in order)}   "
                    f"(행 = 스테이지, 열 = 실제 task)\n")
            f.write("after_stage," + ",".join(f"task{t}" for t in range(K)) + "\n")
            for k in range(K):
                cells = ["" if R[k][pos[t]] is None else f"{R[k][pos[t]]:.1f}"
                         for t in range(K)]
                f.write(f"{k}," + ",".join(cells) + "\n")
    metrics = compute_metrics(R, K)
    metrics["task_order"] = order if order else list(range(K))
    metrics["config"] = {
        "mode": MODE, "p_drop": P_DROP, "lambda_anchor": LAMBDA_ANCHOR,
        "suite": args.suite, "num_tasks": K, "steps_per_task": args.steps_per_task,
        "episodes_per_task": args.episodes_per_task, "eval_episodes": args.eval_episodes,
        "seed": args.seed, "eval_seed": args.eval_seed, "guidance_w": GUIDANCE_W,
    }
    json.dump(metrics, (out_dir / "metrics.json").open("w"), indent=2, ensure_ascii=False)
    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 아래는 l2_codebook.py 원본 (코드북 앵커)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ═════════════════════════════════════════════════════════════════════════════
#  구 시간-bin 경로 차단 (§0.5)
# ═════════════════════════════════════════════════════════════════════════════
class NoAnchor:
    """Seq-FT — 앵커 없음. 코드북 수집·생성·teacher 스냅샷을 전부 건너뛴다.

    B1 이 요구하는 프로토콜(describe/on_task_start/loss/on_task_end)만 만족시킨다.
    loss 가 항상 0 이므로 최종 손실이 FM 항 그대로다: L = L_FM.
    """

    name = "seq_ft"

    def __init__(self, args):
        self.a = args
        self.cls = None
        self.step = 0
        self.t_anchor = 0.0

    def describe(self):
        return "Seq-FT — 앵커 없음 (순차 파인튜닝 베이스라인)"

    def on_task_start(self, policy, k, args, instructions, device, **kw):
        logging.info(f"[SEQFT] task {k} 시작 — 앵커 없음")

    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        return torch.zeros((), device=device)

    def on_task_end(self, policy, k, args, instructions, device, **kw):
        logging.info(f"[SEQFT] task {k} 종료 — 코드북/teacher 없음")

    def restore(self, policy, n_done, args, device):
        logging.info(f"[SEQFT] resume — 스테이지 {n_done} 부터, 복원할 코드북/teacher 없음")


def _forbid_time_bin():
    """R10 의 시간-bin 통계 함수가 이 런에서 호출되면 즉시 터지게 한다."""
    try:
        import R10
    except Exception:
        return          # 독립 실행본에는 시간-bin 경로 자체가 없다

    def _dead(*a, **kw):
        raise RuntimeError(
            "시간-bin 경로(R10.compute_stats/phase_bins)는 L2_codebook 에서 쓰지 않는다")

    R10.compute_stats = _dead
    R10.phase_bins = _dead


# ═════════════════════════════════════════════════════════════════════════════
#  k-means (표준화된 s 공간, 16-d) — torch, GPU
# ═════════════════════════════════════════════════════════════════════════════
def kmeans(x: torch.Tensor, K: int, n_init: int = 10, iters: int = 100,
           seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, float]:
    """k-means++ 초기화 x n_init. (centers(K,D), labels(N,), inertia) 를 돌려준다.

    x: (N,D) — 여기서는 표준화된 s 공간이라 D = Dz = 16, N <= n_pairs = 8000.
    """
    N, D = x.shape                                # x: (N,D) = (N,16)
    best = None
    for init in range(n_init):
        g = torch.Generator(device="cpu").manual_seed(seed * 1000 + init)
        # k-means++
        idx = [int(torch.randint(N, (1,), generator=g))]   # 중심 인덱스 리스트, 최종 길이 K
        d2 = (x - x[idx[0]]).pow(2).sum(1)                 # (N,) 최근접 중심까지 거리²
        for _ in range(K - 1):
            p = (d2 / d2.sum().clamp_min(1e-12)).cpu()     # (N,) 다음 중심 추첨 확률
            idx.append(int(torch.multinomial(p, 1, generator=g)))
            d2 = torch.minimum(d2, (x - x[idx[-1]]).pow(2).sum(1))    # (N,)
        c = x[idx].clone()                                 # (K,D) 중심
        prev = None
        for _ in range(iters):
            lab = torch.cdist(x, c).argmin(1)              # (N,K) -> (N,) 소속 라벨
            cnt = torch.bincount(lab, minlength=K).clamp_min(1)       # (K,)
            cnew = torch.zeros_like(c).index_add_(0, lab, x) / cnt[:, None]   # (K,D)
            empty = torch.bincount(lab, minlength=K) == 0  # (K,) bool
            if empty.any():                       # 빈 셀은 가장 먼 점으로 재배치
                far = torch.cdist(x, c).min(1).values.argsort(descending=True)   # (N,)
                cnew[empty] = x[far[:int(empty.sum())]]    # (n_empty,D) 를 덮어쓴다
            inertia = float((x - cnew[lab]).pow(2).sum())  # (N,D) -> 스칼라
            if prev is not None and abs(prev - inertia) < 1e-7 * max(prev, 1.0):
                c = cnew
                break
            prev, c = inertia, cnew
        lab = torch.cdist(x, c).argmin(1)                  # (N,)
        inertia = float((x - c[lab]).pow(2).sum())         # 스칼라
        if best is None or inertia < best[2]:
            best = (c, lab, inertia)
    return best                                   # ((K,D), (N,), float)


# ═════════════════════════════════════════════════════════════════════════════
#  코드북
# ═════════════════════════════════════════════════════════════════════════════
def build_codebook(s: torch.Tensor, o: torch.Tensor, K: int, seed: int,
                   min_n: int = 5, h_scale: float = 1.0, ridge_rho: float = 0.05,
                   grad_min_frames: int = 24) -> dict:
    """(s(N,16), o(N,3072)) -> 코드북 dict. §2 그대로.

    입력  s: (N,Dz) = (N,16)      같은 프레임의 상태
          o: (N,Do) = (N,3072)    같은 프레임의 DINOv2 CLS 를 펼친 것
    출력  dict — 각 항목의 shape 은 함수 끝의 out 조립부에 적어 두었다.
    """
    N = s.shape[0]
    mean_s, std_s = s.mean(0), s.std(0).clamp_min(1e-8)   # 각 (16,)
    zs = (s - mean_s) / std_s                             # (N,16) 표준화된 s
    c, lab, inertia = kmeans(zs, K, n_init=10, seed=seed) # c (K,16), lab (N,), float

    # n_k < min_n 셀은 최근접 셀로 병합
    cnt = torch.bincount(lab, minlength=K)     # (K,) — 원소 j = j번 셀에 속한 표본 수
    keep = (cnt >= min_n).nonzero().squeeze(1) # (n_keep,) 살아남는 셀 인덱스
    drop = (cnt < min_n).nonzero().squeeze(1)  # (n_drop,) 병합될 셀 인덱스
    n_merged = int(drop.numel())
    if n_merged and keep.numel():
        # 버려지는 셀의 소속 점들을 남는 셀 중 최근접으로
        remap = torch.arange(K, device=s.device)   # (K,) 옛 라벨 -> 새 라벨 표
        d = torch.cdist(c[drop], c[keep])          # (n_drop, n_keep)
        remap[drop] = keep[d.argmin(1)]            # (n_drop,)
        lab = remap[lab]                           # (N,)
        # 라벨을 0..K_eff-1 로 압축
        uniq, lab = torch.unique(lab, return_inverse=True)   # uniq (Ke,), lab (N,)
        c = c[uniq]                                # (Ke,16)
    K_eff = int(c.shape[0])                        # Ke

    pi = torch.bincount(lab, minlength=K_eff).float()   # (Ke,) — 원소 j = j번 셀의 표본 수
    n_k = pi.clone()                                    # (Ke,)
    pi = pi / pi.sum()                                  # (Ke,) 합이 1 인 셀 확률

    def per_cell(v, D):
        """v: (N,D) -> (평균, 표준편차) 각 (Ke,D). 누적은 float64 로 한다.

        N 개 행을 lab 로 그룹지어 Ke 개 행으로 접는다. 반환 텐서의 j번째 행은
        "j번 셀에 속한 n_k[j] 개 표본의 평균/표준편차" 이지 표본 하나가 아니다.
        """
        m = torch.zeros(K_eff, D, device=v.device, dtype=torch.float64)   # (Ke,D) Σv
        sq = torch.zeros_like(m)                                          # (Ke,D) Σv²
        m.index_add_(0, lab, v.double())
        sq.index_add_(0, lab, v.double() ** 2)
        cnt = n_k.double().clamp_min(1.0)[:, None]      # (Ke,1) 브로드캐스트용
        mu = m / cnt                                    # (Ke,D)
        sd = (sq / cnt - mu ** 2).clamp_min(0.0).sqrt() # (Ke,D)
        return mu.float(), sd.float()

    # ★ 여기서 N 이 사라진다: (N,16) -> (Ke,16). 행 하나가 표본이 아니라 **셀** 이다.
    mu_s, sig_s = per_cell(s, s.shape[1])          # 각 (Ke,16)   = (셀 개수, Dz)
    m_o, sig_o = per_cell(o, o.shape[1])           # 각 (Ke,3072) = (셀 개수, Do)

    # ── v2: 셀별 선형 기울기 A_k (3072x16) ──────────────────────────────────
    # v1 은 셀 안에서 o 가 s 를 따라 움직이는 성분을 m_k 하나로 뭉개고, 그 변동을
    # σ_o,k 에 분산으로 흡수한다(스미어). A_k 를 두면 õ 의 중심이 s̃ 를 서브셀
    # 스케일로 따라가고, σ_o,k 는 **선형 잔차** 기준이 되어 얇아진다.
    n_demoted = 0
    stats_v1_var = sig_o.pow(2).sum(1).clone()      # (Ke,) 셀별 0차 총분산(로그용)
    D = o.shape[1]                                          # Do = 3072
    zbar = (mu_s - mean_s) / std_s                          # (Ke,16) 셀 평균의 표준화 좌표
    A = torch.zeros(K_eff, D, s.shape[1], device=o.device)  # (Ke,3072,16) 셀별 기울기
    sig_lin = sig_o.clone()                                 # (Ke,3072)
    for kk in range(K_eff):
        sel = (lab == kk).nonzero().squeeze(1)              # (n_k,) 이 셀의 표본 인덱스
        if sel.numel() == 0:
            continue
        dl = zs[sel] - zbar[kk]                             # (n_k,16)  δ = z − z̄_k
        Y = o[sel] - m_o[kk]                                # (n_k,3072) 중심화된 o
        if int(sel.numel()) < grad_min_frames:
            n_demoted += 1                                  # A_k = 0 (0차 강등)
        else:
            S = dl.T @ dl                                   # (16,n_k)@(n_k,16) -> (16,16)
            lam_r = ridge_rho * float(S.diagonal().sum()) / s.shape[1]   # 스칼라 ρ
            G = Y.T @ dl                                    # (3072,n_k)@(n_k,16) -> (3072,16)
            A[kk] = G @ torch.linalg.inv(                   # (3072,16)@(16,16) -> (3072,16)
                S + lam_r * torch.eye(s.shape[1], device=o.device))      # (16,16)
            Y = Y - dl @ A[kk].T             # (n_k,16)@(16,3072) -> (n_k,3072) 선형 잔차
        sig_lin[kk] = Y.std(0, unbiased=False)              # (n_k,3072) -> (3072,)
    sig_o = sig_lin                                         # (Ke,3072) 잔차 기준으로 교체

    # σ 하한 (§2)
    sig_s = sig_s.clamp_min(1e-3 * float(s.std()))          # (Ke,16)
    sig_o = sig_o.clamp_min(1e-3 * float(o.std()))          # (Ke,3072)
    # 대역폭 h = median(각 zs -> 최근접 중심 거리)
    # 스펙: h = median(각 zs -> 최근접 중심 거리).
    # 그런데 softmax(−d²/h²) 에 이 값을 그대로 쓰면 2등 셀도 거의 같은 가중을 받아
    # w 가 20여 개 셀에 퍼진다(합성 테스트 실측: 참여 셀 median 20.7/96, 최대가중 0.147).
    # 그러면 w@m 이 여러 중심의 평균이 되어 구조가 뭉개지고, 남은 분산은 대각 잡음이
    # 채워 d_eff 가 폭발한다 — R13 의 등방 실패 모드로 되돌아간다.
    # h_scale 로 대역폭을 좁힐 수 있게 열어 둔다(1.0 = 스펙 그대로).
    # ★ c 와 h 는 커널 가중(softmax(−‖zs−c‖²/h²))용이다. L2_codebook_bayes 는 가중을
    #   GMM 사후확률로 바꾸므로 그 팔에서는 **deprecated** — 파일에는 남긴다
    #   (스키마 호환 + sanity 의 구가중 대비용).
    # cdist (N,Ke) -> 행별 최소 (N,) -> 중앙값 -> 스칼라 h
    h = float(torch.cdist(zs, c).min(1).values.median().clamp_min(1e-6)) * h_scale

    # 코드북 dict — 이 파일에서 cb[...] 로 읽히는 모든 항목의 shape 일람 (Ke = K_eff)
    out = {"h_scale": h_scale,             # float   대역폭 배율
            "pi": pi.cpu(),                # (Ke,)   셀 확률 (합 1)
            "mu_s": mu_s.cpu(),            # (Ke,16) 셀별 s 평균
            "sig_s": sig_s.cpu(),          # (Ke,16) 셀별 s 표준편차 (대각 가우시안)
            "c": c.cpu(),                  # (Ke,16) 표준화 s 공간의 중심
            "m": m_o.cpu(),                # (Ke,3072) 셀별 o 평균
            "sig_o": sig_o.cpu(),          # (Ke,3072) 선형 잔차 표준편차
            "mean_s": mean_s.cpu(),        # (16,)   s 표준화 통계
            "std_s": std_s.cpu(),          # (16,)
            "h": h,                        # float   softmax 대역폭
            "K_eff": K_eff, "N": N,        # int, int
            "n_k": n_k.cpu(),              # (Ke,)   원소 j = j번 셀의 표본 수 (Σ = N)
            "inertia": inertia,            # float
            "n_merged": n_merged}          # int
    out["A"] = A.half().cpu()              # (Ke,3072,16) fp16 저장(~10MB), 연산은 fp32
    out["zbar"] = ((mu_s - mean_s) / std_s).cpu()   # (Ke,16) 셀 평균의 표준화 좌표
    out["ridge_rho"] = ridge_rho                    # float
    out["n_demoted"] = n_demoted                    # int
    # (Ke,)/(Ke,) -> (Ke,)  셀별 0차분산 / 1차잔차분산
    out["var_ratio_cells"] = (stats_v1_var / sig_o.pow(2).sum(1).clamp_min(1e-12)).cpu()
    out["A_fro"] = A.flatten(1).norm(dim=1).cpu()   # (Ke,3072*16)->(Ke,)  ‖A_k‖_F
    # ── ablation(Independent Gaussian) 재료 ────────────────────────────────
    # 셀을 무시한 **주변** 통계다. mean_s/std_s 는 표준화용으로 이미 있지만 o 쪽은
    # 셀별 m/sig_o 밖에 없어서 따로 담는다. 다른 샘플러는 이 키를 읽지 않으므로
    # 기존 팔의 동작은 그대로다.
    out["mean_o"] = o.mean(0).cpu()                      # (3072,)
    out["std_o"] = o.std(0).clamp_min(1e-8).cpu()        # (3072,)
    return out


def sample_codebook(cb: dict, n: int, gen=None) -> tuple[torch.Tensor, torch.Tensor]:
    """§3. (s̃(n,16), õ(n,3072))."""
    # 아래 첫 축은 전부 **셀 인덱스** j = 0..Ke-1 이다 (표본 축이 아니다).
    pi, mu_s, sig_s = cb["pi"], cb["mu_s"], cb["sig_s"]    # (Ke,), (Ke,16), (Ke,16)
    c, m, sig_o = cb["c"], cb["m"], cb["sig_o"]            # (Ke,16), (Ke,3072), (Ke,3072)
    k = torch.multinomial(pi, n, replacement=True, generator=gen)   # (n,) 셀 인덱스
    eps = torch.randn(n, mu_s.shape[1], device=mu_s.device, generator=gen)   # (n,16)
    s = mu_s[k] + sig_s[k] * eps              # (n,16)  p(s|j) 는 대각 가우시안
    z = (s - cb["mean_s"]) / cb["std_s"]      # (n,16)  표준화 좌표
    d2 = torch.cdist(z, c).pow(2)                          # (n,Ke)
    w = torch.softmax(-d2 / (cb["h"] ** 2), dim=1)         # (n,Ke) 행마다 합 1
    o_mu = w @ m                              # (n,Ke)@(Ke,3072) -> (n,3072)
    # õ_mean = Σ_j w_j (m_j + A_j δ_j)  —  δ_j = z − z̄_j
    # 두 번째 항을 (B,K,16) -> (B,K·16) @ (K·16, D) 한 번의 matmul 로 접는다.
    # (B,K,D) 중간 텐서를 만들지 않으므로 top-m 절단이 필요 없다.
    A = cb["A"].float()                                   # (Ke,3072,16)
    d = z[:, None, :] - cb["zbar"][None]                  # (n,1,16)-(1,Ke,16) -> (n,Ke,16)
    u = (w[:, :, None] * d).reshape(z.shape[0], -1)       # (n,Ke,16) -> (n,Ke*16)
    # A.permute -> (Ke,16,3072) -> (Ke*16,3072);  (n,Ke*16)@(Ke*16,3072) -> (n,3072)
    o_mu = o_mu + u @ A.permute(0, 2, 1).reshape(-1, A.shape[1])
    o_sd = (w @ sig_o.pow(2)).clamp_min(0.0).sqrt()       # (n,Ke)@(Ke,3072) -> (n,3072)
    o = o_mu + o_sd * torch.randn(n, m.shape[1], device=m.device, generator=gen)  # (n,3072)
    return s, o                                           # (n,16), (n,3072)


def to_device_cb(cb: dict, device) -> dict:
    return {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in cb.items()}


# ═════════════════════════════════════════════════════════════════════════════
#  앵커
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
#  ★ v4 변경점 — Lotus_l2cb_cl.py / L2_V4.py 와 같은 구현
# ═════════════════════════════════════════════════════════════════════════════
POSE_DIMS = 6      # action = [x,y,z,roll,pitch,yaw,gripper] -> 0..5 pose, 6 gripper


def sample_codebook_hard(cb: dict, n: int, gen=None):
    """v4 (1) 경성 배정 + 1차 항만. 커널 가중 w·대역폭 h·잡음 항 σ_o 를 쓰지 않는다."""
    pi, mu_s, sig_s = cb["pi"], cb["mu_s"], cb["sig_s"]
    k = torch.multinomial(pi, n, replacement=True, generator=gen)      # (n,)
    eps = torch.randn(n, mu_s.shape[1], device=mu_s.device, generator=gen)
    s = mu_s[k] + sig_s[k] * eps                                      # (n,16)
    z = (s - cb["mean_s"]) / cb["std_s"]                              # (n,16)
    d = (z - cb["zbar"][k]).unsqueeze(2)                              # (n,16,1)
    A = cb["A"][k].float()                                            # (n,3072,16)
    o = cb["m"][k] + torch.bmm(A, d).squeeze(2)                       # (n,3072)
    return s, o


def sample_codebook_indep(cb: dict, n: int, gen=None):
    """ablation A/B — Independent Gaussian.

    s 와 o 를 **각자의 주변 가우시안**에서 따로 뽑는다. 셀도, A_k 도, 짝도 없다.
      s̃ ~ N(mean_s, std_s²)   (대각)
      õ ~ N(mean_o, std_o²)   (대각, s̃ 와 독립)
    joint reconstruction 과 **주변 분포는 대체로 같고 결합 구조만 없다** — 그래서
    "조건 좌표가 서로 안 맞을 때 무슨 일이 생기는가"를 가르는 대조군이 된다.
    x_t 는 바뀌지 않는다: 여기서 나온 (s̃,õ) 를 teacher 에 넣어 얻은 â 로 그대로 깐다.
    """
    ms, ss = cb["mean_s"], cb["std_s"]                                # (16,), (16,)
    mo, so = cb["mean_o"], cb["std_o"]                                # (3072,), (3072,)
    s = ms + ss * torch.randn(n, ms.shape[0], device=ms.device, generator=gen)
    o = mo + so * torch.randn(n, mo.shape[0], device=mo.device, generator=gen)
    return s, o                                                       # (n,16), (n,3072)


def sample_codebook_shuffled(cb: dict, n: int, group: int = 4, gen=None):
    """ablation C/D — Shuffled pairing.

    joint reconstruction 과 **똑같이** 뽑은 뒤(hard_linear), 크기 group 묶음 안에서
    õ 만 자리를 바꿔 (s̃,õ) 짝을 깬다. 주변 분포는 한 글자도 안 바뀌고 결합만 깨지므로,
    A 와 달리 "분포가 달라져서"라는 반론이 원천적으로 막힌다.

    묶음 안에서 **완전 순열(derangement)** 을 쓴다 — 무작위 순열이면 제자리가 남아
    일부 행이 원래 짝 그대로 학습에 들어간다. group 이 1 이면 섞을 게 없다.
    """
    s, o = sample_codebook_hard(cb, n, gen)                           # (n,16), (n,3072)
    if group < 2:
        return s, o
    idx = torch.arange(n, device=o.device)
    for a in range(0, n, group):
        b = min(a + group, n)
        m = b - a
        if m < 2:
            continue            # 끝에 홀로 남은 행은 섞을 짝이 없다 — 그대로 둔다
        # 순환 이동은 항상 완전 순열이고, 이동량을 매번 새로 뽑아 고정 패턴을 피한다
        shift = int(torch.randint(1, m, (1,), generator=gen)) if m > 2 else 1
        idx[a:b] = torch.roll(idx[a:b], shifts=shift, dims=0)
    return s, o[idx]                                                  # (n,16), (n,3072)


def speed_weights(v_ref, eps_noise, c: float, w_clip: float, wnorm: str = "mean"):
    """기준 속도의 역제곱 가중. (B,H,A) -> (B,H). θ 와 무관해 detach.

    wnorm="mean"  (v4)  배치 평균 1 로 정규화한다. 앵커 항의 **크기**가 c 와
                        무관해지고 가중의 **모양**만 바뀐다.
    wnorm="none"  (v5)  w~_h 를 그대로 쓴다. 모양뿐 아니라 **크기도** 바뀐다 —
                        w ~ 1/(rho^2+c) 라 c 가 작을수록 앵커가 세진다.
                        즉 λ 를 c 에 따라 같이 움직이는 것과 같다.
    """
    with torch.no_grad():
        rho = (v_ref + eps_noise)[..., :POSE_DIMS].norm(dim=-1)        # (B,H)
        w = 1.0 / (rho.pow(2) + c)
        w = torch.minimum(w, w.median() * w_clip)
        if wnorm == "mean":
            w = w / w.mean().clamp_min(1e-12)
        return w.detach()


def weighted_sq(resid, w):
    """가중 제곱오차. pose 6 차원에만 w 를 곱하고 gripper 는 그대로 둔다."""
    H, A = resid.shape[1], resid.shape[2]
    pose = (resid[..., :POSE_DIMS].pow(2).sum(-1) * w).sum(1)          # (B,)
    grip = resid[..., POSE_DIMS:].pow(2).sum(-1).sum(1)                # (B,)
    return ((pose + grip) / (H * A)).mean()


def action_speed_med2(actions) -> float:
    """c_j 의 재료 — median_{데모·스텝} ‖a_s[0:6]‖². (N,A) 또는 (N,H,A)."""
    a = actions.reshape(-1, actions.shape[-1])
    return float(a[:, :POSE_DIMS].pow(2).sum(-1).median())


class L2CodebookAnchor:
    """(s,o) 코드북 좌표 위의 level 앵커 + teacher 부트스트랩 x_t.

    R13Anchor 를 상속하지 않는다 — 시간-bin 통계를 아예 만들지 않기 위해서다.
    B1 이 요구하는 프로토콜(describe/on_task_start/loss/on_task_end)만 구현한다.
    """

    name = "L2_codebook"

    def __init__(self, args):
        self.a = args
        self.teacher = None
        self.books: dict[int, dict] = {}          # j -> 코드북(GPU 상주)
        self.K = args.codebook_k
        self.n_pairs = args.n_pairs
        self.h_scale = args.h_scale
        self.ridge_rho = args.ridge_rho
        self.grad_min_frames = args.grad_min_frames
        self.lam_lvl = args.lambda_level
        self.step = 0
        self.t_anchor = 0.0
        self.out = Path(args.out_dir); self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "codebooks").mkdir(exist_ok=True)
        self.log = (self.out / "l2cb.jsonl").open("a")
        # 수집 저수지 — 채워지면 s (n_pairs,16), o (n_pairs,3072), id (n_pairs,2) 로 상주(CPU)
        self._res_s = self._res_o = self._res_id = None
        self._seen = 0
        # v4 앵커 가중의 c_j 재료. 프레임별 ‖a[0:6]‖² 를 모아 태스크 끝에서 median.
        # 저수지가 아니라 전부 쌓는다 — 5000 스텝 x B x H 라 몇 MB 밖에 안 된다.
        self._sp2: list = []
        self._dataset = None
        self._prep = None
        self._sanity_done = False
        self._xt_sanity = False
        # teacher 의 언어 벡터 캐시 (j -> (1,512)). teacher 가 바뀌면 반드시 비운다.
        self._langT_cache: dict[int, torch.Tensor] = {}

    def describe(self):
        return (f"L2_codebook+grad (v3) — (s,o) 코드북 앵커  K={self.K}, "
                f"수집 {self.n_pairs}쌍, λ_lvl={self.lam_lvl}, "
                f"코드북 {len(self.books)}개")

    def _sample(self, cb, n, gen=None):
        """(s̃, õ). v4 는 경성 배정 + 1차 항, v3 는 커널 가중.

        ablation 두 모드는 여기서만 갈린다. 아래 단계 — teacher 부트스트랩 x_t,
        가중, λ, 과거 태스크 합산, 과거 태스크당 query 수 n=B — 는 전부 공통이다.
        """
        if self.a.sampler_mode == "indep_gauss":
            return sample_codebook_indep(cb, n, gen)
        if self.a.sampler_mode == "shuffled":
            return sample_codebook_shuffled(cb, n, self.a.shuffle_group, gen)
        if self.a.sampler_mode == "hard_linear":
            return sample_codebook_hard(cb, n, gen)
        return sample_codebook(cb, n, gen)

    def _c_of(self, cb) -> float:
        """앵커 가중의 분모 상수 c. --anchor_c 가 0 이면 태스크별 median 규칙.

        median 규칙은 c 가 태스크마다 달라진다(속도 분포가 다르니까). 고정하면
        태스크 간 가중 스케일이 같아지는 대신, 느린 태스크에서 w 가 매우 커질 수 있다.
        """
        if getattr(self.a, "anchor_c", 0.0) > 0:
            return float(self.a.anchor_c)
        return float(cb["speed_med2"]) * self.a.kappa

    def reduce_level(self, x):
        """anchor_norm = mean 고정. 원소 전체 평균이라 청크 길이·액션 차원으로도 나뉜다.

        x: (B,H,A) -> flatten (B,H*A) -> 행별 평균 (B,) -> 스칼라 ()
        """
        return x.flatten(1).pow(2).mean(1).mean()

    # ── 태스크 시작 ─────────────────────────────────────────────────────────
    def on_task_start(self, policy, k, args, instructions, device, **kw):
        self._dataset = kw.get("dataset")
        self._prep = kw.get("prep")
        self._res_s = self._res_o = self._res_id = None
        self._seen = 0
        self._sp2 = []
        self.step = 0
        self._xt_sanity = k > 0
        logging.info(f"[L2CB] task {k} 시작 — 수집 목표 {self.n_pairs}쌍, "
                     f"보유 코드북 {sorted(self.books)}")

    # ── (s,o) 쌍 수집 (§1) ──────────────────────────────────────────────────
    @torch.no_grad()
    def _collect(self, batch, cls, device):
        """저수지 표본. s 와 o 는 **같은 배치 행**에서 온 짝이다."""
        # 입력 cls: (B*S*M, Dc) = (B*4,768)
        s = batch["observation.state"].detach().flatten(1).float()  # (B,S,Ds) -> (B,16)
        n = s.shape[0]                                                   # B
        o = cls.detach().reshape(n, -1).float()             # (B*4,768) -> (B,3072)
        ident = torch.stack([batch["episode_index"].detach().long(),     # (B,)
                             batch["frame_index"].detach().long()], 1)   # (B,) -> (B,2)
        if "action" in batch:            # (B,H,A) 정규화된 액션 — pose 속도 제곱
            a = batch["action"].detach().float()
            self._sp2.append(a[..., :POSE_DIMS].pow(2).sum(-1).flatten().cpu())
        if self._res_s is None:
            self._res_s = torch.zeros(self.n_pairs, s.shape[1])            # (n_pairs,16)
            self._res_o = torch.zeros(self.n_pairs, o.shape[1])            # (n_pairs,3072)
            self._res_id = torch.zeros(self.n_pairs, 2, dtype=torch.long)  # (n_pairs,2)
        s, o, ident = s.cpu(), o.cpu(), ident.cpu()         # (B,16), (B,3072), (B,2)
        # 배치 단위 벡터화 저수지 — 행 루프를 돌면 스테이지당 16만 회가 되어 느리다.
        base = self._seen
        pos = torch.arange(base, base + n)               # (B,) 전역 표본 번호
        R = self.n_pairs
        fill = pos < R                                   # (B,) bool — 아직 안 찬 자리
        if fill.any():
            dst = pos[fill]                              # (n_fill,) 저수지 목적지 인덱스
            self._res_s[dst], self._res_o[dst], self._res_id[dst] = s[fill], o[fill], ident[fill]
        rest = ~fill                                     # (B,) bool
        if rest.any():
            # j 번째 원소는 확률 R/(j+1) 로 채택, 채택 시 [0,R) 균등 위치를 덮어쓴다
            j = pos[rest]                                # (n_rest,)
            r = (torch.rand(int(rest.sum())) * (j + 1).float()).long()   # (n_rest,)
            take = r < R                                 # (n_rest,) bool
            if take.any():
                idx_src = rest.nonzero().squeeze(1)[take]   # (n_take,) 배치 내 원본 행
                dst = r[take]                               # (n_take,) 저수지 목적지
                # 같은 자리에 여러 번 쓰이면 마지막 것이 남는다(저수지 정의와 동치)
                self._res_s[dst], self._res_o[dst], self._res_id[dst] = (
                    s[idx_src], o[idx_src], ident[idx_src])
        self._seen += n

    # ── 손실 ────────────────────────────────────────────────────────────────
    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        # 입력 shape:  tail (B,2064),  x_t (B,H,A),  t (B,)  — 반환은 스칼라 ()
        cls = getattr(self, "cls", None)                     # (B*4,768) 또는 None
        if cls is None:
            cls = rgb_cls(policy, batch)                     # (B*4,768)
        self._collect(batch, cls, device)          # k==0 포함 — 매 스텝 수집

        if k == 0 or self.teacher is None or not self.books:
            return torch.zeros((), device=device)            # 스칼라 ()
        t0w = time.perf_counter()
        n = batch["observation.state"].shape[0]              # n = B
        st_shape = batch["observation.state"].shape          # (B,S,Ds) = (B,2,8)
        tcol = t[:, None, None]                              # (B,) -> (B,1,1) 브로드캐스트용
        # 청크 backward 고정. 과거 태스크 하나를 계산할 때마다 즉시 backward 해서
        # 그래프를 버린다 -> 활성값이 O(1). ★ 그 결과 앵커 그래디언트가 과거에 대한
        # **합**이 된다(일괄이었다면 평균). 스테이지 9 에서 실효 가중이 λ_lvl 의 9배다.
        if getattr(policy.config, "use_amp", False):
            raise RuntimeError("이 팔은 청크 backward 고정이라 use_amp=True 와 못 쓴다")

        if getattr(self.a, "anchor_sched", "naive") == "batched" and len(self.books) > 1:
            return self._loss_batched(policy, batch, x_t, t, k, instructions,
                                      device, cls, n, st_shape, tcol, t0w)

        lvl = []
        teach = self.teacher
        for j in sorted(self.books):
            cb = self.books[j]
            s_t, o_t = self._sample(cb, n)                          # (n,16), (n,3072)
            s_t = s_t.detach(); o_t = o_t.detach()
            bb = dict(batch)
            # (n,16) -> (B,S,Ds): 합성 s̃ 를 관측 자리에 그대로 꽂는다
            bb["observation.state"] = s_t.view(st_shape).to(batch["observation.state"].dtype)
            cls_j = o_t.reshape(-1, cls.shape[-1]).to(x_t.dtype)     # (n,3072) -> (n*4,768)
            past = [instructions[f"task{j}"]] * n                    # 길이 n 의 문자열 리스트

            tl_S = cond_tail(policy, bb, cls_j)                      # (n,2064) student
            with torch.no_grad():
                tl_T = cond_tail(teach, bb, cls_j)                   # (n,2064) teacher
                cond_T = make_cond(encode_lang(teach, past), tl_T)   # (n,2576)
            cond_S = make_cond(encode_lang(policy, past), tl_S)      # (n,2576)

            def vel(pol, xx, tt, cond):
                """(n,H,A) x (n,) x (n,2576) -> (n,H,A) 속도장."""
                return pol.dit_flow.velocity_net(noisy_actions=xx, time=tt, global_cond=cond)

            # ── teacher 부트스트랩 x_t (L2 그대로) ──────────────────────────
            # 현재 배치의 x_t 를 쓰지 않는다. teacher 가 1-step 오일러로 만든 â 로
            # 경로를 다시 깔아, teacher 가 실제로 지나가는 지점에서 두 속도장을 잰다.
            with torch.no_grad():
                eps0 = torch.randn_like(x_t)                     # (n,H,A)
                # 1-step 오일러: â = ε₀ + v_T(ε₀, t=0)
                a_hat = (eps0 + vel(teach, eps0, torch.zeros_like(t), cond_T)).detach()  # (n,H,A)
            eps1 = torch.randn_like(x_t)                         # (n,H,A)
            x_j = ((1 - tcol) * eps1 + tcol * a_hat).detach()    # (n,1,1)*(n,H,A) -> (n,H,A)

            with torch.no_grad():
                vt0 = vel(teach, x_j, t, cond_T)                 # (n,H,A) teacher 속도
            vs0 = vel(policy, x_j, t, cond_S)                    # (n,H,A) student 속도
            resid = vs0 - vt0.to(vs0.dtype)                      # (n,H,A)
            if self.a.anchor_weight == "speedinv":
                # ṽ_T = v_T + ε₁ 프레임에서 잰 teacher 속도로 가중.
                # 잔차는 프레임을 옮겨도 같다: ṽ_S − ṽ_T = v_S − v_T.
                w = speed_weights(vt0, eps1, self._c_of(cb), self.a.w_clip,
                                  getattr(self.a, "anchor_wnorm", "mean"))
                L_j = weighted_sq(resid, w)
                if self.step % 500 == 0:
                    logging.info(f"[L2CB][v4] task{j} anc_w mean={float(w.mean()):.3f} "
                                 f"max={float(w.max()):.2f} c={self._c_of(cb):.4f}")
            else:
                L_j = self.reduce_level(resid)                   # (n,H,A) -> 스칼라 ()
            (LAMBDA_ANCHOR * self.lam_lvl * L_j).backward()
            lvl.append(L_j.detach())

            if self._xt_sanity and j == min(self.books):
                with torch.no_grad():
                    logging.info(
                        f"[L2CB][sanity] task{k} j={j}  K_eff={cb['K_eff']}  "
                        f"‖s̃‖={float(s_t.norm(dim=1).mean()):.4f} "
                        f"(실배치 {float(batch['observation.state'].flatten(1).norm(dim=1).mean()):.4f})  "
                        f"‖õ‖={float(o_t.norm(dim=1).mean()):.2f} "
                        f"(실배치 {float(cls.reshape(n,-1).norm(dim=1).mean()):.2f})  "
                        f"â유한={bool(torch.isfinite(x_j).all())}  "
                        f"‖r0‖={float((vs0-vt0).flatten(1).norm(dim=1).mean()):.4f}")
                self._xt_sanity = False

        # 이미 backward 했으므로 반환값은 로그용이다. 그래프가 없어 학습 루프가
        # 다시 더해도 그래디언트에 기여하지 않는다.
        L_lvl = sum(lvl) / len(lvl)          # 길이 k 의 스칼라 리스트 -> 스칼라 ()
        out = (self.lam_lvl * L_lvl).detach() # 스칼라 () — 로그용, 그래프 없음
        self.step += 1
        self.t_anchor += time.perf_counter() - t0w
        if self.step % self.a.log_every_anchor == 0:
            rec = {"task": k, "step": self.step, "L_level": float(L_lvl.detach()),
                   "n_past": len(self.books), "K": self.K,
                   "ms_per_step": 1000 * self.t_anchor / max(self.step, 1)}
            self.log.write(json.dumps(rec) + "\n"); self.log.flush()
            logging.info(f"[L2CB] k={k} step={self.step:5d} L_lvl={rec['L_level']:.4f} "
                         f"ms/step={rec['ms_per_step']:.1f}")
        return out

    # ── 앵커 손실: teacher 를 과거 태스크 전체에 대해 한 번에 ──────────────────
    def _lang_T(self, instructions, j, n):
        """teacher 의 언어 벡터. teacher 도 문자열도 stage 내내 고정이라 상수다.

        naive 경로는 이걸 매 스텝 태스크마다 다시 계산한다. LanguageEncoder 에 문자열
        캐시가 있어 비싸진 않지만, 상수를 반복 계산할 이유는 없다.
        """
        if j not in self._langT_cache:
            with torch.no_grad():
                v = encode_lang(self.teacher, [instructions[f"task{j}"]])   # (1,512)
            self._langT_cache[j] = v.detach()
        return self._langT_cache[j].expand(n, -1)

    def _loss_batched(self, policy, batch, x_t, t, k, instructions,
                      device, cls, n, st_shape, tcol, t0w):
        """naive 와 같은 계산을 하되 **teacher 쪽만** 과거 태스크 전체로 묶는다.

        왜 되는가
          · teacher 는 no_grad 라 활성값을 안 남긴다 -> 배치를 k 배로 키워도 VRAM 이
            늘지 않는다. 반면 student 는 backward 그래프를 들고 있어야 하므로
            태스크별 chunk backward 를 그대로 둔다(활성값 O(1) 유지).
          · 그 결과 teacher forward 가 스텝당 3k 회(cond_tail+velocity x2) -> 3 회가 된다.

        순서 주의
          난수(합성 좌표·ε₀·ε₁)는 **네트워크 호출 전에** naive 와 같은 순서로 전부
          뽑는다. 그래야 dropout 을 끈 상태에서 naive 와 값이 일치해 검증이 된다.
        """
        js = sorted(self.books)
        teach = self.teacher

        # ── 1) 난수부터. naive 와 같은 순서: 태스크마다 (표본, ε₀, ε₁) ──────────
        draw = []
        for j in js:
            s_t, o_t = self._sample(self.books[j], n)               # (n,16), (n,3072)
            eps0 = torch.randn_like(x_t)                            # (n,H,A)
            eps1 = torch.randn_like(x_t)                            # (n,H,A)
            draw.append((j, s_t.detach(), o_t.detach(), eps0, eps1))

        # ── 2) teacher: 과거 태스크를 세로로 이어 붙여 한 번씩만 ────────────────
        S_all = torch.cat([s.view(st_shape).to(batch["observation.state"].dtype)
                           for _, s, _, _, _ in draw])              # (k*n,S,Ds)
        CLS_all = torch.cat([o.reshape(-1, cls.shape[-1]).to(x_t.dtype)
                             for _, _, o, _, _ in draw])            # (k*n*4,768)
        eps0_all = torch.cat([e0 for *_, e0, _ in draw])            # (k*n,H,A)
        eps1_all = torch.cat([e1 for *_, e1 in draw])               # (k*n,H,A)
        t_all = t.repeat(len(js))                                   # (k*n,)
        lang_T = torch.cat([self._lang_T(instructions, j, n) for j in js])   # (k*n,512)

        # teacher 를 메모리에 맞는 청크로 나눠 돈다. 0 이면 전체를 한 번에.
        # ★ no_grad 라도 forward 중간 텐서는 배치에 비례한다 — 통째로 넣으면 OOM 날 수 있다.
        rows = S_all.shape[0]
        chunk = getattr(self.a, "anchor_teacher_chunk", 0) or rows
        tcol_all = t_all[:, None, None]
        x_parts, vt_parts = [], []
        with torch.no_grad():
            for c0 in range(0, rows, chunk):
                c1 = min(c0 + chunk, rows)
                sl_c, sl_cls = slice(c0, c1), slice(c0 * 4, c1 * 4)
                bb_c = dict(batch)
                bb_c["observation.state"] = S_all[sl_c]
                tl_T = cond_tail(teach, bb_c, CLS_all[sl_cls])       # (c,2064)
                cond_T = make_cond(lang_T[sl_c], tl_T)               # (c,2576)
                a_hat = (eps0_all[sl_c] + teach.dit_flow.velocity_net(
                    noisy_actions=eps0_all[sl_c], time=torch.zeros_like(t_all[sl_c]),
                    global_cond=cond_T)).detach()                    # (c,H,A)
                x_c = ((1 - tcol_all[sl_c]) * eps1_all[sl_c] + tcol_all[sl_c] * a_hat).detach()
                vt_c = teach.dit_flow.velocity_net(
                    noisy_actions=x_c, time=t_all[sl_c], global_cond=cond_T)
                x_parts.append(x_c); vt_parts.append(vt_c)
        x_all = torch.cat(x_parts)                                   # (k*n,H,A)
        vt_all = torch.cat(vt_parts)                                 # (k*n,H,A)

        # ── 3) student: 태스크별로 잘라 기존과 똑같이 (chunk backward 유지) ─────
        lvl = []
        for i, (j, s_t, o_t, _e0, _e1) in enumerate(draw):
            sl = slice(i * n, (i + 1) * n)
            bb = dict(batch)
            bb["observation.state"] = S_all[sl]
            cls_j = CLS_all[i * n * 4:(i + 1) * n * 4]
            past = [instructions[f"task{j}"]] * n
            tl_S = cond_tail(policy, bb, cls_j)                     # (n,2064)
            cond_S = make_cond(encode_lang(policy, past), tl_S)     # (n,2576)
            vs0 = policy.dit_flow.velocity_net(
                noisy_actions=x_all[sl], time=t, global_cond=cond_S)
            vt0 = vt_all[sl]
            resid = vs0 - vt0.to(vs0.dtype)
            if self.a.anchor_weight == "speedinv":
                w = speed_weights(vt0, eps1_all[sl],
                                  self._c_of(self.books[j]), self.a.w_clip,
                                  getattr(self.a, "anchor_wnorm", "mean"))
                L_j = weighted_sq(resid, w)
            else:
                L_j = self.reduce_level(resid)
            (LAMBDA_ANCHOR * self.lam_lvl * L_j).backward()
            lvl.append(L_j.detach())

        L_lvl = sum(lvl) / len(lvl)
        out = (self.lam_lvl * L_lvl).detach()
        self.step += 1
        self.t_anchor += time.perf_counter() - t0w
        if self.step % self.a.log_every_anchor == 0:
            rec = {"task": k, "step": self.step, "L_level": float(L_lvl.detach()),
                   "n_past": len(self.books), "K": self.K, "sched": "batched",
                   "ms_per_step": 1000 * self.t_anchor / max(self.step, 1)}
            self.log.write(json.dumps(rec) + "\n"); self.log.flush()
            logging.info(f"[L2CB] k={k} step={self.step:5d} L_lvl={rec['L_level']:.4f} "
                         f"ms/step={rec['ms_per_step']:.1f} (batched)")
        return out

    # ── 태스크 종료: 코드북 빌드 + teacher ──────────────────────────────────
    def on_task_end(self, policy, k, args, instructions, device, **kw):
        n = min(self._seen, self.n_pairs)      # N = 실제로 채워진 저수지 행 수
        s = self._res_s[:n].to(device)         # (N,16)
        o = self._res_o[:n].to(device)         # (N,3072)
        t0 = time.perf_counter()
        cb = build_codebook(s, o, self.K, seed=getattr(args, "seed", 42),
                            h_scale=self.h_scale, ridge_rho=self.ridge_rho,
                            grad_min_frames=self.grad_min_frames)
        dt = time.perf_counter() - t0
        # v4 앵커 가중의 c_j. 액션은 MIN_MAX 로 [-1,1] 이라 태스크 간 스케일이 같다.
        cb["speed_med2"] = (float(torch.cat(self._sp2).median()) if self._sp2 else 1.0)
        p = self.out / "codebooks" / f"codebook_task{k}.pt"
        torch.save({kk: vv for kk, vv in cb.items()}, p)
        nk = cb["n_k"]
        logging.info(
            f"[L2CB] task {k} 코드북  N={cb['N']} (본 프레임 {self._seen})  "
            f"K_eff={cb['K_eff']} (병합 {cb['n_merged']})  "
            f"n_k min/med/max={int(nk.min())}/{int(nk.median())}/{int(nk.max())}  "
            f"h={cb['h']:.4f}  {dt:.1f}s  {p.stat().st_size/1e6:.2f}MB"
            + (f"  |  grad: 강등 {cb['n_demoted']}셀  "
               f"‖A‖_F med/max={float(cb['A_fro'].median()):.2f}/"
               f"{float(cb['A_fro'].max()):.2f}  "
               f"분산비(0차/1차) med={float(cb['var_ratio_cells'].median()):.2f}  "
               f"speed_med2={cb['speed_med2']:.4f}"))
        self.books[k] = to_device_cb(cb, device)

        if k == 0 and not self._sanity_done:
            self._sanity_done = True
            try:
                self._sanity_suite(policy, cb, s, o, k, device, instructions, args)
            except Exception as e:
                logging.warning(f"[L2CB][sanity] 실패(실행은 계속): {type(e).__name__}: {e}")

        del self.teacher
        self.teacher = snapshot(policy, args.teacher_bf16)
        self._langT_cache.clear()      # ★ teacher 가 바뀌었다 — 캐시된 언어 벡터는 낡았다
        self._res_s = self._res_o = self._res_id = None
        self._sp2 = []
        torch.cuda.empty_cache()
        logging.info(f"[L2CB] task {k} teacher 갱신 (코드북 {len(self.books)}개)")

    # ── --resume: 끝난 스테이지의 코드북 + teacher 복원 ───────────────────────
    def restore(self, policy, n_done, args, device):
        """on_task_end(n_done-1) 직후 상태를 되살린다.

        코드북은 on_task_end 가 저장한 파일 그대로다(speed_med2 포함). teacher 는
        방금 끝난 정책의 스냅샷인데, policy 가 바로 그 체크포인트를 읽은 것이므로
        스냅샷을 새로 떠도 같다(float32 왕복).
        """
        for j in range(n_done):
            p = self.out / "codebooks" / f"codebook_task{j}.pt"
            cb = torch.load(p, map_location="cpu", weights_only=False)
            self.books[j] = to_device_cb(cb, device)
        del self.teacher
        self.teacher = snapshot(policy, args.teacher_bf16)
        self._langT_cache.clear()    # ★ teacher 가 새로 생겼다
        self._sanity_done = True     # task 0 sanity 는 원래 실행에서 이미 돌았다
        torch.cuda.empty_cache()
        logging.info(f"[L2CB] resume — 코드북 {sorted(self.books)} 복원, teacher = 스테이지 {n_done-1} 체크포인트")

    def _extra_sanity(self, cb, cbd, s_real, o_real, device, W, X, r_real):
        """파생 팔이 sanity 를 덧붙이는 지점. 기본은 없음."""
        return []

    # ── §4 기울기 실재 진단 (v2 전용) ───────────────────────────────────────
    @torch.no_grad()
    def _grad_sanity(self, cb, cbd, s_real, o_real, device, W, X, r_real):
        # 입력  s_real (N,16),  o_real (N,3072),  W (17,3072) ridge 계수,  X (N,17)
        L = ["", "── §4 기울기(A_k) 진단 ──"]
        mean_s, std_s = cbd["mean_s"], cbd["std_s"]        # 각 (16,)
        zs = (s_real - mean_s) / std_s                     # (N,16)
        lab = torch.cdist(zs, cbd["c"]).argmin(1)          # (N,Ke) -> (N,) hard 배정
        zbar, m, A = cbd["zbar"], cbd["m"], cbd["A"].float()  # (Ke,16),(Ke,3072),(Ke,3072,16)
        K = m.shape[0]                                     # Ke

        corrs, r2_0, r2_1, vr, anorm, resid0 = [], [], [], [], [], []   # 셀별 스칼라 리스트
        ss0 = ss1 = sst = 0.0
        for kk in range(K):
            sel = (lab == kk).nonzero().squeeze(1)         # (n_k,)
            if sel.numel() < 20:
                continue
            d = zs[sel] - zbar[kk]                         # (n_k,16)
            Y = o_real[sel] - m[kk]                        # (n_k,3072)
            # 1a. 셀 내 ‖Y‖ vs ‖δ‖ 상관
            a_, b_ = Y.norm(dim=1), d.norm(dim=1)          # 각 (n_k,)
            corrs.append(float(((a_ - a_.mean()) * (b_ - b_.mean())).mean()
                               / (a_.std(unbiased=False) * b_.std(unbiased=False)).clamp_min(1e-12)))
            # 1b/1c. 셀 내 80/20
            pm = torch.randperm(sel.numel(), device=device)   # (n_k,)
            ntr = max(int(0.8 * sel.numel()), 16)
            tr, te = pm[:ntr], pm[ntr:]                    # (n_tr,), (n_te,)
            if te.numel() < 4:
                continue
            dt, Yt = d[tr], Y[tr]                          # (n_tr,16), (n_tr,3072)
            S = dt.T @ dt                                  # (16,16)
            lam = cb.get("ridge_rho", 0.05) * float(S.diagonal().sum()) / d.shape[1]  # 스칼라
            # (3072,n_tr)@(n_tr,16) @ (16,16) -> (3072,16)
            Ak = (Yt.T @ dt) @ torch.linalg.inv(S + lam * torch.eye(d.shape[1], device=device))
            m_tr = Yt.mean(0)                              # (3072,)
            e0 = Y[te] - m_tr                              # (n_te,3072) 0차 예측 잔차
            e1 = Y[te] - (m_tr + d[te] @ Ak.T)             # (n_te,16)@(16,3072) -> (n_te,3072)
            ss0 += float(e0.pow(2).sum()); ss1 += float(e1.pow(2).sum())
            sst += float((Y[te] - Yt.mean(0)).pow(2).sum())
            vr.append(float(e0.pow(2).sum() / e1.pow(2).sum().clamp_min(1e-12)))
            anorm.append(float((d[te] @ Ak.T).norm(dim=1).median()))
            resid0.append(float(e0.norm(dim=1).median()))
        import statistics as st
        q = lambda v, p: float(torch.tensor(v).quantile(p))
        R2_0 = 1 - ss0 / max(sst, 1e-12)
        R2_1 = 1 - ss1 / max(sst, 1e-12)
        gap = R2_1 - R2_0
        vr_med = st.median(vr) if vr else 0.0
        L.append(f"1a 셀내 corr(‖o−m‖,‖δ‖)  중앙값={st.median(corrs):.3f}  "
                 f"Q1/Q3={q(corrs,0.25):.3f}/{q(corrs,0.75):.3f}  (양수 = 기울기 실재)")
        L.append(f"1b held-out R²  0차={R2_0:.4f}  1차={R2_1:.4f}  gap={gap:+.4f}")
        L.append(f"1c 셀별 분산비(0차/1차) 중앙값={vr_med:.3f}  "
                 f"Q1/Q3={q(vr,0.25):.3f}/{q(vr,0.75):.3f}")
        gate = (gap >= 0.05) or (vr_med >= 1.2)
        L.append(f"   ★게이트: gap≥0.05 또는 분산비≥1.2 -> "
                 f"{'통과 — 본 런 진행' if gate else '미달 — 본 런 보류'}")

        # 2. d_eff 3열 (실측 / grad-õ / v1-õ)
        def deff(x):
            """x: (n,D) -> (d_eff, 총분산). 둘 다 스칼라."""
            x = x - x.mean(0)                              # (n,D)
            lam = torch.linalg.svdvals(x.double()).pow(2) / (x.shape[0] - 1)  # (min(n,D),)
            return float(lam.sum() ** 2 / lam.pow(2).sum()), float(lam.sum())
        cb_v1 = {kk: vv for kk, vv in cbd.items() if kk != "A"}
        cb_v1["A"] = None                                  # A 를 끈 v1 대조군
        g = torch.Generator(device=device).manual_seed(1234)
        s1, o1 = sample_codebook(cbd, 1000, g)             # (1000,16), (1000,3072) grad 판
        g = torch.Generator(device=device).manual_seed(1234)
        s0, o0 = sample_codebook(cb_v1, 1000, g)           # (1000,16), (1000,3072) v1 판
        sel = torch.randperm(o_real.shape[0], device=device)[:1000]   # (1000,)
        dr, vrr = deff(o_real[sel]); dg, vg = deff(o1); dv, vv_ = deff(o0)
        L += ["2 d_eff / total variance",
              f"   {'':10}{'d_eff':>10}{'var':>12}{'d_eff 비':>10}",
              f"   {'실측':10}{dr:10.2f}{vrr:12.1f}{1.0:10.2f}",
              f"   {'grad-õ':10}{dg:10.2f}{vg:12.1f}{dg/dr:10.2f}"
              f"   {'OK' if abs(dg/dr-1) <= 0.20 else '★±20% 이탈★'}",
              f"   {'v1-õ':10}{dv:10.2f}{vv_:12.1f}{dv/dr:10.2f}"]

        # 3. 서브셀 정합 (v1 sanity 의 ridge f(s) 재사용)
        def rmse(ss, oo):
            """ss: (n,16), oo: (n,3072) -> 스칼라 RMSE(oo − f(ss))."""
            Xs = torch.cat([ss, torch.ones(ss.shape[0], 1, device=device)], 1).double()  # (n,17)
            return float((oo.double() - Xs @ W).pow(2).mean().sqrt())   # (n,17)@(17,3072)
        rg, rv = rmse(s1, o1), rmse(s0, o0)
        L += [f"3 서브셀 정합  RMSE(õ−f(s̃))  grad={rg:.4f}  v1={rv:.4f}  "
              f"실측 잔차={r_real:.4f}",
              f"   비(실측 대비)  grad={rg/max(r_real,1e-12):.2f}  v1={rv/max(r_real,1e-12):.2f}  "
              f"({'개선' if rg < rv else '개선 없음'})"]

        # 4. 외삽 안전
        L.append(f"4 외삽 안전  ‖A_k δ‖ 중앙값={st.median(anorm):.4f}  "
                 f"0차 잔차 중앙값={st.median(resid0):.4f}  "
                 f"비={st.median(anorm)/max(st.median(resid0),1e-12):.3f} (<1 정상)")
        self._grad_gate = gate
        return L

    # ── Sanity (§5) — task0 종료 시 1회, 로그만 ─────────────────────────────
    @torch.no_grad()
    def _sanity_suite(self, policy, cb, s_real, o_real, k, device, instructions, args):
        L = ["", "═" * 74, f"[L2CB][sanity] task{k} — 본 런 전 점검", "═" * 74]

        # 5.1 수집 짝 검증 — 저장한 (ep, frame) 을 데이터셋에서 다시 읽어 대조
        ok = bad = 0
        if self._dataset is not None and self._prep is not None:
            idx = torch.randperm(min(self._seen, self.n_pairs))[:20]   # (20,) 표본 행
            ei = self._dataset.episode_data_index
            for i in idx.tolist():
                ep, fr = int(self._res_id[i, 0]), int(self._res_id[i, 1])
                gidx = int(ei["from"][ep]) + fr
                raw = self._dataset[gidx]
                bb = to_device({kk: (vv.unsqueeze(0) if torch.is_tensor(vv) else [vv])
                                   for kk, vv in raw.items()}, device)
                bb = self._prep(policy, bb)
                # bb 는 배치 크기 1 짜리 배치다 (state (1,S,Ds), images (1,S,M,C,H,W))
                s_ref = bb["observation.state"].flatten(1)[0].float().cpu()   # (1,16) -> (16,)
                o_ref = rgb_cls(policy, bb).reshape(1, -1)[0].float().cpu()   # (4,768)->(1,3072)->(3072,)
                ds = float((s_ref - self._res_s[i]).norm() / s_ref.norm().clamp_min(1e-8))
                do = float((o_ref - self._res_o[i]).norm() / o_ref.norm().clamp_min(1e-8))
                if ds < 1e-3 and do < 1e-3:
                    ok += 1
                else:
                    bad += 1
                    if bad <= 3:
                        L.append(f"  ✗ i={i} ep{ep} fr{fr}  Δs={ds:.2e}  Δo={do:.2e}")
            L.append(f"5.1 짝 검증  일치 {ok}/20" +
                     ("  ★짝 어긋남 — 치명★" if bad else "  OK"))
        else:
            L.append("5.1 짝 검증  건너뜀 (dataset/prep 없음)")

        # 5.2 빌드 지표
        nk = cb["n_k"]
        L.append(f"5.2 빌드  N={cb['N']}  K_eff={cb['K_eff']}(병합 {cb['n_merged']})  "
                 f"n_k {int(nk.min())}/{int(nk.median())}/{int(nk.max())}  h={cb['h']:.4f}")

        # 5.3 합성 1000개
        cbd = to_device_cb(cb, device)
        s_syn, o_syn = sample_codebook(cbd, 1000)   # (1000,16), (1000,3072)
        # a. s̃ 주변통계 — 차원별 (16,) 비교량을 평균해 스칼라로 만든다
        rel_m = float(((s_syn.mean(0) - s_real.mean(0)).abs()
                       / s_real.std(0).clamp_min(1e-8)).mean())
        rel_s = float(((s_syn.std(0) - s_real.std(0)).abs()
                       / s_real.std(0).clamp_min(1e-8)).mean())
        L.append(f"5.3a s̃ 주변통계  |Δmean|/σ={rel_m:.4f}  |Δstd|/σ={rel_s:.4f} "
                 f"(<0.10 기대)  {'OK' if max(rel_m, rel_s) < 0.10 else '★이탈★'}")

        # b. d_eff = (Σλ)²/Σλ² , total variance
        def deff(x):
            """x: (n,D) -> (d_eff, 총분산). 둘 다 스칼라."""
            x = x - x.mean(0)                              # (n,D)
            lam = torch.linalg.svdvals(x.double()).pow(2) / (x.shape[0] - 1)  # (min(n,D),)
            return float(lam.sum() ** 2 / lam.pow(2).sum()), float(lam.sum())
        m = min(1000, o_real.shape[0])
        sel = torch.randperm(o_real.shape[0])[:m]   # (m,)
        d_r, v_r = deff(o_real[sel])                # (m,3072) -> 스칼라 2개
        d_s, v_s = deff(o_syn[:m])                  # (m,3072) -> 스칼라 2개
        ok_d = abs(d_s - d_r) / max(d_r, 1e-9) <= 0.20
        ok_v = 0.8 <= v_s / max(v_r, 1e-12) <= 1.2
        L.append(f"5.3b o 기하  d_eff 실측={d_r:.1f} 합성={d_s:.1f} (비 {d_s/d_r:.2f}, ±20% 게이트 "
                 f"{'OK' if ok_d else '★이탈★'})   var 비={v_s/v_r:.2f} "
                 f"({'OK' if ok_v else '★이탈★'})")

        # b2. w 의 확산 정도 + 분산 성분 분해 (대역폭 진단)
        zz = (s_syn - cbd["mean_s"]) / cbd["std_s"]    # (1000,16)
        wv = torch.softmax(-torch.cdist(zz, cbd["c"]).pow(2) / cbd["h"] ** 2, 1)   # (1000,Ke)
        pr = float((1.0 / wv.pow(2).sum(1)).median())  # (1000,) 참여셀 수 -> 스칼라 중앙값
        omu = wv @ cbd["m"]                            # (1000,Ke)@(Ke,3072) -> (1000,3072)
        osd2 = (wv @ cbd["sig_o"].pow(2))              # (1000,3072) 잡음 분산
        L.append(f"5.3b2 대역폭  h={cbd['h']:.4f}(scale {cb.get('h_scale',1.0)})  "
                 f"w 참여셀 median={pr:.1f}/{cb['K_eff']}  최대가중 median="
                 f"{float(wv.max(1).values.median()):.3f}")
        L.append(f"      분산 분해  평균부={float(omu.var(0).sum()):.1f}  "
                 f"잡음부={float(osd2.sum(1).mean()):.1f}  실측 total={float(o_real.var(0).sum()):.1f}  "
                 f"(평균부 비중 {float(omu.var(0).sum())/max(float(o_real.var(0).sum()),1e-9):.2f})")

        # c. ridge  o ≈ f(s)  (80/20)
        N = s_real.shape[0]
        pm = torch.randperm(N, device=device)          # (N,)
        ntr = int(0.8 * N)
        tr, te = pm[:ntr], pm[ntr:]                    # (n_tr,), (n_te,)
        X = torch.cat([s_real, torch.ones(N, 1, device=device)], 1).double()  # (N,17) 편향항 포함
        Y = o_real.double()                            # (N,3072)
        Xm, Ym = X[tr], Y[tr]                          # (n_tr,17), (n_tr,3072)
        lam = 1e-3 * float(Xm.shape[0])                # 스칼라
        A = Xm.T @ Xm + lam * torch.eye(X.shape[1], device=device, dtype=torch.float64)  # (17,17)
        W = torch.linalg.solve(A, Xm.T @ Ym)           # (17,17)⁻¹@(17,3072) -> (17,3072)
        res_te = Y[te] - X[te] @ W                     # (n_te,3072) held-out 잔차
        ss_res = float(res_te.pow(2).sum())            # 스칼라
        ss_tot = float((Y[te] - Ym.mean(0)).pow(2).sum())   # 스칼라
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        d_res, _ = deff(res_te.float())                # (n_te,3072) -> 스칼라
        d_pool, _ = deff(Y[te].float())                # (n_te,3072) -> 스칼라
        L.append(f"5.3c ridge o≈f(s)  held-out R²={r2:.4f}  "
                 f"잔차 d_eff={d_res:.1f} vs 풀드 {d_pool:.1f} (비 {d_res/max(d_pool,1e-9):.2f})")
        gate_fail = (r2 < 0.15) and (d_res / max(d_pool, 1e-9) > 0.9)
        L.append(f"     게이트: {'★전제 붕괴 — s 가 o 를 설명하지 못한다★' if gate_fail else 'OK'}")

        # d. 합성 정합
        Xs = torch.cat([s_syn, torch.ones(s_syn.shape[0], 1, device=device)], 1).double()  # (1000,17)
        r_syn = float((o_syn.double() - Xs @ W).pow(2).mean().sqrt())   # (1000,3072) -> 스칼라
        r_real = float(res_te.pow(2).mean().sqrt())                     # 스칼라
        L.append(f"5.3d 합성 정합  RMSE(õ−f(s̃))={r_syn:.4f}  실측 잔차 RMSE={r_real:.4f}  "
                 f"비={r_syn/max(r_real,1e-12):.2f} (1~2 기대)")

        # ── §4 v2 기울기 sanity ────────────────────────────────────────
        L += self._grad_sanity(cb, cbd, s_real, o_real, device, W, X, r_real)

        L += self._extra_sanity(cb, cbd, s_real, o_real, device, W, X, r_real)

        # 5.4 teacher 스모크
        if self.teacher is not None:
            L.append("5.4 teacher 스모크  건너뜀 (teacher 는 이 뒤에 만들어진다)")
        else:
            L.append("5.4 teacher 스모크  task0 종료 시점엔 teacher 가 없다 — task1 sanity 로 대체")

        L.append("═" * 74)
        logging.info("\n".join(L))
        (self.out / "sanity_task0.txt").write_text("\n".join(L) + "\n")
        json.dump({"pair_ok": ok, "pair_bad": bad, "K_eff": cb["K_eff"], "h": cb["h"],
                   "rel_mean": rel_m, "rel_std": rel_s,
                   "d_eff_real": d_r, "d_eff_syn": d_s, "var_ratio": v_s / v_r,
                   "ridge_r2": r2, "resid_deff": d_res, "pooled_deff": d_pool,
                   "resid_ratio": d_res / max(d_pool, 1e-9),
                   "synth_rmse": r_syn, "real_rmse": r_real, "gate_fail": bool(gate_fail)},
                  (self.out / "sanity_task0.json").open("w"), indent=2)


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_args()
    _forbid_time_bin()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor = NoAnchor(args) if args.arm == "seq_ft" else L2CodebookAnchor(args)

    json.dump({
        "arm": {"v4": "TETHER (v4) — hard_linear + speedinv",
                "v3": "L2_codebook+grad (v3)",
                "seq_ft": "Seq-FT — 앵커 없음"}[args.arm],
        "sampler_mode": args.sampler_mode, "anchor_weight": args.anchor_weight,
        "kappa": args.kappa, "anchor_c": args.anchor_c, "anchor_wnorm": args.anchor_wnorm,
        "w_clip": args.w_clip, "base": "L2",
        "base_diff": [
            "1. 앵커의 (o,s) 공급원을 시간-bin 가우시안에서 (s,o) 코드북으로 교체. 시간 축 제거.",
            "2. task j 학습 중 같은 프레임의 (state 16, CLS 3072) 쌍을 저수지 표본으로 수집.",
            "3. task 종료 시 표준화 s 공간 k-means(K=96) -> 셀별 (π, μ_s, σ_s, c, m, σ_o, A_k) 저장.",
            "4. 앵커에서 k~Cat(π) -> s̃ -> softmax(−‖z(s̃)−c‖²/h²) 가중으로 õ. s̃ 가 õ 를 고른다.",
            "5. teacher 부트스트랩 x_t·λ·reduction·teacher 운용·eval 은 L2 와 동일.",
        ],
        "codebook_k": args.codebook_k, "n_pairs": args.n_pairs, "h_scale": args.h_scale,
        "grad_enable": True, "ridge_rho": args.ridge_rho,
        "grad_min_frames": args.grad_min_frames,
        "full_cov_s": False, "seed": args.seed,
        "xt_mode": "teacher", "lambda_level": args.lambda_level,
        "anchor_norm": "mean", "chunk_backward": True,
        "time_bin": "제거 (R10.compute_stats/phase_bins 를 예외로 차단)",
        "teacher": "rolling (1 snapshot)", "embedding": "dinov2_cls_768_frozen",
        "p_drop": P_DROP, "guidance_w": GUIDANCE_W, "mode": MODE,
    }, (out_dir / "l2_codebook_config.json").open("w"), indent=2, ensure_ascii=False)
    # l2_report.py 가 읽는 표준 이름으로도 남긴다
    shutil.copy(out_dir / "l2_codebook_config.json", out_dir / "l2_config.json")

    _train_loop(args, anchor)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
