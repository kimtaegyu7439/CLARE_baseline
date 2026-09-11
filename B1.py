#!/usr/bin/env python
"""B1 — condition dropout + counterfactual conditional anchoring.

가설: 순차 파인튜닝에서 조건부 속도장 v(x_t, t, o, ℓ) 이 현재 태스크의 marginal 로
붕괴한다("condition blindness"). 그래서 명령어 ℓ 을 바꿔도 출력이 안 변하고, 과거
태스크가 소실된다. B1 은 같은 학습 예산에서 두 가지만 추가해 이걸 막는다.

  (1) condition dropout — 확률 p_drop 으로 명령어를 NULL 로 바꿔 학습.
      unconditional 스트림 v(·, ∅) 를 살려 둔다.
  (2) counterfactual conditional anchoring — 현재 배치의 (x_t, t, o) 에 **과거**
      태스크 명령어 ℓ_j 를 물려서 현재 모델과 teacher(직전 태스크 종료 시점 스냅샷)
      출력을 L2 로 붙인다. 과거 관측/액션을 저장하지 않는 rehearsal-free 방식이다.

무엇을 어디서 그대로 가져왔는지는 results/B1/replication_notes.md 에 file:line 으로
적어 두었다. 이 파일은 저장소의 기존 파일을 하나도 건드리지 않는다.

사용법
    python B1.py --smoke
    python B1.py --mode baseline --smoke
    python B1.py                                # full run (ours)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
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

# E0(이전 seq-FT 실행)의 45에피소드 분할 규칙을 복제하지 않고 그대로 import 한다.
from lerobot.scripts.E0 import episode_sampler, split_episodes, to_device  # noqa: E402

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

NULL_TEXT = ""          # NULL 조건(빈 문자열). replication_notes.md §3 참조.
OUT_DIR = REPO / "results" / "B1"      # 지표·SR 행렬·진단 로그가 쌓이는 곳


# ═════════════════════════════════════════════════════════════════════════════
#  설정
# ═════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    # ── B1 의 두 가지 추가분 ─────────────────────────────────────────────────
    p.add_argument("--p_drop", type=float, default=0.1,
                   help="condition dropout 확률. 매 샘플 이 확률로 명령어를 ∅ 로 바꾼다. "
                        "무조건부 스트림 v(·,∅) 를 살리려는 것인데, 그 결과 v(·,∅) 가 "
                        "**현재 태스크**로 고정된다(results/B_default 참조).")
    p.add_argument("--lambda_anchor", type=float, default=1.0,
                   help="앵커 항의 가중치 λ. 최종 손실 L = L_FM + λ·L_anchor. "
                        "0 이면 앵커를 아예 계산하지 않는다.")
    p.add_argument("--anchor_agg", choices=["sum", "mean", "sample"], default="sum",
                   help="과거 태스크를 어떻게 모을지. "
                        "sum(기본): 0..k-1 전부의 합 -> 실효 가중치 k·λ. "
                        "mean: 전부의 평균 -> 실효 가중치 λ. "
                        "sample: 스텝당 하나만 균등 추첨(2026-08-24 이전 동작). "
                        "sample 은 배치 32개 전부에 같은 ℓ_j 를 물려 분산이 컸다.")
    p.add_argument("--mode", choices=["ours", "baseline"], default="ours",
                   help="baseline: p_drop=0, lambda_anchor=0 -> E0 λ=0 경로와 동일")

    # ── 평가 ─────────────────────────────────────────────────────────────────
    p.add_argument("--eval_episodes", type=int, default=20, help="SR 칸당 롤아웃 수 (E0와 동일)")
    p.add_argument("--eval_batch_size", type=int, default=20,
                   help="동시에 띄우는 시뮬 환경 수. 20 이면 한 배치에 다 들어간다.")
    p.add_argument("--eval_after_each_task", type=lambda s: s.lower() != "false", default=True,
                   help="False 면 학습만 하고 SR 행렬을 비워 둔다(나중에 따로 프로브할 때).")
    p.add_argument("--guidance_w", type=float, default=1.0,
                   help="평가 시 CFG. v = v(∅) + w*(v(ℓ)-v(∅)). w=1이면 표준 조건부 추론.")

    p.add_argument("--task_order", default=None,
                   help="학습 순서. 예 \"1,0,2,3\". 기본은 0,1,..,K-1. "
                        "SR 행렬과 teacher 인덱스는 **스테이지 기준**으로 유지되고 "
                        "데이터셋·환경만 이 순서를 따른다(다른 팔과 지표 정의가 같아진다).")
    # ── 실행 환경 ────────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42,
                   help="학습 RNG 시드이자 롤아웃 시작 시드. eval_policy(start_seed=) 로 "
                        "들어가 환경 초기 상태 20개를 정한다.")
    p.add_argument("--log_every", type=int, default=100, help="진단 기록 주기(스텝)")
    p.add_argument("--no_diag", action="store_true",
                   help="조건 민감도 δ 진단을 끈다. 알고리즘에는 영향이 없고 "
                        "diagnostics.jsonl 의 delta_cur/delta_prev 만 빠진다. "
                        "고정 배치(probe_batch, 이미지 포함 약 201MB)를 스테이지 내내 "
                        "들고 있는 것과 100스텝마다의 추가 forward 3회가 사라진다.")
    p.add_argument("--smoke", action="store_true", help="2태스크 x 100스텝 x 2에피소드")
    p.add_argument("--out_dir", default=str(OUT_DIR), help="지표/로그 출력 디렉토리")
    p.add_argument("--ckpt_root", default=str(REPO / "outputs" / "B1"),
                   help="체크포인트 루트. 실제 경로는 <root>/<suite>_seed<seed>_<mode>/task_<k>")
    p.add_argument("--device", default="cuda")
    p.add_argument("--teacher_bf16", action="store_true", help="OOM 시 teacher를 bf16으로")
    p.add_argument("--skip_verify", action="store_true", help="시작 시 등가성 검사 생략(권장 안 함)")
    args = p.parse_args()

    if args.mode == "baseline":
        args.p_drop, args.lambda_anchor = 0.0, 0.0
    if args.smoke:
        args.num_tasks = min(args.num_tasks, 2)
        args.steps_per_task = 100
        args.eval_episodes = 2
        args.eval_batch_size = 2
        if args.task_order:      # 순서도 함께 줄인다. 상대 순서는 보존한다.
            keep = [int(x) for x in args.task_order.split(",")
                    if int(x) < args.num_tasks]
            args.task_order = ",".join(str(x) for x in keep)
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
        job_name=f"B1_{args.mode}_task_{task_k}",
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
    batch = policy.normalize_inputs(batch)          # 관측을 통계로 정규화
    if policy.config.image_features:
        batch = dict(batch)
        # 카메라 여러 대를 한 축으로 쌓는다: (B, n_obs, n_cam, C, H, W)
        batch["observation.images"] = torch.stack(
            [batch[k] for k in policy.config.image_features], dim=-4
        )
    # batch["action"] 을 MIN_MAX 로 [-1,1] 에 넣는다. 이후 모든 수식이 이 스케일이다.
    return policy.normalize_targets(batch)


def encode_lang(policy, texts: list[str]) -> torch.Tensor:
    """명령어 -> 조건 벡터의 앞 512차원. LanguageEncoder 는 문자열 캐시가 있어 싸다."""
    dit = policy.dit_flow
    with torch.no_grad():
        emb = dit.language_encoder(texts)           # (B, 512) CLIP 텍스트 임베딩. 동결.
    # language_embedding_projection 은 학습 대상이므로 no_grad 밖에 둔다.
    return dit.language_embedding_projection(emb)   # (B, 512) cond 의 앞부분


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
    bsz, n_obs = batch["observation.state"].shape[:2]
    images = einops.rearrange(
        batch["observation.images"], "b s n ... -> (b s n) ...",
        b=bsz, s=n_obs, n=len(cfg.image_features),
    )
    with torch.no_grad():
        return dit.pretrained_rgb_encoder(images)


def shares_backbone(a, b) -> bool:
    """두 정책이 동결 이미지 백본을 객체째 공유하는가. bf16 teacher 는 아니다."""
    return a.dit_flow.pretrained_rgb_encoder is b.dit_flow.pretrained_rgb_encoder


def cond_tail(policy, batch: dict, cls: torch.Tensor | None = None) -> torch.Tensor:
    """조건 벡터에서 언어를 뺀 나머지 (상태 + 이미지). 명령어와 무관하다.

    cls 를 주면 DINOv2 를 건너뛴다. rgb_embedding_projection 은 **학습 대상**이라
    정책마다 다르므로 투영은 매번 다시 한다.
    """
    import einops

    dit = policy.dit_flow
    cfg = policy.config
    # bsz = 배치 크기 B,  n_obs = 관측 프레임 수 (n_obs_steps=2, 즉 [t-1, t])
    bsz, n_obs = batch["observation.state"].shape[:2]

    # 8차원 상태 x 2프레임 = 16
    parts = [batch["observation.state"].flatten(start_dim=1)]           # (B, 16)
    if cfg.image_features:
        if cls is None:
            # 배치·프레임·카메라를 한 축으로 접어 인코더에 한 번에 넣는다
            images = einops.rearrange(
                batch["observation.images"], "b s n ... -> (b s n) ...",
                b=bsz, s=n_obs, n=len(cfg.image_features),
            )
            with torch.no_grad():
                cls = dit.pretrained_rgb_encoder(images)      # DINOv2 CLS 토큰. 동결.
        cls_tokens = cls
        emb = dit.rgb_embedding_projection(cls_tokens)        # 학습 대상 투영
        # 다시 펼친다: 512 x 2프레임 x 2카메라 = 2048
        feats = einops.rearrange(
            emb, "(b s n) ... -> b s (n ...)", b=bsz, s=n_obs, n=len(cfg.image_features)
        )
        parts.append(feats.flatten(start_dim=1))                        # (B, 2048)
    if cfg.env_state_feature:                                # libero 에서는 안 쓴다
        parts.append(batch["observation.environment_state"].flatten(start_dim=1))
    return torch.cat(parts, dim=-1)                                     # (B, 2064)


def make_cond(lang_vec: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """(B,512) + (B,2064) -> (B,2576). concat 순서가 원본과 같아야 한다(verify 가 확인)."""
    return torch.cat([lang_vec, tail], dim=-1)


# ═════════════════════════════════════════════════════════════════════════════
#  Flow matching — modeling_dit_flow_mt.py:1281-1297 의 구성을 그대로 복제
# ═════════════════════════════════════════════════════════════════════════════
def sample_fm(policy, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x_t, t, target) 을 만든다. 앵커 항이 x_t, t 를 직접 잡아야 해서 복제했다."""
    dit = policy.dit_flow
    traj = batch["action"]                                   # a: (B,H,A) 정답 청크, [-1,1]
    noise = dit.velocity_net.sample_noise(traj.shape[0], traj.device)   # ε ~ N(0,I)
    t = dit.noise_distribution.sample((traj.shape[0],)).to(traj.device)  # t ~ U(0,1), (B,)
    # 직선 경로 위의 점. t=0 이면 순수 노이즈, t=1 이면 정답.
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * traj
    # 그 직선의 속도. t 와 무관한 상수 벡터라 목표가 "직선을 따라가라"가 된다.
    return x_t, t, traj - noise                              # (x_t, t, target = a - ε)


def fm_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """do_mask_loss_for_padding=False 경로. modeling_dit_flow_mt.py:1298-1320."""
    # 패딩 마스크를 쓰지 않는 경로. (B,H,A) 전 원소 평균이라 청크 길이로도 나뉜다.
    return F.mse_loss(pred, target, reduction="none").mean()



# ═════════════════════════════════════════════════════════════════════════════
#  앵커 타깃 제공자 — B2/B3 가 이 부분만 갈아 끼운다
# ═════════════════════════════════════════════════════════════════════════════
def teacher_tail(policy, teacher, batch, cls=None):
    """teacher 쪽 tail. 백본을 공유하면 student 의 CLS 토큰을 그대로 쓴다."""
    return cond_tail(teacher, batch, cls if shares_backbone(policy, teacher) else None)


def anchor_against(policy, teacher, batch, tail, x_t, t, instruction, t_tail=None):
    """현재 배치의 (x_t, t, o) 에 과거 명령어를 물려 student 와 teacher 를 붙인다.

    t_tail 을 주면 teacher 쪽 tail 계산을 건너뛴다. 과거 태스크 여러 개를 도는
    anchor_over_tasks 가 그렇게 쓴다 — tail 은 명령어와 무관하기 때문이다.
    """
    bsz = x_t.shape[0]
    past = [instruction] * bsz          # ℓ_j — 배치 전체에 같은 과거 명령어를 물린다
    # student: 현재 관측 o_k + 과거 명령어 ℓ_j. 이 조합은 학습 데이터에 없다(반사실).
    cond_j = make_cond(encode_lang(policy, past), tail)
    pred_j = policy.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond_j)
    with torch.no_grad():
        # teacher 는 rgb_embedding_projection 이 student 와 달라 tail 도 다르다.
        # 재사용분이 없으면 여기서 만든다.
        if t_tail is None:
            t_tail = teacher_tail(policy, teacher, batch)
        t_cond = make_cond(encode_lang(teacher, past), t_tail)
        v_tgt = teacher.dit_flow.velocity_net(          # 앵커 목표 v_T(x_t, t, o_k, ℓ_j)
            noisy_actions=x_t.to(t_cond.dtype), time=t, global_cond=t_cond
        ).to(pred_j.dtype)
    return F.mse_loss(pred_j, v_tgt)    # ‖v_θ(ℓ_j) - v_T(ℓ_j)‖²



def past_tasks(k: int, rng, agg: str) -> list[int]:
    """이번 스텝에서 앵커를 걸 과거 스테이지 목록.

      agg="sum"/"mean" : 0..k-1 **전부**.  스텝마다 모든 과거를 한 번씩 본다.
      agg="sample"     : 균등 추첨 하나 (구버전 동작. 재현용으로만 남긴다).

    ★ 구버전은 배치 32개 전부에 같은 ℓ_j 를 물렸다. 추첨이 몰리면 수백 스텝 동안
      특정 과거만 앵커되고 나머지는 방치돼 분산이 컸다. sum/mean 은 그 분산을
      없앤다. 대신 sum 은 앵커의 실효 가중치가 k배가 되므로(λ_eff = k·λ) 구버전과
      같은 λ 라도 세기가 다르다 — λ 를 다시 훑어야 비교가 성립한다.
    """
    if k == 0:
        return []
    if agg == "sample":
        return [rng.randrange(k)]
    return list(range(k))


def reduce_anchor(terms: list, agg: str, device):
    if not terms:
        return torch.zeros((), device=device)
    total = sum(terms)
    return total / len(terms) if agg == "mean" else total


def anchor_over_tasks(policy, teacher_of, batch, tail, x_t, t, k,
                      instructions, rng, args, device, cls=None):
    """과거 태스크들에 대한 앵커 손실을 모아 하나로 만든다.

    teacher_of(j) 가 스테이지 j 의 teacher 를 돌려준다. B1 은 j 와 무관하게 같은
    스냅샷을, B2 는 j 별 스냅샷을 준다. 같은 teacher 가 연속으로 나오면 tail 을
    다시 계산하지 않는다(DINOv2 절약).
    """
    agg = getattr(args, "anchor_agg", "sum")
    terms, cache = [], {}
    for j in past_tasks(k, rng, agg):
        teacher = teacher_of(j)
        if teacher is None:
            continue
        key = id(teacher)
        if key not in cache:
            cache[key] = teacher_tail(policy, teacher, batch, cls)
        terms.append(anchor_against(policy, teacher, batch, tail, x_t, t,
                                    instructions[f"task{j}"], t_tail=cache[key]))
    return reduce_anchor(terms, agg, device)


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


class RollingTeacher:
    """B1 기본 방식. 직전 태스크 종료 시점 스냅샷 하나만 유지한다.

    측정 결과 이 방식은 세대마다 앵커 목표가 오염된다(results/B1_drift/report.txt).
    B2(FrozenTeachers) / B3(CachedTargets) 가 그 지점을 바꾼다.
    """

    name = "rolling"

    def __init__(self):
        self.teacher = None            # 직전 태스크 종료 시점의 정책 스냅샷 (없으면 None)

    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        # k=0 은 과거가 없다. teacher 가 없거나 λ=0 이면 계산 자체를 건너뛴다.
        if k == 0 or self.teacher is None or args.lambda_anchor == 0:
            return torch.zeros((), device=device)
        # rolling 은 스냅샷이 하나뿐이라 어느 j 를 물어도 같은 teacher 다.
        return anchor_over_tasks(policy, lambda j: self.teacher, batch, tail,
                                 x_t, t, k, instructions, rng, args, device,
                                 cls=getattr(self, "cls", None))

    def on_task_end(self, policy, k, args, instructions, device, **kw):
        del self.teacher               # 이전 teacher 를 버린다 = 세대마다 목표가 갱신됨
        self.teacher = snapshot(policy, args.teacher_bf16)
        torch.cuda.empty_cache()

    def describe(self):
        return "rolling teacher — 최신 스냅샷 1개"


ANCHOR = RollingTeacher()      # B2/B3 는 이 전역을 자기 것으로 바꾼다


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
        ref_cond = policy.dit_flow._prepare_global_conditioning(prepped)
        mine = make_cond(encode_lang(policy, list(prepped["task"])), cond_tail(policy, prepped))
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
    cond = make_cond(encode_lang(policy, list(p2["task"])), cond_tail(policy, p2))
    x_t, t, target = sample_fm(policy, p2)
    my_loss = fm_loss(policy.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond), target)

    if not torch.allclose(ref_loss.detach(), my_loss.detach(), atol=1e-5, rtol=1e-4):
        raise AssertionError(
            f"복제한 flow matching 손실이 policy.forward 와 다르다 "
            f"(ref {float(ref_loss):.6f} vs mine {float(my_loss):.6f}). "
            "modeling_dit_flow_mt.py:compute_loss 가 바뀌었는지 확인하라."
        )
    logging.info(f"[B1][verify] flow-matching OK  ref={float(ref_loss):.6f} mine={float(my_loss):.6f}")


# ═════════════════════════════════════════════════════════════════════════════
#  평가 — E0.py:249-273 probe_sr 과 같은 호출
# ═════════════════════════════════════════════════════════════════════════════
def cfg_guidance(policy, w: float, lang_dim: int):
    """평가용 classifier-free guidance 훅.

    w == 1 이면 아무것도 하지 않는다(∅ forward 를 아예 건너뛴다) — 표준 조건부 추론과
    비트 단위로 같아야 하기 때문이다. w != 1 일 때만 velocity_net.forward 를 인스턴스
    수준에서 감싼다. velocity_net.sample() 이 self.forward 를 부르므로(:779) 이 방식이
    적분 루프 100스텝 전부에 적용된다.
    """
    if w == 1.0:
        return nullcontext()

    net = policy.dit_flow.velocity_net
    base = net.forward
    null_vec = encode_lang(policy, [NULL_TEXT]).detach()          # (1, 512)

    def guided(noisy_actions, time, global_cond):
        v_c = base(noisy_actions, time, global_cond)
        uncond = global_cond.clone()
        uncond[:, :lang_dim] = null_vec.to(global_cond.dtype).expand(global_cond.shape[0], lang_dim)
        v_u = base(noisy_actions, time, uncond)
        return v_u + w * (v_c - v_u)

    class _Patch:
        def __enter__(self):
            net.forward = guided

        def __exit__(self, *exc):
            net.forward = base
            return False

    return _Patch()


def rollout_sr(policy, cfg: TrainPipelineConfig, env_task: str, args, lang_dim: int) -> float | None:
    """시뮬레이터 롤아웃 성공률(%). E0 와 같은 인자로 부른다."""
    from lerobot.envs.factory import make_env
    from lerobot.scripts.eval import eval_policy

    env = None
    try:
        env_cfg = copy.deepcopy(cfg.env)
        env_cfg.task = env_task
        env = make_env(env_cfg, n_envs=args.eval_batch_size, use_async_envs=False)
        policy.eval()
        with torch.no_grad(), cfg_guidance(policy, args.guidance_w, lang_dim):
            info = eval_policy(env, policy, args.eval_episodes, start_seed=args.seed)
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
    last = [R[K - 1][i] for i in range(K)]   # 마지막 행 = 전부 배운 뒤의 최종 성능
    diag = [R[i][i] for i in range(K)]       # 대각선 = 각 태스크를 막 배웠을 때의 성능(습득)
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
def main() -> None:
    args = parse_args()
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
    ckpt_root = Path(args.ckpt_root) / f"{args.suite}_seed{args.seed}_{args.mode}"

    json.dump(vars(args), (out_dir / "config.json").open("w"), indent=2, ensure_ascii=False)
    logging.info(colored(f"[B1] mode={args.mode} p_drop={args.p_drop} "
                         f"lambda_anchor={args.lambda_anchor} "
                         f"anchor_agg={getattr(args, 'anchor_agg', 'sum')} tasks=0..{K-1} "
                         f"anchor={ANCHOR.describe()}", "green", attrs=["bold"]))

    policy = None            # 학습 중인 정책. 태스크 0 에서 한 번 만들고 계속 이어 쓴다.
    policy_path = None       # 다음 태스크의 초기값 경로. 처음엔 사전학습, 이후엔 직전 체크포인트.
    t_start = time.perf_counter()
    step_times: list[float] = []      # 스텝당 소요(초). 전체 소요 추정용.
    rollout_times: list[float] = []   # SR 칸 하나당 소요(초).

    for k in range(K):
        task_id = order[k]                 # 실제 태스크. k 는 스테이지 인덱스다.
        repo_id = f"{ds_prefix}{task_id}"
        # instructions 는 **스테이지 기준**으로 담는다. 앵커가 rng.randrange(k) 로
        # 과거 "스테이지"를 뽑으므로, 이렇게 해야 순서를 바꿔도 올바른 명령어가 나온다.
        instructions[f"task{k}"] = task_instruction(repo_id)
        json.dump(instructions, (out_dir / "instructions.json").open("w"), indent=2, ensure_ascii=False)

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
        lang_dim = policy.dit_flow.language_embedding_projection.out_features   # 512

        probe_batch = None           # 진단 δ 를 항상 같은 데이터에서 재기 위한 고정 배치

        # 태스크 시작 훅. 기본 팔(B1/B2/B7/B8/B9)에는 없으므로 아무 일도 하지 않는다.
        # R10 처럼 태스크 데이터 전체로 통계를 먼저 잡아야 하는 팔이 쓴다.
        if hasattr(ANCHOR, "on_task_start"):
            ANCHOR.on_task_start(policy, k, args, instructions, device,
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

            bsz = batch["action"].shape[0]              # B
            # DINOv2 CLS 토큰은 동결이라 값이 안 변한다. 한 번만 뽑아 student 와
            # 모든 teacher 가 돌려 쓴다(앵커가 과거 k개를 돌아도 DINOv2 는 1회).
            cls = rgb_cls(policy, batch)
            ANCHOR.cls = cls
            tail = cond_tail(policy, batch, cls)        # (B,2064) 명령어와 무관한 부분
            x_t, t, target = sample_fm(policy, batch)   # 이번 스텝의 flow 좌표와 정답 속도

            # ── (1) condition dropout ────────────────────────────────────────
            texts = list(batch["task"])                 # 배치에 실려 온 명령어 문자열들
            if args.p_drop > 0:
                # 샘플별 독립 추첨. 뽑힌 자리는 ∅ 로 바꾸되 target 은 그대로라,
                # v(·,∅) 가 **현재 태스크의 정답**을 맞추도록 학습된다.
                texts = [NULL_TEXT if rng.random() < args.p_drop else s for s in texts]
            cond = make_cond(encode_lang(policy, texts), tail)   # (B,2576)

            with torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
                pred = policy.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond)
                loss_fm = fm_loss(pred, target)

                # ── (2) counterfactual conditional anchoring ─────────────────
                # 타깃을 어디서 가져오는지는 ANCHOR 가 정한다(B1 기본 = rolling teacher).
                # B8 은 현재 태스크의 FM 정답 v*_k = a_k − ε 을 가중 계산에 쓰므로
                # 여기서 넘겨 준다. 쓰지 않는 팔은 그냥 무시한다(계산에 영향 없음).
                ANCHOR.fm_target = target
                loss_anchor = ANCHOR.loss(policy, batch, tail, x_t, t, k,
                                          instructions, rng, args, device)

                # 최종 손실. FM 항의 가중이 1 이므로 앞선 이론식의 μ=1 에 해당한다.
                loss = loss_fm + args.lambda_anchor * loss_anchor

            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)     # 클리핑 전에 AMP 스케일을 되돌린다
            grad_norm = torch.nn.utils.clip_grad_norm_(     # 클리핑 **전**의 노름을 돌려준다
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

        ANCHOR.on_task_end(policy, k, args, instructions, device,
                           dl_iter=dl_iter, cfg=cfg, prep=prep_batch)
        torch.cuda.empty_cache()

        # ── SR 평가: 태스크 0..k ────────────────────────────────────────────────
        if args.eval_after_each_task:
            for i in range(k + 1):
                r0 = time.perf_counter()
                sr = rollout_sr(policy, cfg, f"{env_prefix}{order[i]}", args, lang_dim)
                rollout_times.append(time.perf_counter() - r0)
                R[k][i] = sr
                logging.info(colored(
                    f"[B1] SR  after stage {k}(task {task_id})  on stage {i}(task {order[i]}) = {sr}",
                    "yellow"))
                write_outputs(out_dir, R, K, args, order)

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
            tail = cond_tail(policy, probe)
            x_t, t, _ = sample_fm(policy, probe)
            bsz = x_t.shape[0]
            net = policy.dit_flow.velocity_net

            v_null = net(noisy_actions=x_t, time=t,        # 명령어를 뺀 기본 출력 v(·,∅)
                         global_cond=make_cond(encode_lang(policy, [NULL_TEXT] * bsz), tail))
            v_cur = net(noisy_actions=x_t, time=t,         # 현재 태스크 명령어 v(·,ℓ_k)
                        global_cond=make_cond(encode_lang(policy, [instructions[f"task{k}"]] * bsz), tail))
            # δ_cur: p_drop 이 켜져 있으면 0 에 수렴한다 — 기본 출력이 곧 현재 태스크라서.
            out["delta_cur"] = float((v_cur - v_null).flatten(1).norm(dim=1).mean())
            if k >= 1:
                v_prev = net(noisy_actions=x_t, time=t,    # 첫 태스크 명령어 v(·,ℓ_0)
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
        "mode": args.mode, "p_drop": args.p_drop, "lambda_anchor": args.lambda_anchor,
        "anchor_agg": getattr(args, "anchor_agg", "sum"),
        "suite": args.suite, "num_tasks": K, "steps_per_task": args.steps_per_task,
        "episodes_per_task": args.episodes_per_task, "eval_episodes": args.eval_episodes,
        "seed": args.seed, "guidance_w": args.guidance_w,
    }
    json.dump(metrics, (out_dir / "metrics.json").open("w"), indent=2, ensure_ascii=False)
    return metrics


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
