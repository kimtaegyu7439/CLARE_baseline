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
# env_prefix = "_".join(w.capitalize() for w in suite.split("_")) + "_Task_"
# libero_spatial → ["libero","spatial"] → ["Libero","Spatial"] → "Libero_Spatial" → "Libero_Spatial_Task_"
# ═════════════════════════════════════════════════════════════════════════════

def suite_prefixes(suite: str) -> tuple[str, str]:
    """스위트 이름 -> (데이터셋 repo prefix, env task prefix). E0.sh:62-63 과 같은 규칙."""
    # libero_spatial -> "Libero_Spatial_Task_"  (env 이름),
    #                   "continuallearning/libero_spatial_image_task_"  (데이터셋 repo)
    env_prefix = "_".join(w.capitalize() for w in suite.split("_")) + "_Task_" # 첫 글자를 대문자로 바꿔서 재조립 + 데이터 경로 생성
    return f"continuallearning/{suite}_image_task_", env_prefix

def task_instructions(repo_id: str) -> str:
    tasks = LeRobotDatasetMetadata(repo_id).tasks
    if hasattr(tasks, "index"):
        return str(list(tasks.index)[0])
    return str(next(iter(tasks.values())))

def build_cfg(args, task_k: int, policy_path: str, output_dir: Path) -> TrainPipelineConfig:
    # 저장소 설정 객체 직접 조립
    
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

    cfg.optimizer = policy_cfg.get_optimizer.preset()
    cfg.scheduler = policy_cfg.get_scheduler_preset()
    cfg.checkpoint_path = None
    return cfg

def prep_batch(policy, batch: dict) -> dict:
    # DiTFlowMTPolicy.forward 전처리 3단계
    
    batch = policy.normalize_inputs(batch) # 관측을 통계로 정규화
    if policy.config.image_features:
        batch = dict(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in policy.config.image_features], dim=-4) # (B, n_obs, n_cam, C, H, W)
    
    # batch["action"]을 minmax 스케일링: [-1, 1]
    return policy.normalize_targets(batch)

def rgb_cls(policy, batch: dict) -> torch.Tensor | None:
    import einops

    dit = policy.dit_flow
    cfg = policy.config
    if not cfg.image_features:
        return None
    bsz, n_obs = batch["observation.state"].shape[:2]
    image = einops.rearrange(
        batch["observation_images"], "b s n ... -> (b s n) ...", # shape이 어떻게 될까?
        b=bsz, s=n_obs, n=len(cfg.image_features),
    )
    with torch.no_grad():
        return dit.pretrained_rgb_encoder

class RollingTeacher:
    # 종료 시점 하나만 유지 

    name = "rolling"
    
    def __init__(self):
        self.teacher = None # task 종료 직전 snapshot

    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        if k = 0 or self.teacher is None or args.lambda_anchor == 0:
            return torch.zeros((), device=device)
        return anchor_over_tasks(policy, lambda j: self.teacher, batch, tail, x_t, t, k, instructions, rng, args, device, cls=getattr(self, "cls", None)) # 함수

    def on_task_end(self, policy, k, args, instructions, device, **kw):
        del self.teacher # 이전 teacher버리기
        self.teacher = snapshot(policy, args.teacher_bf16) # 함수
        torch.cuda.empty_cache()

    def describe(self):
        return "rolling teacher" # 최신 스냅샷 1    

ANCHOR = RollingTeacher() 


def main() -> None:
    args = parse.args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    init_logging()
    
    device = get_safe_torch_device(args.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)
    rng = random.Random(args.seed) # 앵커 j 추첨?
    
    ds_prefix, env_prefix = suite_prefixes(args.suite)
    K = args.num_tasks
    order = ([int(x) for x in args.task_order.split(",") if args.task_order else list(range(args.num_tasks))])

    hold_out = None

    instructions: dict[str, str] = {} # task{k} : k가 stage index
    R: list[list[float | None]] = [[None]*K for _ in range(K)] # SR matrix
    diag_log: list[dict] = [] # step별 loss 및 delta기록
    ckpt_root = Path(args.ckpt_root) / f"{args.suite}_seed{args.seed}_{args.mode}"

    json.dump(vars(args), (out_dir / "config.json").open("W"), indent=2, ensure_ascii=False) # 왜 하는거지? vars? indent?, ensure ascii?

    policy = None
    policy_path = None
    t_start = time.perf_counter() # ?
    step_times: list[float] = [] # step당 소요 (초)
    rollout_times: list[float] = [] # SR 시간당 소요(초)

    for k in range(K):
        task_id = orker[k]
        repo_id = f"{ds_prefix}{task_id}"
        instructions[f"task{k}"]  =  task_instructions.json(repo_id)
        json.dump(instructions, (out_dir / "instruction.json").open("w"), indent=2, ensure_ascii=False) # 어디에 기록?
        

        if policy_path is None:
            import os 
            policy_path = os.environ.get("PRETRAIN_PATH") or str(
                Path(os.environ.get("CLARE_MODEL_ROOT", str(Path.home() / "Models"))) / "dit_flow_mt_libero90_pretrained"
            ) # PRETRAIN_PATH 를 ./bash/env.sh에 저장 후 각 실험에서 source로 불러옴. 

        stage_dir = ckpt_root / f"task_{k}"
        cfg = build_cfg(args, task_id, policy_path, stage_dir)

        dataset = make_dataset(cfg)
        
        if hold_out is None:
            total = dataset.meta.total_episodes
            holdout = total - args.episodes_per_task
        if holdout > 0: 
            train_eps, _ = split_episodes(cfg.dataset.repo_id, cfg.dataset.root, holdout)
        else:
            train_eps = list(range(dataset.meta.total_episodes))

        if policy is None:
            policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)

        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
        grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)   # AMP 손실 스케일러

        sampler = episode_sampler(cfg, dataset, train_eps)
        loader = torch.utils.DataLoader(
            dataset,
            num_workers = cfg.num_workers,
            batch_size = cfg.batch_size,
            sampler = sampler,
            pin_memory = device.type == "cuda",
            drop_last = False, # epoch돌 때 마지막 batch수보다 적은 데이터 쓸건지
            multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
            persistent_workers=cfg.num_workers > 0,
        )
        dl_iter = cycle(loader)
        lang_dim = policy.dit_fow.language_embedding_projection.out_features # 512

        probe_batch = None
        policy.train()

        for it in range(args.stpes.per_task):
            t0 = time.perf_counter() # 시간 ?
            batch = to_device(next(dl_iter), device)
            batch = prep_batch(policy, batch)
            
            bsz = batch["action"].shape[0] # B
            cls = rgb_cls(policy, batch)
            ANCHOR.cls = cls






        

    
