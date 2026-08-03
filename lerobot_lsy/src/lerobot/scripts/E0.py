#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""E0 — EWC의 λ를 바꿔 가며 MSE와 SR을 같이 재고 한 장으로 비교한다.

train.py 뼈대를 그대로 유지하고 세 가지만 더한다.
    1. EWC 2차 페널티:  loss = mse + 0.5·λ·Σ_i F_i (θ_i - θ*_i)²
       F(Fisher)와 θ*(앵커)는 이전 태스크 끝에서 계산해 파일로 다음 태스크에 넘긴다.
       λ=inf 는 파라미터 완전 동결(학습 스텝 0)로 처리한다.
    2. 태스크 k 학습이 끝나면 지금까지 본 모든 태스크 j에 대해
       held-out MSE(= flow matching loss)와 SR을 재서 JSONL 한 줄씩 남긴다.
    3. --plot_only 로 그 JSONL을 MSE / SR 두 패널 그림으로 만든다.

보고 싶은 것: λ를 올려도 MSE는 완만한데 SR은 절벽처럼 무너지는가.
"""

import json
import logging
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
import torch.multiprocessing as mp
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, get_safe_torch_device, has_method, init_logging
from lerobot.utils.wandb_utils import WandBLogger


@dataclass
class E0Config(TrainPipelineConfig):
    """train.py 인자 전부 + EWC/프로브용 인자."""

    # EWC
    ewc_lambda: float = 0.0                # 0=순차 파인튜닝, inf=완전 동결
    ewc_state_path: str | None = None      # 이전 태스크가 남긴 fisher+anchor
    fisher_batches: int = 100
    fisher_batch_size: int = 8

    # 프로브
    task_ids: str = ""                     # 프로브 대상 "0,1,...,k"
    dataset_prefix: str = "continuallearning/libero_spatial_image_task_"
    env_task_prefix: str = "Libero_Spatial_Task_"
    current_task: int = 0                  # 방금 학습을 끝낸 태스크 k
    run_tag: str = ""                      # 그림 범례 이름 (예: lam100)
    results_path: str = "outputs/E0/e0_results.jsonl"
    holdout_episodes: int = 5              # 태스크당 뒤 N 에피소드는 학습에서 제외
    probe_batches: int = 16
    probe_batch_size: int = 16
    probe_sr: bool = True
    probe_n_episodes: int = 20
    probe_eval_batch_size: int = 20


# ═════════════════════════════════════════════════════════════════════════════
#  EWC
# ═════════════════════════════════════════════════════════════════════════════
def ewc_penalty(policy: PreTrainedPolicy, state: dict) -> torch.Tensor:
    """0.5·Σ_i F_i (θ_i - θ*_i)²"""
    total = None
    for n, p in policy.named_parameters():
        if p.requires_grad and n in state["fisher"] and n in state["anchor"]:
            t = (state["fisher"][n] * (p - state["anchor"][n]).pow(2)).sum()
            total = t if total is None else total + t
    if total is None:
        raise RuntimeError("EWC penalty가 아무 파라미터도 못 잡았다 (이름 불일치)")
    return 0.5 * total


def build_ewc_state(cfg: E0Config, policy, dataset, train_eps, device, prev: dict | None) -> dict:
    """방금 끝낸 태스크의 학습 데이터로 Fisher를 재고 이전 것과 누적, 앵커는 현재 파라미터.

    F_i = E[(∂L/∂θ_i)²].  마지막에 mean(F)=1로 정규화해서 λ 값이 태스크마다
    같은 세기를 뜻하게 만든다(λ 스윕을 읽으려면 필요하다).
    """
    trainable = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    fisher = {n: torch.zeros_like(p) for n, p in trainable.items()}

    logging.info(f"Estimating Fisher over {cfg.fisher_batches} batches")
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=cfg.fisher_batch_size,
        sampler=episode_sampler(cfg, dataset, train_eps),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    it = cycle(loader)
    policy.eval()
    for _ in range(cfg.fisher_batches):
        policy.zero_grad(set_to_none=True)
        policy.forward(to_device(next(it), device))[0].backward()
        for n, p in trainable.items():
            if p.grad is not None:
                fisher[n] += p.grad.detach().pow(2)
    policy.zero_grad(set_to_none=True)
    policy.train()

    for n in fisher:
        fisher[n] /= cfg.fisher_batches
        if prev is not None:
            fisher[n] += prev["fisher"][n]
    scale = (sum(v.sum() for v in fisher.values()) / sum(v.numel() for v in fisher.values())).clamp_min(1e-12)

    return {
        "fisher": {n: v / scale for n, v in fisher.items()},
        "anchor": {n: p.detach().clone() for n, p in trainable.items()},
    }


# ═════════════════════════════════════════════════════════════════════════════
#  데이터: 학습 / held-out 분할
# ═════════════════════════════════════════════════════════════════════════════
def split_episodes(repo_id: str, root: str | None, holdout: int) -> tuple[list[int], list[int]]:
    total = LeRobotDatasetMetadata(repo_id, root=root).total_episodes
    if not 0 < holdout < total:
        raise ValueError(f"holdout_episodes={holdout} invalid for {repo_id} (total={total})")
    return list(range(total - holdout)), list(range(total - holdout, total))


def episode_sampler(cfg: E0Config, dataset, episodes: list[int], shuffle: bool = True):
    """에피소드 부분집합만 뽑는 샘플러.

    ★ LeRobotDataset(episodes=[...])를 쓰면 안 된다.
      episode_data_index는 선택 에피소드 기준 0..N-1로 다시 매겨지는데
      __getitem__은 그걸 **원본** episode_index로 인덱싱한다(lerobot_dataset.py:689).
      그래서 [45..49]처럼 0에서 시작 안 하는 부분집합을 주면 IndexError가 난다.
      데이터셋은 통째로 열고 샘플러에서 가른다.
    """
    return EpisodeAwareSampler(
        dataset.episode_data_index,
        episode_indices_to_use=episodes,
        drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
        shuffle=shuffle,
    )


def to_device(batch: dict, device) -> dict:
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device, non_blocking=device.type == "cuda")
    return batch


# ═════════════════════════════════════════════════════════════════════════════
#  프로브: held-out MSE + SR
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def probe_mse(cfg: E0Config, policy: PreTrainedPolicy, repo_id: str, device) -> float:
    """held-out 전문가 데이터에서의 flow matching MSE. 학습이 최소화하던 바로 그 값."""
    _, holdout_eps = split_episodes(repo_id, None, cfg.holdout_episodes)
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps=resolve_delta_timestamps(cfg.policy, LeRobotDatasetMetadata(repo_id)),
        video_backend=cfg.dataset.video_backend,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=cfg.probe_batch_size,
        sampler=episode_sampler(cfg, dataset, holdout_eps),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    it = cycle(loader)
    policy.eval()
    total = 0.0
    for _ in range(cfg.probe_batches):
        total += float(policy.forward(to_device(next(it), device))[0])
    policy.train()
    return total / cfg.probe_batches


def probe_sr(cfg: E0Config, policy: PreTrainedPolicy, env_task: str) -> float | None:
    """시뮬레이터 롤아웃 성공률(%). 실패하면 None (MSE 쪽은 그대로 살린다)."""
    if not cfg.probe_sr or cfg.env is None:
        return None
    import copy

    env = None
    try:
        from lerobot.envs.factory import make_env
        from lerobot.scripts.eval import eval_policy

        env_cfg = copy.deepcopy(cfg.env)
        env_cfg.task = env_task
        env = make_env(env_cfg, n_envs=cfg.probe_eval_batch_size, use_async_envs=False)
        with torch.no_grad():
            info = eval_policy(env, policy, cfg.probe_n_episodes, start_seed=cfg.seed)
        return float(info["aggregated"]["pc_success"])
    except Exception as e:
        logging.warning(f"SR probe skipped ({env_task}): {type(e).__name__}: {e}")
        return None
    finally:
        if env is not None:
            env.close()
        policy.train()


def run_probe(cfg: E0Config, policy: PreTrainedPolicy, device) -> None:
    """지금까지 본 모든 태스크에 대해 MSE/SR을 재서 JSONL에 append."""
    task_ids = [int(t) for t in cfg.task_ids.split(",") if t.strip()] or [cfg.current_task]
    out = Path(cfg.results_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    for j in task_ids:
        row = {
            "run_tag": cfg.run_tag or f"lam{cfg.ewc_lambda:g}",
            "ewc_lambda": cfg.ewc_lambda if math.isfinite(cfg.ewc_lambda) else "inf",
            "seed": cfg.seed,
            "stage": cfg.current_task,                 # 방금 끝낸 태스크 k
            "probe_task": j,                           # 프로브 대상 태스크 j
            "is_old": j < cfg.current_task,
            "mse": probe_mse(cfg, policy, f"{cfg.dataset_prefix}{j}", device),
            "sr": probe_sr(cfg, policy, f"{cfg.env_task_prefix}{j}"),
        }
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        logging.info(f"[E0] k={row['stage']} j={j} mse={row['mse']:.5f} sr={row['sr']}")


# ═════════════════════════════════════════════════════════════════════════════
#  한 스텝 (train.py의 update_policy + EWC 페널티 한 항)
# ═════════════════════════════════════════════════════════════════════════════
def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
    ewc_state: dict | None = None,
    ewc_lambda: float = 0.0,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        mse, output_dict = policy.forward(batch)
        # mse와 penalty를 따로 로깅한다. 안 그러면 "EWC 실패"와 "EWC 미적용"이 구분되지 않는다.
        # 곱셈을 조건 밖으로 빼면 안 된다: λ=inf 이고 penalty=0 일 때 inf*0 = nan 이 된다.
        # (λ=inf 는 애초에 동결 팔이라 여기 오지 않지만, 앵커가 없는 태스크 0에서는 온다.)
        if ewc_state is not None and 0 < ewc_lambda < float("inf"):
            penalty = ewc_penalty(policy, ewc_state)
            loss = mse + ewc_lambda * penalty
        else:
            penalty = torch.zeros((), device=device)
            loss = mse

    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm, error_if_nonfinite=False)
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()
    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.mse = mse.item()
    train_metrics.penalty = float(penalty.detach())
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


# ═════════════════════════════════════════════════════════════════════════════
#  메인 (train.py와 같은 [1]~[11] 순서)
# ═════════════════════════════════════════════════════════════════════════════
@parser.wrap()
def train(cfg: E0Config):
    # [1] 설정
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))
    logging.info(colored(f"[E0] λ={cfg.ewc_lambda:g} stage k={cfg.current_task}", "green", attrs=["bold"]))

    # [2] 로거
    wandb_logger = WandBLogger(cfg) if (cfg.wandb.enable and cfg.wandb.project) else None

    # [3] 재현성
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # [4] 디바이스
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # [5] 데이터셋 + held-out 분할 (분할은 샘플러가 한다, episode_sampler 주석 참조)
    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    train_eps, holdout_eps = split_episodes(cfg.dataset.repo_id, cfg.dataset.root, cfg.holdout_episodes)
    logging.info(f"Split: train {len(train_eps)} ep / held-out {len(holdout_eps)} ep")

    # [6] 평가 환경은 만들지 않는다(--eval_freq=0). SR은 [11]의 프로브에서 태스크별로 잰다.

    # [7] 정책
    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)

    ewc_state = None
    if cfg.ewc_state_path and Path(cfg.ewc_state_path).exists():
        ewc_state = torch.load(cfg.ewc_state_path, map_location=device, weights_only=False)
        logging.info(colored(f"[E0] EWC state loaded: {cfg.ewc_state_path}", "green"))

    # λ=inf -> 학습 스텝 0. 단 태스크 0에는 앵커가 없어 어떤 λ도 페널티가 0이므로,
    # 그때는 λ에 상관없이 정상 학습한다. 동결은 지킬 이전 태스크가 생긴 뒤부터다.
    frozen = math.isinf(cfg.ewc_lambda) and ewc_state is not None

    # [8] 옵티마이저
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")

    # [9] 데이터로더 — held-out 에피소드가 배제되는 지점
    sampler = episode_sampler(cfg, dataset, train_eps)
    logging.info(f"train sampler: {len(sampler)} frames")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
        persistent_workers=cfg.num_workers > 0,
    )
    dl_iter = cycle(dataloader)

    policy.train()
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        {
            "loss": AverageMeter("loss", ":.3f"),      # mse + λ·penalty
            "mse": AverageMeter("mse", ":.3f"),        # flow matching만
            "penalty": AverageMeter("pen", ":.3e"),    # λ 곱하기 전
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        },
        initial_step=step,
    )

    # [10] 학습 루프
    if frozen:
        logging.info(colored("[E0] λ=inf — 파라미터 동결, 학습 스텝 없음", "magenta", attrs=["bold"]))
        ckpt = get_step_checkpoint_dir(cfg.output_dir, max(cfg.steps, 1), 0)
        save_checkpoint(ckpt, 0, cfg, policy, optimizer, lr_scheduler)
        update_last_checkpoint(ckpt)
    else:
        logging.info("Start offline training on a fixed dataset")
        for _ in range(step, cfg.steps):
            t0 = time.perf_counter()
            batch = to_device(next(dl_iter), device)
            train_tracker.dataloading_s = time.perf_counter() - t0

            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=cfg.policy.use_amp,
                ewc_state=ewc_state,
                ewc_lambda=cfg.ewc_lambda,
            )

            step += 1
            train_tracker.step()

            if cfg.log_freq > 0 and step % cfg.log_freq == 0:
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_logger.log_dict({**train_tracker.to_dict(), **(output_dict or {})}, step)
                train_tracker.reset_averages()

            if cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps):
                logging.info(f"Checkpoint policy after step {step}")
                ckpt = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(ckpt, step, cfg, policy, optimizer, lr_scheduler)
                update_last_checkpoint(ckpt)

    # [11] Fisher 갱신 -> 프로브
    # 둘 다 "이 태스크를 막 끝낸 파라미터"에서 재야 MSE와 SR이 같은 x축 위에 놓인다.
    if cfg.ewc_lambda > 0 and not frozen:
        if math.isinf(cfg.ewc_lambda):
            # 동결 팔은 Fisher를 쓰지 않는다. 다음 스테이지에 "이전 태스크가 있다"만 알리면 된다.
            state = {"fisher": {}, "anchor": {n: p.detach() for n, p in policy.named_parameters()
                                              if p.requires_grad}}
        else:
            state = build_ewc_state(cfg, policy, dataset, train_eps, device, ewc_state)
        path = Path(cfg.output_dir) / "ewc_state.pt"
        torch.save({k: {n: v.cpu() for n, v in d.items()} for k, d in state.items()}, path)
        logging.info(colored(f"[E0] EWC state saved -> {path}", "green"))

    logging.info(colored("[E0] probing (held-out MSE + SR)", "cyan", attrs=["bold"]))
    run_probe(cfg, policy, device)
    logging.info("End of E0 stage")


# ═════════════════════════════════════════════════════════════════════════════
#  그림: λ별 MSE(왼쪽) / SR(오른쪽)
# ═════════════════════════════════════════════════════════════════════════════
def plot_e0(results_path: str, out_path: str) -> None:
    rows = [json.loads(x) for x in Path(results_path).read_text().splitlines() if x.strip()]
    if not rows:
        raise SystemExit(f"no rows in {results_path}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    csv_path = str(Path(out_path).with_suffix(".csv"))
    keys = ["run_tag", "ewc_lambda", "seed", "stage", "probe_task", "is_old", "mse", "sr"]
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in sorted(rows, key=lambda r: (str(r["ewc_lambda"]), r["stage"], r["probe_task"])):
            f.write(",".join("" if r.get(k) is None else str(r[k]) for k in keys) + "\n")
    print(f"saved table  -> {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib 없음 -> 그림 생략 (pip install matplotlib 후 --plot_only 다시)")
        return

    def lam_key(t):
        v = [r["ewc_lambda"] for r in rows if r["run_tag"] == t][0]
        return float("inf") if v == "inf" else float(v)

    tags = sorted({r["run_tag"] for r in rows}, key=lam_key)
    stages = sorted({r["stage"] for r in rows})

    def old_mean(tag, stage, key):
        """이전 태스크(j<k)들의 평균. 망각의 정도를 보는 값."""
        v = [r[key] for r in rows
             if r["run_tag"] == tag and r["stage"] == stage and r["is_old"] and r.get(key) is not None]
        return sum(v) / len(v) if v else None

    fig, (ax_mse, ax_sr) = plt.subplots(1, 2, figsize=(13, 5))
    for tag in tags:
        for ax, key in ((ax_mse, "mse"), (ax_sr, "sr")):
            # stage 0에는 이전 태스크가 없고, SR은 gym_libero가 없으면 통째로 비어 있다.
            pts = [(s, old_mean(tag, s, key)) for s in stages]
            pts = [(s, v) for s, v in pts if v is not None]
            if pts:
                ax.plot(*zip(*pts), "-o", ms=5, label=f"λ={tag}")

    # 그림 안 텍스트는 영어로. 기본 matplotlib 폰트에 한글 글리프가 없어 두부(□)가 된다.
    ax_mse.set(xlabel="training stage k", ylabel="held-out MSE on old tasks",
               title="MSE  (the training objective)")
    ax_sr.set(xlabel="training stage k", ylabel="SR on old tasks (%)",
              title="SR  (what we actually care about)", ylim=(-3, 103))
    for ax in (ax_mse, ax_sr):
        ax.set_xticks(stages)
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9, title="EWC lambda", title_fontsize=9)
        else:
            ax.text(0.5, 0.5, "no data (gym_libero not installed?)", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
    fig.suptitle(
        "EWC lambda sweep: flat MSE with collapsing SR = loss/SR anchor mismatch", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"saved figure -> {out_path}")


if __name__ == "__main__":
    if "--plot_only" in sys.argv:
        kv = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
        init_logging()
        plot_e0(kv.get("results", "outputs/E0/e0_results.jsonl"),
                kv.get("out", "outputs/E0/E0_mse_vs_sr.png"))
    else:
        mp.set_start_method("spawn", force=True)
        init_logging()
        train()
