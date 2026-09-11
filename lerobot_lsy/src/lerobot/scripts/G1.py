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

"""G1 — 망각이 denoising timestep에 따라 어디에 몰려 있는가.

가설
    이전 태스크의 held-out MSE는 별로 안 오르는데 SR은 무너진다. flow matching
    디코더에서는 초기 노이즈에서 출발하는 **이른 적분 구간**이 최종 behavioral mode를
    고르고, 따라서 그 구간의 작은 속도장 변화가 SR 붕괴를 불균형하게 일으킬 수 있다.

이 스크립트는 학습을 하지 않는다. 저장된 체크포인트 두 개(θ_old, θ_new)를 읽어
속도장을 재고, 개입 롤아웃으로 인과를 확인한다.

시간축 정의 (modeling_dit_flow_mt.DiTFlowModel.sample에서 확인한 사실)
    x <- N(0, I)                       t = 0      초기 노이즈
    for k in range(T):
        t = k / T
        x <- x + (1/T) * v(x, t, c)
        x <- clamp(x, -1, 1)           clip_sample=True
    return x                           t = 1      최종 액션
    즉 s = k/T 이고 s=0이 노이즈, s=1이 액션이다. 그림의 x축은 이 s를 쓴다.

구성
    [A] 얼린 궤적 위의 속도 드리프트
        θ_old로 궤적 x_t를 한 번 만들고, **같은 x_t**를 두 모델에 넣어 v를 비교한다.
        각자 롤아웃해서 비교하면 상태 자체가 갈라져 교란이 생긴다.
        조건 벡터 c도 모델마다 다르므로 두 가지를 따로 잰다.
            D_t^full  = ||v_old(x_t,t,c_old) - v_new(x_t,t,c_new)||^2   실제로 벌어지는 일
            D_t^vel   = ||v_old(x_t,t,c_old) - v_new(x_t,t,c_old)||^2   속도장만의 변화
        차이가 곧 조건(인코더) 드리프트의 몫이다.
    [B] held-out MSE (E0와 같은 자로 잰다) — "MSE는 그대로인데 SR만 무너진다"의 좌변
    [C] 개입 롤아웃: 특정 구간만 θ_new의 속도를 쓰고 나머지는 θ_old를 쓴다.
        old / new / 단일 스텝 격자 / 구간(early·mid·late)의 SR을 잰다.
    [D] 에피소드 단위 상관: old 롤아웃 중에 매 재계획마다 D_t를 기록해 두고,
        그 에피소드의 성공/실패와 timestep별로 상관을 낸다.

사용 예
    python G1.py --old_ckpt=<...>/task_0/checkpoints/last/pretrained_model \
                 --new_ckpt=<...>/task_1/checkpoints/last/pretrained_model \
                 --policy.path=<old_ckpt> --dataset.repo_id=continuallearning/libero_spatial_image_task_0 \
                 --env.type=libero --env.benchmark=libero_spatial --env.task=Libero_Spatial_Task_0
    python G1.py --plot_only --run_dir=outputs/G1/<run_tag>
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
import torch.multiprocessing as mp
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.constants import ACTION
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.policies.factory import make_policy
from lerobot.scripts.eval import rollout
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

# held-out 분할과 샘플러는 E0에서 그대로 가져온다. 복사본을 두면 "G1이 잰 MSE"와
# "E0가 잰 MSE"가 조용히 갈라진다.
from lerobot.scripts.E0 import episode_sampler, split_episodes, to_device


@dataclass
class G1Config(TrainPipelineConfig):
    """train.py 인자 전부 + 이 실험용. 학습 인자(steps/batch 등)는 쓰이지 않는다."""

    # ── 무엇을 비교하는가 ────────────────────────────────────────────────────
    old_ckpt: str = ""          # 이전 태스크를 막 끝낸 체크포인트 θ_old
    new_ckpt: str = ""          # 다음 태스크까지 학습한 체크포인트 θ_new
    probe_task: int = 0         # 평가 태스크 = 이전 태스크
    dataset_prefix: str = "continuallearning/libero_spatial_image_task_"
    env_task_prefix: str = "Libero_Spatial_Task_"
    holdout_episodes: int = 5   # E0와 같은 분할

    # ── [A] 얼린 궤적 드리프트 ───────────────────────────────────────────────
    drift_batches: int = 4      # held-out 관측 배치 수
    drift_batch_size: int = 8   # 배치당 관측 수
    drift_n_noise: int = 4      # 관측 하나당 초기 노이즈 개수
    drift_seed: int = 12345

    # ── [B] held-out MSE ─────────────────────────────────────────────────────
    probe_batches: int = 16
    probe_batch_size: int = 16
    probe_seed: int = 12345
    probe_n_tau: int = 10       # 고정 τ 격자 (랜덤 τ 평균이면 팔마다 잡음이 달라진다)

    # ── [C] 개입 롤아웃 ──────────────────────────────────────────────────────
    skip_sr: bool = False       # true면 시뮬레이터 없이 [A][B][D-드리프트]만
    sr_episodes: int = 20       # = n_envs. 한 배치로 돌려 에피소드 짝을 맞춘다
    n_grid: int = 10            # 단일 스텝 개입을 잴 timestep 격자 수
    windows: str = ("early10:0.0-0.1,early20:0.0-0.2,early30:0.0-0.3,"
                    "mid20:0.4-0.6,late20:0.8-1.0,late30:0.7-1.0")
    noise_base_seed: int = 777  # 모든 조건이 같은 초기 노이즈를 쓰게 하는 기준값
    indep_noise_seed: int = 313131   # Condition B(독립 노이즈)용
    env_seed: int = 100000      # env.reset 시드

    # ── 출력 ─────────────────────────────────────────────────────────────────
    out_root: str = "outputs/G1"
    run_tag: str = ""

    def validate(self):
        out = self.output_dir
        if isinstance(out, Path) and out.is_dir():
            self.output_dir = None
            super().validate()
            self.output_dir = out
        else:
            super().validate()


# ═════════════════════════════════════════════════════════════════════════════
#  모델 로드
# ═════════════════════════════════════════════════════════════════════════════
def load_policy(path: str, ds_meta, device):
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path
    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()
    return policy


def norm_hash(policy) -> str:
    """정규화 버퍼의 해시. 두 모델이 다르면 같은 batch를 공유할 수 없다."""
    import hashlib

    h = hashlib.sha1()
    for name, buf in sorted(policy.named_parameters()):
        if "normalize" in name:
            h.update(name.encode())
            h.update(buf.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


# ═════════════════════════════════════════════════════════════════════════════
#  적분기: 스텝마다 어느 모델의 속도를 쓸지 고른다
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def integrate(net_old, net_new, cond_old, cond_new, timesteps: int,
              use_new: np.ndarray | None, noise: torch.Tensor,
              record: bool = False):
    """Euler 적분. use_new[k]가 True인 스텝만 θ_new의 (속도장, 조건)을 쓴다.

    modeling_dit_flow_mt.DiTFlowModel.sample과 **같은 순서**로 돈다:
        x <- x + dt*v(x, t=k/T, c) -> clamp.  use_new가 전부 False면 결과가
    sample()과 비트 단위로 같아야 한다(sanity check 4에서 실제로 확인한다).

    record=True면 매 스텝의 x_t와 두 모델의 속도를 함께 돌려준다([A],[D]용).
    """
    x = noise.clone()
    dt = 1.0 / timesteps
    clip, rng = net_old.clip_sample, net_old.clip_sample_range
    rec_v_old, rec_v_new_cnew, rec_v_new_cold = [], [], []

    for k in range(timesteps):
        t = torch.full((x.shape[0],), k / timesteps, device=x.device, dtype=x.dtype)
        if record:
            v_old = net_old(x, t, cond_old)
            v_new_cnew = net_new(x, t, cond_new)
            v_new_cold = net_new(x, t, cond_old)
            rec_v_old.append(v_old)
            rec_v_new_cnew.append(v_new_cnew)
            rec_v_new_cold.append(v_new_cold)
            v = v_new_cnew if (use_new is not None and use_new[k]) else v_old
        else:
            if use_new is not None and use_new[k]:
                v = net_new(x, t, cond_new)
            else:
                v = net_old(x, t, cond_old)
        x = x + dt * v
        if clip:
            x = torch.clamp(x, -rng, rng)

    if not record:
        return x, None
    return x, {
        "v_old": torch.stack(rec_v_old, dim=1),              # (B, T, horizon, adim)
        "v_new_cnew": torch.stack(rec_v_new_cnew, dim=1),
        "v_new_cold": torch.stack(rec_v_new_cold, dim=1),
    }


def make_schedule(timesteps: int, spec) -> np.ndarray | None:
    """조건 이름 -> 스텝별 bool 마스크.

    spec:
      None          전부 old
      "all"         전부 new
      ("step", k)   k번 스텝만 new
      ("win", lo, hi)  s in [lo, hi) 구간만 new
    """
    if spec is None:
        return None
    m = np.zeros(timesteps, dtype=bool)
    if spec == "all":
        m[:] = True
    elif spec[0] == "step":
        m[spec[1]] = True
    elif spec[0] == "win":
        s = np.arange(timesteps) / timesteps
        m[(s >= spec[1]) & (s < spec[2])] = True
    else:
        raise ValueError(f"unknown schedule spec {spec}")
    return m


# ═════════════════════════════════════════════════════════════════════════════
#  롤아웃용 후크: policy_old.dit_flow.generate_actions를 갈아끼운다
# ═════════════════════════════════════════════════════════════════════════════
class MixedActionGenerator:
    """select_action의 큐/정규화/슬라이싱은 그대로 두고 적분기만 바꾼다.

    generate_actions가 받는 batch는 이미 policy_old.normalize_inputs를 거친 것이다.
    두 모델의 정규화 통계가 같다는 것을 확인했으므로(norm_hash) 그대로 θ_new의
    인코더에도 먹여 c_new를 만든다.

    노이즈는 호출 순번으로 시드를 만든다. 조건이 달라져도 k번째 재계획은 같은
    초기 노이즈를 받는다 -> Condition A(같은 노이즈)의 짝지은 비교가 성립한다.
    """

    def __init__(self, policy_old, policy_new, schedule, base_seed: int, record: bool = False):
        self.p_old, self.p_new = policy_old, policy_new
        self.schedule = schedule
        self.base_seed = base_seed
        self.record = record
        self.calls = 0
        self.drift = []      # record=True일 때 (call, B, T) 배열들

    def reset(self):
        self.calls = 0
        self.drift = []

    @torch.no_grad()
    def __call__(self, batch):
        dit_old, dit_new = self.p_old.dit_flow, self.p_new.dit_flow
        cond_old = dit_old._prepare_global_conditioning(batch)
        cond_new = dit_new._prepare_global_conditioning(batch)

        b = cond_old.shape[0]
        device = cond_old.device
        gen = torch.Generator(device=device).manual_seed(
            (self.base_seed + 9973 * self.calls) % (2**31 - 1))
        noise = torch.randn(b, dit_old.velocity_net.ac_chunk, dit_old.velocity_net.ac_dim,
                            device=device, dtype=cond_old.dtype, generator=gen)

        x, rec = integrate(dit_old.velocity_net, dit_new.velocity_net, cond_old, cond_new,
                           dit_old.num_inference_steps, self.schedule, noise, record=self.record)
        if self.record:
            d = ((rec["v_old"] - rec["v_new_cnew"]) ** 2).mean(dim=(2, 3))   # (B, T)
            self.drift.append(d.float().cpu().numpy())
        self.calls += 1

        start = self.p_old.config.n_obs_steps - 1
        end = start + self.p_old.config.n_action_steps
        return x[:, start:end]


# ═════════════════════════════════════════════════════════════════════════════
#  [A] 얼린 궤적 위의 속도 드리프트 (시뮬레이터 불필요)
# ═════════════════════════════════════════════════════════════════════════════
def make_holdout_loader(cfg: G1Config, policy, repo_id: str, batch_size: int, device):
    ds_meta = LeRobotDatasetMetadata(repo_id)
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps=resolve_delta_timestamps(policy.config, ds_meta),
        video_backend=cfg.dataset.video_backend,
    )
    _, holdout = split_episodes(repo_id, None, cfg.holdout_episodes)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=0, batch_size=batch_size,
        sampler=episode_sampler(cfg, dataset, holdout),
        pin_memory=device.type == "cuda", drop_last=True,
    )
    return dataset, loader


@torch.no_grad()
def encode_both(policy_old, policy_new, batch):
    """같은 원시 batch에서 두 모델의 조건 벡터를 만든다."""
    b = policy_old.normalize_inputs(batch)
    if policy_old.config.image_features:
        b = dict(b)
        b["observation.images"] = torch.stack(
            [b[k] for k in policy_old.config.image_features], dim=-4)
    return (policy_old.dit_flow._prepare_global_conditioning(b),
            policy_new.dit_flow._prepare_global_conditioning(b), b)


@torch.no_grad()
def phase_a_drift(cfg: G1Config, policy_old, policy_new, device) -> dict:
    """held-out 관측 위에서 timestep별 속도 드리프트를 잰다."""
    logging.info(colored("[G1][A] 얼린 궤적 속도 드리프트", "cyan", attrs=["bold"]))
    repo = f"{cfg.dataset_prefix}{cfg.probe_task}"
    _, loader = make_holdout_loader(cfg, policy_old, repo, cfg.drift_batch_size, device)
    torch.manual_seed(cfg.drift_seed)
    it = cycle(loader)

    net_o = policy_old.dit_flow.velocity_net
    net_n = policy_new.dit_flow.velocity_net
    T = policy_old.dit_flow.num_inference_steps

    full, velonly, cond_only, rel_full, norm_v = [], [], [], [], []
    sanity_oo, sanity_nn = [], []

    for bi in range(cfg.drift_batches):
        batch = to_device(next(it), device)
        c_old, c_new, _ = encode_both(policy_old, policy_new, batch)
        for j in range(cfg.drift_n_noise):
            gen = torch.Generator(device=device).manual_seed(cfg.drift_seed + 1000 * bi + j)
            noise = torch.randn(c_old.shape[0], net_o.ac_chunk, net_o.ac_dim,
                                device=device, dtype=c_old.dtype, generator=gen)
            _, rec = integrate(net_o, net_n, c_old, c_new, T, None, noise, record=True)

            v_o, v_nc, v_no = rec["v_old"], rec["v_new_cnew"], rec["v_new_cold"]
            full.append(((v_o - v_nc) ** 2).mean(dim=(2, 3)).float().cpu().numpy())
            velonly.append(((v_o - v_no) ** 2).mean(dim=(2, 3)).float().cpu().numpy())
            cond_only.append(((v_no - v_nc) ** 2).mean(dim=(2, 3)).float().cpu().numpy())
            nrm = v_o.pow(2).mean(dim=(2, 3)).sqrt()
            rel_full.append(((v_o - v_nc).pow(2).mean(dim=(2, 3)).sqrt()
                             / (nrm + 1e-8)).float().cpu().numpy())
            norm_v.append(nrm.float().cpu().numpy())

            # sanity 1/2: 같은 모델끼리는 정확히 0이어야 한다
            _, rec_oo = integrate(net_o, net_o, c_old, c_old, T, None, noise, record=True)
            sanity_oo.append(float(((rec_oo["v_old"] - rec_oo["v_new_cnew"]) ** 2).max()))
            _, rec_nn = integrate(net_n, net_n, c_new, c_new, T, None, noise, record=True)
            sanity_nn.append(float(((rec_nn["v_old"] - rec_nn["v_new_cnew"]) ** 2).max()))

    out = {
        "s": (np.arange(T) / T).astype(np.float32),
        "D_full": np.concatenate(full, axis=0),        # (N, T)
        "D_vel": np.concatenate(velonly, axis=0),
        "D_cond": np.concatenate(cond_only, axis=0),
        "D_rel": np.concatenate(rel_full, axis=0),
        "v_norm": np.concatenate(norm_v, axis=0),
        "sanity_old_old_max": float(np.max(sanity_oo)),
        "sanity_new_new_max": float(np.max(sanity_nn)),
    }
    logging.info(f"[G1][A] N={out['D_full'].shape[0]} 샘플, T={T}")
    logging.info(f"[G1][A] sanity  old-old max D = {out['sanity_old_old_max']:.3e}  "
                 f"new-new max D = {out['sanity_new_new_max']:.3e}  (0이어야 한다)")
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  [B] held-out MSE — E0/R1과 같은 고정 τ 격자
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def heldout_mse(cfg: G1Config, policy, device) -> float:
    repo = f"{cfg.dataset_prefix}{cfg.probe_task}"
    _, loader = make_holdout_loader(cfg, policy, repo, cfg.probe_batch_size, device)
    torch.manual_seed(cfg.probe_seed)
    it = cycle(loader)
    taus = (torch.arange(cfg.probe_n_tau, device=device, dtype=torch.float32) + 0.5) / cfg.probe_n_tau
    net = policy.dit_flow.velocity_net
    total, count = 0.0, 0
    for b in range(cfg.probe_batches):
        batch = to_device(next(it), device)
        nb = policy.normalize_inputs(batch)
        if policy.config.image_features:
            nb = dict(nb)
            nb["observation.images"] = torch.stack(
                [nb[k] for k in policy.config.image_features], dim=-4)
        nb = policy.normalize_targets(nb)
        cond = policy.dit_flow._prepare_global_conditioning(nb)
        traj = nb[ACTION]
        gen = torch.Generator(device=device).manual_seed(cfg.probe_seed + 1000 * b)
        noise = torch.randn(traj.shape, generator=gen, device=device, dtype=traj.dtype)
        target = traj - noise
        for tau in taus:
            t = tau.expand(traj.shape[0])
            noisy = (1 - tau) * noise + tau * traj
            pred = net(noisy_actions=noisy, time=t, global_cond=cond)
            total += float(torch.nn.functional.mse_loss(pred, target))
            count += 1
    return total / max(count, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  [C][D] 개입 롤아웃
# ═════════════════════════════════════════════════════════════════════════════
def episode_success(rollout_data: dict) -> np.ndarray:
    """eval_policy와 **같은 방식**으로 에피소드 성공을 판정한다(첫 done까지 마스킹)."""
    import einops

    n_steps = rollout_data["done"].shape[1]
    done_idx = torch.argmax(rollout_data["done"].to(int), dim=1)
    mask = (torch.arange(n_steps) <= einops.repeat(done_idx + 1, "b -> b s", s=n_steps)).int()
    return einops.reduce((rollout_data["success"] * mask), "b n -> b", "any").numpy()


def run_condition(env, policy_old, policy_new, name: str, spec, cfg: G1Config,
                  init_ids: list[int], seeds: list[int], record: bool = False,
                  base_seed: int | None = None) -> dict:
    """조건 하나를 롤아웃하고 에피소드별 성공을 돌려준다.

    ★ 초기 상태를 조건마다 되돌린다. eval.rollout이 reset 직후 _init_state_id를
      num_envs만큼 전진시키므로, 그냥 연달아 부르면 조건마다 다른 초기 상태를 받아
      짝지은 비교가 깨진다.
    """
    T = policy_old.dit_flow.num_inference_steps
    sched = make_schedule(T, spec)
    gen = MixedActionGenerator(policy_old, policy_new, sched,
                               cfg.noise_base_seed if base_seed is None else base_seed,
                               record=record)
    orig = policy_old.dit_flow.generate_actions
    policy_old.dit_flow.generate_actions = gen
    try:
        for i in range(env.num_envs):
            env.envs[i].env.env._init_state_id = init_ids[i]
        data = rollout(env, policy_old, seeds=list(seeds))
    finally:
        policy_old.dit_flow.generate_actions = orig

    succ = episode_success(data)
    logging.info(f"[G1][C] {name:22s} SR = {100.0 * succ.mean():5.1f}%  "
                 f"({int(succ.sum())}/{len(succ)})  calls={gen.calls}")
    out = {"name": name, "success": succ, "sr": float(100.0 * succ.mean())}
    if record and gen.drift:
        # (calls, B, T) -> 에피소드별 평균. 에피소드가 끝난 뒤의 호출도 섞이지만
        # done 이후에는 env가 자동 리셋되어 같은 태스크를 계속 돌므로 큰 왜곡은 없다.
        # 정확히 자르고 싶으면 done_idx // n_action_steps 까지만 쓰면 된다.
        d = np.stack(gen.drift, axis=0)                       # (calls, B, T)
        import einops as _e

        n_steps = data["done"].shape[1]
        done_idx = torch.argmax(data["done"].to(int), dim=1).numpy()
        n_act = policy_old.config.n_action_steps
        per_ep = np.zeros((d.shape[1], d.shape[2]), dtype=np.float32)
        for i in range(d.shape[1]):
            k = max(1, min(d.shape[0], int(done_idx[i]) // n_act + 1))
            per_ep[i] = d[:k, i, :].mean(axis=0)
        out["drift_per_episode"] = per_ep                     # (B, T)
        out["drift_all_calls"] = d
    return out


def parse_windows(spec: str):
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, rng = item.split(":")
        lo, hi = rng.split("-")
        out.append((name, float(lo), float(hi)))
    return out


def phase_c_interventions(cfg: G1Config, policy_old, policy_new, device) -> dict:
    logging.info(colored("[G1][C] 개입 롤아웃", "cyan", attrs=["bold"]))
    import copy

    env_cfg = copy.deepcopy(cfg.env)
    env_cfg.task = f"{cfg.env_task_prefix}{cfg.probe_task}"
    env = make_env(env_cfg, n_envs=cfg.sr_episodes, use_async_envs=False)
    init_ids = [env.envs[i].env.env._init_state_id for i in range(env.num_envs)]
    seeds = list(range(cfg.env_seed, cfg.env_seed + cfg.sr_episodes))
    logging.info(f"[G1][C] n_envs={env.num_envs} init_state_ids={init_ids} seeds[0]={seeds[0]}")

    T = policy_old.dit_flow.num_inference_steps
    results = {}
    try:
        # 기준 두 개. old는 드리프트도 같이 기록한다([D]).
        results["old"] = run_condition(env, policy_old, policy_new, "old (baseline)", None,
                                       cfg, init_ids, seeds, record=True)
        results["new"] = run_condition(env, policy_old, policy_new, "new (all steps)", "all",
                                       cfg, init_ids, seeds)
        # Condition B: 독립 노이즈. 확률적 변동폭의 기준선.
        results["new_indep"] = run_condition(env, policy_old, policy_new, "new (indep noise)", "all",
                                             cfg, init_ids, seeds, base_seed=cfg.indep_noise_seed)

        # 단일 스텝 교체
        grid = sorted(set(int(round(x)) for x in np.linspace(0, T - 1, cfg.n_grid)))
        for k in grid:
            results[f"step{k}"] = run_condition(
                env, policy_old, policy_new, f"single step k={k} (s={k / T:.2f})",
                ("step", k), cfg, init_ids, seeds)

        # 구간 교체
        for name, lo, hi in parse_windows(cfg.windows):
            results[f"win_{name}"] = run_condition(
                env, policy_old, policy_new, f"window {name} [{lo:.2f},{hi:.2f})",
                ("win", lo, hi), cfg, init_ids, seeds)
    finally:
        env.close()
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  sanity: 개입 없는 적분기가 원래 sample()과 같은가
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def sanity_integrator(policy_old, policy_new, device) -> dict:
    net_o = policy_old.dit_flow.velocity_net
    net_n = policy_new.dit_flow.velocity_net
    T = policy_old.dit_flow.num_inference_steps
    cond = torch.zeros(2, policy_old.dit_flow.config.hidden_dim * 0 + net_o.cond_proj.in_features,
                       device=device)
    seed = 4242
    gen = torch.Generator(device=device).manual_seed(seed)
    ref = net_o.sample(cond, timesteps=T, generator=gen)
    gen2 = torch.Generator(device=device).manual_seed(seed)
    noise = net_o.sample_noise(2, device, gen2)
    mine, _ = integrate(net_o, net_n, cond, cond, T, None, noise, record=False)
    d_all_old = float((ref - mine).abs().max())

    gen3 = torch.Generator(device=device).manual_seed(seed)
    ref_new = net_n.sample(cond, timesteps=T, generator=gen3)
    mine_new, _ = integrate(net_o, net_n, cond, cond, T, np.ones(T, dtype=bool), noise, record=False)
    d_all_new = float((ref_new - mine_new).abs().max())

    # 결정성: 같은 노이즈 두 번 -> 정확히 같아야 한다
    mine2, _ = integrate(net_o, net_n, cond, cond, T, None, noise, record=False)
    d_det = float((mine - mine2).abs().max())

    logging.info(f"[G1] sanity  integrate(all-old) vs sample()  max|Δ| = {d_all_old:.3e}")
    logging.info(f"[G1] sanity  integrate(all-new) vs sample()  max|Δ| = {d_all_new:.3e}")
    logging.info(f"[G1] sanity  determinism                     max|Δ| = {d_det:.3e}")
    return {"integrator_vs_sample_old": d_all_old,
            "integrator_vs_sample_new": d_all_new,
            "determinism": d_det}


# ═════════════════════════════════════════════════════════════════════════════
#  분석 + 그림
# ═════════════════════════════════════════════════════════════════════════════
def band(x: np.ndarray):
    """평균과 95% CI(정규 근사)."""
    m = x.mean(axis=0)
    se = x.std(axis=0, ddof=1) / np.sqrt(max(x.shape[0], 2))
    return m, 1.96 * se


def region_slices(T: int):
    return {"early": slice(0, T // 3), "middle": slice(T // 3, 2 * T // 3), "late": slice(2 * T // 3, T)}


def analyze(run_dir: Path) -> dict:
    z = np.load(run_dir / "g1_raw.npz", allow_pickle=True)
    meta = json.loads((run_dir / "g1_meta.json").read_text())
    s, T = z["s"], len(z["s"])
    reg = region_slices(T)

    summary = {"meta": meta, "regions": {}}
    for name, sl in reg.items():
        summary["regions"][name] = {
            "velocity_drift": float(z["D_full"][:, sl].mean()),
            "velocity_drift_velnet_only": float(z["D_vel"][:, sl].mean()),
            "velocity_drift_cond_only": float(z["D_cond"][:, sl].mean()),
            "relative_velocity_drift": float(z["D_rel"][:, sl].mean()),
        }

    # 개입 SR
    if "sr_names" in z:
        srs = {str(n): float(v) for n, v in zip(z["sr_names"], z["sr_values"])}
        summary["sr"] = srs
        sr_old = srs.get("old (baseline)")
        for name, sl in reg.items():
            wins = [k for k in srs if k.startswith("window") and name in k]
            if wins:
                summary["regions"][name]["intervention_sr"] = float(np.mean([srs[w] for w in wins]))
                summary["regions"][name]["behavioral_impact"] = float(
                    sr_old - np.mean([srs[w] for w in wins]))

    # 에피소드 단위 상관 (timestep별)
    if "ep_drift" in z and "ep_degrade" in z:
        from scipy import stats as sps

        d, deg = z["ep_drift"], z["ep_degrade"]
        pear, spear = [], []
        for t in range(d.shape[1]):
            if np.std(d[:, t]) < 1e-12 or np.std(deg) < 1e-12:
                pear.append(np.nan); spear.append(np.nan); continue
            pear.append(sps.pearsonr(d[:, t], deg)[0])
            spear.append(sps.spearmanr(d[:, t], deg)[0])
        summary["corr_pearson"] = [None if np.isnan(v) else float(v) for v in pear]
        summary["corr_spearman"] = [None if np.isnan(v) else float(v) for v in spear]
        for name, sl in reg.items():
            dm = d[:, sl].mean(axis=1)
            if np.std(dm) > 1e-12 and np.std(deg) > 1e-12:
                r, p = sps.pearsonr(dm, deg)
                rs, ps = sps.spearmanr(dm, deg)
                summary["regions"][name]["corr_pearson"] = float(r)
                summary["regions"][name]["corr_pearson_p"] = float(p)
                summary["regions"][name]["corr_spearman"] = float(rs)
                summary["regions"][name]["corr_spearman_p"] = float(ps)
    return summary


def plot_all(run_dir: Path, summary: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib 없음 -> 그림 생략")
        return

    z = np.load(run_dir / "g1_raw.npz", allow_pickle=True)
    s = z["s"]
    T = len(s)
    XLAB = "Denoising Progress (Initial Noise -> Final Action)"

    # ── Figure 1: timestep별 속도 드리프트 ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for key, lab in (("D_full", "full (velocity + condition)"),
                     ("D_vel", "velocity net only (c fixed to old)"),
                     ("D_cond", "condition drift only")):
        m, ci = band(z[key])
        axes[0].plot(s, m, label=lab)
        axes[0].fill_between(s, m - ci, m + ci, alpha=0.2)
    axes[0].set(xlabel=XLAB, ylabel=r"$\|v_{old}-v_{new}\|^2$", title="Velocity drift")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    m, ci = band(z["D_rel"])
    axes[1].plot(s, m, color="tab:red")
    axes[1].fill_between(s, m - ci, m + ci, alpha=0.2, color="tab:red")
    axes[1].set(xlabel=XLAB, ylabel=r"$\|v_{old}-v_{new}\|/\|v_{old}\|$",
                title="Relative velocity drift")
    axes[1].grid(alpha=0.3)
    fig.suptitle("G1 Fig.1 — timestep-wise velocity drift (mean +/- 95% CI)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "G1_F1_velocity_drift.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    srs = summary.get("sr", {})
    sr_old = srs.get("old (baseline)")

    # ── Figure 2: 단일 스텝 개입의 SR 영향 ───────────────────────────────────
    pts = []
    for name, v in srs.items():
        if name.startswith("single step k="):
            k = int(name.split("k=")[1].split(" ")[0])
            pts.append((k / T, sr_old - v))
    if pts:
        pts.sort()
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot(*zip(*pts), "-o", ms=5)
        ax.axhline(0, color="k", lw=0.8)
        if "new (all steps)" in srs:
            ax.axhline(sr_old - srs["new (all steps)"], color="tab:red", ls="--",
                       label=f"all steps new (SR drop {sr_old - srs['new (all steps)']:.0f})")
            ax.legend(fontsize=8)
        ax.set(xlabel=XLAB, ylabel="SR drop vs old (pp)",
               title="G1 Fig.2 — single-step replacement: behavioral sensitivity")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "G1_F2_single_step_sensitivity.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # ── Figure 3: early / middle / late 구간 개입 ────────────────────────────
    bars = [(k, v) for k, v in srs.items() if k.startswith("window") or k.startswith("old")
            or k.startswith("new (all")]
    if bars:
        order = ["old (baseline)", "new (all steps)"] + sorted(
            [k for k in srs if k.startswith("window")])
        order = [k for k in order if k in srs]
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(order)), 4.6))
        cols = ["tab:green" if k.startswith("old") else "tab:red" if k.startswith("new")
                else "tab:blue" for k in order]
        ax.bar(range(len(order)), [srs[k] for k in order], color=cols)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([k.replace("window ", "").split(" [")[0] for k in order],
                           rotation=30, ha="right", fontsize=8)
        ax.set(ylabel="SR (%)", title="G1 Fig.3 — window replacement (early vs middle vs late)")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(run_dir / "G1_F3_window_intervention.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # ── Figure 4: 초기 노이즈 민감도 ─────────────────────────────────────────
    if "ep_drift" in z and "ep_old_success" in z:
        d = z["ep_drift"]
        reg = region_slices(T)
        early = d[:, reg["early"]].mean(axis=1)
        late = d[:, reg["late"]].mean(axis=1)
        so, sn = z["ep_old_success"], z["ep_new_success"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        idx = np.arange(len(so))
        axes[0].bar(idx - 0.2, so, width=0.4, label="old success")
        axes[0].bar(idx + 0.2, sn, width=0.4, label="new success")
        axes[0].set(xlabel="episode (fixed initial noise & initial state)", ylabel="success",
                    title="per-episode outcome")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="y")
        deg = z["ep_degrade"]
        axes[1].scatter(early, deg, label="early drift", alpha=0.8)
        axes[1].scatter(late, deg, label="late drift", alpha=0.8, marker="^")
        axes[1].set(xlabel="mean velocity drift", ylabel="behavioral degradation (old - new)",
                    title="drift vs degradation")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        fig.suptitle("G1 Fig.4 — initial-noise / episode-level sensitivity", fontweight="bold")
        fig.tight_layout()
        fig.savefig(run_dir / "G1_F4_noise_sensitivity.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # ── Figure 5: early vs late 상관 ─────────────────────────────────────────
    if "corr_pearson" in summary:
        cp = np.array([np.nan if v is None else v for v in summary["corr_pearson"]])
        cs = np.array([np.nan if v is None else v for v in summary["corr_spearman"]])
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        axes[0].plot(s, cp, label="Pearson")
        axes[0].plot(s, cs, label="Spearman")
        axes[0].axhline(0, color="k", lw=0.8)
        axes[0].set(xlabel=XLAB, ylabel="corr(D_t, degradation)",
                    title="timestep-wise correlation")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
        names, vals = [], []
        for rname in ("early", "middle", "late"):
            r = summary["regions"].get(rname, {})
            if "corr_pearson" in r:
                names.append(rname); vals.append(r["corr_pearson"])
        if names:
            axes[1].bar(names, vals, color=["tab:blue", "tab:orange", "tab:green"])
            axes[1].axhline(0, color="k", lw=0.8)
            axes[1].set(ylabel="Pearson r", title="early vs middle vs late")
            axes[1].grid(alpha=0.3, axis="y")
        fig.suptitle("G1 Fig.5 — early vs late drift correlation with degradation",
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(run_dir / "G1_F5_early_vs_late_corr.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    print(f"그림 저장 -> {run_dir}/G1_F*.png")


def write_table(run_dir: Path, summary: dict) -> None:
    rows = ["metric,early,middle,late"]
    keys = [("velocity_drift", "Velocity drift"),
            ("relative_velocity_drift", "Relative velocity drift"),
            ("velocity_drift_velnet_only", "Velocity drift (vel-net only)"),
            ("velocity_drift_cond_only", "Velocity drift (condition only)"),
            ("intervention_sr", "Intervention SR"),
            ("behavioral_impact", "Behavioral impact (SR drop)"),
            ("corr_pearson", "Correlation with degradation")]
    for k, label in keys:
        vals = [summary["regions"].get(r, {}).get(k) for r in ("early", "middle", "late")]
        if all(v is None for v in vals):
            continue
        rows.append(label + "," + ",".join("" if v is None else f"{v:.6g}" for v in vals))
    (run_dir / "G1_summary.csv").write_text("\n".join(rows) + "\n")
    print(f"표 저장 -> {run_dir}/G1_summary.csv")
    print("\n".join(rows))


# ═════════════════════════════════════════════════════════════════════════════
#  메인
# ═════════════════════════════════════════════════════════════════════════════
@parser.wrap()
def main(cfg: G1Config):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    run_tag = cfg.run_tag or f"task{cfg.probe_task}"
    run_dir = Path(cfg.out_root) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.seed is not None:
        set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True

    ds_meta = LeRobotDatasetMetadata(f"{cfg.dataset_prefix}{cfg.probe_task}")
    logging.info(f"[G1] old = {cfg.old_ckpt}")
    logging.info(f"[G1] new = {cfg.new_ckpt}")
    policy_old = load_policy(cfg.old_ckpt, ds_meta, device)
    policy_new = load_policy(cfg.new_ckpt, ds_meta, device)

    h_old, h_new = norm_hash(policy_old), norm_hash(policy_new)
    logging.info(f"[G1] 정규화 통계 해시  old={h_old}  new={h_new}")
    if h_old != h_new:
        raise SystemExit(
            "두 체크포인트의 정규화 통계가 다르다. 같은 batch를 두 모델에 먹일 수 없으므로 "
            "드리프트 측정이 성립하지 않는다.")

    # 파라미터가 실제로 얼마나 움직였는지 (드리프트가 0으로 나올 때 원인 구분용)
    po = dict(policy_old.named_parameters())
    pn = dict(policy_new.named_parameters())
    moved = sum(int((po[k] != pn[k]).sum()) for k in po if k in pn)
    total = sum(po[k].numel() for k in po)
    logging.info(f"[G1] 파라미터 변화: {moved:,}/{total:,} ({100.0 * moved / total:.1f}%)")

    T = policy_old.dit_flow.num_inference_steps
    logging.info(colored(f"[G1] num_inference_steps={T}; s=k/T, s=0이 초기 노이즈, "
                         f"s=1이 최종 액션 (sample() 코드에서 확인)", "yellow"))

    sanity = sanity_integrator(policy_old, policy_new, device)

    drift = phase_a_drift(cfg, policy_old, policy_new, device)

    logging.info(colored("[G1][B] held-out MSE", "cyan", attrs=["bold"]))
    mse_old = heldout_mse(cfg, policy_old, device)
    mse_new = heldout_mse(cfg, policy_new, device)
    logging.info(f"[G1][B] MSE old={mse_old:.5f}  new={mse_new:.5f}  "
                 f"증가율={100.0 * (mse_new - mse_old) / max(mse_old, 1e-9):+.1f}%")

    raw = dict(drift)
    meta = {
        "old_ckpt": cfg.old_ckpt, "new_ckpt": cfg.new_ckpt, "probe_task": cfg.probe_task,
        "timesteps": T, "seed": cfg.seed, "sr_episodes": cfg.sr_episodes,
        "mse_old": mse_old, "mse_new": mse_new,
        "param_moved_frac": float(moved) / float(total),
        "sanity": {**sanity, "old_old_max_D": drift["sanity_old_old_max"],
                   "new_new_max_D": drift["sanity_new_new_max"]},
        "norm_hash": h_old,
    }

    if not cfg.skip_sr:
        res = phase_c_interventions(cfg, policy_old, policy_new, device)
        names = [v["name"] for v in res.values()]
        values = [v["sr"] for v in res.values()]
        raw["sr_names"] = np.array(names)
        raw["sr_values"] = np.array(values, dtype=np.float32)
        old_s = res["old"]["success"].astype(np.float32)
        new_s = res["new"]["success"].astype(np.float32)
        raw["ep_old_success"] = old_s
        raw["ep_new_success"] = new_s
        raw["ep_degrade"] = old_s - new_s
        if "drift_per_episode" in res["old"]:
            raw["ep_drift"] = res["old"]["drift_per_episode"]
            raw["ep_drift_all_calls"] = res["old"]["drift_all_calls"]
        meta["sr"] = dict(zip(names, values))
        meta["sanity_check4_old_matches_baseline"] = (
            "old 조건은 개입이 없으므로 표준 old 롤아웃과 같아야 한다 (같은 seed/init_state)")

    np.savez_compressed(run_dir / "g1_raw.npz", **raw)
    (run_dir / "g1_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logging.info(f"[G1] raw 저장 -> {run_dir}/g1_raw.npz")

    summary = analyze(run_dir)
    (run_dir / "g1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    plot_all(run_dir, summary)
    write_table(run_dir, summary)
    logging.info("[G1] done")


if __name__ == "__main__":
    if "--plot_only" in sys.argv:
        kv = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
        init_logging()
        rd = Path(kv.get("run_dir", "outputs/G1/task0"))
        s = analyze(rd)
        (rd / "g1_summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
        plot_all(rd, s)
        write_table(rd, s)
    else:
        mp.set_start_method("spawn", force=True)
        init_logging()
        main()
