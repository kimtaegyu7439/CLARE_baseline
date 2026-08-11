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

"""R3 — end-effector 절대 궤적을 방법 × CL 스테이지로 그린다.

무엇을 그리는가
    2 x 2 패널(seq / EWC λ=100 / ER / PackNet), 각 패널 안에서 CL 스테이지마다 선 색이
    다른 3D 궤적. 축은 로봇 베이스 기준 EE 절대 좌표 (x, y, z) [m].

프로토콜
    네 방법 × 네 스테이지의 체크포인트를 **모두 같은 태스크**(--probe_task, 기본 0)
    환경에서 롤아웃한다. 초기 상태는 시드가 아니라 **인덱스 0..R-1로 지정**하므로
    16개 체크포인트가 완전히 같은 초기 조건 집합을 본다 -> 궤적을 겹쳐 읽을 수 있다.
    stage 0은 "probe_task를 방금 배운 상태", stage 1~3은 "그 뒤로 다른 태스크를 배운
    상태"다. 색이 갈라지는 정도가 곧 망각의 크기다.

EE 좌표는 어디서 오는가
    gym_libero의 obs["agent_pos"] = hstack(robot0_eef_pos(3), axisangle(3), gripper(2)).
    앞 3개가 절대 위치다(env.py:174-178). 시뮬레이터 물리 상태이므로 정책이 망가져도
    자(尺)는 휘지 않는다.

PackNet
    체크포인트에 mask.safetensors가 같이 저장돼 있다. 마스크 값 v는 "v-1번 태스크 소유",
    0은 가지치기로 비워진 슬롯이다. task j를 평가할 때는 v > j+1 인 가중치를 0으로
    만들어야 PackNet이 PackNet으로 동작한다(--packnet_methods로 끌 수 있다).
    제대로 적용되면 네 스테이지 궤적이 거의 겹쳐야 한다 — 안 겹치면 그 자체가 발견이다.

이 스크립트는 학습을 하지 않는다. 저장된 체크포인트를 읽어 롤아웃만 돌린다.

사용 예
    python R3.py \
        --policy.path=<아무 체크포인트나> \
        --dataset.repo_id=continuallearning/libero_spatial_image_task_0 \
        --env.type=libero --env.benchmark=libero_spatial \
        --ckpt_roots="seq=outputs/E0/libero_spatial/seed_42/lam0,ewc=...,er=...,packnet=..." \
        --probe_task=0 --num_rollouts=5 --run_tag=libero_spatial_seed42_probe0
    python R3.py --plot_only --run_dir=outputs/R3/libero_spatial_seed42_probe0
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

# 스테이지 색. 원래는 순서형이라 단일 hue 램프(밝음->어두움)를 썼는데, 3D에서 스테이지당
# 5개씩 20개 선이 겹치니 명도만으로는 구분이 안 됐다. 그래서 **서로 다른 hue**로 바꿨다.
# 순서 정보는 범례 순서와 stage 라벨이 나른다.
#
# 팔레트 슬롯 1/2/3/7 (blue/orange/aqua/violet). 8개 슬롯의 4개 조합을 전수 검사해
# 고른 것이다 — all-pairs 기준(3D에서는 네 색이 동시에 보이므로 인접쌍만으로는 부족)으로
# CVD 최소 ΔE 9.2 (목표 8 이상), 정상시야 최소 ΔE 16.3 (하한 15), 최소 대비 2.74:1.
# 슬롯 4(yellow)는 흰 배경 대비가 2.11까지 떨어져 얇은 선에서 사라져 뺐다.
STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

# 두 번째 그림의 전문가 데모(정답 궤적) 색. 패널마다 **그 스테이지가 방금 배운 태스크의
# 데모 하나만** 그리므로 램프가 필요 없다 — 회색 하나면 된다.
# 채도가 0에 가까워(C=0.008) 채도 있는 stage 색과 절대 헷갈리지 않는다: 회색 = 정답,
# 색 = 정책. 흰 배경 대비 5.36:1.
DEMO_GRAY = "#6a6964"

# R3c의 파지 지점 마커. 노란 원 = 정책이 그리퍼를 닫은 곳, 빨간 세모 = 데모가 닫은 곳.
# ★ 빨강은 팔레트 기본값(#e34948)을 쓰지 않았다. stage 1의 주황(#eb6834)과 ΔE가 7.1밖에
#   안 돼, 하필 가설이 가장 선명한 주황 패널에서 마커와 궤적이 섞인다. #b3123f는 주황과
#   19.6, 데모 회색과 19.0, 노랑과 32.4로 떨어진다(대비 6.65:1).
GRIP_POLICY, GRIP_DEMO = "#eda100", "#b3123f"

# R3d에서 표시할 "지금 이 장면의 실제 물체". 데이터 계열이 아니라 세계의 주석이므로
# 새 색상 슬롯을 쓰지 않고 잉크색 **테두리만 있는 큰 마커**로 그린다. 채워진 마커(정책/데모
# 파지)와 빈 마커(물체)가 역할을 나눈다.
SCENE_OBJECTS = ["akita_black_bowl_1", "plate_1"]

# 패널 제목에 쓸 이름. ckpt_roots의 키를 이 표로 옮긴다(없으면 키를 그대로 쓴다).
METHOD_LABEL = {
    "seq": "Sequential fine-tuning",
    "ewc": "EWC (λ=100)",
    "er": "Experience Replay",
    "packnet": "PackNet",
}


# ═════════════════════════════════════════════════════════════════════════════
#  설정
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class R3Config(TrainPipelineConfig):
    """train.py 인자 전부 + EE 궤적 수집용 인자. 학습 인자(steps/batch 등)는 무시된다."""

    # ── 어떤 체크포인트를 볼 것인가 ──────────────────────────────────────────
    # "seq=<tree>,ewc=<tree>,er=<tree>,packnet=<tree>" 형식.
    # 각 tree 아래에 task_{k}/checkpoints/last/pretrained_model 이 있어야 한다.
    ckpt_roots: str = ""
    num_stages: int = 4                    # 방법당 스테이지 수 (task_0..task_{n-1})
    probe_task: int = 0                    # 모든 체크포인트를 굴릴 태스크

    # ── 환경 / 롤아웃 ────────────────────────────────────────────────────────
    env_task_prefix: str = "Libero_Spatial_Task_"
    num_rollouts: int = 5                  # 스테이지당 궤적 수 (패널당 num_stages배가 그려진다)
    max_steps: int = 0                     # 0 -> cfg.env.episode_length
    # 초기 상태 정착 스텝. LIBERO의 pruned_init은 물체를 테이블 위 ~7cm에 띄운 상태로
    # 저장돼 있어 그대로 시작하면 첫 프레임들이 자유낙하 아티팩트가 된다. R1과 같은 값.
    settle_steps: int = 5
    rollout_seed_base: int = 777000        # rollout_id -> flow matching a0 시드
    # 마스크를 적용할 방법 이름(쉼표 구분). 빈 문자열이면 아무 데도 적용하지 않는다.
    packnet_methods: str = "packnet"

    # ── 두 번째 그림: 정책 궤적 vs 전문가 데모 ───────────────────────────────
    # "망각한 모델의 궤적이 사실 가장 최근에 배운 태스크의 궤적이 아닌가"를 보는 그림.
    # 스테이지 k 패널에 (probe_task에서 굴린 stage k 롤아웃)과 (태스크 0~3의 데모)를
    # 같이 올린다. 롤아웃이 태스크 k의 데모를 따라가면 가설이 맞는 것이다.
    dataset_prefix: str = "continuallearning/libero_spatial_image_task_"
    demo_arm: str = "seq"                  # 어느 방법의 롤아웃을 볼 것인가
    demo_episodes: int = 6                 # 태스크당 그릴 데모 에피소드 수 (0이면 그림 생략)

    # ── 출력 / 제어 ──────────────────────────────────────────────────────────
    out_root: str = "outputs/R3"
    run_tag: str = ""
    recompute: bool = False                # 캐시가 있어도 다시 굴린다
    no_plot: bool = False

    def validate(self):
        """R1Config.validate와 같은 이유 — 캐시 재사용을 위해 output_dir 존재 검사를 우회한다.

        R3는 아무것도 학습하지 않으므로 산출물 덮어쓰기 위험이 없다.
        """
        out = self.output_dir
        if isinstance(out, Path) and out.is_dir():
            self.output_dir = None
            super().validate()
            self.output_dir = out
        else:
            super().validate()


# ═════════════════════════════════════════════════════════════════════════════
#  공통 유틸
# ═════════════════════════════════════════════════════════════════════════════
def parse_kv(spec: str) -> dict[str, str]:
    """'a=1,b=2' -> {'a': '1', 'b': '2'} (입력 순서 보존). R1.parse_kv와 동일."""
    out: dict[str, str] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"'name=value' 형식이 아니다: {item!r} (전체: {spec!r})")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def stage_ckpt(root: str, stage: int) -> Path:
    """E0/ER/PackNet가 남긴 스테이지 체크포인트 경로."""
    return Path(root) / f"task_{stage}" / "checkpoints" / "last" / "pretrained_model"


def load_policy_at(cfg: R3Config, ckpt: Path, ds_meta, device):
    """체크포인트의 파라미터로 정책을 만든다. 항상 eval 모드로 돌려준다."""
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"체크포인트가 없다: {ckpt}\n"
            f"  --ckpt_roots 가 산출물 트리를 가리키는지 확인해라 "
            f"(예: outputs/E0/libero_spatial/seed_42/lam0)."
        )
    pcfg = PreTrainedConfig.from_pretrained(ckpt)
    pcfg.pretrained_path = ckpt
    pcfg.device = cfg.policy.device
    policy = make_policy(cfg=pcfg, ds_meta=ds_meta)
    policy.eval()
    # dropout/BN이 살아 있으면 같은 관측에도 다른 행동이 나온다.
    assert not policy.training, "policy가 eval 모드가 아니다"
    return policy


@torch.no_grad()
def apply_packnet_mask(policy, ckpt: Path, keep_upto_task: int) -> str:
    """task `keep_upto_task` 이후에 배정된 슬롯을 0으로 만든다.

    packnet.py의 규약: mask[name] 값 v는 "v-1번 태스크 소유", v=0은 가지치기로 비워진
    슬롯이다(packnet.py:339에서 v=0인 자리를 current_task+1로 채운다). 따라서 task j를
    평가하려면 1 <= v <= j+1 만 남기고 v > j+1 을 0으로 만든다. v=0은 이미 가지치기로
    0에 가까우므로 건드리지 않는다.

    반환: 로그용 요약 문자열.
    """
    from safetensors.torch import load_file

    path = Path(ckpt) / "mask.safetensors"
    if not path.exists():
        raise SystemExit(
            f"PackNet 마스크가 없다: {path}\n"
            f"  이 체크포인트는 PackNet 산출물이 아니거나, --packnet_methods 에서 빼야 한다."
        )
    mask = load_file(str(path))
    zeroed = total = 0
    hit = 0
    for name, p in policy.named_parameters():
        if name not in mask:
            continue
        hit += 1
        m = mask[name].to(p.device)
        drop = m.gt(keep_upto_task + 1)
        p.data[drop] = 0.0
        zeroed += int(drop.sum())
        total += p.numel()
    if hit == 0:
        raise SystemExit(
            f"마스크 키가 정책 파라미터 이름과 하나도 안 맞는다: {path}\n"
            f"  mask 키 예시: {list(mask)[:3]}"
        )
    pct = 100.0 * zeroed / max(total, 1)
    return f"{hit}개 텐서, {zeroed:,}/{total:,} ({pct:.1f}%) 를 0으로 (task>{keep_upto_task} 소유)"


# ═════════════════════════════════════════════════════════════════════════════
#  환경
# ═════════════════════════════════════════════════════════════════════════════
def make_probe_env(cfg: R3Config):
    """probe_task의 gym_libero 환경 하나. R1.make_probe_env와 같은 이유로 단일 env를 쓴다.

    초기 상태를 rollout_id로 **직접 지정**해야 짝지은 비교가 되는데, LiberoEnv.reset은
    클래스 변수 카운터로 init_state를 고른다(env.py:199). 벡터 래퍼를 거치면 그 카운터를
    개별 env마다 되돌려야 한다.
    """
    import importlib

    import gymnasium as gym

    if cfg.env is None:
        raise SystemExit("--env.type=libero --env.benchmark=libero_spatial 가 필요하다.")
    importlib.import_module("gym_libero")
    handle = f"gym_libero/{cfg.env_task_prefix}{cfg.probe_task}"
    kwargs = dict(cfg.env.gym_kwargs)
    # 정착 스텝이 TimeLimit 예산을 갉아먹지 않게 한도를 늘려 준다.
    kwargs["max_episode_steps"] = (cfg.max_steps or cfg.env.episode_length) + cfg.settle_steps + 1
    return gym.make(handle, disable_env_checker=True, **kwargs)


# ═════════════════════════════════════════════════════════════════════════════
#  롤아웃
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def rollout_ee(cfg: R3Config, env, policy, device, method: str, stage: int, ckpt: Path) -> dict:
    """체크포인트 하나로 num_rollouts개를 굴리고 스텝별 EE 절대좌표를 기록한다.

    초기 상태는 rollout_id로 지정한다(시드가 아니라 인덱스). 모든 체크포인트가 같은
    집합을 쓰므로 궤적을 겹쳐 읽을 수 있다.
    """
    max_steps = cfg.max_steps or cfg.env.episode_length
    R = cfg.num_rollouts

    init_states = env.unwrapped._init_states
    if R > len(init_states):
        raise SystemExit(f"num_rollouts={R} > 사용 가능한 초기 상태 {len(init_states)}개")

    ee = np.full((R, max_steps, 3), np.nan, dtype=np.float32)
    # 그리퍼. R3c(파지 지점 표시)가 쓴다.
    #   grip_cmd : 정책이 **명령한** 그리퍼 (action의 마지막 차원, ±1). 데모의 action과
    #              정의가 완전히 같아 그대로 비교된다 -> "어디서 잡기로 결정했는가".
    #   grip_qpos: 실제 손가락 관절(agent_pos[6:8])의 간격. 명령이 물리적으로 먹혔는지
    #              보는 교차 확인용(명령만 보면 허공에서 쥐는 것도 파지로 세게 된다).
    grip_cmd = np.full((R, max_steps), np.nan, dtype=np.float32)
    grip_qpos = np.full((R, max_steps), np.nan, dtype=np.float32)
    lengths = np.zeros(R, dtype=np.int32)
    success = np.zeros(R, dtype=bool)

    task_text = env.unwrapped.task_description
    # 정착용 null 액션: OSC_POSE 델타 6개 = 0, 그리퍼는 열림(-1).
    null_action = np.zeros(env.action_space.shape, dtype=np.float32)
    null_action[-1] = -1.0

    for rid in range(R):
        env.reset()
        raw = env.unwrapped.set_init_state(init_states[rid])   # 초기 상태를 명시적으로 고정
        obs = env.unwrapped._format_raw_obs(raw)
        for _ in range(cfg.settle_steps):                      # 물체를 테이블에 내려앉힌다
            obs, _r, _term, _trunc, _i = env.step(null_action)
        policy.reset()
        # ★ flow matching의 a0는 전역 RNG에서 나온다. rollout_id로 고정해야 같은 초기
        #   상태에서 체크포인트끼리 짝지은 비교가 된다.
        torch.manual_seed(cfg.rollout_seed_base + rid)

        t = 0
        for t in range(max_steps):
            ee[rid, t] = np.asarray(obs["agent_pos"][:3], dtype=np.float32)   # 스텝 t의 EE 절대좌표
            # gripper_qpos 두 관절의 간격. 데모에서 열림 0.068, 파지 시 0.004까지 좁아진다.
            grip_qpos[rid, t] = float(obs["agent_pos"][6] - obs["agent_pos"][7])

            proc = preprocess_observation(obs)
            proc.pop("task", None)
            batch = {k: v.to(device) for k, v in proc.items() if isinstance(v, torch.Tensor)}
            batch["task"] = [task_text]
            action = policy.select_action(batch).squeeze(0).cpu().numpy()
            grip_cmd[rid, t] = float(action[-1])   # 데모 action의 마지막 차원과 같은 정의

            obs, _reward, terminated, truncated, _info = env.step(
                np.asarray(action, dtype=np.float32))
            if terminated or truncated:
                success[rid] = bool(terminated)                 # LiberoEnv: terminated == 성공
                break

        lengths[rid] = t + 1
        logging.info(f"[R3] {method} stage{stage} rollout {rid + 1}/{R}: "
                     f"len={lengths[rid]} success={bool(success[rid])}")

    sr = float(success.mean())
    logging.info(colored(f"[R3] {method} stage{stage}: SR {sr * 100:.0f}% ({success.sum()}/{R})",
                         "cyan"))
    return {
        "ee": ee,
        "grip_cmd": grip_cmd,
        "grip_qpos": grip_qpos,
        "lengths": lengths,
        "success": success,
        "meta": np.array(json.dumps({
            "method": method,
            "stage": stage,
            "ckpt": str(ckpt),
            "probe_task": cfg.probe_task,
            "num_rollouts": R,
            "max_steps": max_steps,
            "settle_steps": cfg.settle_steps,
            "seed_base": cfg.rollout_seed_base,
            "task": task_text,
        })),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  메인 (train.py / E0 / E1 / R1 과 같은 [1]~ 순서)
# ═════════════════════════════════════════════════════════════════════════════
@parser.wrap()
def main(cfg: R3Config):
    # [1] 설정
    cfg.validate()
    cfg.save_checkpoint = False
    if not cfg.ckpt_roots:
        raise SystemExit(
            '--ckpt_roots 가 필요하다 (예: "seq=outputs/E0/.../lam0,ewc=outputs/E0/.../lam100").')
    logging.info(pformat(cfg.to_dict()))

    roots = parse_kv(cfg.ckpt_roots)
    masked = {m.strip() for m in cfg.packnet_methods.split(",") if m.strip()}
    unknown = masked - set(roots)
    if unknown:
        raise SystemExit(f"--packnet_methods 에 ckpt_roots 에 없는 이름: {sorted(unknown)}")
    run_dir = Path(cfg.out_root) / (cfg.run_tag or f"probe{cfg.probe_task}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(colored(f"[R3] {len(roots)}개 방법 x {cfg.num_stages}스테이지 "
                         f"-> probe_task {cfg.probe_task} 에서 롤아웃  ({run_dir})",
                         "green", attrs=["bold"]))

    # [2] 재현성
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # [3] 디바이스
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # [4] 데이터셋 메타 (정책 생성용 — 데이터는 읽지 않는다)
    ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, revision=cfg.dataset.revision)

    # [5] 환경
    env = make_probe_env(cfg)
    logging.info(f"[R3] env task: {env.unwrapped.task_description!r}")

    # [6] 방법 x 스테이지 롤아웃 (캐시가 있으면 건너뛴다)
    for method, root in roots.items():
        for stage in range(cfg.num_stages):
            cache = run_dir / f"{method}_stage{stage}.npz"
            if cache.exists() and not cfg.recompute:
                logging.info(f"[R3] 캐시 재사용: {cache.name}")
                continue
            ckpt = stage_ckpt(root, stage)
            policy = load_policy_at(cfg, ckpt, ds_meta, device)
            if method in masked:
                summary = apply_packnet_mask(policy, ckpt, cfg.probe_task)
                logging.info(f"[R3] {method} stage{stage} 마스크 적용: {summary}")
            rec = rollout_ee(cfg, env, policy, device, method, stage, ckpt)
            np.savez_compressed(cache, **rec)
            logging.info(f"[R3] saved -> {cache}")
            del policy
            torch.cuda.empty_cache()

    # 장면 물체 좌표(R3d용). 롤아웃과 같은 init_states / 같은 정착 스텝으로 읽는다.
    dump_scene_objects(cfg, env, run_dir / "scene_objects.npz")
    env.close()
    logging.info(colored(f"[R3] 롤아웃 완료 -> {run_dir}", "green", attrs=["bold"]))

    # [7] 그림
    if not cfg.no_plot:
        plot_r3(str(run_dir))
        if cfg.demo_episodes > 0:
            plot_r3_vs_demo(str(run_dir), cfg.demo_arm, cfg.dataset_prefix, cfg.demo_episodes)
            plot_r3_vs_demo(str(run_dir), cfg.demo_arm, cfg.dataset_prefix, cfg.demo_episodes,
                            grip=True)
            plot_r3_vs_demo(str(run_dir), cfg.demo_arm, cfg.dataset_prefix, cfg.demo_episodes,
                            grip=True, scene=True)


# ═════════════════════════════════════════════════════════════════════════════
#  그림
# ═════════════════════════════════════════════════════════════════════════════
def plot_r3(run_dir_str: str) -> None:
    run_dir = Path(run_dir_str)
    caches = sorted(run_dir.glob("*_stage*.npz"))
    if not caches:
        raise SystemExit(f"npz 캐시가 없다: {run_dir}")

    recs: dict[tuple[str, int], dict] = {}
    for p in caches:
        method, tail = p.stem.rsplit("_stage", 1)
        with np.load(p, allow_pickle=False) as z:
            recs[(method, int(tail))] = {k: z[k] for k in z.files}
    methods = sorted({m for m, _ in recs}, key=lambda m: list(METHOD_LABEL).index(m)
                     if m in METHOD_LABEL else 99)
    stages = sorted({s for _, s in recs})
    meta = json.loads(str(next(iter(recs.values()))["meta"]))

    # ── CSV (그림과 같은 수치를 표로도 남긴다) ────────────────────────────────
    csv_path = run_dir / "R3_ee_trajectory.csv"
    with open(csv_path, "w") as f:
        f.write("method,stage,rollout,step,x,y,z\n")
        for (m, s), rec in sorted(recs.items()):
            for r in range(rec["ee"].shape[0]):
                for t in range(int(rec["lengths"][r])):
                    x, y, z = rec["ee"][r, t]
                    f.write(f"{m},{s},{r},{t},{x:.6f},{y:.6f},{z:.6f}\n")
    print(f"saved table  -> {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError:
        print("matplotlib 없음 -> 그림 생략")
        return

    # 모든 패널이 **같은 축 범위**를 써야 궤적을 겹쳐 읽을 수 있다.
    allpts = np.concatenate([rec["ee"][r, :int(rec["lengths"][r])]
                             for rec in recs.values() for r in range(rec["ee"].shape[0])])
    lo, hi = allpts.min(0), allpts.max(0)
    pad = (hi - lo) * 0.08 + 1e-3
    lo, hi = lo - pad, hi + pad

    ncol = 2
    nrow = int(np.ceil(len(methods) / ncol))
    fig = plt.figure(figsize=(13.5, 5.4 * nrow))
    for i, m in enumerate(methods):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        srs = []
        for s in stages:
            rec = recs.get((m, s))
            if rec is None:
                continue
            col = STAGE_COLORS[s % len(STAGE_COLORS)]
            for r in range(rec["ee"].shape[0]):
                n = int(rec["lengths"][r])
                tr = rec["ee"][r, :n]
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=col, lw=1.6, alpha=0.85,
                        solid_capstyle="round")
                # 끝점만 표시한다. 시작점은 초기 상태가 공유되어 모두 같은 자리다.
                ax.scatter(*tr[-1], color=col, s=22, depthshade=False,
                           edgecolors="white", linewidths=0.6, zorder=5)
            srs.append(f"s{s} {rec['success'].mean() * 100:.0f}%")
        # 시작점을 회색 삼각형으로. 초기 상태가 체크포인트끼리 공유되므로 스테이지별로
        # 찍을 필요가 없다 — 아무 스테이지의 rollout별 첫 프레임이면 전부 같은 자리다.
        st = next(rec["ee"][:, 0] for (mm, _), rec in recs.items() if mm == m)
        ax.scatter(st[:, 0], st[:, 1], st[:, 2], color=INK2, s=38, marker="^",
                   depthshade=False, zorder=6)

        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        # zoom>1 로 3D 축의 기본 여백을 줄인다. 그냥 두면 패널 면적의 30%가 빈다.
        ax.set_box_aspect(hi - lo, zoom=1.15)
        ax.set_xlabel("x [m]", color=INK2, fontsize=9, labelpad=1)
        ax.set_ylabel("y [m]", color=INK2, fontsize=9, labelpad=1)
        ax.set_zlabel("z [m]", color=INK2, fontsize=9, labelpad=1)
        ax.tick_params(colors=INK2, labelsize=7.5, pad=0)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor("white")
            pane.pane.set_edgecolor(GRID)
            pane._axinfo["grid"]["color"] = GRID
            pane._axinfo["grid"]["linewidth"] = 0.6
        ax.set_title(f"{METHOD_LABEL.get(m, m)}\nSR: " + "  ".join(srs),
                     color=INK, fontsize=11, pad=6)
        ax.view_init(elev=22, azim=-58)

    handles = [Line2D([0], [0], color=STAGE_COLORS[s % len(STAGE_COLORS)], lw=2.4,
                      label=f"after task {s}") for s in stages]
    handles.append(Line2D([0], [0], color=INK2, lw=0, marker="^", ms=7, label="start (shared)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=10, labelcolor=INK2, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        f"R3: end-effector trajectory on task {meta['probe_task']} across CL stages\n"
        f"{meta['num_rollouts']} rollouts per stage, identical initial states  ·  "
        f"task: {meta['task']}",
        fontsize=13, color=INK, y=0.985)
    # 3D 축에는 tight_layout이 잘 듣지 않는다(내부 여백을 못 읽는다). 직접 잡는다.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.06, wspace=0.02, hspace=0.20)
    out = run_dir / "R3_ee_trajectory.png"
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"saved figure -> {out}")


# ═════════════════════════════════════════════════════════════════════════════
#  그림 2 — 정책 궤적 vs 전문가 데모
# ═════════════════════════════════════════════════════════════════════════════
def load_demo_ee(prefix: str, tasks: list[int], n_episodes: int, cache: Path) -> dict:
    """태스크별 전문가 데모. {task: [ (T,4) x n_episodes ]}  — [:3]=EE xyz, [3]=그리퍼 명령.

    ★ 롤아웃의 obs["agent_pos"]와 데이터셋의 observation.state는 **같은 출처**다
      (gym_libero env.py의 _format_raw_obs가 만든 그 배열이 그대로 기록됐다).
      그래서 좌표계 변환 없이 그대로 겹칠 수 있다. 앞 3차원이 robot0_eef_pos.

    이미지 컬럼을 건드리면 5천 장을 디코딩하게 되므로 select_columns로 잘라 읽는다
    (태스크당 0.1초). 결과는 npz로 캐시해 --plot_only 재실행 시 다시 받지 않는다.
    """
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            out: dict[int, list] = {t: [] for t in tasks}
            for k in sorted(z.files, key=lambda s: (int(s.split("_")[0][1:]),
                                                    int(s.split("_")[1][1:]))):
                t = int(k.split("_")[0][1:])
                if t in out:
                    out[t].append(z[k])
        if all(out[t] for t in tasks):
            logging.info(f"[R3] 데모 캐시 재사용: {cache.name}")
            return out

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = {}
    blob = {}
    for t in tasks:
        ds = LeRobotDataset(f"{prefix}{t}")
        sub = ds.hf_dataset.select_columns(["observation.state", "action", "episode_index"])
        st = np.asarray(sub["observation.state"], dtype=np.float32)[:, :3]
        # action의 마지막 차원이 그리퍼 명령(±1). 롤아웃의 grip_cmd와 정의가 같다.
        gc = np.asarray(sub["action"], dtype=np.float32)[:, -1:]
        st = np.concatenate([st, gc], axis=1)                     # (N, 4)
        ep = np.asarray(sub["episode_index"])
        n = min(n_episodes, int(ep.max()) + 1)
        out[t] = [st[ep == e] for e in range(n)]
        for e, tr in enumerate(out[t]):
            blob[f"t{t}_e{e}"] = tr
        logging.info(f"[R3] task {t} 데모 {n}개 에피소드 (프레임 {len(st)})")
        del ds, sub
    np.savez_compressed(cache, **blob)
    return out


def dump_scene_objects(cfg: R3Config, env, cache: Path) -> None:
    """롤아웃이 실제로 본 장면의 물체 위치를 저장한다. {obj: (R, 3)}.

    왜 필요한가: R3c의 빨간 세모는 **태스크 k 장면**에서 데모가 잡은 자리다. 정책이
    거기서 그리퍼를 닫았다고 해서 곧바로 "빈 자리를 쥐었다"가 되지는 않는다 — 지금
    장면(probe_task)의 그릇이 어디 있는지를 같이 봐야 판정이 된다.

    초기 상태 인덱스가 정해지면 물체 배치도 정해지므로 롤아웃과 같은 init_states를
    같은 정착 스텝만큼 굴려 읽으면 롤아웃이 본 배치와 일치한다.

    ★ 한계: 정착 직후 위치다. 정책이 그릇을 밀친 뒤 잡으려 했다면 어긋난다. 매 스텝
      기록하려면 롤아웃을 다시 굴려야 한다(물체 pose는 캐시에 없다).
    """
    be = env.unwrapped._env
    missing = [o for o in SCENE_OBJECTS if o not in be.obj_body_id]
    if missing:
        raise SystemExit(f"장면에 없는 물체: {missing}  (있는 것: {sorted(be.obj_body_id)})")
    init_states = env.unwrapped._init_states
    null_action = np.zeros(env.action_space.shape, dtype=np.float32)
    null_action[-1] = -1.0

    out = {o: np.zeros((cfg.num_rollouts, 3), dtype=np.float32) for o in SCENE_OBJECTS}
    for rid in range(cfg.num_rollouts):
        env.reset()
        sim = env.unwrapped._env.sim              # reset마다 MjSim이 새로 만들어진다
        be = env.unwrapped._env
        env.unwrapped.set_init_state(init_states[rid])
        for _ in range(cfg.settle_steps):
            env.step(null_action)
        for o in SCENE_OBJECTS:
            out[o][rid] = sim.data.body_xpos[be.obj_body_id[o]]
    np.savez_compressed(cache, **out)
    for o in SCENE_OBJECTS:
        logging.info(f"[R3] scene {o}: " + "  ".join(f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"
                                                     for p in out[o]))
    logging.info(f"[R3] saved -> {cache}")


def grip_events(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(닫는 순간, 여는 순간)의 스텝 인덱스.

    값이 아니라 **전이**를 잡는다. 닫힌 채 유지되는 구간을 전부 세면 오래 쥐고 있는
    궤적에 마커가 수십 개 찍힌다. 보려는 건 "언제 결정이 바뀌었나"다.

    ★ 여는 순간은 i>0에서만 센다. 에피소드는 열린 상태로 시작하므로 t=0을 '여는 사건'으로
      세면 모든 롤아웃에 가짜 마커가 하나씩 생긴다. 반대로 닫는 순간은 t=0도 사건이다
      (시작하자마자 쥐는 건 실제로 일어나는 이상 동작이다).
    """
    g = np.asarray(g, dtype=np.float32)
    g = np.where(np.isfinite(g), g, -1.0)
    closed = g > 0
    prev = np.concatenate([[False], closed[:-1]])
    close = np.flatnonzero(closed & ~prev)
    open_ = np.flatnonzero(~closed & prev)
    return close, open_


def grip_onsets(g: np.ndarray) -> np.ndarray:
    """그리퍼가 '열림 -> 닫힘'으로 바뀌는 스텝 인덱스들.

    g는 ±1 명령열(데모의 action 마지막 차원, 롤아웃의 grip_cmd). 값이 아니라 **전이**를
    잡는 이유: 닫힌 채로 유지되는 구간을 전부 세면 오래 쥐고 있는 궤적에 마커가 수십 개
    찍힌다. 우리가 보려는 건 "어디서 잡기로 결정했는가" 한 점이다.
    재파지(놓았다 다시 잡기)는 별개의 결정이므로 전이가 여러 번이면 전부 표시한다.
    """
    g = np.asarray(g, dtype=np.float32)
    ok = np.isfinite(g)
    if not ok.any():
        return np.array([], dtype=int)
    g = np.where(ok, g, -1.0)
    closed = g > 0
    return np.flatnonzero(closed & ~np.concatenate([[False], closed[:-1]]))


def plot_r3_vs_demo(run_dir_str: str, arm: str, prefix: str, n_episodes: int,
                    grip: bool = False, scene: bool = False) -> None:
    """스테이지 k 패널 = (probe_task에서 굴린 stage k 롤아웃) + (태스크 k의 전문가 데모).

    묻는 것: 망각한 모델이 probe_task에서 그리는 궤적이, 사실은 **가장 최근에 배운
    태스크**의 궤적인가? 그렇다면 stage k의 색 궤적이 task k의 회색 궤적에 붙는다.

    grip=True (R3c)면 파지 지점을 얹는다. 이게 궤적 모양보다 강한 증거다: probe_task의
    그릇은 다른 자리에 있는데도 정책이 **최근 태스크의 그릇 자리**에서 그리퍼를 닫으면,
    이미지와 언어 지시를 무시하고 최근 태스크를 재생하고 있다는 뜻이 된다.
    """
    run_dir = Path(run_dir_str)
    recs = {}
    for p in sorted(run_dir.glob(f"{arm}_stage*.npz")):
        recs[int(p.stem.rsplit("_stage", 1)[1])] = {k: v for k, v in np.load(p).items()}
    if not recs:
        raise SystemExit(f"'{arm}' 팔의 npz가 없다: {run_dir}  (--demo_arm 확인)")
    stages = sorted(recs)
    meta = json.loads(str(recs[stages[0]]["meta"]))
    probe = meta["probe_task"]

    demos = load_demo_ee(prefix, stages, n_episodes, run_dir / "demo_traj.npz")

    objs = {}
    if scene:
        sp = run_dir / "scene_objects.npz"
        if not sp.exists():
            raise SystemExit(
                f"장면 물체 위치가 없다: {sp}\n"
                f"  R3d는 probe_task 장면의 실제 그릇/접시 좌표가 필요하다. 다시 돌려라:\n"
                f"    bash bash/E0/R3.sh   (롤아웃 캐시가 있으면 물체 좌표만 새로 읽는다)"
            )
        with np.load(sp, allow_pickle=False) as z:
            objs = {k: z[k] for k in z.files}

    # ── 눈이 아니라 숫자로 판정한다 ──────────────────────────────────────────
    # 스테이지 k의 롤아웃 점들에서 각 태스크 데모까지의 **평균 최근접거리**.
    # 가설이 맞으면 행 k의 최솟값이 열 k에 온다. 시간 정렬이 필요 없는 척도라
    # (롤아웃과 데모는 길이도 속도도 다르다) 궤적 모양만 비교된다.
    pool = {t: np.concatenate(demos[t])[:, :3] for t in stages}

    def mean_nn(traj, ref):
        return float(np.linalg.norm(traj[:, None, :] - ref[None, :, :], axis=2).min(1).mean())

    dist = {}
    for s in stages:
        rec = recs[s]
        pts = np.concatenate([rec["ee"][r, :int(rec["lengths"][r])]
                              for r in range(rec["ee"].shape[0])])
        dist[s] = {t: mean_nn(pts, pool[t]) * 100 for t in stages}   # cm
    # 데모끼리의 거리도 같이 남긴다 — "자기 태스크가 최소"라는 결론이 얼마나 여유
    # 있는지는 데모들이 서로 얼마나 떨어져 있느냐에 달려 있다.
    csv_path = run_dir / (f"R3d_{arm}_scene.csv" if scene else
                      f"R3c_{arm}_grasp.csv" if grip else f"R3b_{arm}_vs_demo.csv")
    with open(csv_path, "w") as f:
        f.write("row," + ",".join(f"task{t}_demo_cm" for t in stages) + ",argmin\n")
        for s in stages:
            f.write(f"policy_after_task{s}," + ",".join(f"{dist[s][t]:.3f}" for t in stages)
                    + f",{min(stages, key=lambda t: dist[s][t])}\n")
        for a in stages:
            f.write(f"task{a}_demo," + ",".join(f"{mean_nn(pool[a], pool[b]) * 100:.3f}"
                                                for b in stages) + ",\n")
        if grip:
            # ★ 궤적 모양보다 강한 증거: **파지 지점**끼리의 거리.
            #   정책이 그리퍼를 닫은 자리가 어느 태스크 데모의 파지 자리에 가까운가.
            #   probe_task의 그릇이 아니라 최근 태스크의 그릇 자리라면 argmin이 자기
            #   스테이지 번호로 나온다.
            gdemo = {}
            for t in stages:
                pts_t = [tr[k, :3] for tr in demos[t] for k in [grip_onsets(tr[:, 3])] if len(k)]
                gdemo[t] = np.concatenate(pts_t) if pts_t else np.zeros((0, 3), np.float32)
            f.write("\ngrasp points\n")
            f.write("row,n_grasps," + ",".join(f"task{t}_grasp_cm" for t in stages) + ",argmin\n")
            for s in stages:
                rec = recs[s]
                pts_s = []
                for r in range(rec["ee"].shape[0]):
                    n = int(rec["lengths"][r])
                    k = grip_onsets(rec["grip_cmd"][r, :n]) if "grip_cmd" in rec else []
                    if len(k):
                        pts_s.append(rec["ee"][r, :n][k, :3])
                if not pts_s:
                    f.write(f"policy_after_task{s},0," + ",".join("" for _ in stages) + ",\n")
                    continue
                P = np.concatenate(pts_s)
                d = {t: (mean_nn(P, gdemo[t]) * 100 if len(gdemo[t]) else float("nan"))
                     for t in stages}
                f.write(f"policy_after_task{s},{len(P)}," + ",".join(f"{d[t]:.3f}" for t in stages)
                        + f",{min(stages, key=lambda t: d[t])}\n")
    print(f"saved table  -> {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError:
        print("matplotlib 없음 -> 그림 생략")
        return

    # 데모와 롤아웃을 모두 담는 공통 축. 패널마다 다르면 "붙었다/떨어졌다"를 눈으로
    # 비교할 수 없다.
    pts = [rec["ee"][r, :int(rec["lengths"][r])] for rec in recs.values()
           for r in range(rec["ee"].shape[0])]
    pts += [tr[:, :3] for trs in demos.values() for tr in trs]
    allpts = np.concatenate(pts)
    lo, hi = allpts.min(0), allpts.max(0)
    pad = (hi - lo) * 0.08 + 1e-3
    lo, hi = lo - pad, hi + pad

    ncol = 2
    nrow = int(np.ceil(len(stages) / ncol))
    fig = plt.figure(figsize=(13.5, 5.4 * nrow))
    for i, s in enumerate(stages):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        # ★ 회색 = 정답. 패널마다 **그 스테이지가 방금 배운 태스크의 데모 하나만** 그린다.
        #   네 태스크 데모를 다 깔면 어느 회색이 어느 태스크인지 세느라 정작 봐야 할
        #   "색이 회색에 붙었나"가 안 보인다. 네 태스크 전부와의 거리는 아래 숫자 상자가
        #   준다 — 그림은 한 가지만 묻고, 판정은 숫자가 한다.
        for tr in demos[s]:
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=DEMO_GRAY, lw=1.5, alpha=0.75, zorder=1)
            if grip:
                # 세모 = 닫힘(집기), 원 = 열림(놓기). 데모는 한 번씩만 나오므로 두 마커가
                # 곧 "어디서 집어 어디에 놓는 태스크인가"를 그대로 보여 준다.
                cl, op = grip_events(tr[:, 3])
                for idx, mk in ((cl, "^"), (op, "o")):
                    if len(idx):
                        ax.scatter(tr[idx, 0], tr[idx, 1], tr[idx, 2], color=GRIP_DEMO, s=95,
                                   marker=mk, depthshade=False, edgecolors="white",
                                   linewidths=1.0, zorder=7)
        rec = recs[s]
        col = STAGE_COLORS[s % len(STAGE_COLORS)]
        if grip and "grip_cmd" not in rec:
            raise SystemExit(
                f"이 캐시에는 그리퍼 기록이 없다: {arm}_stage{s}.npz\n"
                f"  R3c는 grip_cmd가 필요하다. 해당 팔을 다시 굴려라:\n"
                f"    rm {run_dir}/{arm}_stage*.npz && bash bash/E0/R3.sh"
            )
        for r in range(rec["ee"].shape[0]):
            n = int(rec["lengths"][r])
            tr = rec["ee"][r, :n]
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=col, lw=1.8, alpha=0.9,
                    solid_capstyle="round", zorder=4)
            ax.scatter(*tr[-1], color=col, s=24, depthshade=False,
                       edgecolors="white", linewidths=0.6, zorder=5)
            if grip:
                cl, op = grip_events(rec["grip_cmd"][r, :n])
                for idx, mk in ((cl, "^"), (op, "o")):
                    if len(idx):
                        ax.scatter(tr[idx, 0], tr[idx, 1], tr[idx, 2], color=GRIP_POLICY, s=80,
                                   marker=mk, depthshade=False, edgecolors=INK,
                                   linewidths=0.9, zorder=8)
        # 지금 이 장면(probe_task)의 실제 물체. 채움 없는 큰 마커라 파지 마커와 역할이 갈린다.
        if scene:
            for name, mk, sz, ec, lw in (("akita_black_bowl_1", "o", 300, INK, 2.2),
                                         ("plate_1", "s", 330, INK2, 1.8)):
                p = objs.get(name)
                if p is None:
                    continue
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], facecolors="none", edgecolors=ec,
                           marker=mk, s=sz, linewidths=lw, depthshade=False, zorder=9)

        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.15)
        ax.set_xlabel("x [m]", color=INK2, fontsize=9, labelpad=1)
        ax.set_ylabel("y [m]", color=INK2, fontsize=9, labelpad=1)
        ax.set_zlabel("z [m]", color=INK2, fontsize=9, labelpad=1)
        ax.tick_params(colors=INK2, labelsize=7.5, pad=0)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor("white")
            pane.pane.set_edgecolor(GRID)
            pane._axinfo["grid"]["color"] = GRID
            pane._axinfo["grid"]["linewidth"] = 0.6
        ax.set_title(f"policy after task {s}  —  rolled out on task {probe}   (SR "
                     f"{rec['success'].mean() * 100:.0f}%)\n"
                     f"gray = task {s} demo, the task it learned most recently",
                     color=INK, fontsize=10.5, pad=6)
        # 판정 숫자는 그림에 넣지 않는다 — 겹치는지 여부를 눈으로 보는 그림이다.
        # 네 태스크 데모와의 거리는 옆에 저장되는 CSV에 그대로 있다.
        ax.view_init(elev=22, azim=-58)

    handles = [Line2D([0], [0], color=DEMO_GRAY, lw=2.0,
                      label="expert demo of the task learned in that panel")]
    handles += [Line2D([0], [0], color=STAGE_COLORS[s % len(STAGE_COLORS)], lw=2.6,
                       label=f"policy after task {s}") for s in stages]
    if grip:
        # 모양 = 무슨 일(세모 닫힘 / 원 열림), 색 = 누가(노랑 정책 / 빨강 데모).
        handles += [
            Line2D([0], [0], color=GRIP_POLICY, lw=0, marker="^", ms=10, mec=INK, mew=0.9,
                   label="policy CLOSES (grasp)"),
            Line2D([0], [0], color=GRIP_POLICY, lw=0, marker="o", ms=9, mec=INK, mew=0.9,
                   label="policy OPENS (release)"),
            Line2D([0], [0], color=GRIP_DEMO, lw=0, marker="^", ms=10, mec="white", mew=1.0,
                   label="demo CLOSES (picks the bowl up)"),
            Line2D([0], [0], color=GRIP_DEMO, lw=0, marker="o", ms=9, mec="white", mew=1.0,
                   label="demo OPENS (drops it on the plate)"),
        ]
    if scene:
        handles += [
            Line2D([0], [0], lw=0, marker="o", ms=11, mfc="none", mec=INK, mew=2.0,
                   label=f"BOWL actually here (task {probe} scene)"),
            Line2D([0], [0], lw=0, marker="s", ms=11, mfc="none", mec=INK2, mew=1.8,
                   label=f"PLATE actually here (task {probe} scene)"),
        ]
    # 한 줄에 몰면 오른쪽 끝이 잘린다. 항목이 늘어날수록 줄을 나눈다.
    fig.legend(handles=handles, loc="lower center", ncol=4 if len(handles) > 6 else 5,
               frameon=False, fontsize=9.5, labelcolor=INK2, bbox_to_anchor=(0.5, 0.004))
    if scene:
        fig.suptitle(
            f"R3d: gripper events against the objects that are actually there\n"
            f"{METHOD_LABEL.get(arm, arm)}  ·  every panel rolled out on task {probe}  ·  "
            f"△ close  ○ open  ·  yellow = policy, red = demo  ·  hollow = real object",
            fontsize=13, color=INK, y=0.985)
    elif grip:
        fig.suptitle(
            f"R3c: the open/close pattern of the gripper\n"
            f"{METHOD_LABEL.get(arm, arm)}  ·  every panel rolled out on task {probe}  ·  "
            f"shape = what happened (△ close, ○ open),  colour = who (yellow policy, red demo)",
            fontsize=13, color=INK, y=0.985)
    else:
        fig.suptitle(
            f"R3b: does a forgetting policy trace the most recently learned task?\n"
            f"{METHOD_LABEL.get(arm, arm)}  ·  every panel rolled out on task {probe}  ·  "
            f"gray = that panel's own demo ({n_episodes} episodes)",
            fontsize=13, color=INK, y=0.985)
    bottom = 0.13 if len(handles) > 6 else 0.10
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=bottom,
                        wspace=0.02, hspace=0.20)
    out = run_dir / (f"R3d_{arm}_scene.png" if scene else
                 f"R3c_{arm}_grasp.png" if grip else f"R3b_{arm}_vs_demo.png")
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    init_logging()
    if "--plot_only" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--plot_only"]
        kv = dict(a.lstrip("-").split("=", 1) for a in args if "=" in a)
        if "run_dir" not in kv:
            raise SystemExit("--plot_only 에는 --run_dir=<outputs/R3/...> 가 필요하다")
        plot_r3(kv["run_dir"])
        n_ep = int(kv.get("demo_episodes", 6))
        if n_ep > 0:
            arm = kv.get("demo_arm", "seq")
            pre = kv.get("dataset_prefix", "continuallearning/libero_spatial_image_task_")
            plot_r3_vs_demo(kv["run_dir"], arm, pre, n_ep)
            plot_r3_vs_demo(kv["run_dir"], arm, pre, n_ep, grip=True)
            plot_r3_vs_demo(kv["run_dir"], arm, pre, n_ep, grip=True, scene=True)
    else:
        main()
