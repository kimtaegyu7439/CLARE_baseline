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
import warnings
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.utils import get_channel_first_image_shape


def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    # TODO(aliberts, rcadene): refactor this to use features from the environment (no hardcoding)
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            imgs = {f"observation.images.{key}": img for key, img in observations["pixels"].items()}
        else:
            imgs = {"observation.image": observations["pixels"]}

        for imgkey, img in imgs.items():
            # TODO(aliberts, rcadene): use transforms.ToTensor()?
            img = torch.from_numpy(img)

            # When preprocessing observations in a non-vectorized environment, we need to add a batch dimension.
            # This is the case for human-in-the-loop RL where there is only one environment.
            if img.ndim == 3:
                img = img.unsqueeze(0)
            # sanity check that images are channel last
            _, h, w, c = img.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

            # sanity check that images are uint8
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

            # convert to channel first of type float32 in range [0,1]
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255

            return_observations[imgkey] = img

    if "environment_state" in observations:
        env_state = torch.from_numpy(observations["environment_state"]).float()
        if env_state.dim() == 1:
            env_state = env_state.unsqueeze(0)

        return_observations["observation.environment_state"] = env_state

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    agent_pos = torch.from_numpy(observations["agent_pos"]).float()
    if agent_pos.dim() == 1:
        agent_pos = agent_pos.unsqueeze(0)
    return_observations["observation.state"] = agent_pos

    if "task" in observations:
        return_observations["task"] = observations["task"]

    return return_observations


def env_to_policy_features(env_cfg: EnvConfig) -> dict[str, PolicyFeature]:
    # TODO(aliberts, rcadene): remove this hardcoding of keys and just use the nested keys as is
    # (need to also refactor preprocess_observation and externalize normalization from policies)
    policy_features = {}
    for key, ft in env_cfg.features.items():
        if ft.type is FeatureType.VISUAL:
            if len(ft.shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={ft.shape})")

            shape = get_channel_first_image_shape(ft.shape)
            feature = PolicyFeature(type=ft.type, shape=shape)
        else:
            feature = ft

        policy_key = env_cfg.features_map[key]
        policy_features[policy_key] = feature

    return policy_features


def are_all_envs_same_type(env: gym.vector.VectorEnv) -> bool:
    # 래퍼가 아니라 실제 환경의 타입을 비교해야 한다. env.envs[i]는 전부 TimeLimit 등으로
    # 똑같이 감싸여 있어서, 안의 환경이 서로 달라도 래퍼 타입은 항상 같게 나온다.
    # (그러면 아래 check_env_attributes_and_types의 경고가 영영 안 뜬다.)
    first_type = type(env.envs[0].unwrapped)
    return all(type(e.unwrapped) is first_type for e in env.envs)


def check_env_attributes_and_types(env: gym.vector.VectorEnv) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("once", UserWarning)  # Apply filter only in this function

        if not (hasattr(env.envs[0].unwrapped, "task_description") and hasattr(env.envs[0].unwrapped, "task")):
            warnings.warn(
                "The environment does not have 'task_description' and 'task'. Some policies require these features.",
                UserWarning,
                stacklevel=2,
            )
        if not are_all_envs_same_type(env):
            warnings.warn(
                "The environments have different types. Make sure you infer the right task from each environment. Empty task will be passed instead.",
                UserWarning,
                stacklevel=2,
            )


def add_envs_task(env: gym.vector.VectorEnv, observation: dict[str, Any]) -> dict[str, Any]:
    """Adds task feature to the observation dict with respect to the first environment attribute.

    반드시 .unwrapped에서 속성을 확인해야 한다. env.envs[0]는 TimeLimit 같은 래퍼이고,
    gymnasium 1.0부터 Wrapper.__getattr__(속성 전달)이 제거되어 래퍼에서는 hasattr가
    항상 False가 된다. 그러면 아래 else로 빠져 언어 지시문이 빈 문자열이 되고,
    언어 조건 정책(예: DiT-Flow-MT의 CLIP 인코더)은 조건 없이 롤아웃하게 되어
    손실은 멀쩡한데 성공률만 0이 되는 형태로 조용히 실패한다.
    바로 위 check_env_attributes_and_types()가 .unwrapped를 쓰므로 경고도 뜨지 않는다.
    (env.call()은 래퍼를 통과하므로 게이트만 고치면 된다.)
    """
    first_env = env.envs[0].unwrapped
    if hasattr(first_env, "task_description"):
        observation["task"] = env.call("task_description")
    elif hasattr(first_env, "task"):
        observation["task"] = env.call("task")
    else:  #  For envs without language instructions, e.g. aloha transfer cube and etc.
        num_envs = observation[list(observation.keys())[0]].shape[0]
        observation["task"] = ["" for _ in range(num_envs)]
    return observation
