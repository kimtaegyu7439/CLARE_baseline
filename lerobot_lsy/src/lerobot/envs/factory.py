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
import importlib

import gymnasium as gym

from lerobot.envs.configs import AlohaEnv, EnvConfig, LiberoEnv, HILEnvConfig, PushtEnv, XarmEnv


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    if env_type == "aloha":
        return AlohaEnv(**kwargs)
    elif env_type == "libero":
        return LiberoEnv(**kwargs)
    elif env_type == "pusht":
        return PushtEnv(**kwargs)
    elif env_type == "xarm":
        return XarmEnv(**kwargs)
    elif env_type == "hil":
        return HILEnvConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{env_type}' is not available.")


def make_env(cfg: EnvConfig, n_envs: int = 1, use_async_envs: bool = False) -> gym.vector.VectorEnv | None:
    """Makes a gym vector environment according to the config.

    Args:
        cfg (EnvConfig): the config of the environment to instantiate.
        n_envs (int, optional): The number of parallelized env to return. Defaults to 1.
        use_async_envs (bool, optional): Whether to return an AsyncVectorEnv or a SyncVectorEnv. Defaults to
            False.

    Raises:
        ValueError: if n_envs < 1
        ModuleNotFoundError: If the requested env package is not installed

    Returns:
        gym.vector.VectorEnv: The parallelized gym.env instance.
    """
    if n_envs < 1:
        raise ValueError("`n_envs must be at least 1")

    # cfg.type="libero" -> "gym_libero" 패키지를 import한다. 이 import는 부작용이 목적이다:
    # 패키지가 로드되면서 gym 레지스트리에 "gym_libero/Libero_..." 핸들들이 등록된다.
    package_name = f"gym_{cfg.type}"

    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        # gym-libero는 lerobot 의존성에 포함돼 있지 않아 별도 설치가 필요하다.
        #   git clone git@github.com:ZhangYi1999/gym-libero.git && cd gym-libero && pip install -e .
        # 학습만 할 거면 --eval_freq=0으로 두어 이 함수 호출 자체를 피할 수 있다.
        print(f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.type}]'`")
        raise e

    # 예: "gym_libero/Libero_Spatial_Task_0"
    gym_handle = f"{package_name}/{cfg.task}"

    # batched version of the env that returns an observation of shape (b, c)
    # n_envs개를 동시에 굴려 에피소드를 병렬 평가한다(--eval.batch_size).
    # Sync는 한 프로세스에서 순차 실행, Async는 프로세스를 분리해 병렬 실행.
    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    env = env_cls(
        [lambda: gym.make(gym_handle, disable_env_checker=True, **cfg.gym_kwargs) for _ in range(n_envs)]
    )

    return env
