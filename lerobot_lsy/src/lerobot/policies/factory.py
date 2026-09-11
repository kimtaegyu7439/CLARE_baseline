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

import logging

from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import env_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion_transformer.configuration_diffusion_transformer import (
    DiffusionTransformerConfig,
)
from lerobot.policies.dit.configuration_dit import DiTConfig
from lerobot.policies.dit_flow.configuration_dit_flow import DiTFlowConfig
from lerobot.policies.dit_flow_mt.configuration_dit_flow_mt import DiTFlowMTConfig
# from lerobot.policies.dit_meanflow.configuration_dit_meanflow import DiTMeanFlowConfig
from lerobot.policies.dit_mt.configuration_dit_mt import DiTMTConfig
# from lerobot.policies.dit_update_mt.configuration_dit_update_mt import DiTUpdateMTConfig as DiTUpdateMTConfig
# from lerobot.policies.dit_flow_update_mt.configuration_dit_flow_update_mt import DiTFlowUpdateMTConfig as DiTFlowUpdateMTConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
# from lerobot.policies.smolvla_image_only.configuration_smolvla_image_only import SmolVLAImageConfig
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
# from lerobot.policies.unet_meanflow.configuration_unet_meanflow import UnetMeanFlowConfig
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig


def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "tdmpc":
        from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        return TDMPCPolicy
    elif name == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    elif name == "diffusion_transformer":
        from lerobot.policies.diffusion_transformer.modeling_diffusion_transformer import (
            DiffusionTransformerPolicy,
        )

        return DiffusionTransformerPolicy
    elif name == "dit":
        from lerobot.policies.dit.modeling_dit import DiTPolicy

        return DiTPolicy
    elif name == "ditflow":
        from lerobot.policies.dit_flow.modeling_dit_flow import DiTFlowPolicy

        return DiTFlowPolicy
    elif name == "dit_meanflow":
        from lerobot.policies.dit_meanflow.modeling_dit_meanflow import DiTMeanFlowPolicy

        return DiTMeanFlowPolicy
    elif name == "ditflow_mt":
        from lerobot.policies.dit_flow_mt.modeling_dit_flow_mt import DiTFlowMTPolicy

        return DiTFlowMTPolicy
    elif name == "dit_mt":
        from lerobot.policies.dit_mt.modeling_dit_mt import DiTMTPolicy

        return DiTMTPolicy
    elif name == "dit_update_mt":
        from lerobot.policies.dit_update_mt.modeling_dit_update_mt import DiTUpdateMTPolicy

        return DiTUpdateMTPolicy
    elif name == "ditflow_update_mt":
        from lerobot.policies.dit_flow_update_mt.modeling_dit_flow_update_mt import DiTFlowUpdateMTPolicy

        return DiTFlowUpdateMTPolicy
    elif name == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy

        return ACTPolicy
    elif name == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy
    elif name == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    elif name == "pi0fast":
        from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy

        return PI0FASTPolicy
    elif name == "sac":
        from lerobot.policies.sac.modeling_sac import SACPolicy

        return SACPolicy
    elif name == "reward_classifier":
        from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

        return Classifier
    elif name == "unet_meanflow":
        from lerobot.policies.unet_meanflow.modeling_unet_meanflow import UnetMeanFlowPolicy

        return UnetMeanFlowPolicy
    elif name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    elif name == "smolvla_image_only":
        from lerobot.policies.smolvla_image_only.modeling_smolvla_image_only import SmolVLAImagePolicy

        return SmolVLAImagePolicy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "tdmpc":
        return TDMPCConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    elif policy_type == "diffusion_transformer":
        return DiffusionTransformerConfig(**kwargs)
    elif policy_type == "dit":
        return DiTConfig(**kwargs)
    elif policy_type == "ditflow":
        return DiTFlowConfig(**kwargs)
    elif policy_type == "dit_mt":
        return DiTMTConfig(**kwargs)
    elif policy_type == "dit_update_mt":
        return DiTUpdateMTConfig(**kwargs)
    elif policy_type == "ditflow_update_mt":
        return DiTFlowUpdateMTConfig(**kwargs)
    elif policy_type == "ditflow_mt":
        return DiTFlowMTConfig(**kwargs)
    elif policy_type == "dit_meanflow":
        return DiTMeanFlowConfig(**kwargs)
    elif policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "vqbet":
        return VQBeTConfig(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi0fast":
        return PI0FASTConfig(**kwargs)
    elif policy_type == "sac":
        return SACConfig(**kwargs)
    elif policy_type == "unet_meanflow":
        return UnetMeanFlowConfig(**kwargs)
    elif policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
    elif policy_type == "smolvla_image_only":
        return SmolVLAImageConfig(**kwargs)
    elif policy_type == "reward_classifier":
        return RewardClassifierConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    This function exists because (for now) we need to parse features from either a dataset or an environment
    in order to properly dimension and instantiate a policy for that dataset or environment.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.
        env_cfg (EnvConfig | None, optional): The config of a gym environment to parse features from. Must be
            provided if ds_meta is not. Defaults to None.

    Raises:
        ValueError: Either ds_meta or env and env_cfg must be provided.
        NotImplementedError: if the policy.type is 'vqbet' and the policy device 'mps' (due to an incompatibility)

    Returns:
        PreTrainedPolicy: _description_

    핵심 포인트 두 가지.

    1) 정책의 input_features / output_features는 config에 하드코딩된 게 아니라
       여기서 데이터셋(또는 환경)으로부터 채워진다(234-235행). 그래서 같은 정책
       클래스가 데이터셋만 바꿔도 다른 차원으로 만들어진다.

    2) cfg.pretrained_path(= CLI의 --policy.path)가 있으면 from_pretrained로
       체크포인트를 불러오고, 없으면 새로 만든다(238-245행). 즉 순차 파인튜닝
       (이전 태스크 체크포인트를 다음 태스크의 시작점으로) 이 코드 수정 없이 된다.
       CLARE 없는 continual learning 베이스라인을 만들 때 이 점이 중요하다.
    """
    # 둘 중 정확히 하나만 있어야 한다. 학습 시에는 ds_meta, 환경만으로 평가할 때는 env_cfg.
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

    # NOTE: Currently, if you try to run vqbet with mps backend, you'll get this error.
    # TODO(aliberts, rcadene): Implement a check_backend_compatibility in policies?
    # NotImplementedError: The operator 'aten::unique_dim' is not currently implemented for the MPS device. If
    # you want this op to be added in priority during the prototype phase of this feature, please comment on
    # https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment
    # variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be
    # slower than running natively on MPS.
    if cfg.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    # cfg.type 문자열("ditflow_mt") -> 정책 클래스(DiTFlowMTPolicy) 조회.
    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    if ds_meta is not None:
        # 데이터셋 메타 -> PolicyFeature 변환. (H,W,C)를 (C,H,W)로 바꾸는 것도 여기서 한다.
        features = dataset_to_policy_features(ds_meta.features)
        # 정규화 통계. datasets/factory.py에서 이미지 항목이 ImageNet 값으로
        # 덮어씌워진 뒤의 stats가 여기로 들어와 Normalize 레이어의 버퍼가 된다.
        kwargs["dataset_stats"] = ds_meta.stats
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        features = env_to_policy_features(env_cfg)

    # ACTION 타입은 출력, 나머지는 전부 입력으로 분류한다.
    # 주의: 여기서 input_features에는 observation.state.joint도 포함된다. 실제 사용
    # 여부는 각 정책이 PreTrainedConfig.robot_state_feature 등으로 다시 골라 정한다.
    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        # --policy.path로 넘긴 경로(로컬 디렉토리 또는 HF repo id). 순차 파인튜닝의 연결고리.
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        # --policy.type만 준 경우(사전학습 pretrain.sh 경로).
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy
