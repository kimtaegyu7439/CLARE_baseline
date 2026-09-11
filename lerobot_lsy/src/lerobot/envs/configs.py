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

import abc
from dataclasses import dataclass, field
from typing import Any, Optional

import draccus

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_ROBOT
from lerobot.robots import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @property
    @abc.abstractmethod
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }



# gym 핸들 접두사 -> LIBERO 벤치마크 이름.
# gym_libero/__init__.py는 핸들마다 (benchmark, task_id) 짝을 등록해 둔다
# (예: Libero_Goal_Task_3 -> benchmark="libero_goal", task_id="task_3").
# 이 표는 핸들 이름만 보고 그 짝의 benchmark 쪽을 되찾기 위한 것이다.
LIBERO_TASK_PREFIX_TO_BENCHMARK = {
    "Libero_10": "libero_10",
    "Libero_Goal": "libero_goal",
    "Libero_Spatial": "libero_spatial",
    "Libero_Object": "libero_object",
    "Libero_90": "libero_90",
}


@EnvConfig.register_subclass("libero")
@dataclass
class LiberoEnv(EnvConfig):
    """LIBERO 시뮬레이션 환경 설정. --env.type=libero 로 선택된다.

    학습에는 쓰이지 않고 롤아웃 평가에만 쓰인다. gym_libero 패키지가 설치돼 있어야
    make_env()가 동작한다(envs/factory.py). 학습만 할 거면 --eval_freq=0으로 두면
    make_env 자체가 호출되지 않아 이 의존성을 피할 수 있다.

    여기 features/features_map은 "gym이 내놓는 관측"을 "정책이 기대하는 키"로
    번역하는 표다. 데이터셋(LeRobotDataset)이 이미 정책 키를 쓰는 것과 달리,
    gym 환경은 자기 나름의 이름(agent_pos, pixels/...)을 쓰기 때문에 필요하다.
    """

    benchmark: str = "libero_10"          # 과제 묶음: libero_10 / libero_goal / libero_spatial / libero_object
    task: str = "Libero_10_Task_0"        # 평가할 과제. 쉼표로 여러 개 지정하면 순차 평가된다.
    task_id: str = "task_0"
    fps: int = 20                         # 데이터셋 fps와 같아야 한다. 1스텝 = 0.05초.
    episode_length: int = 500             # 최대 500스텝 = 25초 안에 성공해야 함
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )
    # gym 관측 키 -> 정책 키 변환표.
    # joint_state가 주석 처리된 것에 주목: 데이터셋에는 observation.state.joint(7차원)가
    # 들어 있지만 DiT-Flow는 쓰지 않으므로 환경 쪽에서도 아예 넘기지 않는다.
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,                          # -> observation.state (8차원)
            # "joint_state": f"{OBS_ROBOT}.joint",           # 의도적으로 비활성화됨
            "pixels/image": f"{OBS_IMAGES}.image",           # 3인칭 카메라
            "pixels/wrist_image": f"{OBS_IMAGES}.wrist_image",  # 손목 카메라
        }
    )

    def __post_init__(self):
        # dataclass field에서 선언하지 않고 여기서 채우는 이유는 features가 default_factory로
        # 만들어진 뒤에 항목을 덧붙여야 하기 때문. shape은 단일 프레임 기준이다.
        self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
        # self.features["joint_state"] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
        # 주의: 여기는 (H,W,C)이지만 정책에 들어갈 때는 (C,H,W)로 바뀐다.
        self.features["pixels/image"] = PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3))
        self.features["pixels/wrist_image"] = PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3))

    @property
    def resolved_benchmark(self) -> str:
        """task 핸들과 짝이 맞는 벤치마크 이름.

        gym.make(handle, **gym_kwargs)는 benchmark를 덮어쓰지만 task_id는 핸들 등록값을
        그대로 둔다(gym_libero/__init__.py). 그래서 handle과 무관한 benchmark를 넘기면
        (benchmark, task_id) 짝이 깨져 **엉뚱한 태스크가 실행된다** --
        env.py의 _make_env_task(benchmark, task_id)가 이 둘로 bddl/init state/언어지시를
        모두 결정하기 때문이다.

        실제로 libero_40(네 suite를 이어 붙인 40스테이지 시퀀스)에서 이 일이 있었다.
        스테이지마다 --env.benchmark는 현재 suite 하나인데 --env.task에는 이전 suite의
        핸들이 누적돼 넘어가므로, 과거 태스크가 전부 현재 suite의 같은 인덱스 태스크로
        리매핑돼 평가됐다. 핸들에서 benchmark를 되찾아 그 짝을 항상 맞춘다.

        핸들을 해석할 수 없으면(쉼표 목록 등) 설정값을 그대로 쓴다.
        """
        if self.task and "_Task_" in self.task:
            prefix = self.task.rsplit("_Task_", 1)[0]
            if prefix in LIBERO_TASK_PREFIX_TO_BENCHMARK:
                return LIBERO_TASK_PREFIX_TO_BENCHMARK[prefix]
        return self.benchmark

    @property
    def gym_kwargs(self) -> dict:
        """gym.make(handle, **gym_kwargs)로 넘어가는 인자."""
        return {
            "benchmark": self.resolved_benchmark,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("pusht")
@dataclass
class PushtEnv(EnvConfig):
    task: str = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,
            "environment_state": OBS_ENV_STATE,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(type=FeatureType.VISUAL, shape=(384, 384, 3))
        elif self.obs_type == "environment_state_agent_pos":
            self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("xarm")
@dataclass
class XarmEnv(EnvConfig):
    task: str = "XarmLift-v0"
    fps: int = 15
    episode_length: int = 200
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(84, 84, 3)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_STATE,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@dataclass
class VideoRecordConfig:
    """Configuration for video recording in ManiSkill environments."""

    enabled: bool = False
    record_dir: str = "videos"
    trajectory_name: str = "trajectory"


@dataclass
class EnvTransformConfig:
    """Configuration for environment wrappers."""

    # ee_action_space_params: EEActionSpaceConfig = field(default_factory=EEActionSpaceConfig)
    control_mode: str = "gamepad"
    display_cameras: bool = False
    add_joint_velocity_to_observation: bool = False
    add_current_to_observation: bool = False
    add_ee_pose_to_observation: bool = False
    crop_params_dict: Optional[dict[str, tuple[int, int, int, int]]] = None
    resize_size: Optional[tuple[int, int]] = None
    control_time_s: float = 20.0
    fixed_reset_joint_positions: Optional[Any] = None
    reset_time_s: float = 5.0
    use_gripper: bool = True
    gripper_quantization_threshold: float | None = 0.8
    gripper_penalty: float = 0.0
    gripper_penalty_in_reward: bool = False


@EnvConfig.register_subclass(name="gym_manipulator")
@dataclass
class HILSerlRobotEnvConfig(EnvConfig):
    """Configuration for the HILSerlRobotEnv environment."""

    robot: Optional[RobotConfig] = None
    teleop: Optional[TeleoperatorConfig] = None
    wrapper: Optional[EnvTransformConfig] = None
    fps: int = 10
    name: str = "real_robot"
    mode: str = None  # Either "record", "replay", None
    repo_id: Optional[str] = None
    dataset_root: Optional[str] = None
    task: str = ""
    num_episodes: int = 10  # only for record mode
    episode: int = 0
    device: str = "cuda"
    push_to_hub: bool = True
    pretrained_policy_name_or_path: Optional[str] = None
    reward_classifier_pretrained_path: Optional[str] = None
    # For the reward classifier, to record more positive examples after a success
    number_of_steps_after_success: int = 0

    def gym_kwargs(self) -> dict:
        return {}


@EnvConfig.register_subclass("hil")
@dataclass
class HILEnvConfig(EnvConfig):
    """Configuration for the HIL environment."""

    type: str = "hil"
    name: str = "PandaPickCube"
    task: str = "PandaPickCubeKeyboard-v0"
    use_viewer: bool = True
    gripper_penalty: float = 0.0
    use_gamepad: bool = True
    state_dim: int = 18
    action_dim: int = 4
    fps: int = 100
    episode_length: int = 100
    video_record: VideoRecordConfig = field(default_factory=VideoRecordConfig)
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(18,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.image": OBS_IMAGE,
            "observation.state": OBS_STATE,
        }
    )
    ################# args from hilserlrobotenv
    reward_classifier_pretrained_path: Optional[str] = None
    robot_config: Optional[RobotConfig] = None
    teleop_config: Optional[TeleoperatorConfig] = None
    wrapper: Optional[EnvTransformConfig] = None
    mode: str = None  # Either "record", "replay", None
    repo_id: Optional[str] = None
    dataset_root: Optional[str] = None
    num_episodes: int = 10  # only for record mode
    episode: int = 0
    device: str = "cuda"
    push_to_hub: bool = True
    pretrained_policy_name_or_path: Optional[str] = None
    # For the reward classifier, to record more positive examples after a success
    number_of_steps_after_success: int = 0
    ############################

    @property
    def gym_kwargs(self) -> dict:
        return {
            "use_viewer": self.use_viewer,
            "use_gamepad": self.use_gamepad,
            "gripper_penalty": self.gripper_penalty,
        }
