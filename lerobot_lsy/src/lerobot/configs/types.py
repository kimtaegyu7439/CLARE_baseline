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
# Note: We subclass str so that serialization is straightforward
# https://stackoverflow.com/questions/24481852/serialising-an-enum-member-to-json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class FeatureType(str, Enum):
    """데이터 필드의 종류. 정책이 "무엇을 입력으로 받고 무엇을 출력하는지" 구분하는 꼬리표.

    LIBERO + DiT-Flow MT에서의 대응:
        STATE  -> observation.state (8), observation.state.joint (7)
        VISUAL -> observation.images.image, observation.images.wrist_image
        ACTION -> action (7)
        ENV    -> observation.environment_state (LIBERO에는 없음)
        REWARD -> next.reward (모방학습이라 사용 안 함)

    주의: STATE가 두 개인데 정책은 하나만 쓴다. PreTrainedConfig.robot_state_feature가
    type뿐 아니라 key == "observation.state"까지 검사하기 때문(configs/policies.py).
    즉 FeatureType만으로는 실제 사용 여부가 결정되지 않는다.
    """

    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"


class NormalizationMode(str, Enum):
    """정규화 방식. 어떤 필드에 무엇을 쓸지는 정책 config의 normalization_mapping이 정한다.

        MEAN_STD : (x - mean) / std     이미지에 사용. 단 통계는 데이터셋이 아니라
                                        ImageNet 값으로 덮어씌워진다(datasets/factory.py 참조).
        MIN_MAX  : [min,max] -> [-1,1]  state/action에 사용.
        IDENTITY : 정규화 안 함
    """

    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"


class DictLike(Protocol):
    """dict처럼 []로 접근 가능하기만 하면 되는 타입을 나타내는 구조적 타입힌트."""

    def __getitem__(self, key: Any) -> Any: ...


@dataclass
class PolicyFeature:
    """한 필드의 "종류 + 모양". 정책과 데이터셋이 서로를 이해하는 최소 단위.

    shape에는 배치(B)도 시간축(n_obs_steps)도 들어가지 않는다. 단일 프레임의 모양만 담는다.
        observation.state -> shape=(8,)   실제 배치는 (B, 2, 8)
        action            -> shape=(7,)   실제 배치는 (B, 16, 7)
    시간축은 정책 config의 delta_indices가 따로 결정한다(datasets/factory.py).
    """

    type: FeatureType
    shape: tuple
