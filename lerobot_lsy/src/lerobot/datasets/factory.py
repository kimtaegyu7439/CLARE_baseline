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
"""정책 설정(PreTrainedConfig)과 데이터셋(LeRobotDataset)을 이어붙이는 접착제.

하는 일은 세 가지뿐이다.
  1. 인덱스 -> 초 번역: 정책은 "몇 스텝 전/후"로 생각하지만 LeRobotDataset은
     "몇 초"로 질의하므로 fps로 나눠 변환한다. (resolve_delta_timestamps)
  2. 2단계 로딩: fps를 알아야 1번을 할 수 있으므로 메타데이터를 먼저 읽고,
     그다음 본 데이터셋을 만든다. (make_dataset)
  3. 이미지 통계 교체: 비전 백본의 사전학습 분포에 맞춘다. (IMAGENET_STATS)

DiT-Flow MT + LIBERO 조합(fps=20, n_obs_steps=2, horizon=16)에서 최종적으로
DataLoader가 내놓는 배치는 다음과 같다.

    observation.images.image        (B, 2, 3, 256, 256)   과거 1프레임 + 현재
    observation.images.wrist_image  (B, 2, 3, 256, 256)   손목 카메라
    observation.state               (B, 2, 8)             엔드이펙터 절대 자세
    observation.state.joint         (B, 2, 7)             관절 각도 (모델 미사용, 아래 주석 참조)
    action                          (B, 16, 7)            t-1 ~ t+14 이동 명령
    action_is_pad                   (B, 16)               에피소드 경계 패딩 마스크
    task                            list[str]             자연어 명령

state와 action은 성격이 다르다.
    state  = 지금 팔이 어디 있는지 잰 값 (절대, 미터/라디안). 8차원 =
             x,y,z,roll,pitch,yaw + 그리퍼 손가락 2개(각각 측정하므로 2개).
    action = 여기서 얼마나 더 움직일지 시킨 값 (상대 변위, [-1,1] 정규화).
             7차원 = x,y,z,roll,pitch,yaw + 그리퍼 명령 1개(두 손가락 동시 구동).
             그리퍼는 -1=열기 / +1=닫기이며 아래 스케일링에서 제외된다.

             robosuite OSC_POSE는 [-1,1]을 실제 단위로 선형 사상한다.
                 위치: action 1.0 -> 0.05 m 명령
                 회전: action 1.0 -> 0.5 rad 명령

             단 이 값은 "목표"이지 "실제 이동량"이 아니다. OSC는 임피던스 제어라
             목표 지점을 찍어주면 그쪽으로 힘(kp=150)을 가할 뿐 순간이동하지 않는다.
             팔에 질량이 있어 한 스텝(0.05초) 안에 목표에 도달하지 못하고, 다음
             스텝에서 새 위치 기준으로 목표가 다시 찍힌다(움직이는 목표를 추종).
             이 데이터셋에서 측정하면 한 스텝당 명령의 약 23%만 실제로 좁혀진다.
             시연 데이터도 같은 제어기로 수집됐으므로 정책이 "명령"을 재현하는 것으로
             일관성은 유지된다.
"""

import logging
from pprint import pformat

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.transforms import ImageTransforms

# ImageNet 채널별 평균/표준편차. (c,1,1) 모양은 (C,H,W) 이미지에 브로드캐스팅하기 위함.
# 데이터셋 자체 통계 대신 이 값을 쓰는 이유는 make_dataset 맨 아래 주석 참조.
IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.

    DiT-Flow MT + LIBERO(fps=20)에서 실제로 만들어지는 결과:

        observation.images.image        [-0.05, 0.0]                    <- 2개
        observation.images.wrist_image  [-0.05, 0.0]
        observation.state               [-0.05, 0.0]
        observation.state.joint         [-0.05, 0.0]
        action                          [-0.05, 0.0, 0.05, ..., 0.70]   <- 16개

    액션 구간이 t-1부터 시작하는 이유는 관측 창(n_obs_steps=2 -> [t-1, t])의
    원점에 정렬했기 때문이다. 그래서 배열 index 0은 "현재"가 아니라 "0.05초 전"이고,
    추론 시 generate_actions()가 index 1부터 슬라이싱한다.
    """
    delta_timestamps = {}
    # 정책이 가진 delta_indices(스텝 단위)를 데이터셋 질의용 timestamp(초)로 번역한다.
    # 예: action_delta_indices=[-1..14], fps=20 -> [-0.05, 0.0, 0.05, ..., 0.70]
    for key in ds_meta.features:
        # 보상 분기. LIBERO 데이터셋에는 next.reward 키가 없고 DiT-Flow는
        # reward_delta_indices가 None이라 여기서는 실행되지 않는다 (RL 정책용).
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        # 주의: "observation."으로 시작하는 키를 전부 잡으므로 observation.state.joint도
        # 포함된다. 하지만 DiT-Flow는 이를 쓰지 않는다 -- PreTrainedConfig.robot_state_feature가
        # key == "observation.state"로 명시 필터링하기 때문(configs/policies.py). 즉 매 배치마다
        # 로드/정규화되지만 조건 벡터에는 안 들어가는 순수 낭비다.
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    # 빈 dict 대신 None을 넘겨야 LeRobotDataset이 "시간 문맥 없이 단일 프레임"
    # 모드로 동작한다. 과거/미래가 필요 없는 정책을 위한 처리.
    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    # 데이터 증강. ImageTransformsConfig.enable 기본값이 False이고 CLARE 스크립트도
    # 켜지 않으므로 보통 None이다.
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    # DatasetConfig.repo_id의 타입이 str로 고정돼 있어 실질적으로 항상 참이다.
    if isinstance(cfg.dataset.repo_id, str):
        # [1단계] 메타데이터만 로드. 7GB 전체가 아니라 meta/ 폴더만 읽는다.
        # 다음 줄에서 fps가 필요한데 fps를 알려면 메타데이터를 먼저 읽어야 하기 때문이며,
        # LeRobotDatasetMetadata가 별도 클래스로 존재하는 이유가 이것이다.
        #
        # revision: HF Hub의 git revision(브랜치/태그/커밋). None이면 CODEBASE_VERSION
        # (="v2.1")이 쓰인다. 이는 커밋 해시가 아니라 LeRobot 데이터셋 포맷 버전 태그로,
        # 저장 구조가 바뀌어도 이 코드가 읽을 수 있는 버전을 고정하는 장치다.
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        # 정책의 시간 규약 + 데이터셋 fps -> 질의용 timestamp 표. 이 파일의 존재 이유.
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        # [2단계] 실제 데이터셋 생성.
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            # root=None이면 HF_LEROBOT_HOME/<repo_id>로 해석된다(constants.py).
            # 즉 데이터셋 저장 위치는 이 인자 또는 그 환경변수로 결정된다.
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,  # None이면 전체 에피소드 사용
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,  # 데이터셋 버전 고정용
            # mp4로 저장된 이미지 스트림의 디코더(torchcodec, 없으면 pyav).
            # 단 이 LIBERO 데이터셋은 이미지를 개별 파일로 저장해 meta.video_keys가
            # 비어 있으므로 실제로는 사용되지 않는다(__getitem__의 video 분기를 안 탄다).
            video_backend=cfg.dataset.video_backend,
        )
    else:
        # 아래 MultiLeRobotDataset 블록은 이 raise 때문에 도달 불가능한 죽은 코드다.
        # 즉 여러 repo_id를 한 번에 학습할 수 없다. continual learning은 태스크를
        # 하나씩 순차 학습하므로 문제없지만, 이전 태스크 데이터를 섞는 rehearsal
        # 방식을 시도하려면 이 부분을 직접 구현해야 한다.
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    # 이 데이터셋으로 계산된 이미지 통계를 ImageNet 통계로 덮어쓴다(기본값 True).
    # 비전 백본이 facebook/dinov2-base이고 DINOv2는 ImageNet 정규화로 사전학습됐기
    # 때문이다. LIBERO 자체 통계로 정규화하면 백본이 본 적 없는 분포가 들어간다.
    #
    # 이중 루프가 하는 일 (이 데이터셋 기준 총 4번 대입):
    #   camera_keys = ['observation.images.image', 'observation.images.wrist_image']
    #   stats는 {feature_key: {'min','max','mean','std','count'}} 형태의 2중 dict이고
    #   그중 'mean'과 'std'만 교체한다. min/max/count와 state/action 항목은 손대지 않는다
    #   (이미지는 MEAN_STD 정규화라 min/max를 쓰지 않는다).
    #
    # 참고: LIBERO 렌더링은 색 통계가 ImageNet과 우연히 비슷해서 실측상 차이가 거의 없다.
    #   덮어쓰기 전 mean = [0.4955, 0.4649, 0.4324]
    #   덮어쓰기 후 mean = [0.4850, 0.4560, 0.4060]
    # 조명/색감이 특이한 실제 로봇 데이터에서는 이 교체가 훨씬 중요해진다.
    #
    # 주의: dataset.meta.stats를 직접 변형한다. 이후 make_policy(cfg, ds_meta=dataset.meta)가
    # 이 stats로 Normalize 레이어의 버퍼를 채우므로 정책의 이미지 정규화가 ImageNet 값을 쓰게 된다.
    # 정규화 방식은 필드마다 다르다: VISUAL은 MEAN_STD인 (x-mean)/std (z-정규화),
    # STATE/ACTION은 MIN_MAX인 (x-min)/(max-min)*2-1 -> [-1,1] (normalize.py:167-181).
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
