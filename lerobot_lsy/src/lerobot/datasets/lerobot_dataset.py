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
import contextlib
import logging
import shutil
from pathlib import Path
from typing import Callable

import datasets
import numpy as np
import packaging.version
import PIL.Image
import torch
import torch.utils
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.image_writer import AsyncImageWriter, write_image
from lerobot.datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    TASKS_PATH,
    _validate_feature_names,
    append_jsonlines,
    backward_compatible_episodes_stats,
    check_delta_timestamps,
    check_timestamps_sync,
    check_version_compatibility,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    embed_images,
    get_delta_indices,
    get_episode_data_index,
    get_hf_features_from_features,
    get_safe_version,
    hf_transform_to_torch,
    is_valid_version,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
)
from lerobot.datasets.video_utils import (
    VideoFrame,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_info,
)

CODEBASE_VERSION = "v2.1"


class LeRobotDatasetMetadata:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        self.episodes = load_episodes(self.root)
        if self._version < packaging.version.parse("v2.1"):
            self.stats = load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info["chunks_size"]

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        """
        Given a task in natural language, add it to the dictionary of tasks.
        """
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")

        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1

        task_dict = {
            "task_index": task_index,
            "task": task,
        }
        append_jsonlines(task_dict, self.root / TASKS_PATH)

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)

    def update_video_info(self) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        for key in self.video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.get_video_file_path(ep_index=0, vid_key=key)
                self.info["features"][key]["info"] = get_video_info(video_path)

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
    ) -> "LeRobotDatasetMetadata":
        """Creates metadata for a LeRobotDataset."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        obj.root.mkdir(parents=True, exist_ok=False)

        # TODO(aliberts, rcadene): implement sanity check for features
        features = {**features, **DEFAULT_FEATURES}
        _validate_feature_names(features)

        obj.tasks, obj.task_to_task_index = {}, {}
        obj.episodes_stats, obj.stats, obj.episodes = {}, {}, {}
        obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, features, use_videos, robot_type)
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
        return obj


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        """
        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:
            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v2.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v1.6 to v2.0, which you can find at
              lerobot/datasets/v2/convert_dataset_v1_to_v2.py.


        2. Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
           LeRobotDataset with the 'create' classmethod. This can be used for recording a dataset or port an
           existing dataset to the LeRobotDataset format.


        In terms of files, LeRobotDataset encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. The files are split by
        episodes which allows a more granular control over which episodes one wants to use and download. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.
            root (Path | None, optional): Local directory to use for downloading/writing files. You can also
                set the LEROBOT_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/lerobot'.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): _description_. Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            sync_cache_first (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
        episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        #
        # self.delta_indices가 대입되는 유일한 지점이다(그 외에는 456행의 None 초기화뿐).
        # 값의 출처를 끝까지 따라가면:
        #
        #   [1] policies/dit_flow_mt/configuration_dit_flow_mt.py
        #         action_delta_indices  ->  [-1, 0, 1, ..., 14]        (스텝)
        #   [2] datasets/factory.py  resolve_delta_timestamps()
        #         [i / fps]             ->  [-0.05, 0.0, ..., 0.70]    (초)  = delta_timestamps
        #   [3] LeRobotDataset(delta_timestamps=...)  생성자 인자로 전달
        #   [4] 아래 줄
        #         get_delta_indices()   ->  self.delta_indices
        #   [5] datasets/utils.py:628
        #         [round(d * fps)]      ->  [-1, 0, 1, ..., 14]        (다시 스텝)
        #
        # 초로 나눴다가 다시 곱하는 왕복이 낭비처럼 보이지만, delta_timestamps가
        # LeRobotDataset의 공개 API이기 때문이다. 초 단위는 fps에 독립적이라 다른 fps의
        # 데이터셋에도 같은 값을 쓸 수 있고, 그 사이에 check_delta_timestamps가 요청한
        # 시각이 실제 프레임 시각과 맞는지 검증한다. round()는 0.05*20=1.0000000000000002
        # 같은 부동소수 오차를 흡수한다.
        #
        # 결과 형태: {필드명: [정수 오프셋들]}
        #   'observation.state' -> [-1, 0]          (len 2  -> 출력 (2, 8))
        #   'action'            -> [-1, 0, ..., 14] (len 16 -> 출력 (16, 7))
        # value의 길이가 곧 그 필드의 시간축 크기가 된다.
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        if not hub_api.file_exists(self.repo_id, REPOCARD_NAME, repo_type="dataset", revision=branch):
            card = create_lerobot_dataset_card(
                tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
            )
            card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download_episodes(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        files = None
        ignore_patterns = None if download_videos else "videos/"
        if self.episodes is not None:
            files = self.get_episodes_file_paths()

        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        """"어느 행들을 읽을지"를 계산한다. 실제로 데이터를 읽지는 않는다.

        입력
            idx    : 데이터셋 전체 기준 행 번호 (0 ~ num_rows-1). 에피소드 번호가 아니다.
            ep_idx : 그 행이 속한 에피소드 번호. __getitem__이 item["episode_index"]에서 꺼내 넘긴다.

        출력 (튜플 2개)
            query_indices : {키: [읽을 행 번호들]}   예) 'action' -> [89, 90, ..., 97, 97, 97]
            padding       : {키+"_is_pad": BoolTensor} 어느 자리가 복제본인지 표시

        왜 필요한가
            반환 텐서 모양이 (16,7)로 고정이어야 DataLoader가 배치를 쌓을 수 있다. 그런데
            에피소드 끝 근처에서는 idx+14 행이 존재하지 않는다. 그래서 없는 자리를 경계값
            복제로 채우고, 어디가 가짜인지 마스크로 함께 알려준다.

        예시 (에피소드0 = 행 0~97, idx=90, action delta=[-1..14])
            요청  : 89 90 91 ... 97 | 98  99 100 101 102 103 104
            결과  : 89 90 91 ... 97 | 97  97  97  97  97  97  97   <- 전부 97로 잘림
            is_pad:  0  0  0 ...  0 |  1   1   1   1   1   1   1
        """
        # 이 에피소드가 차지하는 행 범위. to는 exclusive이므로 마지막 유효 행은 to-1이다.
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]

        # 중첩 컴프리헨션이라 변수 정의가 사용보다 아래 줄에 나온다. 풀어쓰면:
        #     for key, delta_idx in self.delta_indices.items():   # delta_idx = [-1, 0, ..., 14]
        #         for delta in delta_idx:                          # delta    = -1, 0, ...
        #             max(ep_start, min(ep_end - 1, idx + delta))
        #
        # max(lo, min(hi, v))는 "v를 [lo, hi] 범위에 가두기"의 관용구다.
        #     min(ep_end-1, ...) -> 에피소드 마지막 행보다 크면 끌어내림
        #     max(ep_start, ...) -> 에피소드 첫 행보다 작으면 끌어올림
        #
        # 이 클램프가 결정적인 이유: 없으면 idx+delta가 이웃 에피소드의 행을 읽는다.
        # 그 행은 표에 멀쩡히 존재하므로 에러도 나지 않고, 전혀 다른 장면의 데이터가
        # 조용히 섞여 "팔이 순간이동하는" 물리적으로 불가능한 전이를 학습하게 된다.
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            # 클램프가 값을 바꿨는지를 기록한다. 조건이 위 클램프와 짝을 이룬다:
            #     max(ep_start, ...) <-> < ep_start   (앞으로 벗어남)
            #     min(ep_end-1, ...) <-> >= ep_end    (뒤로 벗어남, to가 exclusive라 >= )
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        """비디오(mp4) 키에 한해, 행 번호를 "몇 초 지점"으로 바꾼다.

        입력
            current_ts    : 현재 프레임의 timestamp(초)
            query_indices : 있으면 그 행들의 timestamp를 읽고, 없으면 현재 시각 하나만 쓴다

        출력
            {비디오키: [디코딩할 시각(초) 목록]}

        비디오는 행 번호로 접근할 수 없고 시각으로 seek해야 하므로 이 변환이 필요하다.
        meta.video_keys를 순회하므로, 이미지 저장 데이터셋(=이 LIBERO)에서는 빈 dict를
        반환하고 __getitem__의 비디오 분기 자체가 실행되지 않는다.
        """
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """_get_query_indices가 계산한 행 번호로 실제 값을 읽어 시간축으로 쌓는다.

        입력
            query_indices : {키: [읽을 행 번호들]}   _get_query_indices의 첫 번째 반환값

        출력
            {키: 텐서}. 각 텐서의 첫 차원이 시간축이다.
                'observation.state' -> (2, 8)
                'action'            -> (16, 7)

        이 함수가 시간축을 만드는 지점이다. hf_dataset은 5068행짜리 평평한 표일 뿐이고
        시간 개념이 없다. 여기서 여러 행을 골라 stack해야 비로소 (2, 8) 같은 모양이 된다.

        한 줄 해부:
            self.hf_dataset.select(q_idx)   HF Dataset에서 해당 행들만 골라낸 부분집합
                          [key]             그중 이 열만 -> 길이 N의 텐서 리스트
            torch.stack(...)                리스트를 하나의 텐서로 -> (N, ...)

        select()는 같은 행 번호가 여러 번 나와도 그대로 여러 번 반환한다. 에피소드 끝에서
        [.., 97, 97, 97]처럼 클램프된 인덱스가 오면 97행 값이 그 횟수만큼 복제되는 이유다.

        video_keys는 제외한다. mp4에 들어 있는 것은 행이 아니라 프레임이라 행 번호로 못 읽고,
        __getitem__이 _get_query_timestamps + _query_videos로 따로 처리한다.
        (이 LIBERO 데이터셋은 이미지를 개별 파일로 저장해 video_keys가 비어 있으므로
         이 필터는 실질적으로 아무것도 걸러내지 않는다.)
        """
        return {
            key: torch.stack(self.hf_dataset.select(q_idx)[key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        """샘플 하나를 만든다. 여기서 "프레임 하나"가 "시간 창"으로 확장된다.

        delta_timestamps(datasets/factory.py가 만든 것)를 받았으면, idx 한 점이 아니라
        그 주변 프레임들을 함께 긁어와 시간축을 가진 텐서로 쌓는다. DiT-Flow + LIBERO에서는

            observation.*  : idx-1, idx     -> (2, ...)
            action         : idx-1 .. idx+14 -> (16, 7)

        에피소드 경계를 넘어가는 요청은 가장 가까운 프레임을 복제해 채우고, 어느 자리가
        가짜인지를 action_is_pad 같은 마스크로 함께 돌려준다.
        (단 DiT-Flow는 do_mask_loss_for_padding=False라 그 마스크를 쓰지 않는다.)

        헬퍼 3개의 역할 분담:
            _get_query_indices  "어느 행을 읽을지" 계산 (읽지는 않음) + 패딩 마스크
            _query_hf_dataset   그 행들을 실제로 읽어 시간축으로 stack
            _get_query_timestamps / _query_videos
                                mp4 저장 데이터셋 전용. 행 번호 대신 시각으로 seek.
                                이 LIBERO 데이터셋은 이미지 저장이라 실행되지 않는다.
        """
        # hf_dataset은 (전체 프레임 수 x 열) 짜리 평평한 표다. 시간 개념이 없다.
        # 여기서 꺼낸 item은 아직 시간축이 없는 단일 프레임이다.
        item = self.hf_dataset[idx]
        # 이 행이 몇 번 에피소드인지. 아래 클램프가 에피소드 경계를 알아야 하므로 필요하다.
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            # [1] 어느 행들을 읽을지 계산. 에피소드 밖으로 나가면 경계값으로 클램프하고
            #     그 사실을 padding(..._is_pad)에 기록한다. 아직 데이터를 읽지는 않는다.
            #     query_indices 예: {'action': [89, 90, ..., 97, 97, 97], ...}
            #     padding      예: {'action_is_pad': [0,...,0,1,1,1], ...}
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            # [2] 그 행들을 실제로 읽어 시간축으로 stack. 여기서 (2,8), (16,7)이 만들어진다.
            query_result = self._query_hf_dataset(query_indices)
            # [3] 패딩 마스크를 item에 합치고,
            item = {**item, **padding}
            # [4] 단일 프레임 값들을 시간축 있는 버전으로 덮어쓴다.
            for key, val in query_result.items():
                item[key] = val

        # 이미지가 mp4로 저장된 데이터셋이면 해당 시각의 프레임을 디코딩한다.
        # (이 LIBERO 데이터셋은 이미지로 저장돼 있어 이 분기를 타지 않는다.)
        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        # 정수 task_index를 자연어 문장으로 바꾼다. 이 문자열이 그대로 배치의 "task" 키가
        # 되어 CLIP 인코더로 들어간다. 예: "pick up the black bowl ... and place it on the plate"
        # 텐서가 아니므로 train.py의 .to(device) 루프에서 건너뛰어진다.
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        # size and task are special cases that are not in self.features
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def _save_image(self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path) -> None:
        if self.image_writer is None:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            write_image(image, fpath)
        else:
            self.image_writer.save_image(image=image, fpath=fpath)

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        # Add frame features to episode_buffer
        for key in frame:
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # delete images
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    def clear_episode_buffer(self) -> None:
        episode_index = self.episode_buffer["episode_index"]
        if self.image_writer is not None:
            for cam_key in self.meta.camera_keys:
                img_dir = self._get_image_file_path(
                    episode_index=episode_index, image_key=cam_key, frame_index=0
                ).parent
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()

    def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
        if isinstance(self.image_writer, AsyncImageWriter):
            logging.warning(
                "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
            )

        self.image_writer = AsyncImageWriter(
            num_processes=num_processes,
            num_threads=num_threads,
        )

    def stop_image_writer(self) -> None:
        """
        Whenever wrapping this dataset inside a parallelized DataLoader, this needs to be called first to
        remove the image_writer in order for the LeRobotDataset object to be picklable and parallelized.
        """
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        """Wait for asynchronous image writer to finish."""
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def encode_videos(self) -> None:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        for ep_idx in range(self.meta.total_episodes):
            self.encode_episode_videos(ep_idx)

    def encode_episode_videos(self, episode_index: int) -> dict:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(img_dir, video_path, self.fps, overwrite=True)

        return video_paths

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
        ]

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features})
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
