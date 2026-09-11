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
import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot import envs
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.hub import HubMixin

TRAIN_CONFIG_NAME = "train_config.json"


@dataclass
class TrainPipelineConfig(HubMixin):
    """학습 실행 하나의 전체 설정. CLI 인자가 그대로 이 dataclass에 매핑된다.

    draccus가 중첩 dataclass를 점 표기법으로 노출한다.
        --batch_size=32              -> TrainPipelineConfig.batch_size
        --dataset.repo_id=xxx        -> DatasetConfig.repo_id
        --policy.path=xxx            -> 사전학습 체크포인트(아래 validate() 참조)
        --env.type=libero            -> LiberoEnv 선택
        --wandb.enable=true          -> WandBConfig.enable

    본인 방법론용 인자를 추가하려면 이 클래스를 상속한 dataclass를 만들고 필드를
    더하면 된다. scripts/clare.py의 PEFTTrainPipelineConfig가 그 예시다.

        @dataclass
        class MyTrainConfig(TrainPipelineConfig):
            my_coef: float = 0.1     # --my_coef=0.3 으로 CLI에서 지정 가능

    실행 시 이 설정 전체가 output_dir/checkpoints/*/train_config.json으로 저장되어
    나중에 재현/재개할 수 있다.
    """

    dataset: DatasetConfig
    env: envs.EnvConfig | None = None      # None이면 롤아웃 평가를 하지 않는다
    policy: PreTrainedConfig | None = None
    # Set `dir` to where you would like to save all of the run outputs. If you run another training session
    # with the same value for `dir` its contents will be overwritten unless you set `resume` to true.
    output_dir: Path | None = None
    job_name: str | None = None
    # Set `resume` to true to resume a previous run. In order for this to work, you will need to make sure
    # `dir` is the directory of an existing run with at least one checkpoint in it.
    # Note that when resuming a run, the default behavior is to use the configuration from the checkpoint,
    # regardless of what's provided with the training command at the time of resumption.
    resume: bool = False
    # `seed` is used for training (eg: model initialization, dataset shuffling)
    # AND for the evaluation environments.
    seed: int | None = 1000
    # Number of workers for the dataloader.
    num_workers: int = 4
    batch_size: int = 8
    steps: int = 100_000        # epoch이 아니라 스텝 수로 학습 길이를 정한다
    # 주의: eval_freq > 0이기만 하면 train.py가 시작 시 make_env를 호출한다.
    # 평가가 실제로 발동하지 않더라도 gym_libero가 필요해진다. 학습만 하려면 0으로.
    eval_freq: int = 20_000
    log_freq: int = 200
    save_checkpoint: bool = True
    # Checkpoint is saved every `save_freq` training iterations and after the last training step.
    save_freq: int = 20_000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    eval_with_dataset: str | None = None

    def __post_init__(self):
        self.checkpoint_path = None

    def validate(self):
        """--policy.path / --resume 처리 후 출력 경로를 확정한다.

        --policy.path를 주면 그 체크포인트의 config.json을 읽어 self.policy를 통째로
        교체한다. 즉 아키텍처 하이퍼파라미터(hidden_dim, horizon 등)는 CLI가 아니라
        체크포인트를 따라간다. 순차 파인튜닝에서 태스크마다 구조가 어긋나지 않는 이유다.
        """
        # HACK: We parse again the cli args here to get the pretrained paths if there was some.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Only load the policy config
            # 체크포인트의 config를 기본값으로 삼고, CLI로 준 --policy.xxx만 덮어쓴다.
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            # 이 값이 make_policy에서 from_pretrained 호출 여부를 가른다.
            self.policy.pretrained_path = policy_path
        elif self.resume:
            # The entire train config is already loaded, we just need to get the checkpoint dir
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"A config_path is expected when resuming a run. Please specify path to {TRAIN_CONFIG_NAME}"
                )
            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} is expected to be a local path. "
                    "Resuming from the hub is not supported for now."
                )
            policy_path = Path(config_path).parent
            self.policy.pretrained_path = policy_path
            self.checkpoint_path = policy_path.parent

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("Optimizer and Scheduler must be set when the policy presets are not used.")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

        if self.policy.push_to_hub and not self.policy.repo_id:
            raise ValueError(
                "'policy.repo_id' argument missing. Please specify it to push the model to the hub."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

    def to_dict(self) -> dict:
        return draccus.encode(self)

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: Type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"{TRAIN_CONFIG_NAME} not found in {Path(model_id).resolve()}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{TRAIN_CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_args)


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    dataset: DatasetConfig | None = None  # NOTE: In RL, we don't need an offline dataset
