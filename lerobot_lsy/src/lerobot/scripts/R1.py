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

"""R1 — 채점 장소를 자기 롤아웃으로 옮긴다: 튜브 이탈 d(t)와 행동 불일치 Δa(t).

왜 이 실험이 필요한가
    E0가 보여준 것은 held-out demo loss가 완만한데 SR은 절벽처럼 무너진다는 해리였다.
    그런데 held-out loss는 **전문가가 밟았던 상태**에서 행동을 채점한다. 정작 SR이
    무너지는 곳은 정책이 스스로 흘러들어간 낯선 상태다. 즉 지표가 나쁜 것이 아니라
    **채점 장소가 틀렸을** 가능성이 있다. R1은 채점 장소를 자기 롤아웃으로 옮기고,
    그렇게 하면 해리가 풀리는지를 직접 확인한다.

무엇을 재는가 (측정 두 개뿐)
    d(t)   자기 롤아웃의 상태가 전문가 데모 튜브에서 얼마나 벗어났는가
           = min_{demo frames} ‖φ(s_t) − φ(s_demo)‖   (z-정규화 물리 상태 공간)
    Δa(t)  자기 롤아웃에서 마주친 **바로 그 관측** 위에서, 현재 정책이 "망각 전의
           자기 자신"(θ*₁)과 얼마나 다른 행동을 내는가
           = ‖ā_π(o_t) − ā_θ*₁(o_t)‖

    허용되는 비교는 이 둘뿐이다. 롤아웃 궤적끼리는 직접 비교하지 않는다 —
    초기 미세 차이가 카오스적으로 증폭되어 시간축 정렬 자체가 성립하지 않기 때문이다.
    (i) 궤적 vs 데모 튜브(상태 기준), (ii) 같은 관측 위에서 정책 vs 정책(행동 기준).

왜 물리 상태 공간인가
    d(t)를 정책의 잠재공간에서 재면 안 된다. 인코더 자체가 망각으로 변하면
    **자(尺)가 같이 휘기** 때문이다. 시뮬레이터 물리 상태(EE 위치/그리퍼/물체 pose)는
    모든 체크포인트에 대해 동일한 자다. φ의 z-정규화 통계와 튜브 폭 τ는 demo_ref.npz에
    한 번만 만들고 이후 모든 체크포인트가 그 파일에서 읽어 쓴다(재계산 금지).

읽는 법
    그림 A  d(t)가 τ 위로 이륙하는 시각이 스테이지에 따라 앞당겨지면 = 망각의 진행
    그림 B  d가 먼저 이륙하고 Δa가 뒤따르면 → 표류가 행동 붕괴를 유발(폐루프 증폭).
            Δa가 t=0부터 크면 → 행동 자체의 손상이 선행. 선후가 곧 진단이다.
    그림 C  왼쪽(held-out loss vs SR)이 구름 + 오른쪽(dAUC vs SR)이 선이면,
            loss-SR 해리는 지표의 문제가 아니라 측정 장소의 문제였다는 뜻이다.

이 스크립트는 **학습을 하지 않는다.** 저장된 체크포인트를 읽어 롤아웃만 돌린다.
(EWC 페널티/Fisher는 E0·H4·H5의 몫이다.)

전제
    E0/ER가 만든 체크포인트 트리:  <root>/task_{k}/checkpoints/last/pretrained_model
    gym_libero (롤아웃) + LIBERO 원본 데모 hdf5 (φ의 물체 pose는 LeRobot 데이터셋에
    실려 있지 않고 hdf5의 flattened sim state에만 있다).

사용 예
    python R1.py \
        --policy.path=<아무 체크포인트나 (보통 ref_ckpt)> \
        --dataset.repo_id=continuallearning/libero_spatial_image_task_0 \
        --env.type=libero --env.benchmark=libero_spatial \
        --ckpt_roots="seq=outputs/E0/libero_spatial/seed_42/lam0,ewc=outputs/E0/libero_spatial/seed_42/lam100,er=outputs/ER/libero_spatial/seed42" \
        --probe_task=0 --num_rollouts=30 --run_tag=libero_spatial_seed42_probe0
    python R1.py --plot_only --run_dir=outputs/R1/libero_spatial_seed42_probe0
"""

import hashlib
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
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

# ★ held-out 분할과 샘플러는 E0에서 그대로 가져온다. 복사본을 두면 "R1이 채점한
#   held-out"과 "E0가 채점한 held-out"이 조용히 갈라져 그림 C의 왼쪽 패널이 무의미해진다.
from lerobot.scripts.E0 import episode_sampler, split_episodes, to_device

# libero_spatial 4개 태스크는 모두 같은 장면(물체 5개)을 쓴다. 기본값은 "움직이는 것들"만
# 넣는다: 검은 그릇 두 개(태스크마다 어느 쪽이 타깃인지 다르다)와 목표 접시.
# 쿠키상자/라메킨은 어느 태스크에서도 움직이지 않는 랜드마크라 z-정규화 후 잡음 바닥만
# 더한다 — 필요하면 --phi_objects로 넣을 수 있다.
DEFAULT_PHI_OBJECTS = "akita_black_bowl_1,akita_black_bowl_2,plate_1"

# z-정규화에서 std의 하한(위치는 m, quat은 단위 없음이라 같은 5e-3을 쓴다).
# 데모 안에서 사실상 움직이지 않는 차원(건드리지 않는 물체의 z나 quat)을 그 차원의
# 미세 std로 나누면 잡음이 100배로 증폭되어 d(t)를 그 차원이 지배한다. 실측: 하한을
# 1e-3으로 두면 정지 물체의 z(std≈1mm)가 7cm 움직였을 때 z-거리 71이 나와 다른 모든
# 차원을 덮어버렸다. 5mm 하한이면 같은 이탈이 14로 잡혀 "크지만 압도하지는 않는" 크기가 된다.
STD_FLOOR = 5e-3


@dataclass
class R1Config(TrainPipelineConfig):
    """train.py 인자 전부 + 롤아웃 드리프트 측정용 인자. 학습 인자(steps/batch 등)는 무시된다."""

    # ── 어떤 체크포인트를 볼 것인가 ──────────────────────────────────────────
    # "seq=<tree>,ewc=<tree>,er=<tree>" 형식. 각 tree는 task_{k}/checkpoints/last/pretrained_model.
    ckpt_roots: str = ""
    num_stages: int = 4                    # 방법당 스테이지 수 (task_0..task_{n-1})
    probe_task: int = 0                    # 어느 태스크의 망각을 볼 것인가 (0 또는 1)
    # reference 정책 θ*₁ = probe_task 학습 완료 직후 체크포인트. "망각 전의 자기 자신".
    # 비우면 ckpt_roots의 첫 방법의 task_{probe_task}를 쓴다.
    ref_ckpt: str = ""
    # 그림 C에 점을 더 찍고 싶을 때. "label@stage=path,..." (stage는 표시용, 생략 가능)
    extra_ckpts: str = ""

    # ── 데이터 / 환경 ────────────────────────────────────────────────────────
    dataset_prefix: str = "continuallearning/libero_spatial_image_task_"
    env_task_prefix: str = "Libero_Spatial_Task_"
    holdout_episodes: int = 5              # E0와 같은 분할이어야 한다
    # LIBERO 원본 데모 hdf5의 루트. 비우면 get_libero_path("datasets") -> ./Datasets 순으로 찾는다.
    demo_root: str = ""
    phi_objects: str = ""                  # 비면 DEFAULT_PHI_OBJECTS

    # ── [B] 롤아웃 ───────────────────────────────────────────────────────────
    num_rollouts: int = 30
    max_steps: int = 0                     # 0 -> cfg.env.episode_length
    num_samples: int = 4                   # K: ā를 만드는 flow matching 샘플 수
    # Δa를 몇 스텝마다 잴 것인가. 0 -> policy.n_action_steps(=8, 정책이 실제로 재계획하는 주기).
    # 1로 두면 매 스텝 재지만 비용이 8배다. 재계획 사이에는 두 정책 모두 질의되지 않으므로
    # 기본값(재계획 주기)이 "정책이 실제로 약속한 행동"을 비교하는 자연스러운 눈금이다.
    action_eval_stride: int = 0
    rollout_seed_base: int = 777000        # (rollout_id, step) -> a0 시드 유도의 기준점
    # 초기 상태 정착 스텝. LIBERO의 pruned_init은 물체를 테이블 위 ~7cm에 띄운 상태로
    # 저장돼 있다(모든 물체 z=0.97, 데모의 안착 z는 0.90). 그대로 재면 롤아웃 첫 프레임의
    # d가 84까지 튀어 "이탈"이 자유낙하 아티팩트에 묻힌다. null 액션으로 몇 스텝 떨어뜨린
    # 뒤부터 기록하면 d(0)≈1.2로 튜브 안(τ≈3.8)에서 시작한다. 실측 정착에 4스텝이 걸린다.
    # (E0의 SR 평가는 이 정착을 하지 않는다 — 그래서 R1의 SR은 E0의 SR과 프로토콜이
    #  조금 다르다. R1 내부에서는 모든 체크포인트가 같은 정착을 거치므로 비교는 성립한다.)
    settle_steps: int = 5
    save_obs: bool = False                 # 기본 off (Δa는 온라인 계산 -> 이미지 저장 불필요)
    save_obs_rollouts: int = 5             # --save_obs 시 앞 N개 rollout만 디버그 저장
    save_obs_stride: int = 10

    # ── [C] held-out FM loss (고정 (τ_fm, a0) 격자) ──────────────────────────
    loss_batches: int = 16
    loss_batch_size: int = 16
    loss_n_tau: int = 10                   # τ_fm 격자 10점
    loss_n_a0: int = 8                     # a0 8개
    loss_seed: int = 12345
    e0_results: str = ""                   # 주면 E0의 mse를 읽어 [C] 계산을 대체한다
    e0_run_tags: str = ""                  # "seq=0,ewc=100" — 방법 -> E0 run_tag 대응

    # ── 출력 / 제어 ──────────────────────────────────────────────────────────
    out_root: str = "outputs/R1"
    run_tag: str = ""
    recompute_demo_ref: bool = False
    recompute_rollouts: bool = False
    split_by_success: bool = False         # 그림 A의 성공/실패 분리 부록판
    figb_ckpts: str = "ewc@1,ewc@2,ewc@3"  # 그림 B의 열 (method@stage, stage는 0-based)
    no_plot: bool = False

    def validate(self):
        """H5Config.validate와 같은 이유로 output_dir 존재 검사만 우회한다(캐시 재사용).

        R1은 아무것도 학습하지 않으므로 산출물 덮어쓰기 위험이 없다. 오히려 재실행 시
        기존 캐시를 재사용해야 하므로 디렉터리가 이미 있는 것이 정상이다.
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
    """'a=1,b=2' -> {'a': '1', 'b': '2'} (입력 순서 보존)."""
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
    """E0/ER가 남긴 스테이지 체크포인트 경로."""
    return Path(root) / f"task_{stage}" / "checkpoints" / "last" / "pretrained_model"


def load_policy_at(cfg: R1Config, ckpt: Path, ds_meta, device):
    """체크포인트의 파라미터로 정책을 만든다. 항상 eval 모드로 돌려준다.

    H5.load_policy_at와 같은 구조. 차이는 eval() 고정과 그 검사뿐이다.
    """
    if not Path(ckpt).exists():
        raise FileNotFoundError(
            f"체크포인트가 없다: {ckpt}\n"
            f"  --ckpt_roots 가 E0/ER 산출물 트리를 가리키는지 확인해라 "
            f"(예: outputs/E0/libero_spatial/seed_42/lam100)."
        )
    pcfg = PreTrainedConfig.from_pretrained(ckpt)
    pcfg.pretrained_path = ckpt
    pcfg.device = cfg.policy.device
    policy = make_policy(cfg=pcfg, ds_meta=ds_meta)
    policy.eval()
    # 위생 체크: dropout/BN이 살아 있으면 같은 관측에도 다른 행동이 나와 Δa가 오염된다.
    assert not policy.training, "policy가 eval 모드가 아니다"
    return policy


def norm_stats_hash(policy) -> str:
    """정규화 통계(입력/타깃/출력)의 해시.

    체크포인트마다 dataset_stats가 다르면 Δa는 "행동의 차이"가 아니라 "단위의 차이"를
    재게 된다. 순차 학습은 태스크마다 다른 데이터셋으로 저장하므로 실제로 갈릴 수 있다.
    같은지 확인하고 다르면 경고한다(중단하지는 않는다 — 경고와 함께 읽을 수 있게).
    """
    h = hashlib.sha1()
    for mod_name in ("normalize_inputs", "normalize_targets", "unnormalize_outputs"):
        mod = getattr(policy, mod_name, None)
        if mod is None:
            continue
        items = list(mod.named_parameters()) + list(mod.named_buffers())
        for name, t in sorted(items, key=lambda kv: kv[0]):
            h.update(name.encode())
            h.update(np.ascontiguousarray(t.detach().float().cpu().numpy()).tobytes())
    return h.hexdigest()[:12]


def chunk_seed(cfg: R1Config, rollout_id: int, step: int) -> int:
    """(rollout_id, step) -> flow matching 초기 노이즈 a0의 시드.

    ★ 여기에 method/stage/ckpt가 들어가지 않는 것이 이 실험의 핵심이다.
      같은 rollout의 같은 스텝이면 모든 체크포인트가, 그리고 현재 정책과 θ*₁이
      **같은 a0**에서 출발한다. 그래야 Δa가 "샘플링 잡음"이 아니라 "정책 차이"가 된다.
      (τ_fm 격자는 추론에서 Euler 100스텝 고정 격자라 따로 뽑을 난수가 없다.
       held-out loss 쪽의 τ_fm 격자는 heldout_fm_loss가 따로 고정한다.)
    """
    return int(cfg.rollout_seed_base + rollout_id * 100003 + step)


def spearman_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman ρ와 선형회귀 R². scipy 없이 계산한다(의존성 추가를 피한다)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), float("nan")

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # 동점은 평균 순위로 (Spearman의 표준 처리)
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        for i, c in enumerate(cnt):
            if c > 1:
                m = inv == i
                r[m] = r[m].mean()
        return r

    def pearson(a, b):
        a = a - a.mean()
        b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den > 0 else float("nan")

    return pearson(rank(x), rank(y)), pearson(x, y) ** 2


# ═════════════════════════════════════════════════════════════════════════════
#  [A] 데모 참조 집합: φ, z-정규화 통계, 튜브 폭 τ
# ═════════════════════════════════════════════════════════════════════════════
def make_probe_env(cfg: R1Config):
    """probe_task의 gym_libero 환경 하나. 벡터 환경을 쓰지 않는 이유가 있다.

    초기 상태를 rollout_id로 **직접 지정**해야 짝지은 비교가 되는데, LiberoEnv.reset은
    클래스 변수 카운터로 init_state를 고른다(env.py:199). 벡터 래퍼를 거치면 그 카운터를
    개별 env마다 되돌려야 하고 실수하기 쉽다. 단일 env + set_init_state가 명시적이다.
    """
    import importlib

    import gymnasium as gym

    if cfg.env is None:
        raise SystemExit("--env.type=libero --env.benchmark=libero_spatial 가 필요하다.")
    importlib.import_module("gym_libero")
    handle = f"gym_libero/{cfg.env_task_prefix}{cfg.probe_task}"
    kwargs = dict(cfg.env.gym_kwargs)
    # 정착 스텝이 TimeLimit 예산을 갉아먹어 마지막 몇 스텝이 잘리지 않게 한도를 늘려 준다.
    # (gym.make의 max_episode_steps는 TimeLimit 래퍼용이고 env 생성자에는 전달되지 않는다.)
    kwargs["max_episode_steps"] = (cfg.max_steps or cfg.env.episode_length) + cfg.settle_steps + 1
    return gym.make(handle, disable_env_checker=True, **kwargs)


def phi_spec(env, objects: list[str]) -> dict:
    """φ를 읽을 sim 인덱스들을 한 번만 뽑아 둔다 (매 스텝 이름 조회를 피한다)."""
    be = env.unwrapped._env
    missing = [o for o in objects if o not in be.obj_body_id]
    if missing:
        raise SystemExit(
            f"이 장면에 없는 물체다: {missing}\n  가능한 이름: {sorted(be.obj_body_id.keys())}"
        )
    labels = ["ee_x", "ee_y", "ee_z", "gripper"]
    for o in objects:
        labels += [f"{o}_px", f"{o}_py", f"{o}_pz", f"{o}_qw", f"{o}_qx", f"{o}_qy", f"{o}_qz"]
    return {
        "objects": list(objects),
        "body_ids": [int(be.obj_body_id[o]) for o in objects],
        "eef_site": int(be.robots[0].eef_site_id),
        "gripper_idx": np.asarray(be.robots[0]._ref_gripper_joint_pos_indexes, dtype=int),
        "labels": labels,
    }


def phi_from_sim(sim, spec: dict) -> np.ndarray:
    """φ(s) = [EE xyz(3), gripper(1), 물체마다 pos(3)+quat(4)].

    quat은 부호 모호성(q와 −q가 같은 회전)이 있어 w>=0으로 정규화한다. 안 하면
    같은 자세가 부호만 뒤집힌 프레임끼리 최대 거리로 잡혀 d(t)가 튄다.
    """
    ee = np.asarray(sim.data.site_xpos[spec["eef_site"]], dtype=np.float64)
    g = np.asarray(sim.data.qpos[spec["gripper_idx"]], dtype=np.float64)
    grip = float(g[0] - g[1])          # 손가락 사이 벌어짐 (열림>0)
    parts = [ee, np.array([grip])]
    for bid in spec["body_ids"]:
        parts.append(np.asarray(sim.data.body_xpos[bid], dtype=np.float64))
        q = np.asarray(sim.data.body_xquat[bid], dtype=np.float64)   # mujoco: (w,x,y,z)
        if q[0] < 0:
            q = -q
        parts.append(q)
    return np.concatenate(parts).astype(np.float32)


def resolve_demo_path(cfg: R1Config, env) -> Path:
    """probe_task의 LIBERO 원본 데모 hdf5.

    물체 pose는 LeRobot 데이터셋(이미지+EE 상태만)에 없고 hdf5의 flattened sim state
    (92차원)에만 있다. 그래서 φ의 참조 집합은 hdf5에서 만든다.
    """
    bench = env.unwrapped._libero_benchmark_instance
    rel = bench.get_task_demonstration(cfg.probe_task)      # "libero_spatial/<name>_demo.hdf5"
    roots = []
    if cfg.demo_root:
        roots.append(Path(cfg.demo_root))
    try:
        from gym_libero.libero.utils import get_libero_path

        roots.append(Path(get_libero_path("datasets")))
    except Exception:
        pass
    roots += [Path("Datasets"), Path.home() / "Datasets"]
    for r in roots:
        p = r / rel
        if p.exists():
            return p
    raise SystemExit(
        f"데모 hdf5를 찾지 못했다: {rel}\n  찾아본 곳: {[str(r) for r in roots]}\n"
        f"  --demo_root=<LIBERO 데모 루트> 로 지정해라."
    )


def build_demo_ref(cfg: R1Config, env, spec: dict, out_path: Path) -> dict:
    """[A] 학습 에피소드로 φ 참조 집합 + z-정규화 통계 + τ를 만들어 demo_ref.npz로 굳힌다.

    - 참조 집합 = probe_task 학습 에피소드 전부 (holdout 5개 제외). 이것이 "데모 튜브"다.
    - z-정규화 통계(mean/std)는 **이 집합에서 한 번만** 계산한다. 이후 모든 체크포인트가
      같은 통계를 쓴다. 체크포인트마다 다시 재면 자가 휘어 비교가 무너진다.
    - τ = holdout 5 에피소드의 각 프레임에서 참조 집합까지의 d의 95퍼센타일.
      "데모끼리의 자연 변동보다 멀면 튜브 밖"이라는 정의. 즉 τ는 정책이 아니라
      데모의 성질이며, 모든 방법·스테이지에 공통으로 적용된다.
    """
    try:
        import h5py
    except ModuleNotFoundError as e:
        raise SystemExit("h5py가 필요하다 (LIBERO 데모 hdf5를 읽는다): pip install h5py") from e

    path = resolve_demo_path(cfg, env)
    sim = env.unwrapped._env.sim
    state_dim = sim.get_state().flatten().shape[0]

    with h5py.File(path, "r") as f:
        demos = f["data"]
        n_demo = len(demos)
        # LeRobot 데이터셋의 episode_index와 hdf5의 demo_i는 같은 순서다(변환이 순서를
        # 보존한다). E0의 holdout(뒤 5개)과 정확히 같은 집합을 빼려면 이 대응이 필요하다.
        meta_eps = LeRobotDatasetMetadata(f"{cfg.dataset_prefix}{cfg.probe_task}").total_episodes
        if meta_eps != n_demo:
            logging.warning(colored(
                f"[R1] 에피소드 수 불일치: LeRobot {meta_eps} vs hdf5 {n_demo}. "
                f"앞에서부터 {min(meta_eps, n_demo)}개만 쓴다.", "yellow"))
        n_use = min(meta_eps, n_demo)
        train_eps, holdout_eps = split_episodes(
            f"{cfg.dataset_prefix}{cfg.probe_task}", None, cfg.holdout_episodes)
        train_eps = [e for e in train_eps if e < n_use]
        holdout_eps = [e for e in holdout_eps if e < n_use]

        def demo_phi(ep: int) -> np.ndarray:
            states = np.asarray(demos[f"demo_{ep}"]["states"])
            if states.shape[1] != state_dim:
                raise SystemExit(
                    f"sim state 차원 불일치 (hdf5 {states.shape[1]} vs env {state_dim}). "
                    f"데모와 환경의 bddl이 다르다: {path}")
            out = np.empty((len(states), len(spec["labels"])), dtype=np.float32)
            for i, s in enumerate(states):
                sim.set_state_from_flattened(s)
                sim.forward()
                out[i] = phi_from_sim(sim, spec)
            return out

        logging.info(f"[R1][A] demo hdf5: {path}")
        ref_raw = np.concatenate([demo_phi(e) for e in train_eps], axis=0)
        hold_raw = np.concatenate([demo_phi(e) for e in holdout_eps], axis=0)

    mean = ref_raw.mean(axis=0)
    std = ref_raw.std(axis=0)
    thin = [spec["labels"][i] for i in np.nonzero(std < STD_FLOOR)[0]]
    if thin:
        # 움직이지 않는 차원. 하한을 씌워 잡음 증폭을 막는다(그래도 "물체가 굴러떨어지는"
        # 큰 이탈은 그대로 잡힌다 — 그때는 분자가 실제로 커지기 때문).
        logging.info(f"[R1][A] std<{STD_FLOOR}인 차원 {len(thin)}개에 하한 적용: {thin}")
    std = np.maximum(std, STD_FLOOR)

    ref_z = (ref_raw - mean) / std
    hold_d = nearest_dist(hold_raw, ref_z, mean, std, torch.device("cpu"))
    tau = float(np.percentile(hold_d, 95))

    ref = {
        "ref_raw": ref_raw,
        "ref_z": ref_z.astype(np.float32),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "tau": np.float32(tau),
        "holdout_d": hold_d.astype(np.float32),
        "labels": np.array(spec["labels"]),
        "objects": np.array(spec["objects"]),
        "train_episodes": np.array(train_eps),
        "holdout_episodes": np.array(holdout_eps),
        "probe_task": np.int32(cfg.probe_task),
        "demo_path": np.array(str(path)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **ref)
    logging.info(colored(
        f"[R1][A] demo_ref 저장 -> {out_path}  (참조 {ref_raw.shape[0]} 프레임, "
        f"D={ref_raw.shape[1]}, τ={tau:.3f})", "green"))
    return ref


def load_demo_ref(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def nearest_dist(phi: np.ndarray, ref_z: np.ndarray, mean, std, device) -> np.ndarray:
    """d = min_{demo frames} ‖φ_z − φ_demo,z‖. brute-force cdist.

    참조가 ~5천~1.4만 프레임, D~25라 정확한 최근접이 근사보다 싸고 오해의 소지도 없다
    (KD-tree/근사 인덱스를 쓰면 "이탈"이 인덱스 오차인지 실제인지 구분이 흐려진다).
    """
    q = torch.as_tensor((phi - mean) / std, dtype=torch.float32, device=device)
    r = torch.as_tensor(ref_z, dtype=torch.float32, device=device)
    if q.ndim == 1:
        q = q[None]
    out = torch.cdist(q, r).min(dim=1).values
    return out.cpu().numpy().astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
#  [B] 롤아웃: 현재 정책이 env를 몰고, θ*₁은 같은 관측을 평가만 한다
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def encode_global_cond(policy, obs_hist: list[dict], task: str, device):
    """관측 히스토리(n_obs_steps개)를 정책의 조건 벡터로 인코딩한다.

    select_action의 큐 로직과 같은 입력을 만들되, 두 정책(현재/θ*₁)에 **동일한 원시 관측**을
    먹이기 위해 큐를 우리가 직접 들고 있다. 정규화는 각 정책 자신의 통계로 한다
    (그래서 통계 해시를 따로 비교한다 — norm_stats_hash 참조).
    """
    batch = {}
    for key in obs_hist[0]:
        if key == "task":
            continue
        batch[key] = torch.stack([o[key] for o in obs_hist], dim=1).to(device)  # (1, n_obs, ...)
    batch["task"] = [task]
    batch = policy.normalize_inputs(batch)
    if policy.config.image_features:
        batch = dict(batch)
        batch["observation.images"] = torch.stack(
            [batch[k] for k in policy.config.image_features], dim=-4)
    return policy.dit_flow._prepare_global_conditioning(batch)


@torch.no_grad()
def sample_chunks(policy, global_cond, k_samples: int, seed: int, device):
    """같은 조건에서 K개 액션 청크를 뽑는다. a0 시드가 고정이라 재현된다.

    반환 (K, horizon, action_dim) — 정규화 공간([-1,1]). 실제 단위 변환은 호출자가 한다.
    """
    gen = torch.Generator(device=device).manual_seed(int(seed) % (2**31 - 1))
    cond = global_cond.expand(k_samples, -1)
    return policy.dit_flow.velocity_net.sample(
        cond, timesteps=policy.dit_flow.num_inference_steps, generator=gen)


def executed_slice(policy, chunk: torch.Tensor) -> torch.Tensor:
    """generate_actions와 같은 슬라이싱 + 역정규화. (…, horizon, adim) -> (…, n_action_steps, adim).

    start=n_obs_steps-1인 이유는 modeling_dit_flow_mt.generate_actions의 주석과 같다
    (index 0은 t-1이라 버린다). 여기서 어긋나면 Δa가 한 스텝 밀린 행동을 비교하게 된다.
    """
    start = policy.config.n_obs_steps - 1
    end = start + policy.config.n_action_steps
    sliced = chunk[..., start:end, :]
    return policy.unnormalize_outputs({ACTION: sliced})[ACTION]


def rollout_checkpoint(cfg: R1Config, env, spec: dict, ref: dict, policy, ref_policy,
                       device, method: str, stage: int, ckpt: Path) -> dict:
    """체크포인트 하나로 num_rollouts개를 돌리고 스텝별 원시 기록을 만든다.

    핵심 규칙
      - 초기 상태는 rollout_id로 **지정**한다(시드가 아니라 인덱스). 모든 체크포인트가
        같은 집합을 쓴다 -> 짝지은 비교.
      - 실행 액션은 K샘플 중 0번(시드 고정). ā는 K개의 평균. 즉 "표준 평가와 같은
        단일 샘플 실행"을 유지하면서 평균/분산을 공짜로 얻는다.
      - θ*₁은 같은 관측·같은 a0로 평가만 하고 env에는 손대지 않는다.
      - 성공으로 조기 종료된 rollout은 마지막 값으로 패딩한다(생존 편향 방지, censored 표기).
    """
    max_steps = cfg.max_steps or cfg.env.episode_length
    n_act = policy.config.n_action_steps
    stride = cfg.action_eval_stride or n_act
    da_steps = np.arange(0, max_steps, stride)
    n_eval = len(da_steps)
    D = len(spec["labels"])
    R = cfg.num_rollouts

    init_states = env.unwrapped._init_states
    if R > len(init_states):
        raise SystemExit(f"num_rollouts={R} > 사용 가능한 초기 상태 {len(init_states)}개")
    init_ids = np.arange(R)
    init_hash = hashlib.sha1(
        np.ascontiguousarray(np.asarray(init_states)[:R], dtype=np.float64).tobytes()).hexdigest()[:12]

    phi_all = np.zeros((R, max_steps, D), dtype=np.float32)
    da_all = np.full((R, n_eval), np.nan, dtype=np.float32)
    var_cur_all = np.full((R, n_eval), np.nan, dtype=np.float32)
    var_ref_all = np.full((R, n_eval), np.nan, dtype=np.float32)
    lengths = np.zeros(R, dtype=np.int32)
    success = np.zeros(R, dtype=bool)
    obs_debug: dict[str, np.ndarray] = {}

    task_text = env.unwrapped.task_description
    # 정착용 null 액션: OSC_POSE 델타 6개 = 0, 그리퍼는 열림(-1).
    null_action = np.zeros(env.action_space.shape, dtype=np.float32)
    null_action[-1] = -1.0

    for rid in range(R):
        env.reset()
        # ★ robosuite는 reset마다 모델을 다시 올려 MjSim 객체를 새로 만든다.
        #   sim/스펙을 루프 밖에서 캐시하면 두 번째 에피소드에서 죽은 객체를 참조한다.
        sim = env.unwrapped._env.sim
        spec = phi_spec(env, spec["objects"])
        raw = env.unwrapped.set_init_state(init_states[rid])     # 초기 상태를 명시적으로 고정
        obs = env.unwrapped._format_raw_obs(raw)
        for _ in range(cfg.settle_steps):                        # 물체를 테이블에 내려앉힌다
            obs, _r, _term, _trunc, _i = env.step(null_action)
        policy.reset()
        ref_policy.reset()

        hist: list[dict] = []
        queue: list[np.ndarray] = []
        frames = []
        t = 0
        for t in range(max_steps):
            proc = preprocess_observation(obs)                    # (1,3,H,W) / (1,8)
            proc.pop("task", None)
            if not hist:
                hist = [proc] * policy.config.n_obs_steps         # 첫 스텝은 복제해 창을 채운다
            else:
                hist = (hist + [proc])[-policy.config.n_obs_steps:]

            phi_all[rid, t] = phi_from_sim(sim, spec)

            replan = len(queue) == 0
            evaluate = (t % stride) == 0
            if replan or evaluate:
                cond_cur = encode_global_cond(policy, hist, task_text, device)
                chunk_cur = sample_chunks(policy, cond_cur, cfg.num_samples, chunk_seed(cfg, rid, t), device)
                exec_cur = executed_slice(policy, chunk_cur)      # (K, n_act, adim) 실제 단위

            if evaluate:
                cond_ref = encode_global_cond(ref_policy, hist, task_text, device)
                chunk_ref = sample_chunks(ref_policy, cond_ref, cfg.num_samples,
                                          chunk_seed(cfg, rid, t), device)
                exec_ref = executed_slice(ref_policy, chunk_ref)
                a_cur = exec_cur.mean(dim=0)
                a_ref = exec_ref.mean(dim=0)
                j = t // stride
                da_all[rid, j] = float(torch.linalg.vector_norm(a_cur - a_ref))
                # K샘플 분산 (bias/variance 분해용 부산물): Δa가 커도 분산이 같이 크면
                # "정책이 흔들린다"는 뜻이고, 분산이 작은데 Δa만 크면 "확신을 갖고 다른
                # 행동을 한다"는 뜻이다.
                var_cur_all[rid, j] = float(exec_cur.var(dim=0, unbiased=False).mean())
                var_ref_all[rid, j] = float(exec_ref.var(dim=0, unbiased=False).mean())

            if replan:
                queue = list(exec_cur[0].cpu().numpy())            # 실행은 0번 샘플(시드 고정)

            action = queue.pop(0)
            if cfg.save_obs and rid < cfg.save_obs_rollouts and t % cfg.save_obs_stride == 0:
                frames.append(obs["pixels"]["image"].copy())

            obs, _reward, terminated, truncated, _info = env.step(np.asarray(action, dtype=np.float32))
            if terminated or truncated:
                success[rid] = bool(terminated)                    # LiberoEnv: terminated == 성공
                break

        lengths[rid] = t + 1
        # 조기 종료분은 마지막 값으로 패딩한다. 시간축 집계에서 "성공해서 사라진 rollout"이
        # 평균을 낮추는 생존 편향을 막기 위함이다(censored 표기는 요약에서 따로 남긴다).
        if lengths[rid] < max_steps:
            phi_all[rid, lengths[rid]:] = phi_all[rid, lengths[rid] - 1]
        last = np.where(np.isfinite(da_all[rid]))[0]
        if len(last):
            da_all[rid, last[-1] + 1:] = da_all[rid, last[-1]]
            var_cur_all[rid, last[-1] + 1:] = var_cur_all[rid, last[-1]]
            var_ref_all[rid, last[-1] + 1:] = var_ref_all[rid, last[-1]]
        if frames:
            obs_debug[f"rollout_{rid}"] = np.stack(frames)

        logging.info(
            f"[R1][B] {method} stage{stage} rollout {rid + 1}/{R}: "
            f"len={lengths[rid]} success={bool(success[rid])} "
            f"d_end={nearest_dist(phi_all[rid, lengths[rid] - 1], ref['ref_z'], ref['mean'], ref['std'], device)[0]:.2f}")

    d_all = np.stack([
        nearest_dist(phi_all[r], ref["ref_z"], ref["mean"], ref["std"], device) for r in range(R)
    ]).astype(np.float32)

    return {
        "phi": phi_all,
        "d": d_all,
        "da": da_all,
        "da_steps": da_steps.astype(np.int32),
        "var_cur": var_cur_all,
        "var_ref": var_ref_all,
        "lengths": lengths,
        "success": success,
        "init_ids": init_ids.astype(np.int32),
        "init_hash": np.array(init_hash),
        "meta": np.array(json.dumps({
            "method": method,
            "stage": stage,
            "ckpt": str(ckpt),
            "probe_task": cfg.probe_task,
            "num_rollouts": R,
            "max_steps": max_steps,
            "num_samples": cfg.num_samples,
            "stride": stride,
            "settle_steps": cfg.settle_steps,
            "seed_base": cfg.rollout_seed_base,
            "norm_hash": norm_stats_hash(policy),
            "ref_norm_hash": norm_stats_hash(ref_policy),
            "labels": spec["labels"],
        })),
        "_obs_debug": obs_debug,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  [C] held-out FM loss (고정 (τ_fm, a0) 격자)
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def heldout_fm_loss(cfg: R1Config, policy, device) -> float:
    """probe_task holdout 에피소드에서의 flow matching loss.

    E0의 probe_mse와 같은 데이터를 보지만 난수를 더 세게 묶는다: τ_fm을 U(0,1)에서
    뽑는 대신 **고정 격자 10점**, 노이즈 a0도 **고정 8개**를 쓴다. 그림 C의 x축은
    체크포인트 간 아주 작은 차이를 읽어야 하는데, 샘플링 잡음이 그 차이보다 크면
    산점도가 통째로 흐려지기 때문이다.
    """
    repo_id = f"{cfg.dataset_prefix}{cfg.probe_task}"
    _, holdout_eps = split_episodes(repo_id, None, cfg.holdout_episodes)
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps=resolve_delta_timestamps(policy.config, LeRobotDatasetMetadata(repo_id)),
        video_backend=cfg.dataset.video_backend,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=cfg.loss_batch_size,
        sampler=episode_sampler(cfg, dataset, holdout_eps),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    torch.manual_seed(cfg.loss_seed)     # 샘플러 셔플까지 재현되도록 iterator 생성 전에 고정
    it = cycle(loader)

    # τ_fm 격자: 중점 규칙 (0.05, 0.15, ..., 0.95). 끝점 0/1을 피하는 이유는 t=0에서
    # 목표 속도가 조건과 무관해지고 t=1에서 노이즈 항이 사라져 둘 다 정보가 적기 때문.
    taus = (torch.arange(cfg.loss_n_tau, device=device, dtype=torch.float32) + 0.5) / cfg.loss_n_tau
    net = policy.dit_flow.velocity_net
    total, count = 0.0, 0
    for b in range(cfg.loss_batches):
        batch = to_device(next(it), device)
        batch = policy.normalize_inputs(batch)
        if policy.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[k] for k in policy.config.image_features], dim=-4)
        batch = policy.normalize_targets(batch)
        cond = policy.dit_flow._prepare_global_conditioning(batch)
        traj = batch[ACTION]
        for j in range(cfg.loss_n_a0):
            gen = torch.Generator(device=device).manual_seed(cfg.loss_seed + 1000 * b + j)
            noise = torch.randn(traj.shape, generator=gen, device=device, dtype=traj.dtype)
            target = traj - noise
            for tau in taus:
                t = tau.expand(traj.shape[0])
                noisy = (1 - tau) * noise + tau * traj
                pred = net(noisy_actions=noisy, time=t, global_cond=cond)
                total += float(torch.nn.functional.mse_loss(pred, target))
                count += 1
    return total / max(count, 1)


def e0_heldout_loss(cfg: R1Config, method: str, stage: int) -> float | None:
    """E0 결과 JSONL에서 같은 (run_tag, stage, probe_task)의 mse를 읽어 [C]를 대체한다."""
    if not cfg.e0_results or not Path(cfg.e0_results).exists():
        return None
    tags = parse_kv(cfg.e0_run_tags) if cfg.e0_run_tags else {}
    tag = tags.get(method)
    if tag is None:
        return None
    val = None
    for line in Path(cfg.e0_results).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r.get("run_tag") == tag and r.get("stage") == stage
                and r.get("probe_task") == cfg.probe_task):
            val = r.get("mse")            # append-only이므로 뒤쪽(최신)이 이긴다
    return val


# ═════════════════════════════════════════════════════════════════════════════
#  [D] 요약 스칼라
# ═════════════════════════════════════════════════════════════════════════════
def rollout_metrics(rec: dict, tau: float) -> list[dict]:
    """rollout별 스칼라. t_star의 검열 처리가 여기 있다.

    dAUC/dwell은 두 벌을 낸다.
      *_padded : 공통 시간축(0..T) 위에서. 조기 성공분은 마지막 값으로 채워져 있다.
                 그림과 요약은 이쪽을 쓴다(rollout 길이가 달라도 같은 자로 잰다).
      *_alive  : 실제로 살아 있던 스텝만. "살아 있는 동안의 평균 이탈"이라는 다른 질문.
    """
    d, lengths, success = rec["d"], rec["lengths"], rec["success"]
    T = d.shape[1]
    out = []
    for i in range(d.shape[0]):
        li = int(lengths[i])
        real = d[i, :li]
        over = np.nonzero(real > tau)[0]
        censored = len(over) == 0
        out.append({
            "rollout": i,
            "success": bool(success[i]),
            "length": li,
            "dauc": float(d[i].mean()),
            "dauc_alive": float(real.mean()),
            "dwell": float((d[i] <= tau).mean()),
            "dwell_alive": float((real <= tau).mean()),
            # 이탈하지 않은 rollout은 t_star=T + censored=True. 이걸 빼고 중앙값을 내면
            # "이탈한 것들만의 중앙값"이 되어 좋은 체크포인트가 나쁘게 보인다.
            "t_star": T if censored else int(over[0]),
            "censored": bool(censored),
            "da_mean": float(np.nanmean(rec["da"][i])) if np.isfinite(rec["da"][i]).any() else None,
        })
    return out


def checkpoint_summary(rec: dict, tau: float, per: list[dict]) -> dict:
    return {
        "sr": float(np.mean([p["success"] for p in per])),
        "dauc": float(np.median([p["dauc"] for p in per])),
        "dauc_alive": float(np.median([p["dauc_alive"] for p in per])),
        "t_star": float(np.median([p["t_star"] for p in per])),
        "t_star_censored_frac": float(np.mean([p["censored"] for p in per])),
        "dwell": float(np.median([p["dwell"] for p in per])),
        "dwell_alive": float(np.median([p["dwell_alive"] for p in per])),
        "median_length": float(np.median([p["length"] for p in per])),
        "tau": float(tau),
        "T": int(rec["d"].shape[1]),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  메인 (train.py / E0 / H5와 같은 [1]~ 순서)
# ═════════════════════════════════════════════════════════════════════════════
@parser.wrap()
def main(cfg: R1Config):
    # ── [1] 설정 ─────────────────────────────────────────────────────────────
    cfg.validate()
    cfg.save_checkpoint = False               # R1은 아무것도 저장하지 않는다(학습 없음)
    if not cfg.ckpt_roots:
        raise SystemExit(
            "--ckpt_roots 가 필요하다. 예: "
            "--ckpt_roots=\"seq=outputs/E0/.../lam0,ewc=outputs/E0/.../lam100,er=outputs/ER/...\"")
    roots = parse_kv(cfg.ckpt_roots)
    run_tag = cfg.run_tag or f"{getattr(cfg.env, 'benchmark', 'libero')}_probe{cfg.probe_task}"
    run_dir = Path(cfg.out_root) / run_tag
    (run_dir / "cache").mkdir(parents=True, exist_ok=True)
    results = run_dir / "r1_results.jsonl"
    logging.info(pformat(cfg.to_dict()))
    logging.info(colored(
        f"[R1] probe_task={cfg.probe_task}  methods={list(roots)}  "
        f"stages=0..{cfg.num_stages - 1}  rollouts={cfg.num_rollouts}  -> {run_dir}",
        "green", attrs=["bold"]))

    # ── [2] 로거: 스칼라 표와 그림만 내므로 wandb를 쓰지 않는다 ───────────────

    # ── [3] 재현성 ───────────────────────────────────────────────────────────
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # ── [4] 디바이스 ─────────────────────────────────────────────────────────
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── [5] 데이터셋 메타 (정책 생성용). 전체 데이터셋은 [C]에서만 연다 ───────
    ds_meta = LeRobotDatasetMetadata(f"{cfg.dataset_prefix}{cfg.probe_task}")

    # ── [6] 환경 + φ 정의 ────────────────────────────────────────────────────
    env = make_probe_env(cfg)
    objects = [s.strip() for s in (cfg.phi_objects or DEFAULT_PHI_OBJECTS).split(",") if s.strip()]
    spec = phi_spec(env, objects)
    logging.info(f"[R1] φ 차원 {len(spec['labels'])}: {spec['labels']}")

    # ── [7] [A] 데모 참조 집합 (있으면 재사용 — τ와 z통계는 절대 다시 재지 않는다) ──
    ref_path = run_dir / "demo_ref.npz"
    if ref_path.exists() and not cfg.recompute_demo_ref:
        ref = load_demo_ref(ref_path)
        logging.info(colored(
            f"[R1][A] demo_ref 재사용: {ref_path} (τ={float(ref['tau']):.3f}, "
            f"참조 {ref['ref_z'].shape[0]} 프레임)", "cyan"))
        if list(ref["objects"]) != objects:
            raise SystemExit(
                f"demo_ref의 물체 목록이 다르다: 저장 {list(ref['objects'])} vs 요청 {objects}\n"
                f"  --recompute_demo_ref 로 다시 만들거나 --phi_objects 를 맞춰라.")
    else:
        ref = build_demo_ref(cfg, env, spec, ref_path)
    tau = float(ref["tau"])

    # ── [8] reference 정책 θ*₁ — "망각 전의 자기 자신" ────────────────────────
    ref_ckpt = Path(cfg.ref_ckpt) if cfg.ref_ckpt else stage_ckpt(next(iter(roots.values())), cfg.probe_task)
    logging.info(colored(f"[R1] reference policy θ*₁ = {ref_ckpt}", "cyan"))
    ref_policy = load_policy_at(cfg, ref_ckpt, ds_meta, device)
    ref_hash = norm_stats_hash(ref_policy)

    # ── [9] 볼 체크포인트 목록 ───────────────────────────────────────────────
    targets: list[tuple[str, int, Path]] = []
    for m, root in roots.items():
        for k in range(cfg.num_stages):
            targets.append((m, k, stage_ckpt(root, k)))
    for label, path in parse_kv(cfg.extra_ckpts).items():
        m, _, s = label.partition("@")
        targets.append((m, int(s) if s else -1, Path(path)))

    def emit(row: dict):
        with results.open("a") as f:
            f.write(json.dumps(row) + "\n")

    # ── [10] 체크포인트마다: 롤아웃(캐시) -> held-out loss -> 요약 ────────────
    init_hashes: dict[str, str] = {}
    for method, stage, ckpt in targets:
        cache = run_dir / "cache" / f"rollouts_{method}_stage{stage}.npz"
        if not Path(ckpt).exists():
            logging.warning(colored(f"[R1] 체크포인트 없음, 건너뜀: {ckpt}", "yellow"))
            continue

        if cache.exists() and not cfg.recompute_rollouts:
            z = np.load(cache, allow_pickle=False)
            rec = {k: z[k] for k in z.files}
            logging.info(colored(f"[R1][B] 캐시 재사용: {cache}", "cyan"))
            policy = None
        else:
            logging.info(colored(
                f"[R1][B] rollout {method} stage{stage}: {ckpt}", "cyan", attrs=["bold"]))
            policy = load_policy_at(cfg, ckpt, ds_meta, device)
            h = norm_stats_hash(policy)
            if h != ref_hash:
                # 중단하지는 않는다. 다만 Δa의 일부가 "단위 차이"일 수 있음을 알린다.
                logging.warning(colored(
                    f"[R1] 정규화 통계가 θ*₁과 다르다 ({h} vs {ref_hash}). "
                    f"Δa에 단위 차이가 섞일 수 있다 — {ckpt}", "yellow"))
            rec = rollout_checkpoint(cfg, env, spec, ref, policy, ref_policy, device,
                                     method, stage, Path(ckpt))
            obs_debug = rec.pop("_obs_debug", {})
            np.savez_compressed(cache, **rec)
            logging.info(f"[R1][B] 원시 기록 저장 -> {cache}")
            if obs_debug:
                p = run_dir / "cache" / f"obs_{method}_stage{stage}.npz"
                np.savez_compressed(p, **obs_debug)
                logging.info(f"[R1][B] 디버그 관측 저장 -> {p}")

        # 위생 체크: 초기 상태 집합이 모든 체크포인트에서 같은가 (짝지은 비교의 전제)
        ih = str(rec["init_hash"])
        init_hashes[f"{method}@{stage}"] = ih
        first_key, first_hash = next(iter(init_hashes.items()))
        assert ih == first_hash, (
            f"초기 상태가 체크포인트마다 다르다: {method}@{stage}={ih} vs {first_key}={first_hash}. "
            f"짝지은 비교가 성립하지 않는다.")
        meta = json.loads(str(rec["meta"]))
        # 캐시와 현재 설정이 어긋난 채 섞이면 "체크포인트 차이"로 읽힐 값이 사실은
        # "프로토콜 차이"가 된다. 조용히 섞이느니 여기서 멈춘다.
        want = {"num_samples": cfg.num_samples,
                "max_steps": cfg.max_steps or cfg.env.episode_length,
                "settle_steps": cfg.settle_steps,
                "seed_base": cfg.rollout_seed_base,
                "probe_task": cfg.probe_task}
        bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
        assert not bad, (
            f"캐시와 현재 설정이 다르다 (캐시값, 현재값): {bad} — "
            f"--recompute_rollouts 로 다시 돌려라 ({cache})")
        # ★ 기계장치 자체를 검산하는 지점: 굴린 체크포인트가 θ*₁ 바로 그것이면
        #   Δa는 **정확히 0**이어야 한다. 0이 아니면 (rollout_id, step)에서 유도한 a0
        #   시드가 두 정책에 다르게 먹혔거나 관측이 갈렸다는 뜻이다.
        if Path(ckpt).resolve() == Path(ref_ckpt).resolve():
            worst = float(np.nanmax(np.abs(rec["da"])))
            assert worst < 1e-4, (
                f"Δa self-check 실패: {method} stage{stage}는 θ*₁ 자신인데 Δa 최대 {worst:.3e}. "
                f"a0 시드/관측 경로가 두 정책에서 갈렸다.")
            logging.info(colored(f"[R1] Δa self-check 통과 (θ*₁ 자신, max|Δa|={worst:.1e})", "green"))

        # [C] held-out loss: E0 결과가 있으면 그것을, 없으면 여기서 고정 격자로 잰다.
        loss = e0_heldout_loss(cfg, method, stage)
        if loss is None:
            if policy is None:
                policy = load_policy_at(cfg, ckpt, ds_meta, device)
            logging.info(f"[R1][C] held-out FM loss (고정 격자) — {method} stage{stage}")
            loss = heldout_fm_loss(cfg, policy, device)

        per = rollout_metrics(rec, tau)
        for p in per:
            emit({"kind": "rollout_summary", "method": method, "stage": stage,
                  "probe_task": cfg.probe_task, "seed": cfg.seed, **p})
        summary = checkpoint_summary(rec, tau, per)
        emit({"kind": "checkpoint", "method": method, "stage": stage,
              "probe_task": cfg.probe_task, "seed": cfg.seed, "ckpt": str(ckpt),
              "ref_ckpt": str(ref_ckpt), "num_rollouts": cfg.num_rollouts,
              "heldout_loss": loss, "norm_hash": meta["norm_hash"], **summary})
        logging.info(colored(
            f"[R1][D] {method} stage{stage}: SR={summary['sr']:.2f}  dAUC={summary['dauc']:.2f}  "
            f"t*={summary['t_star']:.0f} (censored {summary['t_star_censored_frac']:.0%})  "
            f"dwell={summary['dwell']:.2f}  loss={loss:.5f}", "green"))

        del policy, rec
        if device.type == "cuda":
            torch.cuda.empty_cache()

    env.close()
    logging.info(colored(f"[R1] done -> {results}", "green", attrs=["bold"]))

    if not cfg.no_plot:
        plot_r1(str(run_dir), split_by_success=cfg.split_by_success, figb=cfg.figb_ckpts)


# ═════════════════════════════════════════════════════════════════════════════
#  그림 (--plot_only에서도 여기만 돈다)
# ═════════════════════════════════════════════════════════════════════════════
METHOD_LABEL = {"seq": "Seq (fine-tune)", "ewc": "EWC", "er": "ER", "frozen": "Frozen (lambda=inf)"}
METHOD_COLOR = {"seq": "#4c72b0", "ewc": "#dd8452", "er": "#55a868", "frozen": "#8172b3"}
# 이름을 모르는 팔(lam10, 다른 시드, λ 스윕 점 등)에도 서로 다른 색이 가야 한다.
# 예전처럼 fallback 색 하나를 공유하면 팔 두 개가 그림 C에서 같은 색으로 찍혀 구분이 안 된다.
FALLBACK_COLORS = ["#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd", "#c44e52"]
STAGE_MARKER = ["o", "s", "^", "D", "v", "P"]


def method_color(method: str, methods: list[str]) -> str:
    """알려진 팔은 고정 색, 나머지는 등장 순서대로 팔레트를 돌려 쓴다."""
    if method in METHOD_COLOR:
        return METHOD_COLOR[method]
    unknown = [m for m in methods if m not in METHOD_COLOR]
    return FALLBACK_COLORS[unknown.index(method) % len(FALLBACK_COLORS)] if method in unknown \
        else FALLBACK_COLORS[0]


def load_checkpoint_rows(run_dir: Path) -> list[dict]:
    """r1_results.jsonl의 checkpoint 행. 같은 (method, stage)는 뒤쪽(최신)만 남긴다.

    ★ JSONL은 append-only다. 같은 체크포인트를 다시 돌리면 옛 행이 남아 산점도에
      점이 두 개 찍힌다(E0.load_rows와 같은 이유).
    """
    p = run_dir / "r1_results.jsonl"
    if not p.exists():
        raise SystemExit(f"결과가 없다: {p}")
    uniq: dict[tuple, dict] = {}
    n = 0
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") != "checkpoint":
            continue
        n += 1
        uniq[(r["method"], r["stage"])] = r
    if n != len(uniq):
        print(f"deduped: dropped {n - len(uniq)} stale checkpoint row(s)")
    return list(uniq.values())


def load_caches(run_dir: Path) -> dict[tuple[str, int], dict]:
    out = {}
    for p in sorted((run_dir / "cache").glob("rollouts_*.npz")):
        stem = p.stem[len("rollouts_"):]
        method, _, s = stem.rpartition("_stage")
        z = np.load(p, allow_pickle=False)
        out[(method, int(s))] = {k: z[k] for k in z.files}
    return out


def shades(base: str, n: int):
    """같은 색의 진하기 n단계 (연함 -> 진함). 스테이지 순서를 색 하나로 읽게 한다."""
    import matplotlib.colors as mc

    rgb = mc.to_rgb(base)
    return [tuple(1 - (1 - c) * w for c in rgb) for w in np.linspace(0.35, 1.0, max(n, 1))]


def band(ax, x, mat, color, label=None, lw=1.8):
    """중앙값 실선 + 25~75퍼센타일 밴드.

    ★ 표준편차 밴드를 쓰지 않는 이유: d(t)는 소수의 rollout이 크게 발산하는 꼬리 분포다.
      평균/표준편차는 그 소수에 끌려가 "대부분의 rollout이 어디 있는지"를 감춘다.
    """
    med = np.median(mat, axis=0)
    q25, q75 = np.percentile(mat, [25, 75], axis=0)
    ax.fill_between(x, q25, q75, color=color, alpha=0.3, linewidth=0)
    (ln,) = ax.plot(x, med, color=color, lw=lw, label=label)
    return ln, med, q25, q75


def write_csv(path: Path, header: list[str], rows: list[list]):
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")
    print(f"saved table  -> {path}")


def plot_A(run_dir: Path, caches, rows, tau, plt, suffix="", mask_fn=None, title_extra=""):
    """그림 A — 튜브 이탈 곡선. x=롤아웃 시각 t, y=d(t)."""
    methods = [m for m in dict.fromkeys(r["method"] for r in rows)
               if any(k[0] == m for k in caches)]
    if not methods:
        return
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4.4),
                             squeeze=False, sharey=True)
    csv_rows = []
    for col, m in enumerate(methods):
        ax = axes[0][col]
        stages = sorted(s for mm, s in caches if mm == m)
        cols = shades(method_color(m, methods), len(stages))
        for i, s in enumerate(stages):
            rec = caches[(m, s)]
            d = rec["d"]
            if mask_fn is not None:
                keep = mask_fn(rec)
                if keep.sum() == 0:
                    continue
                d = d[keep]
            x = np.arange(d.shape[1])
            _, med, q25, q75 = band(ax, x, d, cols[i], label=f"stage{s + 1}")
            for t in range(0, d.shape[1], 5):
                csv_rows.append([m, s + 1, t, f"{med[t]:.4f}", f"{q25[t]:.4f}", f"{q75[t]:.4f}",
                                 f"{tau:.4f}", d.shape[0]])
        ax.axhline(tau, color="k", ls="--", lw=1.2)
        ax.text(0.99, tau, " demo tube width (95th pct of held-out demos)", ha="right", va="bottom",
                transform=ax.get_yaxis_transform(), fontsize=8, color="k")
        ax.set(xlabel="rollout time t (env steps)",
               ylabel="d(t): distance to demo tube (z-normalized)" if col == 0 else "",
               title=f"{METHOD_LABEL.get(m, m)}{title_extra}")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, title="learned tasks 0..k", title_fontsize=8)
        else:
            # --split_by_success에서 한쪽이 통째로 비는 경우(성공 0개 등)를 명시한다.
            ax.text(0.5, 0.5, "no rollouts in this subset", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
    fig.suptitle("R1-A: when does the policy leave the states the expert ever visited?"
                 "  (median + IQR over rollouts)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = run_dir / f"R1_A_tube_departure{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out}")
    plt.close(fig)
    write_csv(run_dir / f"R1_A_tube_departure{suffix}.csv",
              ["method", "stage", "t", "d_median", "d_q25", "d_q75", "tau", "n_rollouts"], csv_rows)


def plot_B(run_dir: Path, caches, rows, tau, plt, figb: str):
    """그림 B — 상태 이탈(d)과 행동 불일치(Δa)의 선후. 위/아래가 x축을 공유한다."""
    want = []
    for item in figb.split(","):
        item = item.strip()
        if not item:
            continue
        m, _, s = item.partition("@")
        key = (m, int(s))
        if key in caches:
            want.append(key)
        else:
            print(f"[R1-B] 캐시 없음, 건너뜀: {item}")
    if not want:
        want = sorted(caches)[-3:]
    if not want:
        return

    by_key = {(r["method"], r["stage"]): r for r in rows}
    all_methods = list(dict.fromkeys([r["method"] for r in rows] + [k[0] for k in caches]))
    fig, axes = plt.subplots(2, len(want), figsize=(4.8 * len(want), 7.2),
                             squeeze=False, sharex=True, sharey="row")
    csv_rows = []
    for col, (m, s) in enumerate(want):
        rec = caches[(m, s)]
        base = method_color(m, all_methods)
        d = rec["d"]
        x = np.arange(d.shape[1])
        _, dmed, dq25, dq75 = band(axes[0][col], x, d, base)
        axes[0][col].axhline(tau, color="k", ls="--", lw=1.2)

        da = rec["da"]
        xs = rec["da_steps"]
        ok = np.isfinite(da).all(axis=0)
        _, amed, aq25, aq75 = band(axes[1][col], xs[ok], da[:, ok], base)

        t_star = by_key.get((m, s), {}).get("t_star")
        if t_star is not None:
            for ax in (axes[0][col], axes[1][col]):
                ax.axvline(t_star, color="crimson", ls=":", lw=1.5)
            axes[0][col].text(t_star, 0.98, " median $t^*$", transform=axes[0][col].get_xaxis_transform(),
                              va="top", fontsize=8, color="crimson")

        axes[0][col].set(title=f"{METHOD_LABEL.get(m, m)}  stage{s + 1}")
        axes[0][col].grid(alpha=0.3)
        axes[1][col].set(xlabel="rollout time t (env steps)")
        axes[1][col].grid(alpha=0.3)
        if col == 0:
            axes[0][col].set_ylabel("d(t): distance to demo tube")
            axes[1][col].set_ylabel(r"$\Delta a(t)=\|\bar a_\pi(o_t)-\bar a_{\theta^*_1}(o_t)\|$")
        for i, t in enumerate(xs[ok]):
            csv_rows.append([m, s + 1, int(t), f"{dmed[t]:.4f}", f"{dq25[t]:.4f}", f"{dq75[t]:.4f}",
                             f"{amed[i]:.4f}", f"{aq25[i]:.4f}", f"{aq75[i]:.4f}",
                             "" if t_star is None else t_star])
    fig.suptitle("R1-B: does state drift lead action damage, or the other way round?\n"
                 "top = drift away from the demo tube, bottom = disagreement with the pre-forgetting self "
                 "on the SAME observations", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = run_dir / "R1_B_state_vs_action.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out}")
    plt.close(fig)
    write_csv(run_dir / "R1_B_state_vs_action.csv",
              ["method", "stage", "t", "d_median", "d_q25", "d_q75",
               "da_median", "da_q25", "da_q75", "t_star_median"], csv_rows)


def plot_C(run_dir: Path, rows, plt, right_key="dauc", suffix="", right_label=None):
    """그림 C — 무엇이 SR을 예측하는가. 왼쪽=전문가 상태에서 잰 loss, 오른쪽=자기 롤아웃 지표."""
    pts = [r for r in rows if r.get("heldout_loss") is not None and r.get("sr") is not None
           and r.get(right_key) is not None]
    if len(pts) < 3:
        print(f"[R1-C] 점이 부족하다 ({len(pts)}개)")
        return
    right_label = right_label or {
        "dauc": "dAUC: mean tube departure on the policy's OWN rollouts",
        "t_star": r"$t^*$: first step outside the tube (censored at T)",
        "dwell": r"dwell: fraction of steps with $d(t)\leq\tau$",
    }.get(right_key, right_key)

    methods = list(dict.fromkeys(r["method"] for r in pts))
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for ax, key, xlabel, title in (
        (ax_l, "heldout_loss", "held-out demo FM loss (probe task, fixed grid)",
         "graded where the EXPERT was"),
        (ax_r, right_key, right_label, "graded where the POLICY went"),
    ):
        for r in pts:
            m, s = r["method"], r["stage"]
            ax.scatter(r[key], r["sr"], s=90,
                       color=method_color(m, methods),
                       marker=STAGE_MARKER[s % len(STAGE_MARKER)] if s >= 0 else "X",
                       edgecolors="k", linewidths=0.7, zorder=3)
        rho, r2 = spearman_r2(np.array([r[key] for r in pts]), np.array([r["sr"] for r in pts]))
        # 한쪽 축이 상수면(예: 모든 SR이 0) 상관은 정의되지 않는다. nan을 그대로 찍지 않고
        # 그렇게 말해 준다 — "상관 없음"과 "계산 불가"는 다른 이야기이기 때문.
        stat = ("n/a (a variable is constant)" if not np.isfinite(rho)
                else f"Spearman $\\rho$ = {rho:+.2f}\n$R^2$ = {r2:.2f}")
        ax.text(0.03, 0.05, f"{stat}\nn = {len(pts)} checkpoints",
                transform=ax.transAxes, fontsize=11, va="bottom",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"})
        ax.set(xlabel=xlabel, title=title)
        ax.grid(alpha=0.3)
    ax_l.set(ylabel="measured SR on the same rollouts", ylim=(-0.05, 1.05))

    handles = []
    from matplotlib.lines import Line2D

    for m in methods:
        handles.append(Line2D([], [], marker="o", ls="", color=method_color(m, methods),
                              markeredgecolor="k", label=METHOD_LABEL.get(m, m)))
    for s in sorted({r["stage"] for r in pts}):
        handles.append(Line2D([], [], marker=STAGE_MARKER[s % len(STAGE_MARKER)] if s >= 0 else "X",
                              ls="", color="0.5", markeredgecolor="k",
                              label=f"stage{s + 1}" if s >= 0 else "extra"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97),
               ncol=len(handles), fontsize=9, frameon=False)
    fig.suptitle("R1-C: a cloud on the left and a line on the right means the loss-SR dissociation "
                 "was about WHERE we measured, not WHAT we measured", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = run_dir / f"R1_C_predictor{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {out}")
    plt.close(fig)
    write_csv(run_dir / f"R1_C_predictor{suffix}.csv",
              ["method", "stage", "heldout_loss", "sr", "dauc", "t_star", "dwell", "ckpt"],
              [[r["method"], r["stage"] + 1, r["heldout_loss"], r["sr"], r["dauc"],
                r["t_star"], r["dwell"], r.get("ckpt", "")] for r in pts])


def plot_r1(run_dir_str: str, split_by_success: bool = False, figb: str = "ewc@1,ewc@2,ewc@3"):
    run_dir = Path(run_dir_str)
    rows = load_checkpoint_rows(run_dir)
    caches = load_caches(run_dir)
    tau = float(np.load(run_dir / "demo_ref.npz")["tau"]) if (run_dir / "demo_ref.npz").exists() \
        else float(rows[0]["tau"])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib 없음 -> 그림 생략 (pip install matplotlib 후 --plot_only 다시)")
        return

    plot_A(run_dir, caches, rows, tau, plt)
    if split_by_success:
        # 부록: 성공/실패를 나눠 본다. "실패한 rollout만 이탈한다"면 d는 결과이지 원인이
        # 아닐 수 있고, 성공 rollout도 스테이지에 따라 이탈이 커지면 원인 쪽에 무게가 실린다.
        plot_A(run_dir, caches, rows, tau, plt, suffix="_success",
               mask_fn=lambda rec: rec["success"].astype(bool), title_extra="  (success only)")
        plot_A(run_dir, caches, rows, tau, plt, suffix="_failure",
               mask_fn=lambda rec: ~rec["success"].astype(bool), title_extra="  (failure only)")
    plot_B(run_dir, caches, rows, tau, plt, figb)
    plot_C(run_dir, rows, plt)
    plot_C(run_dir, rows, plt, right_key="t_star", suffix="_tstar")
    plot_C(run_dir, rows, plt, right_key="dwell", suffix="_dwell")


if __name__ == "__main__":
    if "--plot_only" in sys.argv:
        kv = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
        init_logging()
        plot_r1(kv.get("run_dir", "outputs/R1"),
                split_by_success="--split_by_success" in sys.argv,
                figb=kv.get("figb", "ewc@1,ewc@2,ewc@3"))
    else:
        mp.set_start_method("spawn", force=True)
        init_logging()
        main()
