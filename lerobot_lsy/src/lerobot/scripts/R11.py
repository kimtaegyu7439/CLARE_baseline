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

"""R11 — prefix handover: joint이 앞 K스텝을 몰고 seq CL이 이어받으면 SR은 어떻게 되는가.

R9/R9_A는 forward 프로브였다. 조건 라우팅이 joint와 CL에서 거의 같게 보였는데,
그것만으로는 "그래서 행동이 왜 무너지나"에 답할 수 없다. R11은 시뮬레이터에서
직접 개입해 **인과**를 본다.

측정
    한 에피소드를 이렇게 굴린다.

        step 0 … K-1     joint 정책이 행동을 낸다
        step K … 끝      seq CL 정책이 이어받는다

    K를 0부터 키우며 성공률 SR을 잰다. 태스크마다 그림 한 장.

        K = 0            CL이 처음부터 끝까지 = 순수 CL
        K = max          joint이 처음부터 끝까지 = 순수 joint

    ★ 이 두 끝점은 따로 측정한 joint/CL SR과 일치해야 한다. 일치하지 않으면 인계
      기계장치 자체가 틀린 것이므로 sanity 검사로 쓴다(summary.json에 기록).

읽는 법 — 곡선 모양이 곧 진단이다
    K를 조금만 줘도 SR이 확 오른다   CL의 고장은 **초반 접근 구간**에 몰려 있다.
                                      제자리를 잡아 주면 나머지는 CL이 해낸다.
    K를 끝까지 줘야 오른다            CL은 궤적 전 구간에서 고장나 있다.
                                      joint가 사실상 전부 대신한 것이라 정보가 없다.
    아무리 줘도 안 오른다             CL이 **좋은 상태에서도** 태스크를 못 끝낸다.
                                      조건이 아니라 정책 자체가 지워졌다는 뜻.
    중간에 꺾인다                     그 K 부근에 결정적 분기점이 있다.

★ 주장 범위: 이 실험은 "CL의 실패가 궤적의 어느 구간에서 결정되는가"만 말한다.
  왜 그 구간이 망가졌는지(조건 라우팅인지, 시각 표현인지, readout인지)는 말하지 않는다.

인계할 때 주의한 것 두 가지
  1) 액션 큐. 이 정책은 8스텝치를 한 번에 생성해 큐에서 꺼내 쓴다(n_action_steps=8).
     step K에서 joint의 남은 큐를 그대로 두면 인계가 다음 청크 경계까지 미뤄진다.
     그래서 인계 시점에 CL의 큐를 비운 상태에서 새로 생성하게 한다.
  2) 관측 큐. 이 정책은 최근 2스텝 관측을 본다(n_obs_steps=2). CL을 step K에
     그냥 투입하면 큐가 비어 있어 populate_queues가 obs_K를 두 번 복제해 채운다 —
     즉 "직전 스텝을 못 본" 상태로 시작한다. 그래서 joint이 모는 동안에도 CL의
     **관측 큐만** 매 스텝 데워 둔다(액션 큐는 건드리지 않으므로 CL은 행동하지 않는다).
     --warm_follower=false 로 끄면 1)만 적용된 판을 볼 수 있다.

실행
    python R11.py --task=0 \
      --joint_ckpt=... --cl_ckpt=... \
      --policy.path=<둘 중 아무거나: 설정 로딩용> \
      --dataset.repo_id=continuallearning/libero_spatial_image_task_0 \
      --env.type=libero --env.benchmark=libero_spatial \
      --switch_steps=0,10,20,30,45,60,80,110,150,200,500

    그림(4개 태스크 전부 끝난 뒤):
    python R11.py --plot_only=true --out_root=outputs/R11 --run_tag=...
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.factory import make_env
from lerobot.envs.utils import check_env_attributes_and_types, preprocess_observation
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging
from termcolor import colored

# ★ 평가 규약을 다시 구현하지 않는다. 성공 판정(첫 done까지 마스킹)이 G1/eval_policy와
#   한 글자라도 달라지면 SR 숫자가 조용히 어긋난다.
from lerobot.scripts.G1 import episode_success
from lerobot.scripts.R7 import assert_shared_norm, load_policy_at, norm_stats


# ═════════════════════════════════════════════════════════════════════════════
#  설정
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class R11Config(TrainPipelineConfig):
    """train.py 인자 전부 + 이 실험용. 학습 인자(steps/batch 등)는 쓰이지 않는다."""

    joint_ckpt: str = ""        # 앞 구간을 모는 정책 (lead)
    cl_ckpt: str = ""           # 인계받는 정책 (follow)
    task: int = 0               # 평가 태스크
    env_task_prefix: str = "Libero_Spatial_Task_"
    dataset_prefix: str = "continuallearning/libero_spatial_image_task_"

    # 인계 시점 격자. 0 = 순수 CL, 아주 큰 값 = 순수 joint.
    switch_steps: str = "0,10,20,30,45,60,80,110,150,200,500"
    sr_episodes: int = 50       # = n_envs. 한 배치로 돌려 K끼리 초기 상태를 맞춘다
    env_seed: int = 100000      # env.reset 시드
    # 인계받는 정책의 관측 큐를 joint 구간에도 채워 둘 것인가 (위 docstring 2번).
    warm_follower: bool = True

    # 그림 아래 표에 쓸 "단독 SR" 로그 위치 (SR4 eval 산출물). 없으면 곡선 끝점으로 채운다.
    sr_dir: str = "outputs/R9_A/_sr"

    out_root: str = "outputs/R11"
    run_tag: str = ""
    plot_only: bool = False
    redo: bool = False           # 이미 잰 K도 다시 잰다

    def validate(self):
        out = self.output_dir
        if isinstance(out, Path) and out.is_dir():
            self.output_dir = None
            super().validate()
            self.output_dir = out
        else:
            super().validate()


def run_dir(cfg: R11Config) -> Path:
    tag = cfg.run_tag or "run"
    return Path(cfg.out_root) / tag


def result_path(cfg: R11Config) -> Path:
    return run_dir(cfg) / f"R11_task{cfg.task}.json"


def parse_steps(spec: str) -> list[int]:
    out = sorted({int(x) for x in spec.split(",") if x.strip() != ""})
    if any(k < 0 for k in out):
        raise SystemExit(f"[R11] switch_steps 에 음수가 있다: {out}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  인계 롤아웃
# ═════════════════════════════════════════════════════════════════════════════
def _push_obs_queue(policy, observation: dict) -> None:
    """정책의 **관측 큐만** 갱신한다. 액션 큐는 건드리지 않는다.

    select_action의 앞 세 줄(normalize -> 카메라 stack -> populate_queues)과 같은 일을
    한다. 여기를 손으로 다시 쓰는 대신 select_action을 부르면 액션 큐가 소비되어
    청크 경계가 어긋나므로 이 경로가 필요하다.

    ★ modeling_dit_flow_mt.select_action이 바뀌면 여기도 같이 바뀌어야 한다.
      그걸 놓치지 않도록 아래 _assert_warm_matches가 첫 스텝에서 실측으로 확인한다.
    """
    from lerobot.constants import OBS_IMAGES
    from lerobot.policies.utils import populate_queues

    batch = policy.normalize_inputs(observation)
    if policy.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack(
            [batch[key] for key in policy.config.image_features], dim=-4)
    policy._queues = populate_queues(policy._queues, batch)


def _assert_warm_matches(policy, observation: dict) -> None:
    """_push_obs_queue가 select_action과 **같은 관측 큐**를 만드는지 실측 확인.

    깨끗한 큐 두 개에 각각 한 번씩 밀어 넣고 관측 큐를 비교한다. 다르면 select_action의
    전처리가 바뀐 것이므로 조용히 잘못된 조건으로 굴리는 대신 여기서 죽는다.
    """
    policy.reset()
    _push_obs_queue(policy, observation)
    mine = {k: [t.clone() for t in v] for k, v in policy._queues.items() if k != "action"}
    policy.reset()
    with torch.inference_mode():
        policy.select_action(observation)
    theirs = {k: list(v) for k, v in policy._queues.items() if k != "action"}
    if set(mine) != set(theirs):
        raise AssertionError(f"[R11] 관측 큐 키가 다르다: {sorted(mine)} vs {sorted(theirs)}")
    for k in mine:
        if len(mine[k]) != len(theirs[k]):
            raise AssertionError(f"[R11] 관측 큐 길이가 다르다 ({k}): "
                                 f"{len(mine[k])} vs {len(theirs[k])}")
        for a, b in zip(mine[k], theirs[k]):
            if not torch.equal(a, b):
                raise AssertionError(
                    f"[R11] _push_obs_queue가 select_action과 다른 관측 큐를 만든다 ({k}). "
                    f"modeling_dit_flow_mt.select_action의 전처리가 바뀌었는지 확인해라.")
    policy.reset()
    logging.info("[R11]   관측 큐 데우기 경로가 select_action과 일치함 (실측 확인)")


@torch.no_grad()
def rollout_handover(env, lead, follow, switch: int, seeds: list[int],
                     warm_follower: bool = True) -> dict:
    """eval.rollout과 같은 루프. 다만 step >= switch 부터 follow가 행동한다.

    반환 형식은 eval.rollout과 같다(success/done만 쓴다).

    ★ eval.rollout을 복사한 이유: 정책이 하나라는 전제가 그 함수에 박혀 있고
      (policy.reset, select_action, get_device_from_parameters), 래퍼로 감싸면
      eval_policy의 isinstance(PreTrainedPolicy) 검사에 걸린다. 복사하는 대신
      성공 판정만은 G1.episode_success로 공유해 규약이 갈라지지 않게 했다.
    """
    device = get_device_from_parameters(lead)

    lead.reset()
    follow.reset()
    observation, info = env.reset(seed=seeds)

    # eval.rollout과 동일: reset 직후 초기 상태 인덱스를 num_envs만큼 전진시킨다.
    for i in range(env.num_envs):
        init_id = env.envs[i].env.env._init_state_id
        env.envs[i].env.env._init_state_id = (init_id + env.num_envs) % 50

    all_rewards, all_successes, all_dones = [], [], []
    step = 0
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    check_env_attributes_and_types(env)

    while not np.all(done):
        observation = preprocess_observation(observation)
        for key in observation:
            if isinstance(observation[key], torch.Tensor):
                observation[key] = observation[key].to(
                    device, non_blocking=device.type == "cuda")

        with torch.inference_mode():
            if step < switch:
                # joint 구간. CL은 관측만 보고 행동하지 않는다.
                if warm_follower:
                    _push_obs_queue(follow, observation)
                action = lead.select_action(observation)
            else:
                action = follow.select_action(observation)

        action = action.to("cpu").numpy()
        assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

        observation, reward, terminated, truncated, info = env.step(action)

        if "final_info" in info:
            successes = [i["is_success"] if i is not None else False for i in info["final_info"]]
        elif "is_success" in info:
            valid = info.get("_is_success", np.ones(env.num_envs, dtype=bool))
            successes = [bool(s) and bool(m)
                         for s, m in zip(info["is_success"], valid, strict=False)]
        else:
            successes = [False] * env.num_envs

        done = terminated | truncated | done
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))
        step += 1
        if step > max_steps + 5:                      # 방어: 무한 루프
            raise RuntimeError(f"[R11] 롤아웃이 max_steps({max_steps})를 넘겼다")

    return {
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
        "n_steps": step,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  측정
# ═════════════════════════════════════════════════════════════════════════════
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 Wilson 95% 구간(%). n=50이라 정규근사는 끝에서 어긋난다."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100.0 * max(0.0, c - h), 100.0 * min(1.0, c + h))


def measure(cfg: R11Config) -> dict:
    device = get_safe_torch_device(cfg.policy.device, log=True)
    set_seed(cfg.seed)
    steps = parse_steps(cfg.switch_steps)

    for name, ck in (("joint_ckpt", cfg.joint_ckpt), ("cl_ckpt", cfg.cl_ckpt)):
        if not ck or not Path(ck).exists():
            raise SystemExit(f"[R11] --{name} 가 없다: {ck!r}")

    meta = LeRobotDatasetMetadata(cfg.dataset.repo_id)
    logging.info(colored(f"[R11] task {cfg.task} · lead=joint · follow=CL", "cyan",
                         attrs=["bold"]))
    lead = load_policy_at(cfg, cfg.joint_ckpt, meta, device)
    follow = load_policy_at(cfg, cfg.cl_ckpt, meta, device)
    # ★ 두 정책이 같은 액션 정규화 좌표계를 써야 한 궤적을 이어받는 것이 성립한다.
    assert_shared_norm(norm_stats(lead), norm_stats(follow), "cl vs joint")
    for p in (lead, follow):
        p.eval()
        assert not p.training

    import copy

    env_cfg = copy.deepcopy(cfg.env)
    env_cfg.task = f"{cfg.env_task_prefix}{cfg.task}"
    env = make_env(env_cfg, n_envs=cfg.sr_episodes, use_async_envs=False)
    # ★ 초기 상태 인덱스를 **직접 고정**한다. env가 주는 기본값은 실행마다 다른 지점에서
    #   시작해(관측: [30, 31, 1, 19]) 재현이 안 되고, 따로 잰 joint/CL SR과 끝점을
    #   대조할 수도 없다. 0..n-1로 박으면 n=50일 때 LIBERO의 초기 상태 50개를 정확히
    #   한 번씩 덮는다 — eval.py가 배치를 돌며 덮는 집합과 같은 집합이다.
    given = [env.envs[i].env.env._init_state_id for i in range(env.num_envs)]
    init_ids = [i % 50 for i in range(env.num_envs)]
    logging.info(f"[R11] 초기 상태 인덱스 고정: env 기본값 {given} -> {init_ids}")
    seeds = list(range(cfg.env_seed, cfg.env_seed + cfg.sr_episodes))
    max_steps = int(env.call("_max_episode_steps")[0])
    logging.info(f"[R11] n_envs={env.num_envs}  init_state_ids={init_ids}  "
                 f"max_episode_steps={max_steps}")

    # 데우기 경로가 select_action과 같은지 실측 확인 (한 번, 진짜 관측으로).
    if cfg.warm_follower:
        obs0, _ = env.reset(seed=seeds)
        for i in range(env.num_envs):
            env.envs[i].env.env._init_state_id = init_ids[i]
        obs0 = preprocess_observation(obs0)
        obs0 = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in obs0.items()}
        _assert_warm_matches(follow, obs0)

    out = load_results(cfg)
    out.update({
        "task": cfg.task, "joint_ckpt": str(cfg.joint_ckpt), "cl_ckpt": str(cfg.cl_ckpt),
        "n_episodes": int(cfg.sr_episodes), "env_seed": int(cfg.env_seed),
        "init_state_ids": [int(x) for x in init_ids], "max_episode_steps": max_steps,
        "warm_follower": bool(cfg.warm_follower), "switch_steps": steps,
    })
    out.setdefault("points", {})

    try:
        for K in steps:
            key = str(K)
            if key in out["points"] and not cfg.redo:
                logging.info(f"[R11] K={K:<4d} 건너뜀 (이미 있음: "
                             f"{out['points'][key]['sr']:.1f}%)")
                continue
            # ★ K마다 초기 상태를 되돌린다. rollout이 reset 직후 전진시키므로
            #   되돌리지 않으면 K마다 다른 장면을 받아 곡선이 무의미해진다.
            for i in range(env.num_envs):
                env.envs[i].env.env._init_state_id = init_ids[i]
            data = rollout_handover(env, lead, follow, K, seeds, cfg.warm_follower)
            succ = episode_success(data)
            n_ok, n = int(succ.sum()), int(len(succ))
            lo, hi = wilson(n_ok, n)
            label = ("pure CL" if K == 0 else
                     "pure joint" if K >= max_steps else f"joint 0..{K - 1} → CL")
            out["points"][key] = {
                "switch": K, "sr": 100.0 * n_ok / n, "n_success": n_ok, "n": n,
                "ci_lo": lo, "ci_hi": hi, "label": label,
                "success": [bool(x) for x in succ], "rollout_steps": int(data["n_steps"]),
            }
            logging.info(colored(
                f"[R11] K={K:<4d} SR = {100.0 * n_ok / n:5.1f}%  ({n_ok}/{n})  "
                f"[{lo:.0f}, {hi:.0f}]  {label}", "green"))
            save_results(cfg, out)          # 매 K마다 저장 — 중간에 죽어도 이어서 간다
    finally:
        env.close()

    _sanity(out, max_steps)
    save_results(cfg, out)
    return out


def _sanity(out: dict, max_steps: int) -> None:
    """끝점이 순수 정책과 맞는지, 곡선이 단조인지 등을 기록만 한다(강제하지 않는다)."""
    pts = out["points"]
    notes = []
    if "0" in pts:
        notes.append(f"K=0 (순수 CL) SR = {pts['0']['sr']:.1f}%  "
                     f"— 따로 잰 CL SR과 같아야 한다")
    big = [k for k in pts if int(k) >= max_steps]
    if big:
        k = big[0]
        notes.append(f"K={k} (순수 joint) SR = {pts[k]['sr']:.1f}%  "
                     f"— 따로 잰 joint SR과 같아야 한다")
    sr = [pts[k]["sr"] for k in sorted(pts, key=int)]
    if len(sr) > 2 and all(b >= a - 1e-9 for a, b in zip(sr, sr[1:])):
        notes.append("SR이 K에 대해 단조 증가 — 인계를 늦출수록 좋아진다")
    out["sanity"] = notes
    for n in notes:
        logging.info(colored(f"[R11][sanity] {n}", "yellow"))


def load_results(cfg: R11Config) -> dict:
    p = result_path(cfg)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_results(cfg: R11Config, out: dict) -> None:
    p = result_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))


# ═════════════════════════════════════════════════════════════════════════════
#  기준 SR (그림 아래 표)
# ═════════════════════════════════════════════════════════════════════════════
def parse_sr_logs(sr_dir: Path) -> dict:
    """SR4 eval 로그에서 pc_success를 긁는다. {"joint": {0: 100.0, ...}, "cl": {...}}

    ★ 없으면 조용히 비워 둔다. 그림 아래 표에는 '측정 안 됨'으로 나가고, 곡선의
      끝점(K=0, K=max)이 대신 기준선 노릇을 한다.
    """
    import re

    refs: dict[str, dict[int, float]] = {"joint": {}, "cl": {}}
    if not sr_dir.is_dir():
        return refs
    for f in sorted(sr_dir.glob("*_task*.log")):
        m = re.match(r"(joint|cl3?)_task(\d+)\.log$", f.name)
        if not m:
            continue
        who = "joint" if m.group(1) == "joint" else "cl"
        hits = re.findall(r"'pc_success':\s*([0-9.]+)", f.read_text(errors="ignore"))
        if hits:
            refs[who][int(m.group(2))] = float(hits[-1])
    return refs


def demo_lengths(repo: str) -> np.ndarray | None:
    """전문가 데모 에피소드의 길이(스텝 수). 곡선을 읽는 기준선이 된다.

    ★ 이게 없으면 그림을 오독한다. "K=80에서 SR이 뛴다"는 사실은 데모가 평균 95스텝일
      때와 300스텝일 때 정반대의 뜻이다. 전자면 joint이 태스크를 거의 끝낸 것이고,
      후자면 CL이 뒷구간을 스스로 해낸 것이다.
    ★ 비디오를 디코딩하지 않는다 — episode_data_index만 읽는다.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(repo)
        f = np.asarray(ds.episode_data_index["from"])
        t = np.asarray(ds.episode_data_index["to"])
        del ds
        return (t - f).astype(int)
    except Exception as e:                      # 데이터셋이 없어도 그림은 나와야 한다
        logging.warning(f"[R11] 데모 길이를 못 읽었다 ({repo}): {e}")
        return None


def collect(cfg: R11Config) -> tuple[dict, dict]:
    """태스크별 R11 결과와 기준 SR을 모은다."""
    d = run_dir(cfg)
    runs = {}
    for f in sorted(d.glob("R11_task*.json")):
        r = json.loads(f.read_text())
        runs[int(r["task"])] = r
    refs = parse_sr_logs(Path(cfg.sr_dir))
    # 로그가 없으면 곡선 끝점으로 채운다 (같은 프로토콜이라 오히려 더 정합적이다).
    for t, r in runs.items():
        pts, ms = r["points"], r.get("max_episode_steps", 500)
        if "0" in pts:
            refs["cl"].setdefault(t, pts["0"]["sr"])
        big = [k for k in pts if int(k) >= ms]
        if big:
            refs["joint"].setdefault(t, pts[big[0]]["sr"])
    return runs, refs


# ═════════════════════════════════════════════════════════════════════════════
#  그림 — 태스크당 한 장
# ═════════════════════════════════════════════════════════════════════════════
def draw(cfg: R11Config) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from lerobot.scripts.R8 import GRID, INK, INK2, MODEL_COLORS, _style

    runs, refs = collect(cfg)
    if not runs:
        raise SystemExit(f"[R11] {run_dir(cfg)} 에 R11_task*.json 이 없다.")

    C_CURVE, C_JOINT, C_CL = MODEL_COLORS["cl"], MODEL_COLORS["joint"], MODEL_COLORS["pretrain"]
    tasks_all = sorted(set(refs["joint"]) | set(refs["cl"]) | set(runs))
    written = []

    for t in sorted(runs):
        r = runs[t]
        ms = r.get("max_episode_steps", 500)
        pts = r["points"]
        # 곡선에는 '순수 joint'(K>=max)를 점으로 찍지 않는다 — x축이 그쪽으로 늘어나
        # 정작 봐야 할 초반 구간이 뭉개진다. 대신 수평 기준선으로 그린다.
        ks = sorted((int(k) for k in pts if int(k) < ms))
        sr = np.array([pts[str(k)]["sr"] for k in ks], dtype=float)
        lo = np.array([pts[str(k)]["ci_lo"] for k in ks], dtype=float)
        hi = np.array([pts[str(k)]["ci_hi"] for k in ks], dtype=float)
        n = r.get("n_episodes", 50)

        fig = plt.figure(figsize=(7.6, 6.9))
        gs = GridSpec(2, 1, height_ratios=[3.0, 1.5], hspace=0.42,
                      left=0.115, right=0.965, top=0.822, bottom=0.075)
        ax = fig.add_subplot(gs[0])
        _style(ax)

        j_ref = refs["joint"].get(t)
        c_ref = refs["cl"].get(t)
        if j_ref is not None:
            ax.axhline(j_ref, color=C_JOINT, lw=1.4, ls=(0, (5, 3)), zorder=2)
            ax.text(0.995, j_ref, f"joint alone\n{j_ref:.0f}%", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=8.5, color=C_JOINT, zorder=6,
                    linespacing=1.35,
                    bbox=dict(fc="white", ec="none", pad=1.5))
        if c_ref is not None:
            ax.axhline(c_ref, color=C_CL, lw=1.4, ls=(0, (5, 3)), zorder=2)
            ax.text(0.995, c_ref, f"seq CL alone\n{c_ref:.0f}%", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=8.5, color=C_CL, zorder=6,
                    linespacing=1.35,
                    bbox=dict(fc="white", ec="none", pad=1.5))

        dl = demo_lengths(f"{cfg.dataset_prefix}{t}")
        if dl is not None and len(dl):
            q1, med, q3 = (float(np.percentile(dl, 25)), float(np.median(dl)),
                           float(np.percentile(dl, 75)))
            ax.axvspan(q1, q3, color=GRID, alpha=0.55, lw=0, zorder=1)
            ax.axvline(med, color=INK2, lw=0.9, ls=":", zorder=2)
            ax.text(med, 0.915, f" expert demo length\n median {med:.0f}  "
                    f"(IQR {q1:.0f}–{q3:.0f}) ",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7.6, color=INK2, zorder=7, linespacing=1.4)

        ax.fill_between(ks, lo, hi, color=C_CURVE, alpha=0.16, lw=0, zorder=3)
        ax.plot(ks, sr, "-o", color=C_CURVE, lw=1.9, ms=4.6, mfc="white", mew=1.5, zorder=4)
        for k, v in zip(ks, sr):
            ax.annotate(f"{v:.0f}", (k, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7.6, color=INK2, zorder=5)

        ax.set_xlim(-max(ks) * 0.035, max(ks) * 1.30)
        ax.set_ylim(-4, 104)
        ax.set_xlabel("handover step  K       (joint drives steps 0…K−1,  seq CL takes over at K)",
                  fontsize=9.5)
        ax.set_ylabel(f"success rate  (%,  {n} episodes)", fontsize=9.5)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)

        fig.text(0.115, 0.972, f"R11 — prefix handover · task {t}",
                 fontsize=13.5, color=INK, fontweight="bold", ha="left", va="top")
        fig.text(0.115, 0.928,
                 "joint drives the first K steps, then seq CL finishes the episode.  "
                 "K = 0 is CL alone.\n"
                 "Grey band = how long this task's expert demos are.  A rise only at or past "
                 "it\nmeans CL never finishes the task on its own.  "
                 "Shaded curve = Wilson 95% interval.",
                 fontsize=8.4, color=INK2, ha="left", va="top", linespacing=1.5)

        # ── 아래 표: 두 모델 × 태스크 0..3 단독 SR ────────────────────────────
        axt = fig.add_subplot(gs[1])
        axt.axis("off")
        cols = tasks_all
        x0, dx, y0 = 0.40, min(0.155, 0.60 / max(len(cols), 1)), 0.80
        axt.text(0.0, y0 + 0.16, "success rate without handover", fontsize=9.5, color=INK,
                 fontweight="bold", transform=axt.transAxes)
        for ci, tt in enumerate(cols):
            axt.text(x0 + ci * dx, y0, f"task {tt}", fontsize=9, color=INK,
                     ha="center", transform=axt.transAxes)
        for ri, (who, lab, col) in enumerate((("joint", "joint  (tasks 0–3 mixed)", C_JOINT),
                                              ("cl", "seq CL  (task 0 → … → 3)", C_CL))):
            y = y0 - 0.30 - ri * 0.30
            axt.text(0.0, y, lab, fontsize=9, color=col, ha="left", transform=axt.transAxes)
            for ci, tt in enumerate(cols):
                v = refs[who].get(tt)
                txt = "—" if v is None else f"{v:.0f}%"
                axt.text(x0 + ci * dx, y, txt, fontsize=9.5,
                         color=INK if tt != t else col,
                         fontweight="normal" if tt != t else "bold",
                         ha="center", transform=axt.transAxes)
        axt.plot([0.0, 1.0], [y0 - 0.11] * 2, color=GRID, lw=0.9,
                 transform=axt.transAxes, clip_on=False)
        note = ("bold = the task this figure is about.   '—' = not measured yet."
                if any(refs[w].get(tt) is None for w in ("joint", "cl") for tt in cols)
                else "bold = the task this figure is about.")
        axt.text(0.0, y0 - 0.78, note, fontsize=7.8, color=INK2, transform=axt.transAxes)

        for ext in ("png", "pdf"):
            p = run_dir(cfg) / f"R11_task{t}.{ext}"
            fig.savefig(p, dpi=190 if ext == "png" else None,
                        facecolor="white", bbox_inches=None)
            if ext == "png":
                written.append(p)
        plt.close(fig)
        logging.info(colored(f"[R11] 그림 -> {run_dir(cfg)}/R11_task{t}.png", "green"))

    _write_method(cfg, runs, refs)
    return written


def _write_method(cfg: R11Config, runs: dict, refs: dict) -> None:
    d = run_dir(cfg)
    L = ["# R11 — prefix handover  (method)", "",
         "joint이 롤아웃의 앞 K스텝을 몰고, step K부터 seq CL이 이어받는다. "
         "K를 바꿔 가며 성공률을 잰다.", ""]
    for t in sorted(runs):
        r = runs[t]
        L += [f"## task {t}", "",
              f"- 에피소드 {r['n_episodes']}개 · env_seed {r['env_seed']} · "
              f"max_episode_steps {r.get('max_episode_steps')}",
              f"- 초기 상태 인덱스 {r['init_state_ids']}  (K마다 되돌려 같은 장면을 쓴다)",
              f"- 관측 큐 데우기: {'켬' if r.get('warm_follower') else '끔'}",
              f"- lead(joint) `{r['joint_ckpt']}`", f"- follow(CL) `{r['cl_ckpt']}`", "",
              "| K | SR % | 성공/전체 | 95% CI | 의미 |", "|---|---|---|---|---|"]
        for k in sorted(r["points"], key=int):
            p = r["points"][k]
            L.append(f"| {p['switch']} | {p['sr']:.1f} | {p['n_success']}/{p['n']} | "
                     f"[{p['ci_lo']:.0f}, {p['ci_hi']:.0f}] | {p['label']} |")
        L.append("")
        for s in r.get("sanity", []):
            L.append(f"- sanity: {s}")
        L.append("")
    L += ["## 단독 SR (그림 아래 표)", "", "| 모델 | " +
          " | ".join(f"task {t}" for t in sorted(set(refs['joint']) | set(refs['cl']))) + " |",
          "|---|" + "---|" * len(set(refs["joint"]) | set(refs["cl"]))]
    for who, lab in (("joint", "joint (0–3 혼합)"), ("cl", "seq CL (0→…→3)")):
        row = [f"{refs[who][t]:.0f}%" if t in refs[who] else "—"
               for t in sorted(set(refs["joint"]) | set(refs["cl"]))]
        L.append(f"| {lab} | " + " | ".join(row) + " |")
    L += ["", "## 주장 범위", "",
          "이 실험은 CL의 실패가 **궤적의 어느 구간에서 결정되는가**만 말한다. "
          "그 구간이 왜 망가졌는지(조건 라우팅·시각 표현·readout 중 무엇인지)는 말하지 않는다."]
    (d / "R11.method.md").write_text("\n".join(L))
    logging.info(f"[R11] method -> {d}/R11.method.md")


# ═════════════════════════════════════════════════════════════════════════════
@parser.wrap()
def main(cfg: R11Config):
    init_logging()
    # ★ draccus는 validate()를 부르지 않는다. 이걸 빼면 --policy.path 가 반영되지 않아
    #   cfg.policy 가 None으로 남는다 (G1/R1/H5도 main에서 직접 부른다).
    cfg.validate()
    logging.info(pformat({k: v for k, v in vars(cfg).items()
                          if k in ("task", "joint_ckpt", "cl_ckpt", "switch_steps",
                                   "sr_episodes", "warm_follower", "out_root", "run_tag")}))
    run_dir(cfg).mkdir(parents=True, exist_ok=True)
    if cfg.plot_only:
        draw(cfg)
        return
    measure(cfg)
    logging.info(colored(f"[R11] task {cfg.task} 완료 -> {result_path(cfg)}", "green"))


if __name__ == "__main__":
    main()
