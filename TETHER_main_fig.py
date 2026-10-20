#!/usr/bin/env python
"""TETHER 메인 그림 — 2x4 패널. 모든 패널에서 **task 0 을 평가**한다.

주장  "학습이 진행될수록 task 0 의 행동이 최신 task 쪽으로 끌려가는가, 아니면
      task 0 에 맞게 유지되는가." 조건-행동 대응이 시간에 걸쳐 보존되는지를 본다.
      (조건 변화에 따른 궤적 분리는 interpolation 그림이 따로 담당한다.)

배치
                After Tasks 0-6   After 0-7   After 0-8   After 0-9
      Seq-FT      task 0 수행       task 0      task 0      task 0
      TETHER      task 0 수행       task 0      task 0      task 0

  LOTUS 셋팅이라 base(t0~t5)를 joint 로 배운 체크포인트에서 출발한다. 열 k 는
  lifelong stage k 직후이고, 그 열의 "최신 task"는 t(5+k) 다. 두 행은 **같은 base
  체크포인트**에서 갈라져 나왔으므로 차이는 전적으로 lifelong 학습 방식 차이다.

좌표계  x축 = end-effector x, y축 = end-effector y.  시뮬레이터가 보고하는
      observation.state[0:3] 를 그대로 쓴다(robosuite world 좌표, 단위 m).
      expert 시연과 policy rollout 이 **같은 키**에서 나오므로 좌표계가 섞이지
      않는다. 8 패널 전부 동일한 축 범위 + equal aspect 다.

선
      Task 0 reference     파랑 #0072B2  굵은 점선     보존해야 할 expert 궤적
      Latest-task ref      주황 #D55E00  굵은 일점쇄선  그 열에서 가장 최근 배운 task
      Policy rollout       검정 #000000  실선          그 패널 모델이 task 0 을 수행
      시작 ● / 종료 ▲
      반복 rollout 은 연한 실선, 대표 하나만 굵게. 대표는 **medoid** 로 정한다
      (길이 L 로 리샘플 후 서로간 L2 거리 합이 최소인 궤적). 성공 사례를 임의로
      고르지 않는다.

      rollout 색은 두 행이 같다 — 모델은 이미 행으로 구분되므로 색까지 나누면
      비교가 어려워진다. 주황은 특정 task 번호가 아니라 "그 열의 최신 task"다.

평가  초기 상태와 시드를 모델·체크포인트 사이에서 대응시킨다. eval seed 를 고정하고
      매번 새 env 를 만들므로 8 패널의 10개 초기 상태가 서로 같다. 패널 구석의
      SR 은 **그 10회 rollout** 에서 센 값이라 report.txt 의 20회 SR 과 다를 수 있다.

    bash TETHER_main_fig.sh                 # 롤아웃 + 그림
    bash TETHER_main_fig.sh --plot_only     # 캐시로 그림만 다시
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lerobot_lsy" / "src"))

from Lotus_l2cb_cl import SUITE_DIR, build_eval_cfg, suite_spec          # noqa: E402
from l2_codebook_id import suite_prefixes, task_instruction              # noqa: E402
from lerobot.envs.factory import make_env                                # noqa: E402
from lerobot.envs.utils import preprocess_observation                    # noqa: E402
from lerobot.policies.factory import make_policy                         # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset              # noqa: E402
from lerobot.utils.utils import init_logging                             # noqa: E402

# rollout 은 **순수 검정**. 처음 쓰던 #30343B 는 파란기가 섞인 진회색이라 연한
# 선이 task 0 의 파랑 다발과 구분되지 않았고, 빨강(#C1121F)으로 바꿨더니 이번엔
# 최신 task 의 주황(#D55E00)과 붙어 보였다. 검정은 두 색 어느 쪽과도 안 섞인다.
C_T0, C_LATEST, C_ROLL = "#0072B2", "#D55E00", "#000000"
A_REF, A_ROLL = 0.16, 0.38     # 연한 선의 투명도 — 롤아웃을 더 진하게 둔다
RESAMPLE = 120          # medoid 계산용 리샘플 길이


# ═════════════════════════════════════════════════════════════════════════════
#  롤아웃 — lerobot.scripts.eval.rollout 을 복제하되 **상태만** 남긴다
# ═════════════════════════════════════════════════════════════════════════════
def rollout_states(env, policy, seeds: list[int]):
    """(states, success). states: 길이 B 리스트, 각 (T_b, 8). success: (B,) bool.

    원본 rollout 은 return_observations=True 면 이미지까지 쌓아 10 env x 500 스텝에서
    수 GB 를 먹는다. 여기서는 observation.state 만 모은다. 나머지 절차 — 시드 부여,
    _init_state_id 전진, done 마스킹, is_success 수집 — 는 원본과 같게 맞췄다.
    그래야 SR 이 report.txt 와 같은 초기 상태 위에서 나온다.
    """
    policy.reset()
    obs, _ = env.reset(seed=seeds)
    for i in range(env.num_envs):                     # 원본과 동일한 초기상태 전진
        e = env.envs[i].env.env
        e._init_state_id = (e._init_state_id + env.num_envs) % 50

    device = next(policy.parameters()).device
    n = env.num_envs
    done = np.zeros(n, dtype=bool)
    succ = np.zeros(n, dtype=bool)
    states: list[list[np.ndarray]] = [[] for _ in range(n)]
    alive = ~done
    max_steps = env.call("_max_episode_steps")[0]

    for _ in range(max_steps):
        po = preprocess_observation(obs)
        st = po["observation.state"].numpy()          # (B,8) 정규화 전 원값
        for b in range(n):
            if alive[b]:
                states[b].append(st[b].copy())
        po = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in po.items()}
        with torch.inference_mode():
            act = policy.select_action(po).to("cpu").numpy()
        obs, _, term, trunc, info = env.step(act)

        if "final_info" in info:
            s = [i["is_success"] if i is not None else False for i in info["final_info"]]
        elif "is_success" in info:
            valid = info.get("_is_success", np.ones(n, dtype=bool))
            s = [bool(a) and bool(m) for a, m in zip(info["is_success"], valid, strict=False)]
        else:
            s = [False] * n
        succ |= np.asarray(s, dtype=bool)
        done = term | trunc | done
        alive = ~done
        if done.all():
            break
    return [np.asarray(s, dtype=np.float64) for s in states], succ


# ═════════════════════════════════════════════════════════════════════════════
def resample(tr: np.ndarray, L: int = RESAMPLE) -> np.ndarray:
    """궤적을 길이 L 로 균등 리샘플 (인덱스 기준). medoid 거리 계산용."""
    if len(tr) == 0:
        return np.zeros((L, tr.shape[1] if tr.ndim == 2 else 3))
    idx = np.linspace(0, len(tr) - 1, L)
    return np.stack([np.interp(idx, np.arange(len(tr)), tr[:, d])
                     for d in range(tr.shape[1])], axis=1)


def medoid(trs: list[np.ndarray]) -> int:
    """서로간 L2 거리 합이 최소인 궤적의 인덱스. 규칙이 고정돼 있어 자의성이 없다."""
    if len(trs) <= 1:
        return 0
    R = np.stack([resample(t) for t in trs])                       # (N,L,D)
    D = np.linalg.norm(R[:, None] - R[None, :], axis=-1).sum(-1)   # (N,N)
    return int(D.sum(1).argmin())


def grasp_idx(tr: np.ndarray) -> int:
    """파지 순간 = 궤적의 z 최소점. 로봇이 물체까지 내려갔다가 집어 올리는 지점이다.

    ★ 왜 gripper 명령이 아니라 z 최소점인가: 시연에는 action[6] 이 있지만 롤아웃
      캐시에는 EE 위치만 있다. 두 곡선에 **같은 정의**를 써야 비교가 성립하므로
      양쪽 모두 z 최소점을 쓴다. 시연에서 두 정의를 맞춰 본 결과 차이가
      평균 0.004~0.014 m 로, task 간 파지점 거리(0.045 m)보다 한참 작다.
    """
    return int(np.argmin(tr[:, 2]))


def expert_traj(root: Path, task_pos: int, n_demos: int, policy_cfg) -> list[np.ndarray]:
    """스플릿 하나에서 에피소드 블록 task_pos 의 시연 EE 궤적들.

    ★ 에피소드 블록으로 가른다 — task_index 가 아니다. LIBERO-50 stage3 처럼
      명령어가 같은 두 태스크가 한 task_index 에 겹치는 경우가 있다.
    """
    ds = LeRobotDataset(root.name, root=str(root), episodes=None)
    hf = ds.hf_dataset
    S = np.asarray(hf.data.column("observation.state").to_pylist(), dtype=np.float64)
    ep = np.asarray(hf.data.column("episode_index").to_pylist(), dtype=np.int64)
    want = ([e for e in np.unique(ep) if e // n_demos == task_pos]
            if task_pos is not None else list(np.unique(ep))[:n_demos])
    return [S[ep == e][:, :3] for e in want]


# ═════════════════════════════════════════════════════════════════════════════
def seq_spec(suite: str, n_tasks: int) -> dict:
    """원본(CLARE) 셋팅의 spec. task 0 부터 하나씩 배우므로 열 k = task k 직후다.

    LOTUS 와 달리 base joint 단계가 없어서 1열이 "task 0 만 배운 직후"가 된다 —
    2x4 설계의 원래 의도 그대로다. 1열은 앵커가 걸릴 과거가 없으므로 두 행이
    같아야 정상이고, 2열부터 갈라진다.
    """
    ds_prefix, env_prefix = suite_prefixes(suite)
    root = Path(os.environ.get("HF_LEROBOT_HOME",
                               str(Path.home() / "Datasets" / "lerobot")))
    return {"env_task": {t: f"{env_prefix}{t}" for t in range(n_tasks)},
            "instr": {t: task_instruction(f"{ds_prefix}{t}") for t in range(n_tasks)},
            "ds_root": {t: root / f"{ds_prefix}{t}" for t in range(n_tasks)},
            "n_task": n_tasks, "n_base": 1}


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", choices=["lotus", "seq"], default="lotus",
                   help="lotus = base joint + stage. seq = 원본 CLARE 셋팅(task 0 부터 순차)")
    p.add_argument("--suite", default="object",
                   help="lotus: goal/object/50.  seq: libero_spatial 등 스위트 이름")
    p.add_argument("--n_ref_demos", type=int, default=20, help="기준선으로 그릴 시연 수")
    p.add_argument("--no_suptitle", action="store_true",
                   help="그림 전체 제목을 빼고 패널 소제목만 남긴다. 논문 본문에 넣을 때는 "
                        "캡션이 그 역할을 하므로 중복이다.")
    p.add_argument("--no_grasp_box", action="store_true",
                   help="패널 구석의 파지 높이(z) 상자를 빼고 별만 남긴다. 파지 z 가 "
                        "task 를 가르지 못하는 스위트(libero_spatial — 그릇이 전부 같은 "
                        "테이블 높이)에서는 정보가 없고 오해만 준다. object 처럼 물체 "
                        "높이가 유일한 판별축인 경우에는 켜 두어야 한다.")
    p.add_argument("--seq_run", default="results/lotus_seq_object_seed42")
    p.add_argument("--tether_run", default="results/l2_codebook_v4_hard_sf_object_k96")
    p.add_argument("--stages", default="1,2,3,4",
                   help="열로 쓸 단계. lotus 면 stage 번호, seq 면 task 번호")
    p.add_argument("--row_names", default="Seq-FT,TETHER")
    p.add_argument("--eval_task", type=int, default=0, help="모든 패널에서 평가할 task")
    p.add_argument("--n_rollout", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=7)
    p.add_argument("--axes", default="xy", choices=["xy", "xz", "yz"])
    p.add_argument("--out", default="results/TETHER_main_fig")
    p.add_argument("--plot_only", action="store_true", help="캐시로 그림만 다시 그린다")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def collect(a, spec, out: Path) -> dict:
    """8 패널의 rollout 궤적을 모아 캐시에 저장한다."""
    cache = out / "rollouts.npz"
    if a.plot_only and cache.exists():
        z = np.load(cache, allow_pickle=True)
        logging.info(f"[FIG] 캐시 사용 {cache}")
        return z["data"].item()

    stages = [int(s) for s in a.stages.split(",")]
    rn = [x.strip() for x in a.row_names.split(",")]
    rows = [(rn[0], Path(a.seq_run)), (rn[1], Path(a.tether_run))]
    data: dict = {}
    for name, run in rows:
        for k in stages:
            # lotus: <run>/checkpoints/stage{k}/...   seq: <ckpt_root>/task_{k}/...
            ck = (run / f"task_{k}" / "checkpoints" / "last" / "pretrained_model"
                  if a.setting == "seq" else
                  run / "checkpoints" / f"stage{k}" / "checkpoints" / "last"
                  / "pretrained_model")
            if not ck.exists():
                raise SystemExit(f"체크포인트 없음: {ck}")
            t0 = time.perf_counter()
            cfg = build_eval_cfg(argparse.Namespace(
                suite=a.suite, device=a.device, eval_seed=a.eval_seed, num_workers=0,
                batch_size=8, steps=1, log_every=100, eval_episodes=a.n_rollout,
                eval_batch_size=a.n_rollout), str(ck), spec["env_task"][a.eval_task])
            policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
            policy.eval()
            env = make_env(cfg.env, n_envs=a.n_rollout, use_async_envs=False)
            try:
                seeds = list(range(a.eval_seed, a.eval_seed + a.n_rollout))
                with torch.no_grad():
                    st, sc = rollout_states(env, policy, seeds)
            finally:
                env.close()
            del policy
            torch.cuda.empty_cache()
            data[f"{name}|{k}"] = {"traj": [s[:, :3] for s in st],
                                   "success": sc.tolist(),
                                   "sr": 100.0 * float(sc.mean())}
            logging.info(f"[FIG] {name} stage{k}  SR={100*sc.mean():.0f}%  "
                         f"길이 {[len(s) for s in st]}  ({time.perf_counter()-t0:.0f}s)")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, data=np.array(data, dtype=object))
    return data


def main():
    a = parse()
    init_logging()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stages = [int(s) for s in a.stages.split(",")]
    seq = a.setting == "seq"
    spec = seq_spec(a.suite, max(stages) + 1) if seq else suite_spec(a.suite)
    data = collect(a, spec, out)

    # ── expert 기준선 ────────────────────────────────────────────────────────
    if seq:
        # 스위트 하나가 태스크 하나라 에피소드 블록을 가를 필요가 없다
        ref_t0 = expert_traj(spec["ds_root"][a.eval_task], None, a.n_ref_demos, None)
        ref_latest = {k: (k, expert_traj(spec["ds_root"][k], None, a.n_ref_demos, None))
                      for k in stages}
    else:
        root = Path(spec["root"])
        ref_t0 = expert_traj(root / "base", a.eval_task, spec["base_demos"], None)
        ref_latest = {}
        for k in stages:
            new = spec["stage_tasks"][k - 1]
            ref_latest[k] = (new[0],
                             expert_traj(root / f"stage{k}", 0, spec["stage_demos"], None))

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    AX = {"xy": (0, 1, "x", "y"), "xz": (0, 2, "x", "z"), "yz": (1, 2, "y", "z")}[a.axes]
    i, j, nx, ny = AX

    # 8 패널 공통 축 범위 — 모든 곡선을 모아 한 번에 정한다
    allpts = [t[:, [i, j]] for v in data.values() for t in v["traj"]]
    allpts += [t[:, [i, j]] for t in ref_t0]
    allpts += [t[:, [i, j]] for _, trs in ref_latest.values() for t in trs]
    P = np.concatenate(allpts)
    pad = 0.05 * max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))   # NumPy 2.0: ndarray.ptp 제거됨
    xl = (P[:, 0].min() - pad, P[:, 0].max() + pad)
    yl = (P[:, 1].min() - pad, P[:, 1].max() + pad)

    nrow, ncol = 2, len(stages)
    # equal aspect 라 패널 높이는 데이터 비율이 정한다. 그림 높이를 거기 맞추지
    # 않으면 (xz 처럼 세로 폭이 좁은 투영에서) 행 사이가 휑하게 벌어진다.
    pw = 3.5
    ph = pw * float(np.clip((yl[1] - yl[0]) / (xl[1] - xl[0]), 0.45, 1.6))
    top_in, bot_in = (0.42 if a.no_suptitle else 1.15), 1.30
    fw, fh = pw * ncol + 0.9, ph * nrow + top_in + bot_in
    fig, axes = plt.subplots(nrow, ncol, figsize=(fw, fh), squeeze=False)
    fig.subplots_adjust(left=0.9 / fw, right=1 - 0.12 / fw,
                        top=1 - top_in / fh, bottom=bot_in / fh, wspace=.13, hspace=.17)

    def draw(ax, tr, **kw):
        ax.plot(tr[:, i], tr[:, j], **kw)

    rn = [x.strip() for x in a.row_names.split(",")]
    for r, name in enumerate(rn):
        for c, k in enumerate(stages):
            ax = axes[r][c]
            d = data[f"{name}|{k}"]
            # 기준선 — 같은 열의 두 모델에 동일하게 들어간다
            for t in ref_t0:
                draw(ax, t, color=C_T0, lw=.7, alpha=A_REF, zorder=2)
            tt = ref_t0[medoid(ref_t0)]
            draw(ax, tt, color=C_T0, lw=2.4, ls=(0, (5, 2)), zorder=5)
            lat_id, lat = ref_latest[k]
            for t in lat:
                draw(ax, t, color=C_LATEST, lw=.7, alpha=A_REF, zorder=2)
            lt = lat[medoid(lat)]
            draw(ax, lt, color=C_LATEST, lw=2.4, ls=(0, (7, 2, 1.4, 2)), zorder=5)
            # policy rollout
            for t in d["traj"]:
                draw(ax, t, color=C_ROLL, lw=.8, alpha=A_ROLL, zorder=6)
            m = medoid(d["traj"]); rt = d["traj"][m]
            draw(ax, rt, color=C_ROLL, lw=2.2, zorder=8)
            ax.plot(rt[0, i], rt[0, j], "o", color=C_ROLL, ms=7, zorder=9)
            ax.plot(rt[-1, i], rt[-1, j], "^", color=C_ROLL, ms=8, zorder=9)
            # 파지점 — task 정체성이 있는 곳. 세 곡선에 같은 정의(z 최소점)를 쓴다.
            gp = {}
            for tr, col, z in ((tt, C_T0, 10), (lt, C_LATEST, 10), (rt, C_ROLL, 11)):
                q = tr[grasp_idx(tr)]
                gp[col] = q
                ax.plot(q[i], q[j], "*", color=col, ms=15, mec="white", mew=.8, zorder=z)
            # 롤아웃 파지점이 어느 쪽 높이로 갔는가 — 이 그림의 핵심 수치다
            d0 = abs(gp[C_ROLL][2] - gp[C_T0][2])
            dl = abs(gp[C_ROLL][2] - gp[C_LATEST][2])
            # 오른쪽 아래 — 왼쪽 위는 최신 task 기준선이, 오른쪽 위는 SR 상자가,
            # 왼쪽 아래는 시작 마커가 지난다. spatial/object 양쪽에서 비는 곳이다.
            if not a.no_grasp_box:
                ax.text(.97, .04,
                        f"grasp z   rollout {gp[C_ROLL][2]:.3f}\n"
                        f"  task0 {gp[C_T0][2]:.3f} (Δ{d0:.3f})\n"
                        f"  latest {gp[C_LATEST][2]:.3f} (Δ{dl:.3f})",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.6,
                        family="monospace", linespacing=1.25,
                        bbox=dict(fc="white", ec="0.8", alpha=.9, pad=2.2))
            # 오른쪽 위 — 왼쪽 위는 grasp 상자, 아래쪽은 파지점 별이 지나간다
            ax.text(.97, .96, f"Task 0 SR {d['sr']:.0f}%  ({sum(d['success'])}/{len(d['success'])})",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.6,
                    bbox=dict(fc="white", ec="0.75", alpha=.92, pad=2.4))
            ax.set_xlim(*xl); ax.set_ylim(*yl)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.22); ax.tick_params(labelsize=8)
            if r == 0:
                lab = (("After Task 0" if k == 0 else f"After Tasks 0–{k}") if seq
                       else f"After Tasks 0–{spec['n_base'] - 1 + k}")
                ax.set_title(f"{lab}   (latest: t{lat_id})", fontsize=10.5)
            if r == nrow - 1:
                ax.set_xlabel(f"End-effector {nx} position (m)", fontsize=9)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(f"{name}\nEnd-effector {ny} position (m)", fontsize=9.5)
            else:
                ax.set_yticklabels([])

    # 범례는 세 다발을 각각 "굵은 = medoid / 연한 = 개별" 로 묶어 설명한다.
    # 연한 파랑·주황이 개별 expert 시연이라는 걸 적지 않으면 그림만 보고 알 수 없다.
    # 핸들을 (굵은, 연한) 튜플로 주면 한 항목에 두 선이 겹쳐 그려진다.
    from matplotlib.legend_handler import HandlerTuple
    n_d0, n_dl = len(ref_t0), len(ref_latest[stages[0]][1])
    pair = lambda col, ls: (plt.Line2D([], [], color=col, lw=2.4, ls=ls),   # noqa: E731
                            plt.Line2D([], [], color=col, lw=.8, alpha=.5))
    h = [pair(C_T0, (0, (5, 2))),
         pair(C_LATEST, (0, (7, 2, 1.4, 2))),
         pair(C_ROLL, "-"),
         plt.Line2D([], [], ls="none", marker="o", ms=6, color=C_ROLL),
         plt.Line2D([], [], ls="none", marker="^", ms=7, color=C_ROLL),
         plt.Line2D([], [], ls="none", marker="*", ms=13, color="0.35")]
    lb = [f"Task 0 expert — medoid (thick) + {n_d0} demos (thin)",
          f"Latest-task expert — medoid (thick) + {n_dl} demos (thin)",
          f"Policy rollout on Task 0 — medoid (thick) + {a.n_rollout} rollouts (thin)",
          "start", "end", "grasp point (lowest z)"]
    fig.legend(h, lb, loc="lower center", ncol=3, fontsize=9.5, frameon=False,
               bbox_to_anchor=(.5, .14 / fh), columnspacing=2.2, handlelength=3.0,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    if not a.no_suptitle:
        fig.suptitle(
            "Evaluation: Task 0 at every checkpoint\n"
            f"{a.suite if seq else 'LOTUS ' + SUITE_DIR[a.suite]}, "
            f"seed 42 / eval seed {a.eval_seed}, "
            f"{a.n_rollout} rollouts per panel   ·   "
            f"task 0: “{spec['instr'][a.eval_task]}”",
            fontsize=12, y=1 - .24 / fh)

    for ext in ("png", "pdf"):
        fp = out / f"TETHER_main_fig.{ext}"
        fig.savefig(fp, dpi=150 if ext == "png" else None)
        print(f"saved -> {fp}")
    plt.close(fig)

    L = ["패널별 Task 0 SR (이 그림의 10 rollout 기준)", ""]
    for name in rn:
        L.append(f"  {name:<8}" + "".join(
            f"  {'task' if seq else 'stage'}{k}: {data[f'{name}|{k}']['sr']:5.0f}%"
            for k in stages))
    L += ["", "파지 높이 — 롤아웃이 task 0 과 최신 task 중 어느 쪽으로 갔는가",
          "  (z 최소점 기준. Δ0 = |rollout − task0|, ΔL = |rollout − latest|.",
          "   Δ0 < ΔL 이면 task 0 쪽을 유지한 것이다.)", "",
          f"{'':<9}" + "".join(f"{('task' if seq else 'stage')+str(k):>22}"
                                for k in stages)]
    for name in rn:
        row = f"{name:<9}"
        for k in stages:
            rt = data[f"{name}|{k}"]["traj"][medoid(data[f"{name}|{k}"]["traj"])]
            zr = rt[grasp_idx(rt), 2]
            z0 = ref_t0[medoid(ref_t0)][grasp_idx(ref_t0[medoid(ref_t0)]), 2]
            lat = ref_latest[k][1]
            zl = lat[medoid(lat)][grasp_idx(lat[medoid(lat)]), 2]
            row += f"  Δ0 {abs(zr-z0):.3f} ΔL {abs(zr-zl):.3f}"
        L.append(row)
    (out / "panel_sr.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
