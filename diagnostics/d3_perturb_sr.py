"""D3 — 방향 오차 대 크기 오차의 SR 민감도 (rollout).

가설: 앵커가 방향을 덜 보호한다면, 같은 크기의 벡터 변화라도 **방향** 교란이
**크기** 교란보다 SR 을 더 떨어뜨려야 한다 (P6).

교란은 정책이 낸 행동 청크를 env 에 넣기 직전에 적용한다. 후킹 지점은
DiTFlowMTPolicy.predict_action_chunk — (B, n_action_steps, 7) 을 **실제 로봇 단위로
unnormalize 한 뒤** 돌려주는 자리다(modeling_dit_flow_mt.py:875-903). 여기를 감싸면
eval_policy 의 평가 프로토콜(init state, 에피소드 수, 시드)을 그대로 둘 수 있다.

  scale  : pose dims 0..5 에 (1+δ). δ ∈ {±0.1, ±0.2, ±0.3}
  rotate : translation dims 0..2 만 청크마다 고정한 무작위 단위축 둘레로
           φ = 2·arcsin(δ/2) 회전. 노름을 보존하므로 scale 의 |δ| 와 **같은 크기**의
           벡터 변화다 (‖Rv − v‖ = 2 sin(φ/2)‖v‖ = δ‖v‖).
           orientation dims 3..5 는 회전시키지 않는다 — 축각 표현이라 회전의 의미가
           모호하고, 물리적으로 해석 가능한 교란이 아니다. (REPORT 3절에 명시)
  gripper: dim 6 은 어떤 조건에서도 건드리지 않는다.
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import pandas as pd
import torch

from common import (OUT, RUN, SUITE, SEED_RUN, PolicyCache, TRANS, POSE, stage_ckpt)

TASKS = [3, 5, 7]
DELTAS_SCALE = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]
DELTAS_ROT = [0.1, 0.2, 0.3]
ENV_PREFIX = "Libero_Spatial_Task_"


def rodrigues(v: torch.Tensor, axis: torch.Tensor, phi: float) -> torch.Tensor:
    """v: (..., 3) 을 단위축 axis (…,3) 둘레로 phi 라디안 회전. Rodrigues 공식."""
    c, s = torch.cos(torch.tensor(phi)), torch.sin(torch.tensor(phi))
    c, s = c.to(v), s.to(v)
    cross = torch.cross(axis.expand_as(v), v, dim=-1)
    dot = (axis.expand_as(v) * v).sum(-1, keepdim=True)
    return v * c + cross * s + axis.expand_as(v) * dot * (1 - c)


def make_hook(policy, kind: str, delta: float, gen: torch.Generator):
    """predict_action_chunk 를 감싼다. 원본은 그대로 두고 인스턴스 속성만 덮어쓴다."""
    base = policy.predict_action_chunk

    if kind == "baseline" or delta == 0.0:
        return base

    def hooked(batch):
        a = base(batch)                       # (B, n_action_steps, 7) 실제 단위
        a = a.clone()
        if kind == "scale":
            a[..., POSE] = a[..., POSE] * (1.0 + delta)
        elif kind == "rotate":
            # 청크마다 고정한 무작위 단위축 (배치 원소마다 하나)
            ax = torch.randn(a.shape[0], 1, 3, device=a.device, generator=gen)
            ax = ax / ax.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            phi = 2.0 * float(torch.arcsin(torch.tensor(delta / 2.0)))
            a[..., TRANS] = rodrigues(a[..., TRANS], ax, phi)
        else:
            raise ValueError(kind)
        return a

    return hooked


def run_cell(policy, cfg, env_task, kind, delta, args, device):
    from lerobot.envs.factory import make_env
    from lerobot.scripts.eval import eval_policy

    gen = torch.Generator(device=device).manual_seed(args.seed)
    orig = policy.predict_action_chunk
    policy.predict_action_chunk = make_hook(policy, kind, delta, gen)
    env = None
    try:
        env_cfg = copy.deepcopy(cfg.env)
        env_cfg.task = env_task
        env = make_env(env_cfg, n_envs=args.eval_batch_size, use_async_envs=False)
        policy.eval()
        with torch.no_grad():
            # guidance_w = 1.0 이라 cfg_guidance 는 no-op (B1.py:562-570) — 생략한다
            info = eval_policy(env, policy, args.eval_episodes, start_seed=args.eval_start_seed)
        return float(info["aggregated"]["pc_success"])
    except Exception as e:
        print(f"[D3] 실패 {env_task} {kind} δ={delta}: {type(e).__name__}: {e}", flush=True)
        return None
    finally:
        if env is not None:
            env.close()
        policy.predict_action_chunk = orig


def build_cfg(policy_path, env_task, args):
    from lerobot.configs.default import DatasetConfig, EvalConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.envs.configs import LiberoEnv

    pc = PreTrainedConfig.from_pretrained(policy_path)
    pc.pretrained_path = policy_path
    pc.device = args.device
    pc.push_to_hub = False
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=f"continuallearning/{SUITE}_image_task_0"),
        env=LiberoEnv(benchmark=SUITE, task=env_task),
        policy=pc, output_dir=None, job_name="d3", seed=args.eval_start_seed,
        num_workers=0, batch_size=8, steps=1, eval_freq=0,
        eval=EvalConfig(n_episodes=args.eval_episodes,
                        batch_size=args.eval_batch_size, max_episodes_rendered=0),
    )
    cfg.optimizer = pc.get_optimizer_preset()
    cfg.scheduler = pc.get_scheduler_preset()
    cfg.checkpoint_path = None
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=20)
    ap.add_argument("--eval_start_seed", type=int, default=SEED_RUN,
                    help="metrics.json 을 만든 프로토콜과 같게 (B1.rollout_sr: start_seed=args.seed)")
    ap.add_argument("--seed", type=int, default=0, help="교란 축 추첨 시드")
    ap.add_argument("--tasks", default=",".join(str(t) for t in TASKS))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    pc = PolicyCache(device)
    tasks = [int(x) for x in args.tasks.split(",")]

    conds = [("baseline", 0.0)] + [("scale", d) for d in DELTAS_SCALE] \
        + [("rotate", d) for d in DELTAS_ROT]

    rows, t0 = [], time.perf_counter()
    for j in tasks:
        policy = pc.get(j)
        cfg = build_cfg(str(stage_ckpt(j)), f"{ENV_PREFIX}{j}", args)
        for kind, d in conds:
            sr = run_cell(policy, cfg, f"{ENV_PREFIX}{j}", kind, d, args, device)
            rows.append({"task": j, "kind": kind, "delta": d, "sr": sr,
                         "n_episodes": args.eval_episodes})
            print(f"[D3] task{j} {kind:9s} δ={d:+.1f}  SR={sr}  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "d3_sr.csv", index=False)

    make_plot()
    verdict()


def make_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(OUT / "d3_sr.csv")
    ts = sorted(df.task.unique())
    fig, axes = plt.subplots(1, len(ts), figsize=(4.2 * len(ts), 3.6), squeeze=False)
    for ax, j in zip(axes[0], ts):
        s = df[df.task == j]
        base = s[s.kind == "baseline"].sr
        b = float(base.iloc[0]) if len(base) else float("nan")
        for kind, sel, lab in (("scale", lambda x: x > 0, "scale (+)"),
                               ("scale", lambda x: x < 0, "scale (-)"),
                               ("rotate", lambda x: x > 0, "rotate")):
            g = s[(s.kind == kind) & (s.delta.map(sel))].sort_values("delta", key=abs)
            if g.empty:
                continue
            ax.plot([0] + list(g.delta.abs()), [b] + list(g.sr), marker="o", ms=4, label=lab)
        ax.set_title(f"task {j}"); ax.set_xlabel("|delta|"); ax.set_ylabel("SR (%)")
        ax.grid(alpha=.3); ax.set_ylim(-2, 102)
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "d3_sr_vs_delta.png", dpi=140); plt.close(fig)
    print("[D3] 그림 저장", flush=True)


def verdict():
    """P6: 같은 δ 에서 rotate 의 SR 하락이 scale 보다 크다."""
    df = pd.read_csv(OUT / "d3_sr.csv")
    rows = []
    for j in sorted(df.task.unique()):
        s = df[df.task == j]
        base = s[s.kind == "baseline"].sr
        if base.empty or pd.isna(base.iloc[0]):
            continue
        b = float(base.iloc[0])
        for d in DELTAS_ROT:
            rot = s[(s.kind == "rotate") & (s.delta == d)].sr
            sp = s[(s.kind == "scale") & (s.delta == d)].sr
            sm = s[(s.kind == "scale") & (s.delta == -d)].sr
            if rot.empty or pd.isna(rot.iloc[0]):
                continue
            drop_rot = b - float(rot.iloc[0])
            cand = [float(x.iloc[0]) for x in (sp, sm) if not x.empty and not pd.isna(x.iloc[0])]
            if not cand:
                continue
            drop_scale = b - max(cand)          # scale 쪽에서 더 나쁜(=큰) 하락과 비교
            rows.append({"task": j, "delta": d, "baseline": b,
                         "drop_rotate": drop_rot, "drop_scale_worst": drop_scale,
                         "rotate_worse": bool(drop_rot > drop_scale)})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "d3_p6.csv", index=False)
    v = {"P6": {"n_cmp": len(R),
                "frac_rotate_worse": float(R.rotate_worse.mean()) if len(R) else None,
                "mean_drop_rotate": float(R.drop_rotate.mean()) if len(R) else None,
                "mean_drop_scale_worst": float(R.drop_scale_worst.mean()) if len(R) else None}}
    (OUT / "d3_verdicts.json").write_text(json.dumps(v, indent=2, ensure_ascii=False))
    print(json.dumps(v, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
