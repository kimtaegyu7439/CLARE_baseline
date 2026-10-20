#!/usr/bin/env python
"""TETHER main figure 2 — 2x4 panels. **One checkpoint, four conditions.**

Claim  Figure 1 fixed the condition (task 0) and swept the checkpoint, so it
      showed "task 0 decays over time". That leaves an escape hatch: maybe the
      model just forgot task 0 and is fine elsewhere. This figure removes it.
      We take the **final** checkpoint (after tasks 0-3) and feed it each of the
      four task conditions t0..t3. If the condition still drives behaviour, each
      column should track its own blue reference. If the policy has collapsed
      onto the most recently learned task, every column tracks the orange one.

Layout
                  Condition t0     Condition t1     Condition t2     Condition t3
      Seq-FT      final ckpt       final ckpt       final ckpt       final ckpt
      TETHER      final ckpt       final ckpt       final ckpt       final ckpt

  Both rows use the checkpoint after the *same* number of tasks, so the only
  difference between rows is the lifelong method. The last column is the
  degenerate one (condition == latest task): blue and orange coincide there by
  construction, and it is kept as the control — it is where both methods should
  succeed.

Coordinates  x axis = end-effector x, y axis = end-effector y.  We take
      observation.state[0:3] straight from the simulator (robosuite world
      frame, metres). Expert demos and policy rollouts come from the **same**
      key, so the frames cannot get mixed. All 8 panels share one axis range
      and equal aspect.

Lines
      Condition expert   blue  #0072B2  thick dashed     what task t_j should look like
      Latest-task expert orange #D55E00 thick dash-dot   the last task learned (t3)
      Policy rollout     black #000000  solid            this panel's model under t_j
      start o / end ^

      Repeats are drawn thin, one representative thick. The representative is
      the **medoid** (resample to fixed length, pick the trajectory minimising
      summed L2 to the others) — we do not hand-pick a successful episode.

Numbers  Each panel prints only SR — the success rate over this panel's
      rollouts, which may differ from report.txt (more episodes there).
      panel_sr.txt additionally records, per panel, the mean point-to-point
      distance from the rollout medoid to the condition expert medoid (d_cond)
      and to the latest-task expert medoid (d_latest). Those stay out of the
      figure on purpose: a 0% panel flails, so its medoid is not meaningfully
      near either reference and a small d_cond there means nothing.

    bash TETHER_main_fig2.sh                 # rollouts + figure
    bash TETHER_main_fig2.sh --plot_only     # redraw from cache
"""
from __future__ import annotations

import argparse
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

from Lotus_l2cb_cl import build_eval_cfg                                 # noqa: E402
from l2_codebook_id import suite_prefixes, task_instruction              # noqa: E402
from TETHER_main_fig import (C_LATEST, C_ROLL, C_T0, A_REF, A_ROLL,      # noqa: E402
                             RESAMPLE, expert_traj, grasp_idx, medoid,
                             resample, rollout_states)
from lerobot.envs.factory import make_env                                # noqa: E402
from lerobot.policies.factory import make_policy                         # noqa: E402
from lerobot.utils.utils import init_logging                             # noqa: E402


def traj_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Mean point-to-point L2 between two trajectories after common resampling.

    Both curves are resampled to RESAMPLE points by index, so this compares
    shape and placement, not duration. Durations differ (a failed rollout runs
    to the step limit), and we do not want that to dominate the number.
    """
    A, B = resample(a, RESAMPLE), resample(b, RESAMPLE)
    return float(np.linalg.norm(A - B, axis=1).mean())


def seq_spec(suite: str, n_tasks: int) -> dict:
    """Sequential (CLARE) setting: one suite entry per task, learned in order."""
    ds_prefix, env_prefix = suite_prefixes(suite)
    root = Path(os.environ.get("HF_LEROBOT_HOME",
                               str(Path.home() / "Datasets" / "lerobot")))
    return {"env_task": {t: f"{env_prefix}{t}" for t in range(n_tasks)},
            "instr": {t: task_instruction(f"{ds_prefix}{t}") for t in range(n_tasks)},
            "ds_root": {t: root / f"{ds_prefix}{t}" for t in range(n_tasks)},
            "n_task": n_tasks}


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--seq_run",
                   default="outputs/SEQFT_spatial/libero_spatial_seed42_ours")
    p.add_argument("--tether_run",
                   default="outputs/V4SEQ_spatial/libero_spatial_seed42_ours")
    p.add_argument("--row_names", default="Seq-FT,TETHER")
    p.add_argument("--ckpt_task", type=int, default=3,
                   help="checkpoint to evaluate — the model right after this task. "
                        "This is the LAST task it saw, so it is also the orange reference.")
    p.add_argument("--eval_tasks", default="0,1,2,3",
                   help="conditions to feed that one checkpoint, one per column")
    p.add_argument("--n_ref_demos", type=int, default=20)
    p.add_argument("--n_rollout", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=7)
    p.add_argument("--axes", default="xy", choices=["xy", "xz", "yz"])
    p.add_argument("--grasp_box", action="store_true",
                   help="also print grasp height (z at the lowest point). Off by "
                        "default: in libero_spatial every bowl sits at the same "
                        "table height, so grasp z cannot tell the tasks apart.")
    p.add_argument("--no_suptitle", action="store_true")
    p.add_argument("--out", default="results/TETHER_main_fig2")
    p.add_argument("--plot_only", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def collect(a, spec, out: Path) -> dict:
    """Roll out the final checkpoint of each row under every condition."""
    cache = out / "rollouts.npz"
    if a.plot_only and cache.exists():
        z = np.load(cache, allow_pickle=True)
        logging.info(f"[FIG2] using cache {cache}")
        return z["data"].item()

    tasks = [int(s) for s in a.eval_tasks.split(",")]
    rn = [x.strip() for x in a.row_names.split(",")]
    rows = [(rn[0], Path(a.seq_run)), (rn[1], Path(a.tether_run))]
    data: dict = {}
    for name, run in rows:
        ck = run / f"task_{a.ckpt_task}" / "checkpoints" / "last" / "pretrained_model"
        if not ck.exists():
            raise SystemExit(f"checkpoint missing: {ck}")
        # One checkpoint per row. The policy reads its condition from the env's
        # language instruction, so the same weights are reused for every column;
        # only the env changes. Rebuilding the policy per column would load the
        # identical state dict four times.
        policy = None
        try:
            for j in tasks:
                t0 = time.perf_counter()
                cfg = build_eval_cfg(argparse.Namespace(
                    suite=a.suite, device=a.device, eval_seed=a.eval_seed,
                    num_workers=0, batch_size=8, steps=1, log_every=100,
                    eval_episodes=a.n_rollout, eval_batch_size=a.n_rollout),
                    str(ck), spec["env_task"][j])
                if policy is None:
                    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
                    policy.eval()
                env = make_env(cfg.env, n_envs=a.n_rollout, use_async_envs=False)
                try:
                    seeds = list(range(a.eval_seed, a.eval_seed + a.n_rollout))
                    with torch.no_grad():
                        st, sc = rollout_states(env, policy, seeds)
                finally:
                    env.close()
                data[f"{name}|{j}"] = {"traj": [s[:, :3] for s in st],
                                       "success": sc.tolist(),
                                       "sr": 100.0 * float(sc.mean())}
                logging.info(f"[FIG2] {name} cond t{j}  SR={100*sc.mean():.0f}%  "
                             f"len {[len(s) for s in st]}  ({time.perf_counter()-t0:.0f}s)")
        finally:
            del policy
            torch.cuda.empty_cache()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, data=np.array(data, dtype=object))
    return data


def main():
    a = parse()
    init_logging()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks = [int(s) for s in a.eval_tasks.split(",")]
    rn = [x.strip() for x in a.row_names.split(",")]
    spec = seq_spec(a.suite, max(tasks + [a.ckpt_task]) + 1)
    data = collect(a, spec, out)

    # Expert references. One suite directory per task here, so no episode-block
    # split is needed — take the first n_ref_demos episodes of each.
    ref = {j: expert_traj(spec["ds_root"][j], None, a.n_ref_demos, None) for j in tasks}
    if a.ckpt_task not in ref:
        ref[a.ckpt_task] = expert_traj(spec["ds_root"][a.ckpt_task], None,
                                       a.n_ref_demos, None)
    ref_latest = ref[a.ckpt_task]
    med = {j: ref[j][medoid(ref[j])] for j in ref}
    lat_med = med[a.ckpt_task]

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    AX = {"xy": (0, 1, "x", "y"), "xz": (0, 2, "x", "z"), "yz": (1, 2, "y", "z")}[a.axes]
    i, j_, nx, ny = AX

    allpts = [t[:, [i, j_]] for v in data.values() for t in v["traj"]]
    allpts += [t[:, [i, j_]] for trs in ref.values() for t in trs]
    P = np.concatenate(allpts)
    pad = 0.05 * max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
    xl = (P[:, 0].min() - pad, P[:, 0].max() + pad)
    yl = (P[:, 1].min() - pad, P[:, 1].max() + pad)

    nrow, ncol = 2, len(tasks)
    pw = 3.5
    ph = pw * float(np.clip((yl[1] - yl[0]) / (xl[1] - xl[0]), 0.45, 1.6))
    top_in, bot_in = (0.42 if a.no_suptitle else 1.15), 1.30
    fw, fh = pw * ncol + 0.9, ph * nrow + top_in + bot_in
    fig, axes = plt.subplots(nrow, ncol, figsize=(fw, fh), squeeze=False)
    fig.subplots_adjust(left=0.9 / fw, right=1 - 0.12 / fw,
                        top=1 - top_in / fh, bottom=bot_in / fh, wspace=.13, hspace=.17)

    def draw(ax, tr, **kw):
        ax.plot(tr[:, i], tr[:, j_], **kw)

    dist = {}
    for r, name in enumerate(rn):
        for c, j in enumerate(tasks):
            ax = axes[r][c]
            d = data[f"{name}|{j}"]
            is_latest = (j == a.ckpt_task)
            # Latest-task reference goes down first so that in the degenerate
            # last column the blue condition curve sits visibly on top of it.
            for t in ref_latest:
                draw(ax, t, color=C_LATEST, lw=.7, alpha=A_REF, zorder=2)
            draw(ax, lat_med, color=C_LATEST, lw=2.4, ls=(0, (7, 2, 1.4, 2)), zorder=4)
            for t in ref[j]:
                draw(ax, t, color=C_T0, lw=.7, alpha=A_REF, zorder=3)
            draw(ax, med[j], color=C_T0, lw=2.4, ls=(0, (5, 2)), zorder=5)
            for t in d["traj"]:
                draw(ax, t, color=C_ROLL, lw=.8, alpha=A_ROLL, zorder=6)
            rt = d["traj"][medoid(d["traj"])]
            draw(ax, rt, color=C_ROLL, lw=2.2, zorder=8)
            ax.plot(rt[0, i], rt[0, j_], "o", color=C_ROLL, ms=7, zorder=9)
            ax.plot(rt[-1, i], rt[-1, j_], "^", color=C_ROLL, ms=8, zorder=9)

            # The distances still go to panel_sr.txt, but not onto the panel —
            # a 0% panel flails, so its number invites exactly the misreading
            # the caveat in that file warns about. SR carries the claim here.
            dist[f"{name}|{j}"] = (traj_dist(rt, med[j]), traj_dist(rt, lat_med))
            # Bottom right — top left is crossed by the references, top right
            # holds the SR box, bottom left by the start marker.
            if a.grasp_box:
                q = rt[grasp_idx(rt)]
                ax.text(.97, .04, f"grasp z {q[2]:.3f}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7.8, family="monospace",
                        bbox=dict(fc="white", ec="0.8", alpha=.9, pad=2.2))
            ax.text(.97, .96,
                    f"Task {j} SR {d['sr']:.0f}%  ({sum(d['success'])}/{len(d['success'])})",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.6,
                    bbox=dict(fc="white", ec="0.75", alpha=.92, pad=2.4))
            ax.set_xlim(*xl); ax.set_ylim(*yl)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.22); ax.tick_params(labelsize=8)
            if r == 0:
                tag = "  (= latest)" if is_latest else ""
                ax.set_title(f"Condition: Task {j}{tag}", fontsize=10.5)
            if r == nrow - 1:
                ax.set_xlabel(f"End-effector {nx} position (m)", fontsize=9)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(f"{name}\nEnd-effector {ny} position (m)", fontsize=9.5)
            else:
                ax.set_yticklabels([])

    from matplotlib.legend_handler import HandlerTuple
    pair = lambda col, ls: (plt.Line2D([], [], color=col, lw=2.4, ls=ls),   # noqa: E731
                            plt.Line2D([], [], color=col, lw=.8, alpha=.5))
    h = [pair(C_T0, (0, (5, 2))),
         pair(C_LATEST, (0, (7, 2, 1.4, 2))),
         pair(C_ROLL, "-"),
         plt.Line2D([], [], ls="none", marker="o", ms=6, color=C_ROLL),
         plt.Line2D([], [], ls="none", marker="^", ms=7, color=C_ROLL)]
    lb = [f"Condition-task expert — medoid (thick) + {a.n_ref_demos} demos (thin)",
          f"Latest-task (t{a.ckpt_task}) expert — medoid (thick) + {a.n_ref_demos} demos (thin)",
          f"Policy rollout under that condition — medoid (thick) + {a.n_rollout} rollouts (thin)",
          "start", "end"]
    fig.legend(h, lb, loc="lower center", ncol=3, fontsize=9.5, frameon=False,
               bbox_to_anchor=(.5, .14 / fh), columnspacing=2.2, handlelength=3.0,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    if not a.no_suptitle:
        fig.suptitle(
            f"One checkpoint (after Tasks 0–{a.ckpt_task}), four conditions\n"
            f"{a.suite}, seed 42 / eval seed {a.eval_seed}, "
            f"{a.n_rollout} rollouts per panel   ·   "
            f"latest task t{a.ckpt_task}: “{spec['instr'][a.ckpt_task]}”",
            fontsize=12, y=1 - .24 / fh)

    for ext in ("png", "pdf"):
        fp = out / f"TETHER_main_fig2.{ext}"
        fig.savefig(fp, dpi=150 if ext == "png" else None)
        print(f"saved -> {fp}")
    plt.close(fig)

    L = [f"Final checkpoint (after tasks 0-{a.ckpt_task}) evaluated under each condition",
         f"({a.n_rollout} rollouts per panel, eval seed {a.eval_seed})", "",
         "Success rate", "",
         f"{'':<9}" + "".join(f"{'cond t'+str(j):>12}" for j in tasks)]
    for name in rn:
        L.append(f"{name:<9}" + "".join(
            f"{data[f'{name}|{j}']['sr']:11.0f}%" for j in tasks))
    L += ["", "Trajectory distance of the rollout medoid (mean point-to-point L2, m)",
          "  d_cond   = distance to that column's own expert",
          f"  d_latest = distance to the latest task (t{a.ckpt_task}) expert",
          "  d_cond < d_latest means the condition still steers the policy.",
          f"  Column t{a.ckpt_task} is degenerate (condition == latest) and is the control.",
          "  CAVEAT: this number is only readable when the rollouts are coherent.",
          "  A 0% panel flails, so its medoid is not meaningfully near either",
          "  reference and d_cond < d_latest there does NOT mean the condition was",
          "  followed. Read SR first; use the distance to separate the surviving cases.", "",
          f"{'':<9}" + "".join(f"{'cond t'+str(j):>24}" for j in tasks)]
    for name in rn:
        row = f"{name:<9}"
        for j in tasks:
            dc, dl = dist[f"{name}|{j}"]
            row += f"   d_cond {dc:.3f} d_lat {dl:.3f}"
        L.append(row)
    L += ["", "Verdict per panel (excluding the degenerate column)",
          "  Only meaningful for panels with non-zero SR — see the caveat above.", ""]
    for name in rn:
        v = [f"t{j}:{'cond' if dist[f'{name}|{j}'][0] < dist[f'{name}|{j}'][1] else 'LATEST'}"
             for j in tasks if j != a.ckpt_task]
        L.append(f"  {name:<9} " + "  ".join(v))
    (out / "panel_sr.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
