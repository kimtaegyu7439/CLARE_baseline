#!/usr/bin/env python
"""δ 가 작은 이유 분해 — 조건부가 무너졌나, null 이 안 움직였나.

δ = ‖v(ℓ_0) − v(∅)‖ 가 작아지는 경로는 둘이다.
  (a) v(ℓ_0) 가 marginal 쪽으로 무너짐   -> 조건부를 못 지킨 것
  (b) v(∅) 가 새 태스크로 안 이사함      -> null 이 붙들려 있는 것
ckpt_0 대비 두 항의 이동량을 따로 재면 갈린다.

  null_move = ‖v_k(o,∅)   − v_0(o,∅)‖   / ‖v_0(o,∅)‖
  cond_move = ‖v_k(o,ℓ_0) − v_0(o,ℓ_0)‖ / ‖v_0(o,ℓ_0)‖

관측 분포 두 곳에서 잰다: o_0(과거, SR 결정) / o_k(현재, B1 앵커 위치).
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1

from lerobot.datasets.factory import make_dataset                    # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.sampler import EpisodeAwareSampler             # noqa: E402
from lerobot.policies.factory import make_policy                     # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging  # noqa: E402


def _ns(a):
    return argparse.Namespace(
        suite=a.suite, device=a.device, seed=a.seed, num_workers=0,
        batch_size=a.batch_size, steps_per_task=1, log_every=100,
        eval_episodes=1, eval_batch_size=1, mode="null", p_drop=0.0, lambda_anchor=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="B1,B2,B4,B5")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--steps_tag", default="005000")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/B_null")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(a.suite)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    K = a.stage
    instr = {m: B1.task_instruction(f"{ds_prefix}{m}") for m in (0, K)}
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")

    def ck(arm, k):
        return (REPO / "outputs" / arm / f"{a.suite}_seed42_ours" /
                f"task_{k}" / "checkpoints" / a.steps_tag / "pretrained_model")

    def load(p):
        cfg = B1.build_cfg(_ns(a), 0, str(p), Path("/tmp/b_null_unused"))
        pol = make_policy(cfg=cfg.policy, ds_meta=meta); pol.eval(); return pol

    first = load(ck(arms[0], 0))
    obs, fm = {}, {}
    for m in (0, K):
        cfg = B1.build_cfg(_ns(a), 0, str(ck(arms[0], 0)), Path("/tmp/b_null_unused"))
        cfg.dataset.repo_id = f"{ds_prefix}{m}"
        ds = make_dataset(cfg)
        sp = EpisodeAwareSampler(ds.episode_data_index,
                                 drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                                 shuffle=True)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, sampler=sp,
                                         num_workers=0, drop_last=True)
        torch.manual_seed(a.seed); it = iter(dl)
        obs[m] = [B1.prep_batch(first, B1.to_device(next(it), device)) for _ in range(a.n_batches)]
        f = []
        for b in obs[m]:
            torch.manual_seed(a.seed + 100 + m)
            f.append(B1.sample_fm(first, b)[:2])
        fm[m] = f
        del ds, dl, it
    del first; torch.cuda.empty_cache()

    @torch.no_grad()
    def outs(pol, m, text):
        r = []
        for b, (x, t) in zip(obs[m], fm[m]):
            tl = B1.cond_tail(pol, b)
            n = x.shape[0]
            r.append(pol.dit_flow.velocity_net(
                noisy_actions=x, time=t,
                global_cond=B1.make_cond(B1.encode_lang(pol, [text] * n), tl)).clone())
        return torch.cat([z.flatten(1) for z in r])

    rows = []
    for arm in arms:
        p0, pk = ck(arm, 0), ck(arm, K)
        if not (p0.is_dir() and pk.is_dir()):
            print(f"[null] SKIP {arm}"); continue
        m0 = load(p0)
        ref = {m: {"null": outs(m0, m, B1.NULL_TEXT), "cond": outs(m0, m, instr[0])} for m in obs}
        del m0; torch.cuda.empty_cache()
        mk = load(pk)
        cur = {m: {"null": outs(mk, m, B1.NULL_TEXT), "cond": outs(mk, m, instr[0])} for m in obs}
        del mk; torch.cuda.empty_cache()
        for m in obs:
            def rel(x, y):
                return float((x - y).norm(dim=1).mean() / y.norm(dim=1).mean())
            rows.append({
                "arm": arm, "obs": f"o_{m}",
                "null_move": rel(cur[m]["null"], ref[m]["null"]),
                "cond_move": rel(cur[m]["cond"], ref[m]["cond"]),
                "delta_0": float((ref[m]["cond"] - ref[m]["null"]).norm(dim=1).mean()),
                "delta_k": float((cur[m]["cond"] - cur[m]["null"]).norm(dim=1).mean()),
            })
            r = rows[-1]
            print(f"[null] {arm} {r['obs']}  null_move {r['null_move']:.3f}  "
                  f"cond_move {r['cond_move']:.3f}  δ0 {r['delta_0']:.3f} -> δk {r['delta_k']:.3f}")

    sr = {}
    for arm in arms:
        p = REPO / "results" / arm / "sr_matrix.csv"
        if p.exists():
            for line in p.read_text().splitlines()[1:]:
                f_ = line.split(",")
                if f_[0] == str(K) and len(f_) > 1 and f_[1].strip():
                    sr[arm] = float(f_[1])

    L = ["=" * 88, f"δ 분해 — ckpt_{K} vs ckpt_0, 보존 대상 task 0", "=" * 88, "",
         "null_move = null 스트림이 ckpt_0 대비 이동한 상대량",
         "cond_move = task0 조건부가 ckpt_0 대비 이동한 상대량",
         "δ = ‖v(ℓ_0) − v(∅)‖ 절대값", "",
         "-" * 88,
         f"{'arm':>5}{'obs':>7}{'null_move':>12}{'cond_move':>12}{'δ0':>9}{'δk':>9}{'SR':>7}   해석",
         "-" * 88]
    for r in rows:
        s = sr.get(r["arm"])
        if r["null_move"] > 2 * r["cond_move"]:
            tag = "null 이 크게 이사, 조건부는 붙듦 -> δ 증가"
        elif r["cond_move"] > 2 * r["null_move"]:
            tag = "조건부가 무너짐 -> δ 감소"
        else:
            tag = "둘이 함께 이동 -> δ 유지"
        L.append(f"{r['arm']:>5}{r['obs']:>7}{r['null_move']:>12.3f}{r['cond_move']:>12.3f}"
                 f"{r['delta_0']:>9.3f}{r['delta_k']:>9.3f}"
                 f"{('%.0f' % s) if s is not None else '   .':>7}   {tag}")
    rep = "\n".join(L)
    (out / "report.txt").write_text(rep)
    json.dump(rows, (out / "rows.json").open("w"), indent=2)
    print("\n" + rep + f"\n\nsaved -> {out/'report.txt'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
