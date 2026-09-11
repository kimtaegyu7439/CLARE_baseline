#!/usr/bin/env python
"""오차의 분포 — 평균은 같은데 SR 이 갈리는 이유를 찾는다.

관측: task 1 에서 ER err 0.319 / SR 100,  B2λ3 err 0.297 / SR 40.
      평균 상대오차가 같거나 오히려 ER 이 큰데 결과가 정반대다.
      -> 평균 스칼라 하나로는 설명이 안 된다. 어디가 다른지 쪼갠다.

분해 축 셋
  ① t 구간별   롤아웃은 t=0→1 로 적분한다. 초반 오차는 이후 스텝이 일부 교정하지만
                후반 오차는 곧장 액션에 실린다. 같은 평균도 t 분포가 다르면 결과가 다르다.
  ② 방향 vs 크기   err = ‖v − v*‖ 는 방향 오차와 크기 오차가 섞여 있다.
                cos(v, v*) 는 적분에서 누적되고, 크기 오차는 부분적으로 상쇄된다.
  ③ 분위수      평균이 같아도 소수의 큰 오차(p90)가 지배하면 롤아웃이 그 지점에서 깨진다.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
from B_merge import ARMS, _ns

from lerobot.datasets.factory import make_dataset                    # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.sampler import EpisodeAwareSampler             # noqa: E402
from lerobot.policies.factory import make_policy                     # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging  # noqa: E402

BINS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="ER,seq-FT,B2λ3,B8,B1")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--steps_tag", default="005000")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--tasks", default="0,1,2,3")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/B_errprofile")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(a.suite)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    tasks = [int(x) for x in a.tasks.split(",")]
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    instr = [B1.task_instruction(f"{ds_prefix}{i}") for i in range(4)]

    def ckpt(root, k):
        if isinstance(root, dict):
            return REPO / root["tmpl"].format(k=k)
        return (REPO / root / f"{a.suite}_seed42_ours" / f"task_{k}"
                / "checkpoints" / a.steps_tag / "pretrained_model")

    def load(p):
        cfg = B1.build_cfg(_ns(a), 0, str(p), Path("/tmp/b_errp"))
        pol = make_policy(cfg=cfg.policy, ds_meta=meta); pol.eval(); return pol

    seed_pol = load(ckpt(ARMS[arms[0]], 0))
    data = {}
    for j in tasks:
        cfg = B1.build_cfg(_ns(a), j, str(ckpt(ARMS[arms[0]], 0)), Path("/tmp/b_errp"))
        ds = make_dataset(cfg)
        sp = EpisodeAwareSampler(ds.episode_data_index,
                                 drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                                 shuffle=True)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, sampler=sp,
                                         num_workers=0, drop_last=True)
        torch.manual_seed(a.seed); it = iter(dl)
        bs, fm = [], []
        for _ in range(a.n_batches):
            b = B1.prep_batch(seed_pol, B1.to_device(next(it), device))
            torch.manual_seed(a.seed + j)
            bs.append(b); fm.append(B1.sample_fm(seed_pol, b))
        data[j] = (bs, fm)
        del ds, dl, it
    del seed_pol; torch.cuda.empty_cache()

    rows = []
    for name in arms:
        p = ckpt(ARMS[name], a.stage)
        if not p.is_dir():
            print(f"[err] SKIP {name}"); continue
        pol = load(p); net = pol.dit_flow.velocity_net
        for j in tasks:
            V, T_, G = [], [], []
            with torch.no_grad():
                for b, (x_t, t, tgt) in zip(*data[j]):
                    tl = B1.cond_tail(pol, b); n = x_t.shape[0]
                    v = net(noisy_actions=x_t, time=t,
                            global_cond=B1.make_cond(
                                B1.encode_lang(pol, [instr[j]] * n), tl))
                    V.append(v.flatten(1)); T_.append(t); G.append(tgt.flatten(1))
                v = torch.cat(V); t = torch.cat(T_); g = torch.cat(G)
                d = (v - g).norm(dim=1)
                gn = g.norm(dim=1)
                rel = (d / gn.clamp_min(1e-8))
                cos = F.cosine_similarity(v, g, dim=1)
                mag = v.norm(dim=1) / gn.clamp_min(1e-8)
                prof = []
                for lo, hi in BINS:
                    m = (t >= lo) & (t < hi)
                    prof.append({"n": int(m.sum()),
                                 "rel": float(rel[m].mean()) if m.any() else None,
                                 "cos": float(cos[m].mean()) if m.any() else None,
                                 "mag": float(mag[m].mean()) if m.any() else None})
                q = torch.quantile(rel, torch.tensor([0.5, 0.9], device=rel.device))
                rows.append({"arm": name, "task": j,
                             "rel_mean": float(rel.mean()), "rel_p50": float(q[0]),
                             "rel_p90": float(q[1]), "cos_mean": float(cos.mean()),
                             "mag_mean": float(mag.mean()), "by_t": prof})
                r = rows[-1]
                print(f"[err] {name:>6} t{j}  rel {r['rel_mean']:.3f} "
                      f"(p50 {r['rel_p50']:.3f} p90 {r['rel_p90']:.3f})  "
                      f"cos {r['cos_mean']:.3f}  mag {r['mag_mean']:.3f}")
        del pol; torch.cuda.empty_cache()

    L = ["=" * 100, f"오차 분포 분해 — stage {a.stage}", "=" * 100, "",
         "rel = ‖v(ℓ_j)−v*_j‖/‖v*_j‖   cos = 방향 일치   mag = ‖v‖/‖v*‖ 크기비", "",
         "-" * 100,
         f"{'arm':>7}{'task':>5}{'rel':>8}{'p50':>8}{'p90':>8}{'cos':>8}{'mag':>8}"
         + "".join(f"{'t'+str(i):>16}" for i in range(len(BINS))),
         f"{'':>52}" + "".join(f"{'rel/cos':>16}" for _ in BINS),
         "-" * 100]
    for r in rows:
        cells = "".join(
            f"{b['rel']:>8.3f}{b['cos']:>8.3f}" if b["rel"] is not None else f"{'—':>16}"
            for b in r["by_t"])
        L.append(f"{r['arm']:>7}{r['task']:>5}{r['rel_mean']:>8.3f}{r['rel_p50']:>8.3f}"
                 f"{r['rel_p90']:>8.3f}{r['cos_mean']:>8.3f}{r['mag_mean']:>8.3f}" + cells)
    L += ["", f"t 구간: " + "  ".join(f"t{i}=[{lo:.2f},{hi:.2f})" for i, (lo, hi) in enumerate(BINS))]
    rep = "\n".join(L)
    (out / "report.txt").write_text(rep)
    json.dump(rows, (out / "rows.json").open("w"), indent=2)
    print("\n" + rep + f"\n\nsaved -> {out/'report.txt'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
