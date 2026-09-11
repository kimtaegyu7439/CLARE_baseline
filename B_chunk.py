#!/usr/bin/env python
"""velocity 오차가 아니라 '실제로 뱉는 액션'의 오차를 잰다.

B_errprofile 은 ‖v(x_t,t,c) − v*‖ 을 t~U(0,1) 로 평균냈다. 그런데 롤아웃은
v 를 쓰지 않는다. t=0 노이즈에서 100 스텝 Euler 로 적분해 나온 (16,7) 청크를
쓰고, 그중 앞 n_action_steps=8 개만 실행한다.

그래서 여기서는 policy 의 실제 추론 경로(velocity_net.sample)를 그대로 돌려
전문가 청크와 비교한다. 노이즈 generator 를 팔마다 같은 시드로 고정해서
같은 상태·같은 노이즈에서 나온 결과만 비교한다.

  chunk16 = ‖â − a*‖ / ‖a*‖        전체 16 스텝
  chunk8  = 앞 8 스텝만             실제 실행되는 구간
  step0   = 첫 스텝만               다음 상태를 결정하는 지점
  spread  = 노이즈 시드를 바꿨을 때 â 끼리의 상대 표준편차 (모드 분산)
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
from B_merge import ARMS, _ns

from lerobot.datasets.factory import make_dataset                    # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.sampler import EpisodeAwareSampler             # noqa: E402
from lerobot.policies.factory import make_policy                     # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="ER,seq-FT,B2λ3,B1,B2")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--steps_tag", default="005000")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--tasks", default="0,1,2,3")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/B_chunk")
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
        cfg = B1.build_cfg(_ns(a), 0, str(p), Path("/tmp/b_chunk"))
        pol = make_policy(cfg=cfg.policy, ds_meta=meta); pol.eval(); return pol

    seed_pol = load(ckpt(ARMS[arms[0]], 0))
    data = {}
    for j in tasks:                       # 모든 팔이 정확히 같은 상태를 본다
        cfg = B1.build_cfg(_ns(a), j, str(ckpt(ARMS[arms[0]], 0)), Path("/tmp/b_chunk"))
        ds = make_dataset(cfg)
        sp = EpisodeAwareSampler(ds.episode_data_index,
                                 drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                                 shuffle=True)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, sampler=sp,
                                         num_workers=0, drop_last=True)
        torch.manual_seed(a.seed); it = iter(dl)
        data[j] = [B1.prep_batch(seed_pol, B1.to_device(next(it), device))
                   for _ in range(a.n_batches)]
        del ds, dl, it
    del seed_pol; torch.cuda.empty_cache()

    def rel(pred, gt):                    # 샘플별 상대오차의 평균
        num = (pred - gt).flatten(1).norm(dim=1)
        den = gt.flatten(1).norm(dim=1).clamp_min(1e-8)
        return float((num / den).mean())

    rows = []
    for name in arms:
        pol = load(ckpt(ARMS[name], a.stage))
        net = pol.dit_flow.velocity_net
        for j in tasks:
            acc = {"c16": 0.0, "c8": 0.0, "c1": 0.0, "spread": 0.0}
            nb = 0
            with torch.no_grad():
                for b in data[j]:
                    n = b["action"].shape[0]
                    gt = b["action"]                                   # (n,16,7) in [-1,1]
                    cond = B1.make_cond(B1.encode_lang(pol, [instr[j]] * n),
                                        B1.cond_tail(pol, b))
                    preds = []
                    for s in range(a.n_seeds):
                        g = torch.Generator(device=device).manual_seed(1000 + s)
                        preds.append(net.sample(cond, timesteps=pol.dit_flow.num_inference_steps,
                                                generator=g))
                    P = torch.stack(preds)                             # (S,n,16,7)
                    ah = P.mean(0)
                    acc["c16"] += rel(ah, gt)
                    acc["c8"] += rel(ah[:, :8], gt[:, :8])
                    acc["c1"] += rel(ah[:, :1], gt[:, :1])
                    # 노이즈 시드에 따른 출력 산포 (모드가 갈리면 커진다)
                    sd = P.std(0).flatten(1).norm(dim=1)
                    acc["spread"] += float((sd / gt.flatten(1).norm(dim=1).clamp_min(1e-8)).mean())
                    nb += 1
            r = {k: v / nb for k, v in acc.items()}
            r.update(arm=name, task=j)
            rows.append(r)
            print(f"[chunk] {name:>7} task{j}  c16 {r['c16']:.3f}  c8 {r['c8']:.3f} "
                  f" c1 {r['c1']:.3f}  spread {r['spread']:.3f}", flush=True)
        del pol; torch.cuda.empty_cache()

    SR = {  # B_compare.txt stage-3 행
        "ER": [85, 100, 90, 100], "seq-FT": [0, 0, 50, 90],
        "B2λ3": [90, 40, 100, 90], "B1": [50, 25, 90, 85], "B2": [85, 40, 100, 80]}
    L = ["=" * 88,
         "적분 후 액션 청크 오차 — 롤아웃이 실제로 쓰는 양 (stage 3 체크포인트)", "=" * 88, "",
         "velocity MSE 가 아니라 policy 의 실제 추론 경로(100 스텝 Euler)를 돌린 결과다.",
         "모든 팔이 같은 전문가 상태·같은 노이즈 시드를 본다. 값은 상대오차 ‖â−a*‖/‖a*‖.",
         "",
         "chunk16 전체 16 스텝 | chunk8 실행되는 앞 8 스텝 | step0 첫 스텝",
         "spread  노이즈 시드 3개 간 출력 표준편차(상대) — 모드가 갈리면 커진다",
         "",
         "-" * 88,
         f"{'arm':>8}{'task':>6}{'chunk16':>10}{'chunk8':>9}{'step0':>8}{'spread':>9}{'SR':>7}",
         "-" * 88]
    for r in rows:
        sr = SR.get(r["arm"], [None] * 4)[r["task"]]
        L.append(f"{r['arm']:>8}{r['task']:>6}{r['c16']:>10.3f}{r['c8']:>9.3f}"
                 f"{r['c1']:>8.3f}{r['spread']:>9.3f}"
                 + (f"{sr:>7}" if sr is not None else f"{'—':>7}"))
    rep = "\n".join(L)
    (out / "report.txt").write_text(rep)
    json.dump(rows, (out / "rows.json").open("w"), indent=2)
    print("\n" + rep + f"\n\nsaved -> {out/'report.txt'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
