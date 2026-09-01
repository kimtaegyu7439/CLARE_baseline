#!/usr/bin/env python
"""task 0 만 학습했을 때, 아직 안 배운 태스크들의 속도장이 얼마나 남아 있는가.

가설: 오래 학습할수록 무조건부 필드가 task 0 으로 더 넓게 덮여, 다음 태스크를
      배울 때 '이미 맞는 지점'이 줄어든다.

세 모델을 같은 좌표에서 비교한다.
  pretrain      libero_90 (90 태스크) 사전학습. 범용 필드의 기준선.
  task0_5k      거기서 task 0 만 5000 스텝  (E0 lam0/task_0)
  task0_20k     거기서 task 0 만 20000 스텝 (ER_20k task_0)

각 태스크 j 의 전문가 상태에서 자기 명령어 ℓ_j 를 넣고 잰다.
  rel   = ‖v(o_j,ℓ_j) − v*_j‖ / ‖v*_j‖        정답에서 얼마나 먼가
  delta = ‖v(o_j,ℓ_j) − v(o_j,ℓ_0)‖ / ‖v*_j‖  ℓ_j 와 ℓ_0 을 구분하는가
  drift = ‖v(o_j,ℓ_j) − v_pre(o_j,ℓ_j)‖/‖v*_j‖ pretrain 필드에서 얼마나 밀려났나
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
from B_merge import _ns

from lerobot.datasets.factory import make_dataset                    # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.sampler import EpisodeAwareSampler             # noqa: E402
from lerobot.policies.factory import make_policy                     # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging  # noqa: E402

MODELS = {
    "pretrain":  "/home/sa090180/Models/dit_flow_mt_libero_90_pretrain",
    "task0_5k":  "outputs/E0/libero_spatial/seed_42/lam0/task_0/checkpoints/last/pretrained_model",
    "task0_20k": "outputs/ER_20k/libero_spatial/seed42/task_0/checkpoints/last/pretrained_model",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0,1,2,3")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/B_plasticity")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(a.suite)
    tasks = [int(x) for x in a.tasks.split(",")]
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    instr = [B1.task_instruction(f"{ds_prefix}{i}") for i in range(4)]

    def load(p):
        cfg = B1.build_cfg(_ns(a), 0, str(p if str(p).startswith("/") else REPO / p),
                           Path("/tmp/b_plast"))
        pol = make_policy(cfg=cfg.policy, ds_meta=meta); pol.eval(); return pol

    seed_pol = load(MODELS["pretrain"])
    data = {}
    for j in tasks:                       # 모든 모델이 같은 좌표·같은 노이즈를 본다
        cfg = B1.build_cfg(_ns(a), j, MODELS["pretrain"], Path("/tmp/b_plast"))
        ds = make_dataset(cfg)
        sp = EpisodeAwareSampler(ds.episode_data_index,
                                 drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                                 shuffle=True)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, sampler=sp,
                                         num_workers=0, drop_last=True)
        torch.manual_seed(a.seed); it = iter(dl)
        bs, fm = [], []
        for i in range(a.n_batches):
            b = B1.prep_batch(seed_pol, B1.to_device(next(it), device))
            torch.manual_seed(a.seed * 7 + j * 131 + i)      # 배치마다 다른 (ε,t)
            bs.append(b); fm.append(B1.sample_fm(seed_pol, b))
        data[j] = (bs, fm)
        del ds, dl, it
    del seed_pol; torch.cuda.empty_cache()

    def field(pol, j, lang):
        vs = []
        with torch.no_grad():
            for b, (x_t, t, _) in zip(*data[j]):
                n = x_t.shape[0]
                vs.append(pol.dit_flow.velocity_net(
                    noisy_actions=x_t, time=t,
                    global_cond=B1.make_cond(
                        B1.encode_lang(pol, [instr[lang]] * n), B1.cond_tail(pol, b))).clone())
        return vs

    pre = load(MODELS["pretrain"])
    ref = {j: field(pre, j, j) for j in tasks}
    del pre; torch.cuda.empty_cache()

    def rel(us, vs, j):
        num = tot = 0.0
        for u, v, (_, _, tgt) in zip(us, vs, data[j][1]):
            num += float((u - v).flatten(1).norm(dim=1).mean())
            tot += float(tgt.flatten(1).norm(dim=1).mean())
        return num / tot

    rows = []
    for name, path in MODELS.items():
        pol = load(path)
        for j in tasks:
            vj = field(pol, j, j)
            v0 = field(pol, j, 0)
            tgt = [t for (_, _, t) in data[j][1]]
            rows.append({"model": name, "task": j,
                         "rel": rel(vj, tgt, j),
                         "delta": rel(vj, v0, j),
                         "drift": rel(vj, ref[j], j)})
            r = rows[-1]
            print(f"[plast] {name:>10} task{j}  rel {r['rel']:.3f}  "
                  f"δ(ℓ_j vs ℓ_0) {r['delta']:.3f}  pretrain드리프트 {r['drift']:.3f}", flush=True)
        del pol; torch.cuda.empty_cache()

    L = ["=" * 84,
         "task 0 만 학습했을 때 남아 있는 범용 속도장 — 학습량에 따른 변화", "=" * 84, "",
         "모든 모델이 같은 전문가 상태·같은 (ε,t) 를 본다. 값은 ‖v*_j‖ 로 정규화.",
         "",
         "rel    ‖v(o_j,ℓ_j) − v*_j‖    정답에서 얼마나 먼가 (작을수록 좋다)",
         "δ      ‖v(o_j,ℓ_j) − v(o_j,ℓ_0)‖  ℓ_j 와 ℓ_0 을 구분하는가 (0 이면 무조건부 = task0)",
         "drift  ‖v(o_j,ℓ_j) − v_pretrain(o_j,ℓ_j)‖  범용 필드에서 밀려난 정도",
         "",
         "task 0 은 학습한 태스크, task 1~3 은 아직 안 배운 태스크다.",
         "-" * 84,
         f"{'model':>11}{'task':>6}{'rel':>9}{'δ':>9}{'drift':>9}",
         "-" * 84]
    for r in rows:
        L.append(f"{r['model']:>11}{r['task']:>6}{r['rel']:>9.3f}{r['delta']:>9.3f}{r['drift']:>9.3f}")
    rep = "\n".join(L)
    (out / "report.txt").write_text(rep)
    json.dump(rows, (out / "rows.json").open("w"), indent=2)
    print("\n" + rep + f"\n\nsaved -> {out/'report.txt'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
