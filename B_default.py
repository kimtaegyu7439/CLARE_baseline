#!/usr/bin/env python
"""무조건부 속도장은 누구의 것인가 — "거의 모든 지점이 방금 배운 태스크가 된다" 검증.

가설(사용자): 태스크 k 를 학습하고 나면 입력 공간의 거의 모든 지점에서 모델의 출력이
task k 의 속도가 되어 버린다. 그래서 다음 태스크를 배울 때 '이미 맞는 지점'이 없다.

이걸 재려면 명령어를 뺀 기본 출력 v(o,∅) 이 누구를 가리키는지 보면 된다.
    d_j(o) = ‖v(o, ℓ_j) − v(o, ∅)‖ / ‖v*(o)‖
d_j 가 0 이면 "명령어 j 를 넣어도 기본값과 같다" = 기본 필드가 곧 태스크 j 다.

★ 핵심은 o 의 출처다. task k 자기 관측에서만 d_k≈0 이면 국소적인 현상이고,
  모든 태스크의 관측에서 d_k≈0 이면 공간 전체가 task k 로 덮인 것이다.
  그래서 관측을 태스크 0..3 에서 모두 모아 출처별로 쪼갠다.
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
    ap.add_argument("--arms", default="seq-FT,B2λ3,ER")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--steps_tag", default="005000")
    ap.add_argument("--num_tasks", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=2)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/B_default")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    K = a.num_tasks
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(a.suite)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    instr = [B1.task_instruction(f"{ds_prefix}{i}") for i in range(K)]

    def ckpt(root, k):
        if isinstance(root, dict):
            return REPO / root["tmpl"].format(k=k)
        return (REPO / root / f"{a.suite}_seed42_ours" / f"task_{k}"
                / "checkpoints" / a.steps_tag / "pretrained_model")

    def load(p):
        cfg = B1.build_cfg(_ns(a), 0, str(p), Path("/tmp/b_default"))
        pol = make_policy(cfg=cfg.policy, ds_meta=meta); pol.eval(); return pol

    seed_pol = load(ckpt(ARMS[arms[0]], 0))
    data = {}
    for j in range(K):                    # 모든 팔·모든 스테이지가 같은 좌표를 본다
        cfg = B1.build_cfg(_ns(a), j, str(ckpt(ARMS[arms[0]], 0)), Path("/tmp/b_default"))
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
            torch.manual_seed(a.seed * 7 + j * 131 + i)   # 배치마다 다른 (ε,t)
            bs.append(b); fm.append(B1.sample_fm(seed_pol, b))
        data[j] = (bs, fm)
        del ds, dl, it
    del seed_pol; torch.cuda.empty_cache()

    rows = []
    for name in arms:
        root = ARMS[name]
        for k in range(K):
            p = ckpt(root, k)
            if not p.is_dir():
                print(f"[skip] {name} stage{k}"); continue
            pol = load(p)
            net = pol.dit_flow.velocity_net
            with torch.no_grad():
                for obs_j in range(K):        # 관측 출처
                    per = {j: [] for j in range(K)}
                    for b, (x_t, t, tgt) in zip(*data[obs_j]):
                        n = x_t.shape[0]
                        tail = B1.cond_tail(pol, b)
                        scale = tgt.flatten(1).norm(dim=1).clamp_min(1e-8)
                        v0 = net(noisy_actions=x_t, time=t,
                                 global_cond=B1.make_cond(
                                     B1.encode_lang(pol, [B1.NULL_TEXT] * n), tail))
                        for j in range(K):
                            vj = net(noisy_actions=x_t, time=t,
                                     global_cond=B1.make_cond(
                                         B1.encode_lang(pol, [instr[j]] * n), tail))
                            per[j].append(((vj - v0).flatten(1).norm(dim=1) / scale).cpu())
                    for j in range(K):
                        d = torch.cat(per[j])
                        rows.append({"arm": name, "stage": k, "obs": obs_j, "instr": j,
                                     "mean": float(d.mean()), "p10": float(d.quantile(0.10)),
                                     "p50": float(d.median()), "p90": float(d.quantile(0.90)),
                                     "frac_lt_0p5": float((d < 0.5).float().mean()),
                                     "vals": [round(float(x), 4) for x in d]})
                    print(f"[default] {name:>7} stage{k} obs{obs_j}  "
                          + "  ".join(f"d{j}={rows[-K+j]['mean']:.2f}" for j in range(K)),
                          flush=True)
            del pol; torch.cuda.empty_cache()

    json.dump(rows, (out / "rows.json").open("w"))
    print(f"saved -> {out/'rows.json'}  ({len(rows)} rows)")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
