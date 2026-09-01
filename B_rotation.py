#!/usr/bin/env python
"""회전 가설 검증 — Δ 의 크기는 사는데 방향이 죽는가.

B1 의 시그니처는 "δ 크기 5.4~5.9 인데 SR 0" 이었다. 이론의 해석: raw v-공간 앵커가
이사하는 null 을 target 에 섞어 Δ 를 **회전**시킨다. 크기가 아니라 방향이 죽는다.

B3(질의점 위 raw v-앵커)의 stage 1 이 60 으로 낮게 나온 것도 같은 병일 수 있다.
"질의점이 부족하다"가 아니라 "**회전 보정 없는** 질의점 방식"의 한계라는 가설.

측정: 각 팔의 ckpt_1 이 ckpt_0(태스크 0 에 유능하던 시점) 대비
    mag = ‖Δ_1‖ / ‖Δ_0‖              크기 보존율
    cos = cos(Δ_1, Δ_0)               방향 보존율
  Δ_· = v_·(x,t,o,ℓ_0) − v_·(x,t,o,∅)
기준 ckpt_0 는 팔마다 자기 것을 쓴다(각 팔이 자기 유능 상태에서 얼마나 움직였는가).

관측 분포 두 곳에서 잰다.
    o_0 : 태스크 0 관측 — SR 이 결정되는 곳, B3 가 질의점으로 얼린 영역
    o_1 : 태스크 1 관측 — B1/B2 의 앵커가 실제로 붙잡는 곳

판정
    mag 높은데 cos 낮다  -> 회전 가설. Δ-공간 앵커(B4 축 1)가 처방.
    mag 도 낮다          -> 단순 수축. 이득 w 가 처방.
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


def _ns(args, k):
    return argparse.Namespace(
        suite=args.suite, device=args.device, seed=args.seed, num_workers=0,
        batch_size=args.batch_size, steps_per_task=1, log_every=100,
        eval_episodes=1, eval_batch_size=1, mode="rot", p_drop=0.0, lambda_anchor=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="B1,B2,B3")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--steps_tag", default="005000")
    ap.add_argument("--pairs", default="1:0",
                    help='"k:j" 쌍 목록. 기준은 ckpt_j (태스크 j 를 막 배운 시점). 예 "2:1,3:1,3:2"')
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/B_rotation")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    init_logging()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(args.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(args.suite)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    pairs = []
    for tok in args.pairs.split(","):
        k, j = tok.strip().split(":")
        pairs.append((int(k), int(j)))
    tasks = sorted({t for k, j in pairs for t in (k, j)})
    instr = {m: B1.task_instruction(f"{ds_prefix}{m}") for m in tasks}
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")

    def ckpt_path(arm, k):
        return (REPO / "outputs" / arm / f"{args.suite}_seed42_ours" /
                f"task_{k}" / "checkpoints" / args.steps_tag / "pretrained_model")

    def load(p):
        cfg = B1.build_cfg(_ns(args, 0), 0, str(p), Path("/tmp/b_rot_unused"))
        pol = make_policy(cfg=cfg.policy, ds_meta=meta)
        pol.eval()
        return pol

    first = None
    for arm in arms:
        if ckpt_path(arm, 0).is_dir():
            first = load(ckpt_path(arm, 0)); break
    if first is None:
        raise SystemExit("체크포인트를 찾지 못했다")

    obs, fm = {}, {}
    for m in tasks:
        cfg = B1.build_cfg(_ns(args, m), m, str(ckpt_path(arms[0], 0)), Path("/tmp/b_rot_unused"))
        ds = make_dataset(cfg)
        sampler = EpisodeAwareSampler(
            ds.episode_data_index,
            drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0), shuffle=True)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=True)
        torch.manual_seed(args.seed)
        it = iter(loader)
        obs[m] = [B1.prep_batch(first, B1.to_device(next(it), device)) for _ in range(args.n_batches)]
        f = []
        for b in obs[m]:
            torch.manual_seed(args.seed + 1000 + m)
            f.append(B1.sample_fm(first, b)[:2])
        fm[m] = f
        del ds, loader, it
    del first; torch.cuda.empty_cache()

    @torch.no_grad()
    def deltas(pol, m, j):
        outs = []
        for b, (x_t, t) in zip(obs[m], fm[m]):
            tail = B1.cond_tail(pol, b)
            bsz = x_t.shape[0]
            u = pol.dit_flow.velocity_net(
                noisy_actions=x_t, time=t,
                global_cond=B1.make_cond(B1.encode_lang(pol, [B1.NULL_TEXT] * bsz), tail))
            v = pol.dit_flow.velocity_net(
                noisy_actions=x_t, time=t,
                global_cond=B1.make_cond(B1.encode_lang(pol, [instr[j]] * bsz), tail))
            outs.append((v - u).clone())
        return outs

    # 필요한 (ckpt, j, obs) 조합만 계산한다
    need = set()
    for k, j in pairs:
        for m in (j, k):
            need.add((j, j, m))     # 기준: ckpt_j
            need.add((k, j, m))     # 현재: ckpt_k
    rows = []
    for arm in arms:
        D = {}
        for c in sorted({c for c, _, _ in need}):
            p = ckpt_path(arm, c)
            if not p.is_dir():
                continue
            pol = load(p)
            for (cc, j, m) in sorted(need):
                if cc == c:
                    D[(c, j, m)] = deltas(pol, m, j)
            del pol; torch.cuda.empty_cache()
        for k, j in pairs:
            for m in (j, k):
                if (j, j, m) not in D or (k, j, m) not in D:
                    continue
                b = torch.cat([x.flatten(1) for x in D[(j, j, m)]])
                a = torch.cat([x.flatten(1) for x in D[(k, j, m)]])
                rows.append({
                    "arm": arm, "k": k, "j": j, "obs": f"o_{m}",
                    "ref_norm": float(b.norm(dim=1).mean()),
                    "cur_norm": float(a.norm(dim=1).mean()),
                    "mag": float(a.norm(dim=1).mean() / b.norm(dim=1).mean()),
                    "cos": float(F.cosine_similarity(a, b, dim=1).mean())})
                r = rows[-1]
                print(f"[rot] {arm} k={k} j={j} obs={r['obs']}  "
                      f"‖Δ_j‖={r['ref_norm']:.3f} ‖Δ_k‖={r['cur_norm']:.3f} "
                      f"mag={r['mag']:.2f} cos={r['cos']:.3f}")

    sr = {}
    for arm in arms:
        p = REPO / "results" / arm / "sr_matrix.csv"
        if p.exists():
            for line in p.read_text().splitlines()[1:]:
                f_ = line.split(",")
                for t_, v in enumerate(f_[1:]):
                    if v.strip():
                        sr[(arm, int(f_[0]), t_)] = float(v)

    def tag(r):
        if r["ref_norm"] < 0.3:
            return "기준 Δ 가 거의 0 — 보존할 것이 없음(퇴화)"
        if r["cos"] > 0.85 and r["mag"] > 0.8:
            return "크기·방향 모두 보존"
        if r["cos"] > 0.85:
            return "방향 보존·크기 수축 -> 이득 w 가 처방"
        if r["mag"] > 0.8:
            return "크기 유지·방향 죽음 -> 회전"
        return "수축 + 회전"

    L = ["=" * 92, "회전 vs 수축 진단 — 기준은 ckpt_j (태스크 j 를 막 배운 시점)", "=" * 92, "",
         "Δ = v(x,t,o,ℓ_j) − v(x,t,o,∅)",
         "mag = ‖Δ_k‖/‖Δ_j‖ (크기비)   cos = cos(Δ_k, Δ_j) (방향)",
         "★ ‖Δ_j‖ 자체가 작으면 기준이 퇴화라 cos 는 노이즈다. 그 행은 해석하면 안 된다.", "",
         "-" * 92,
         f"{'arm':>4}{'k':>3}{'j':>3}{'obs':>6}{'‖Δ_j‖':>9}{'‖Δ_k‖':>9}{'mag':>8}{'cos':>8}{'SR':>6}   해석",
         "-" * 92]
    for r in rows:
        s_ = sr.get((r["arm"], r["k"], r["j"]))
        L.append(f"{r['arm']:>4}{r['k']:>3}{r['j']:>3}{r['obs']:>6}"
                 f"{r['ref_norm']:>9.3f}{r['cur_norm']:>9.3f}{r['mag']:>8.2f}{r['cos']:>8.3f}"
                 f"{('%.0f' % s_) if s_ is not None else '    .':>6}   {tag(r)}")
    rep = "\n".join(L)
    (out / "report.txt").write_text(rep)
    json.dump(rows, (out / "rows.json").open("w"), indent=2)
    print("\n" + rep)
    print(f"\nsaved -> {out/'report.txt'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
