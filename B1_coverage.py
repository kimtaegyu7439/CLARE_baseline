#!/usr/bin/env python
"""B1 진단 — 1세대 붕괴가 '사슬 지배'인가 'coverage 지배'인가.

문제 제기: 스테이지 k 에서 앵커는 **현재 태스크 관측 o_k** 위에서 걸린다
(B1.py 의 anchor_against 는 현재 배치의 tail 을 쓴다). 그런데 drift 와 SR 은
**과거 태스크 관측 o_j** 위에서 측정된다. anchor_loss 가 0.004 로 낮은데
1세대 drift 가 이미 0.276 인 것은, 앵커가 엉뚱한 입력 영역을 지키고 있다는
신호일 수 있다.

측정: 같은 (k, j) 조합의 drift 를 관측 분포를 바꿔 가며 두 번 잰다.

    drift[k][j | o_j]  = ‖v_k(x,t,o_j,ℓ_j) − v_j(x,t,o_j,ℓ_j)‖ / ‖v_j(...)‖
                         과거 태스크 영역 — SR 이 결정되는 곳
    drift[k][j | o_k]  = 같은 식을 현재 태스크 관측 위에서
                         앵커가 실제로 붙잡고 있는 영역

판정
    두 값이 비슷           -> 사슬(세대 증류) 지배. 연산자 역보정으로 해결 가능.
    o_j 쪽이 훨씬 크다     -> coverage 지배. 앵커가 지키는 영역이 애초에 다르다.
                              질의점 없이는 원리적으로 한계.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

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
        eval_episodes=1, eval_batch_size=1, mode="cov", p_drop=0.0, lambda_anchor=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default="outputs/B1_libero_spatial/libero_spatial_seed42_ours")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--num_tasks", type=int, default=10)
    ap.add_argument("--steps_tag", default="020000")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/B1_coverage")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    init_logging()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(args.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(args.suite)

    ckpt = {}
    for k in range(args.num_tasks):
        p = Path(args.ckpt_root) / f"task_{k}" / "checkpoints" / args.steps_tag / "pretrained_model"
        if p.is_dir():
            ckpt[k] = str(p)
    K = sorted(ckpt)
    print(f"[cov] 스테이지 {K}")

    instr = [B1.task_instruction(f"{ds_prefix}{j}") for j in range(args.num_tasks)]
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")

    def load(k):
        cfg = B1.build_cfg(_ns(args, 0), 0, ckpt[k], Path("/tmp/b1_cov_unused"))
        pol = make_policy(cfg=cfg.policy, ds_meta=meta)
        pol.eval()
        return pol

    # 관측 분포별 고정 probe (모든 모델이 같은 x_t, t 를 본다)
    tmp = load(K[0])
    obs, fm = {}, {}
    for m in K:
        cfg = B1.build_cfg(_ns(args, m), m, ckpt[K[0]], Path("/tmp/b1_cov_unused"))
        ds = make_dataset(cfg)
        sampler = EpisodeAwareSampler(
            ds.episode_data_index,
            drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0), shuffle=True)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, drop_last=True)
        torch.manual_seed(args.seed)
        it = iter(loader)
        bs = [B1.prep_batch(tmp, B1.to_device(next(it), device)) for _ in range(args.n_batches)]
        obs[m] = bs
        f = []
        for b in bs:
            torch.manual_seed(args.seed + 1000 + m)
            f.append(B1.sample_fm(tmp, b)[:2])          # (x_t, t)
        fm[m] = f
        del ds, loader, it
    del tmp; torch.cuda.empty_cache()

    # v[c][j][m] : 체크포인트 c 가 관측 m 위에서 명령어 ℓ_j 에 내놓는 속도장
    # 기준값 V[(j, j, m)] 도 함께 필요하다 — 체크포인트 j 를 관측 m 위에서 평가한 것.
    need = set()
    for c in K:
        for j in K:
            if j > c:
                continue
            for m in (j, c):
                need.add((c, j, m))
                need.add((j, j, m))      # ★ 기준. 빠뜨리면 drift(o_k) 가 전부 NaN 이 된다.
    V = {}
    for c in K:
        pol = load(c)
        with torch.no_grad():
            for (cc, j, m) in sorted(need):
                if cc != c:
                    continue
                outs = []
                for b, (x_t, t) in zip(obs[m], fm[m]):
                    tail = B1.cond_tail(pol, b)
                    cond = B1.make_cond(B1.encode_lang(pol, [instr[j]] * x_t.shape[0]), tail)
                    outs.append(pol.dit_flow.velocity_net(
                        noisy_actions=x_t, time=t, global_cond=cond).clone())
                V[(c, j, m)] = outs
        del pol; torch.cuda.empty_cache()
        print(f"[cov] ckpt {c} 계산 완료")

    def rel(a, b):
        num = sum(float((x - y).flatten(1).norm(dim=1).sum()) for x, y in zip(a, b))
        den = sum(float(y.flatten(1).norm(dim=1).sum()) for y in b)
        return num / den

    rows = []
    for k in K:
        for j in K:
            if j > k:
                continue
            # 기준은 항상 ckpt_j (태스크 j 에 유능했던 시점)
            d_tgt = rel(V[(k, j, j)], V[(j, j, j)]) if (j, j, j) in V else np.nan
            d_anc = rel(V[(k, j, k)], V[(j, j, k)]) if (j, j, k) in V and (k, j, k) in V else np.nan
            rows.append((k, j, d_tgt, d_anc))

    sr = {}
    p = Path("results/B1_libero_spatial/sr_matrix.csv")
    if p.exists():
        for line in p.read_text().splitlines()[1:]:
            f_ = line.split(",")
            for t_, v in enumerate(f_[1:]):
                if v.strip():
                    sr[(int(f_[0]), t_)] = float(v)

    L = ["=" * 78, "B1 coverage 진단 — 앵커가 지키는 영역 vs SR 이 결정되는 영역", "=" * 78, "",
         "drift(o_j) : 과거 태스크 j 의 관측 위에서 잰 드리프트  ← SR 이 결정되는 곳",
         "drift(o_k) : 현재 태스크 k 의 관측 위에서 잰 드리프트  ← 앵커가 붙잡는 곳",
         "둘 다 기준은 ckpt_j (태스크 j 를 막 배운 시점).", "",
         "-" * 78,
         f"{'k':>3}{'j':>4}{'k-j':>5}{'drift(o_j)':>13}{'drift(o_k)':>13}{'비율':>9}{'SR':>8}",
         "-" * 78]
    for k, j, a, b in rows:
        r = a / b if b and not np.isnan(b) and b > 0 else np.nan
        s = sr.get((k, j))
        L.append(f"{k:>3}{j:>4}{k-j:>5}{a:>13.3f}{b:>13.3f}"
                 f"{('%.2f' % r) if not np.isnan(r) else '   .':>9}"
                 f"{('%.0f' % s) if s is not None else '   .':>8}")
    gen1 = [(a, b) for k, j, a, b in rows if k - j == 1 and not np.isnan(b)]
    L += ["", "-" * 78, "판정", "-" * 78]
    if gen1:
        ma = float(np.mean([a for a, _ in gen1])); mb = float(np.mean([b for _, b in gen1]))
        L.append(f"1세대(k-j=1) 평균:  drift(o_j) {ma:.3f}   drift(o_k) {mb:.3f}   비율 {ma/mb:.2f}")
        if ma / mb > 1.5:
            L.append("-> o_j 쪽이 뚜렷이 크다. coverage 지배. 앵커가 다른 영역을 지키고 있다.")
        elif ma / mb < 1.2:
            L.append("-> 두 영역이 비슷하다. 사슬(세대 증류) 지배. 연산자 역보정이 유효할 것.")
        else:
            L.append("-> 중간. 두 요인이 섞여 있다.")
    rep = "\n".join(L)
    (out / "report.txt").write_text(rep)
    json.dump([{"k": k, "j": j, "drift_oj": a, "drift_ok": b} for k, j, a, b in rows],
              (out / "rows.json").open("w"), indent=2)
    print("\n" + rep)
    print(f"\nsaved -> {out/'report.txt'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
