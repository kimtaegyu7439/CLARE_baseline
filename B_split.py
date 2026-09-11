#!/usr/bin/env python
"""전제 검증 — blind 해의 손실 바닥과 실제 손실, 그리고 조건 분화가 일어났는가.

이론
  같은 좌표에서 과거 정답 v_T(ℓ_j) 와 현재 정답 v*_k 가 충돌하면(충돌량 Ĝ),
  조건을 무시하는 해(v(ℓ_j)=v(ℓ_k)=u)의 손실은 아래로 내려갈 수 없다:
      min_u [ ‖u − v*_k‖² + λ‖u − v_T(ℓ_j)‖² ] = λ/(1+λ) · Ĝ²        ... (바닥)
  조건별로 갈라지면 원리상 0 까지 갈 수 있으므로, 이 바닥이 분화 압력이 된다.

그런데 실험에서는 붕괴가 일어난다. 그래서 두 가지를 직접 잰다.

  ① 실제 손실이 바닥보다 낮은가        actual = ‖v(ℓ_k)−v*_k‖² + λ‖v(ℓ_j)−v_T(ℓ_j)‖²
     actual < floor  -> 분화가 실제로 이득을 냈다
     actual ≈ floor  -> 모델이 blind 해에 머물러 있다
  ② 갈라지긴 했는가                   split = ‖v(ℓ_j) − v(ℓ_k)‖ / Ĝ
     1 에 가까우면 완전 분화, 0 이면 두 조건이 같은 출력

  ★ δ = ‖v(ℓ)−v(∅)‖ 과 다르다. 그건 "명령어 유무"의 차이이고, 여기서 필요한 것은
    "명령어끼리" 구분하는가다.

각 팔의 스테이지 k 체크포인트를 student, k-1 을 teacher 로 놓고(=j=k-1 이라
rolling/frozen 이 일치해 teacher 선택 교란이 없다) 태스크 k 데이터에서 잰다.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1

from lerobot.datasets.factory import make_dataset                    # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.sampler import EpisodeAwareSampler             # noqa: E402
from lerobot.policies.factory import make_policy                     # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging  # noqa: E402

ARMS = {  # 표시명 -> (ckpt 루트, λ, results 디렉토리)
    "B1λ1":    ("outputs/B1",       1.0,  "results/B1"),
    "B1λ3":    ("outputs/B1_lam3",  3.0,  "results/B1_lam3"),
    "B1λ10":   ("outputs/B1_lam10", 10.0, "results/B1_lam10"),
    "B1λ30":   ("outputs/B1_lam30", 30.0, "results/B1_lam30"),
    "B2λ1":    ("outputs/B2",       1.0,  "results/B2"),
    "B2λ3":    ("outputs/B2_lam3",  3.0,  "results/B2_lam3"),
    "B2λ10":   ("outputs/B2_lam10", 10.0, "results/B2_lam10"),
    "B2λ30":   ("outputs/B2_lam30", 30.0, "results/B2_lam30"),
    "B8λ1":    ("outputs/B8",       1.0,  "results/B8"),
    "B8λ3":    ("outputs/B8_lam3",  3.0,  "results/B8_lam3"),
    "B8λ10":   ("outputs/B8_lam10", 10.0, "results/B8_lam10"),
}


def logged_loss(res_dir: str, task: int, last_n: int = 5):
    """학습 중 실제로 기록된 손실. diagnostics.jsonl 의 마지막 last_n 개 평균."""
    import json as _j
    p = REPO / res_dir / "diagnostics.jsonl"
    if not p.exists():
        return None, None
    rows = [_j.loads(l) for l in p.read_text().splitlines()]
    rows = [r for r in rows if r["task"] == task]
    if not rows:
        return None, None
    tail = rows[-last_n:]
    return (sum(r["fm_loss"] for r in tail) / len(tail),
            sum(r["anchor_loss"] for r in tail) / len(tail))


def _ns(a):
    return argparse.Namespace(
        suite=a.suite, device=a.device, seed=a.seed, num_workers=0,
        batch_size=a.batch_size, steps_per_task=1, log_every=100,
        eval_episodes=1, eval_batch_size=1, mode="split", p_drop=0.0, lambda_anchor=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--steps_tag", default="005000")
    ap.add_argument("--stages", default="1,2,3")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/B_split")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(a.suite)
    stages = [int(x) for x in a.stages.split(",")]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    instr = {m: B1.task_instruction(f"{ds_prefix}{m}") for m in range(4)}

    def ckpt(root, k):
        return (REPO / root / f"{a.suite}_seed42_ours" / f"task_{k}"
                / "checkpoints" / a.steps_tag / "pretrained_model")

    def load(p):
        cfg = B1.build_cfg(_ns(a), 0, str(p), Path("/tmp/b_split"))
        pol = make_policy(cfg=cfg.policy, ds_meta=meta); pol.eval(); return pol

    # 태스크별 고정 배치 + FM 구성 (모든 팔이 같은 좌표를 본다)
    seed_pol = None
    for name in arms:
        p = ckpt(ARMS[name][0], 0)
        if p.is_dir():
            seed_pol = load(p); break
    if seed_pol is None:
        raise SystemExit("체크포인트를 찾지 못했다")
    data = {}
    for k in stages:
        cfg = B1.build_cfg(_ns(a), k, str(ckpt(ARMS[arms[0]][0], 0)), Path("/tmp/b_split"))
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
            torch.manual_seed(a.seed + k)
            x_t, t, tgt = B1.sample_fm(seed_pol, b)
            bs.append(b); fm.append((x_t, t, tgt))
        data[k] = (bs, fm)
        del ds, dl, it
    del seed_pol; torch.cuda.empty_cache()

    rows = []
    for name in arms:
        root, lam, res_dir = ARMS[name]
        for k in stages:
            j = k - 1
            ps, pt = ckpt(root, k), ckpt(root, j)
            if not (ps.is_dir() and pt.is_dir()):
                continue
            teach = load(pt)
            tv = []
            with torch.no_grad():
                for b, (x_t, t, _) in zip(*data[k]):
                    tl = B1.cond_tail(teach, b); n = x_t.shape[0]
                    tv.append(teach.dit_flow.velocity_net(
                        noisy_actions=x_t, time=t,
                        global_cond=B1.make_cond(
                            B1.encode_lang(teach, [instr[j]] * n), tl)).clone())
            del teach; torch.cuda.empty_cache()

            stud = load(ps)
            e_fm = e_anc = g2 = spl = 0.0
            nb = 0
            with torch.no_grad():
                for (b, (x_t, t, tgt), v_T) in zip(data[k][0], data[k][1], tv):
                    tl = B1.cond_tail(stud, b); n = x_t.shape[0]
                    net = stud.dit_flow.velocity_net
                    v_k = net(noisy_actions=x_t, time=t,
                              global_cond=B1.make_cond(
                                  B1.encode_lang(stud, [instr[k]] * n), tl))
                    v_j = net(noisy_actions=x_t, time=t,
                              global_cond=B1.make_cond(
                                  B1.encode_lang(stud, [instr[j]] * n), tl))
                    e_fm += float((v_k - tgt).pow(2).mean())
                    e_anc += float((v_j - v_T).pow(2).mean())
                    g2 += float((v_T - tgt).pow(2).mean())
                    spl += float((v_j - v_k).flatten(1).norm(dim=1).mean()
                                 / (v_T - tgt).flatten(1).norm(dim=1).mean().clamp_min(1e-8))
                    nb += 1
            del stud; torch.cuda.empty_cache()
            e_fm, e_anc, g2, spl = e_fm / nb, e_anc / nb, g2 / nb, spl / nb
            actual = e_fm + lam * e_anc
            floor = lam / (1.0 + lam) * g2
            lg_fm, lg_anc = logged_loss(res_dir, k)
            rows.append({"arm": name, "k": k, "lam": lam, "e_fm": e_fm, "e_anc": e_anc,
                         "G2": g2, "actual": actual, "floor": floor,
                         "ratio": actual / max(1e-12, floor), "split": spl,
                         "log_fm": lg_fm, "log_anc": lg_anc,
                         "log_total": (lg_fm + lam * lg_anc) if lg_fm is not None else None})
            r = rows[-1]
            print(f"[split] {name:>6} k={k}  actual {r['actual']:.4f}  floor {r['floor']:.4f}  "
                  f"ratio {r['ratio']:.2f}  split {r['split']:.3f}")

    sr = {}
    for name in arms:
        p = REPO / ARMS[name][2] / "metrics.json"
        if p.exists():
            sr[name] = json.load(open(p)).get("AvgSR_final")

    L = ["=" * 92,
         "blind 손실 바닥 vs 실제 손실, 그리고 조건 분화 정도", "=" * 92, "",
         "floor  = λ/(1+λ)·Ĝ²   조건을 무시하는 해가 내려갈 수 있는 최소 손실",
         "actual = ‖v(ℓ_k)−v*_k‖² + λ‖v(ℓ_j)−v_T(ℓ_j)‖²   실제 도달한 손실",
         "         (μ=1. 우리 손실이 L = L_FM + λ·L_anchor 이므로 FM 항 가중이 1이다)",
         "ratio  = actual/floor   <1 이면 분화가 이득을 냈다는 뜻, ≈1 이면 blind 해 근처",
         "학습log = 학습 중 실제로 기록된 손실 (diagnostics.jsonl 마지막 5개 평균,",
         "          fm_loss + λ·anchor_loss). 체크포인트 재계산(actual)과 교차검증용.",
         "split  = ‖v(ℓ_j)−v(ℓ_k)‖ / Ĝ   1=완전 분화, 0=두 명령어가 같은 출력",
         "",
         "-" * 92,
         f"{'arm':>7}{'λ':>5}{'k':>3}{'Ĝ²':>9}{'floor':>9}{'actual':>9}{'ratio':>7}"
         f"{'학습log':>10}{'log/floor':>11}{'split':>8}{'AvgSR':>8}",
         "-" * 92]
    for r in rows:
        lg = f"{r['log_total']:10.4f}" if r["log_total"] is not None else f"{'—':>10}"
        lr = (f"{r['log_total']/max(1e-12, r['floor']):11.2f}"
              if r["log_total"] is not None else f"{'—':>11}")
        L.append(f"{r['arm']:>7}{r['lam']:>5.0f}{r['k']:>3}{r['G2']:>9.4f}{r['floor']:>9.4f}"
                 f"{r['actual']:>9.4f}{r['ratio']:>7.2f}{lg}{lr}{r['split']:>8.3f}"
                 + (f"{sr[r['arm']]:>8.1f}" if sr.get(r['arm']) is not None else f"{'—':>8}"))
    rep = "\n".join(L)
    (out / "report.txt").write_text(rep)
    json.dump(rows, (out / "rows.json").open("w"), indent=2)
    print("\n" + rep + f"\n\nsaved -> {out/'report.txt'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
