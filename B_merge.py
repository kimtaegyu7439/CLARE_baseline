#!/usr/bin/env python
"""task 1 은 어디로 갔는가 — 조건별 출력의 병합 상대를 찾는다.

관측: 앵커 계열 전부에서 task 1 만 최종 SR 이 0~40 으로 무너진다(ER 은 100).
libero_spatial 은 같은 장면·같은 물체에서 그릇 위치만 다르므로, 틀린 그릇을 집으면
부분점수 없이 SR=0 이다. 즉 연속적 열화가 아니라 **이산적 라우팅 실패**다.

두 가설
  (A) 유사성 흡수  ℓ_1 이 명령어가 가장 가까운 ℓ_0 (CLIP cos 0.952) 쪽으로 병합
  (B) 최신성 흡수  ℓ_1 이 방금 배운 ℓ_3 쪽으로 병합 (task 3 학습 후 무너졌으므로)

측정 (스테이지 3 체크포인트, 태스크 j 자신의 관측 위에서)
  err[i]     = ‖v(o_j, ℓ_i) − v*_j‖ / ‖v*_j‖
               명령어 i 를 줬을 때 태스크 j 의 정답에 얼마나 가까운가.
               argmin 이 j 가 아니면 라우팅이 틀린 것이다.
  pair[i][i'] = ‖v(o_j,ℓ_i) − v(o_j,ℓ_i')‖ / ‖v*_j‖
               조건별 출력 간 거리. 작으면 두 명령어가 같은 행동을 낸다.
  병합 상대   = argmin_{i≠j} pair[j][i]

CLIP 명령어 유사도와 대조하면 (A)/(B) 가 갈린다.
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

# 값이 문자열이면 B-계열 경로 규칙, dict 면 스테이지별 절대 경로(ER/seq-FT 는 트리가 다르다)
ARMS = {
    "B1": "outputs/B1", "B2": "outputs/B2", "B2λ3": "outputs/B2_lam3",
    "B8": "outputs/B8", "B7": "outputs/B7",
    "ER":     {"tmpl": "outputs/ER/libero_spatial/seed42/task_{k}/checkpoints/last/pretrained_model"},
    "seq-FT": {"tmpl": "outputs/E0/libero_spatial/seed_42/lam0/task_{k}/checkpoints/last/pretrained_model"},
}


def _ns(a):
    return argparse.Namespace(
        suite=a.suite, device=a.device, seed=a.seed, num_workers=0,
        batch_size=a.batch_size, steps_per_task=1, log_every=100,
        eval_episodes=1, eval_batch_size=1, mode="merge", p_drop=0.0, lambda_anchor=0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="B1,B2,B2λ3,B8")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--steps_tag", default="005000")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--n_tasks", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/B_merge")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(a.suite)
    T = a.n_tasks
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    instr = [B1.task_instruction(f"{ds_prefix}{i}") for i in range(T)]

    def ckpt(root, k):
        if isinstance(root, dict):
            return REPO / root["tmpl"].format(k=k)
        return (REPO / root / f"{a.suite}_seed42_ours" / f"task_{k}"
                / "checkpoints" / a.steps_tag / "pretrained_model")

    def load(p):
        cfg = B1.build_cfg(_ns(a), 0, str(p), Path("/tmp/b_merge"))
        pol = make_policy(cfg=cfg.policy, ds_meta=meta); pol.eval(); return pol

    seed_pol = load(ckpt(ARMS[arms[0]], 0))
    data = {}
    for j in range(T):
        cfg = B1.build_cfg(_ns(a), j, str(ckpt(ARMS[arms[0]], 0)), Path("/tmp/b_merge"))
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
    with torch.no_grad():
        emb = seed_pol.dit_flow.language_encoder(instr)
        e = F.normalize(emb.float(), dim=-1)
        sim = (e @ e.T).cpu()
    del seed_pol; torch.cuda.empty_cache()

    rows = []
    for name in arms:
        p = ckpt(ARMS[name], a.stage)
        if not p.is_dir():
            print(f"[merge] SKIP {name}"); continue
        pol = load(p)
        net = pol.dit_flow.velocity_net
        for j in range(T):
            V, tnorm = [], 0.0
            with torch.no_grad():
                for b, (x_t, t, tgt) in zip(*data[j]):
                    tl = B1.cond_tail(pol, b); n = x_t.shape[0]
                    V.append([net(noisy_actions=x_t, time=t,
                                  global_cond=B1.make_cond(
                                      B1.encode_lang(pol, [instr[i]] * n), tl)).flatten(1)
                              for i in range(T)])
                    tnorm += float(tgt.flatten(1).norm(dim=1).mean())
                tnorm /= len(V)
                tg = torch.cat([f[2].flatten(1) for f in data[j][1]])
                Vc = [torch.cat([v[i] for v in V]) for i in range(T)]
                err = [float((Vc[i] - tg).norm(dim=1).mean()) / tnorm for i in range(T)]
                pair = [[float((Vc[i] - Vc[i2]).norm(dim=1).mean()) / tnorm
                         for i2 in range(T)] for i in range(T)]
            best = int(min(range(T), key=lambda i: err[i]))
            others = [i for i in range(T) if i != j]
            partner = int(min(others, key=lambda i: pair[j][i]))
            rows.append({"arm": name, "eval_task": j, "err": err, "pair": pair,
                         "best_instr": best, "merge_partner": partner,
                         "partner_dist": pair[j][partner]})
            print(f"[merge] {name:>5} task{j}  err(ℓ_j)={err[j]:.3f}  "
                  f"최적명령어=ℓ_{best}  병합상대=ℓ_{partner} (d={pair[j][partner]:.3f})")
        del pol; torch.cuda.empty_cache()

    sr = {}
    for name in arms:
        root = ARMS[name]
        if isinstance(root, dict):
            continue
        p = REPO / "results" / root.split("/")[-1] / "metrics.json"
        if p.exists():
            sr[name] = json.load(open(p))["final_row"]
    # ER / seq-FT 는 jsonl 산출물에서 최종 행을 읽는다
    for name, path, tag in (("ER", "results/ER_task0123/er_results.jsonl", "er"),
                            ("seq-FT", "outputs/E0/libero_spatial/seed_42/e0_results.jsonl", "0")):
        f = REPO / path
        if name in arms and f.exists():
            row = {}
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r.get("run_tag") == tag and r.get("stage") == a.stage and r.get("sr") is not None:
                    row[f"task{r['probe_task']}"] = float(r["sr"])
            if row:
                sr[name] = {k2: row[k2] for k2 in sorted(row)}

    L = ["=" * 96, f"조건별 출력의 병합 상대 — stage {a.stage} 체크포인트", "=" * 96, "",
         "err[i]  = ‖v(o_j, ℓ_i) − v*_j‖ / ‖v*_j‖   명령어 i 로 태스크 j 정답에 얼마나 가까운가",
         "최적명령어 = argmin_i err[i]. j 가 아니면 라우팅 실패.",
         "병합상대  = argmin_{i≠j} ‖v(ℓ_j) − v(ℓ_i)‖. 그 거리가 작을수록 두 조건이 같은 행동.",
         ""]
    L += ["-" * 96, "CLIP 명령어 코사인 유사도", "-" * 96,
          "     " + "".join(f"{'ℓ'+str(i):>8}" for i in range(T))]
    for i in range(T):
        L.append(f"  ℓ{i} " + "".join(f"{float(sim[i, i2]):>8.3f}" for i2 in range(T)))
    L.append("")
    for name in arms:
        rs = [r for r in rows if r["arm"] == name]
        if not rs:
            continue
        L += ["-" * 96, f"{name}   (최종 행 SR: " +
              " ".join(f"t{i}={v:.0f}" for i, v in enumerate(sr.get(name, {}).values())) + ")",
              "-" * 96,
              f"{'평가태스크':>10}" + "".join(f"{'err(ℓ'+str(i)+')':>10}" for i in range(T))
              + f"{'최적':>6}{'병합상대':>10}{'거리':>8}"]
        for r in rs:
            j = r["eval_task"]
            L.append(f"{'task'+str(j):>10}"
                     + "".join(f"{r['err'][i]:>10.3f}" for i in range(T))
                     + f"{'ℓ'+str(r['best_instr']):>6}"
                     + f"{'ℓ'+str(r['merge_partner']):>10}{r['partner_dist']:>8.3f}")
        L.append("")
        L.append("  조건별 출력 쌍거리 (task 1 관측 위)")
        r1 = [r for r in rs if r["eval_task"] == 1]
        if r1:
            L.append("       " + "".join(f"{'ℓ'+str(i):>8}" for i in range(T)))
            for i in range(T):
                L.append(f"    ℓ{i} " + "".join(f"{r1[0]['pair'][i][i2]:>8.3f}" for i2 in range(T)))
        L.append("")
    rep = "\n".join(L)
    (out / "report.txt").write_text(rep)
    json.dump(rows, (out / "rows.json").open("w"), indent=2)
    print("\n" + rep + f"\n\nsaved -> {out/'report.txt'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
