#!/usr/bin/env python
"""B 계열 체크포인트의 롤아웃 재측정 — 학습 없이 시드만 바꾼다.

B_compare.txt 의 모든 칸은 시드 42 로 20 롤아웃 한 번씩 잰 값이다. ER 은
ER_reprobe.sh 로 시드 43/44/45 를 더해 80 롤아웃으로 확인했으므로, 같은 자를
B 계열에도 대야 비교가 성립한다.

가중치는 outputs/<arm>/<suite>_seed42_ours/task_k/checkpoints/005000 고정.
--seed 가 B1.rollout_sr -> eval_policy(start_seed=) 로 들어가므로 초기 상태와
ODE 노이즈만 달라진다.

주의: --task_order 로 돌린 팔(B9)은 스테이지 k 가 실제 태스크 order[k] 다.
      메타의 task_order 를 읽어 열을 실제 태스크 번호로 되돌린다.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1

from lerobot.policies.factory import make_policy                     # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging  # noqa: E402

ARMS = {  # 표시명 -> (ckpt 루트, results 디렉토리)
    "B1":     ("outputs/B1",       "results/B1"),
    "B1λ3":   ("outputs/B1_lam3",  "results/B1_lam3"),
    "B1λ10":  ("outputs/B1_lam10", "results/B1_lam10"),
    "B1λ30":  ("outputs/B1_lam30", "results/B1_lam30"),
    "B2":     ("outputs/B2",       "results/B2"),
    "B2λ3":   ("outputs/B2_lam3",  "results/B2_lam3"),
    "B2λ10":  ("outputs/B2_lam10", "results/B2_lam10"),
    "B2λ30":  ("outputs/B2_lam30", "results/B2_lam30"),
    "B8":     ("outputs/B8",       "results/B8"),
    "B8λ3":   ("outputs/B8_lam3",  "results/B8_lam3"),
    "B8λ10":  ("outputs/B8_lam10", "results/B8_lam10"),
    "B7":     ("outputs/B7",       "results/B7"),
    "B9-1023": ("outputs/B9_1023", "results/B9_1023"),
    "B9-0321": ("outputs/B9_0321", "results/B9_0321"),
    "B9-2103": ("outputs/B9_2103", "results/B9_2103"),
    "B9-3210": ("outputs/B9_3210", "results/B9_3210"),
}


def order_of(res_dir: str, K: int) -> list[int]:
    """B9 처럼 학습 순서를 바꾼 팔의 order. 없으면 항등."""
    p = REPO / res_dir / "metrics.json"
    if p.exists():
        o = json.load(open(p)).get("task_order")
        if o:
            return [int(x) for x in o][:K]
    return list(range(K))


def _ns(a, order):
    return argparse.Namespace(
        suite=a.suite, device=a.device, seed=a.seed, num_workers=0,
        batch_size=8, steps_per_task=1, log_every=100, mode="reprobe",
        eval_episodes=a.eval_episodes, eval_batch_size=a.eval_batch_size,
        guidance_w=1.0, p_drop=0.0, lambda_anchor=0.0, task_order=order)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--steps_tag", default="005000")
    ap.add_argument("--num_tasks", type=int, default=4)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=20)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    K = a.num_tasks
    out = Path(a.out or f"results/B_reprobe/seed{a.seed}")
    out.mkdir(parents=True, exist_ok=True)
    get_safe_torch_device(a.device, log=True)
    _, env_prefix = B1.suite_prefixes(a.suite)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    rows_path = out / "rows.jsonl"

    done = set()
    if rows_path.exists():                      # 재시작 대비
        for l in rows_path.read_text().splitlines():
            r = json.loads(l); done.add((r["arm"], r["stage"], r["task"]))

    for name in arms:
        if name not in ARMS:
            print(f"[skip] 모르는 팔 {name}"); continue
        root, res_dir = ARMS[name]
        order = order_of(res_dir, K)
        for k in range(K):
            ck = (REPO / root / f"{a.suite}_seed42_ours" / f"task_{k}"
                  / "checkpoints" / a.steps_tag / "pretrained_model")
            if not ck.is_dir():
                print(f"[skip] {name} stage{k}: 체크포인트 없음"); continue
            if all((name, k, order[i]) in done for i in range(k + 1)):
                continue
            ns = _ns(a, order)
            cfg = B1.build_cfg(ns, order[k], str(ck), Path("/tmp/b_reprobe"))
            meta = LeRobotDatasetMetadata(f"{B1.suite_prefixes(a.suite)[0]}{order[k]}")
            policy = make_policy(cfg=cfg.policy, ds_meta=meta)
            lang_dim = policy.dit_flow.language_embedding_projection.out_features
            for i in range(k + 1):
                t = order[i]                     # 스테이지 i 에서 배운 실제 태스크
                if (name, k, t) in done:
                    continue
                sr = B1.rollout_sr(policy, cfg, f"{env_prefix}{t}", ns, lang_dim)
                with rows_path.open("a") as f:
                    f.write(json.dumps({"arm": name, "stage": k, "task": t,
                                        "sr": sr, "seed": a.seed}) + "\n")
                print(f"[reprobe s{a.seed}] {name:>8} stage{k} task{t}  SR={sr}", flush=True)
            del policy; torch.cuda.empty_cache()
    print(f"완료 -> {rows_path}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
