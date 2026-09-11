#!/usr/bin/env python
"""추론 시 guidance 증폭 w 스윕 — 재학습 없이 롤아웃만 다시 한다.

배경
  이론이 예측한 역연산: 망각이 guidance 를 (1−ε)배 수축시키므로 추론에서 w=1/(1−ε)
  로 되키운다.  v = v(∅) + w·(v(ℓ) − v(∅))
  B1.py 에 cfg_guidance 훅으로 구현돼 있는데(velocity_net.forward 를 인스턴스 수준에서
  감싸 적분 100스텝 전부에 적용) **지금까지 전 팔이 w=1.0 으로만 평가됐다.**

근거가 하나 더 생겼다: results/B_errprofile/report.txt 에서 앵커 계열만
  mag = ‖v‖/‖v*‖ 가 0.92~0.93 이다 (ER 0.98, seq-FT 1.02).
속도가 일관되게 8% 작으면 100스텝 적분에서 액션이 목표에 못 미치고, 그건 부분 성공이
없는 실패다. w>1 이 정확히 그 부족분을 메운다.

학습 없음. 스테이지 3 체크포인트로 태스크 0..3 을 w 마다 롤아웃한다.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
from B_merge import ARMS

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_policy                     # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="B2λ3")
    ap.add_argument("--ws", default="1.0,1.25,1.5,2.0")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--steps_tag", default="005000")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--tasks", default="0,1,2,3")
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    out = Path(a.out or f"results/B_wsweep/{a.arm}")
    out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    ds_prefix, env_prefix = B1.suite_prefixes(a.suite)
    ws = [float(x) for x in a.ws.split(",")]
    tasks = [int(x) for x in a.tasks.split(",")]

    root = ARMS[a.arm]
    p = (REPO / root["tmpl"].format(k=a.stage)) if isinstance(root, dict) else \
        (REPO / root / f"{a.suite}_seed42_ours" / f"task_{a.stage}"
         / "checkpoints" / a.steps_tag / "pretrained_model")
    if not p.is_dir():
        raise SystemExit(f"체크포인트 없음: {p}")

    ns = argparse.Namespace(
        suite=a.suite, device=a.device, seed=a.seed, num_workers=0, batch_size=32,
        steps_per_task=1, log_every=100, eval_episodes=a.eval_episodes,
        eval_batch_size=a.eval_batch_size, mode="wsweep", p_drop=0.0, lambda_anchor=0.0,
        guidance_w=1.0, eval_after_each_task=True, teacher_bf16=False, skip_verify=True)
    cfg = B1.build_cfg(ns, 0, str(p), Path("/tmp/b_wsweep"))
    meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
    policy = make_policy(cfg=cfg.policy, ds_meta=meta)
    lang_dim = policy.dit_flow.language_embedding_projection.out_features

    res = {}
    for w in ws:
        ns.guidance_w = w
        row = {}
        for j in tasks:
            t0 = time.perf_counter()
            sr = B1.rollout_sr(policy, cfg, f"{env_prefix}{j}", ns, lang_dim)
            row[j] = sr
            print(f"[w] {a.arm}  w={w:.2f}  task{j}  SR={sr}  ({time.perf_counter()-t0:.0f}s)",
                  flush=True)
            json.dump(res | {str(w): row}, (out / "sr.json").open("w"), indent=2)
        res[str(w)] = row

    L = ["=" * 66, f"guidance 증폭 스윕 — {a.arm}, stage {a.stage}, 칸당 {a.eval_episodes} 롤아웃",
         "=" * 66, "",
         "v = v(∅) + w·(v(ℓ) − v(∅))   w=1 은 표준 조건부 추론", "",
         f"{'w':>6}" + "".join(f"{'task'+str(j):>9}" for j in tasks) + f"{'평균':>9}"]
    for w in ws:
        r = res[str(w)]
        vals = [r[j] for j in tasks if r[j] is not None]
        L.append(f"{w:>6.2f}" + "".join(
            f"{r[j]:>9.0f}" if r[j] is not None else f"{'—':>9}" for j in tasks)
            + (f"{sum(vals)/len(vals):>9.1f}" if vals else f"{'—':>9}"))
    rep = "\n".join(L)
    (out / "report.txt").write_text(rep)
    print("\n" + rep + f"\n\nsaved -> {out/'report.txt'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
