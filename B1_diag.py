#!/usr/bin/env python
"""B1 진단 — 명령어 유사도 vs 조건 민감도 vs 유지 성능.

관측된 현상: B1 이 libero_spatial 에서 task 1 만 선택적으로 잃는다
(10태스크 stage 3 기준 task0 37 / task1 3 / task2 98). 4태스크 실행에서도 같았다.

가설: 앵커는 명령어 ℓ_j 를 조건으로 걸어 teacher 출력을 맞추는데, 두 태스크의 명령어
      임베딩이 가까우면 앵커가 둘을 분리하지 못하고 뒤에 배운 쪽이 밀려난다.

측정 (학습 없음, 저장된 체크포인트만 읽는다):
  1. delta[k][j] = ‖v(x,t,o_j,ℓ_j) − v(x,t,o_j,∅)‖  — 스테이지 k 에서 태스크 j 에 대한
     조건 민감도. B1 학습 중 진단은 task 0 만 기록해서 j 별 비교가 불가능했다.
  2. sim[i][j] = CLIP 명령어 임베딩 코사인 유사도
  3. 둘과 SR 유지의 상관

사용법
    python B1_diag.py --ckpt_root outputs/B1_libero_spatial/libero_spatial_seed42_ours
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1  # 조건 조립 / FM 구성 / 전처리를 그대로 재사용한다

from lerobot.datasets.factory import make_dataset                    # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.sampler import EpisodeAwareSampler             # noqa: E402
from lerobot.policies.factory import make_policy                     # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging  # noqa: E402



def _ns(args, task_k):
    """B1.build_cfg 가 참조하는 필드만 채운 네임스페이스."""
    return argparse.Namespace(
        suite=args.suite, device=args.device, seed=args.seed,
        num_workers=0, batch_size=args.batch_size, steps_per_task=1,
        log_every=100, eval_episodes=1, eval_batch_size=1,
        mode="diag", p_drop=0.0, lambda_anchor=0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root",
                    default="outputs/B1_libero_spatial/libero_spatial_seed42_ours")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--num_tasks", type=int, default=10)
    ap.add_argument("--steps_tag", default="020000")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=4, help="δ를 평균낼 배치 수")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/B1_diag")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    init_logging()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(args.device, log=True)
    ds_prefix, _ = B1.suite_prefixes(args.suite)
    N = args.num_tasks

    # ── 1. 명령어와 CLIP 유사도 ──────────────────────────────────────────────
    instr = [B1.task_instruction(f"{ds_prefix}{j}") for j in range(N)]
    (out / "instructions.json").write_text(
        json.dumps({f"task{j}": instr[j] for j in range(N)}, indent=2, ensure_ascii=False))

    # ── 2. 존재하는 스테이지 체크포인트 찾기 ─────────────────────────────────
    stages = []
    for k in range(N):
        p = Path(args.ckpt_root) / f"task_{k}" / "checkpoints" / args.steps_tag / "pretrained_model"
        if p.is_dir():
            stages.append((k, str(p)))
    if not stages:
        raise SystemExit(f"체크포인트가 없다: {args.ckpt_root}")
    print(f"[diag] 스테이지 {[k for k, _ in stages]}")

    # ── 3. 태스크별 고정 probe 배치 (관측은 그 태스크 자신의 데이터) ─────────
    delta = np.full((N, N), np.nan)
    sim = np.full((N, N), np.nan)
    policy = None
    meta = None

    for k, ckpt in stages:
        cfg = B1.build_cfg(_ns(args, 0), 0, ckpt, Path("/tmp/b1_diag_unused"))
        # 스테이지마다 그 체크포인트의 가중치를 새로 읽는다. 정규화 통계는 태스크와
        # 무관하게 데이터셋 meta 에서 오므로 task 0 것을 계속 쓴다.
        if meta is None:
            meta = LeRobotDatasetMetadata(f"{ds_prefix}0")
        del policy
        torch.cuda.empty_cache()
        policy = make_policy(cfg=cfg.policy, ds_meta=meta)
        policy.eval()

        for j in range(k + 1):
            cfg_j = B1.build_cfg(_ns(args, j), j, ckpt, Path("/tmp/b1_diag_unused"))
            ds = make_dataset(cfg_j)
            sampler = EpisodeAwareSampler(
                ds.episode_data_index,
                drop_n_last_frames=getattr(cfg_j.policy, "drop_n_last_frames", 0),
                shuffle=True)
            loader = torch.utils.data.DataLoader(
                ds, batch_size=args.batch_size, sampler=sampler,
                num_workers=0, drop_last=True)
            torch.manual_seed(args.seed)
            it = iter(loader)

            vals = []
            with torch.no_grad():
                for _ in range(args.n_batches):
                    batch = B1.to_device(next(it), device)
                    batch = B1.prep_batch(policy, batch)
                    tail = B1.cond_tail(policy, batch)
                    x_t, t, _ = B1.sample_fm(policy, batch)
                    bsz = x_t.shape[0]
                    net = policy.dit_flow.velocity_net
                    v0 = net(noisy_actions=x_t, time=t,
                             global_cond=B1.make_cond(
                                 B1.encode_lang(policy, [B1.NULL_TEXT] * bsz), tail))
                    vj = net(noisy_actions=x_t, time=t,
                             global_cond=B1.make_cond(
                                 B1.encode_lang(policy, [instr[j]] * bsz), tail))
                    vals.append(float((vj - v0).flatten(1).norm(dim=1).mean()))
            delta[k, j] = float(np.mean(vals))
            print(f"[diag] stage {k}  task {j}  delta = {delta[k, j]:.3f}")
            del ds, loader, it

    # ── 4. CLIP 유사도 행렬 ──────────────────────────────────────────────────
    with torch.no_grad():
        emb = policy.dit_flow.language_encoder(instr)          # (N, 512) CLIP pooled
        e = torch.nn.functional.normalize(emb.float(), dim=-1)
        sim = (e @ e.T).cpu().numpy()

    np.save(out / "delta.npy", delta)
    np.save(out / "sim.npy", sim)

    # ── 5. 리포트 ────────────────────────────────────────────────────────────
    L = ["=" * 74, "B1 진단 — 조건 민감도 δ 와 명령어 유사도", "=" * 74, "",
         "δ[k][j] = ‖v(x,t,o_j,ℓ_j) − v(x,t,o_j,∅)‖   (태스크 j 자신의 관측 위에서)",
         "  δ가 0 에 가까우면 그 태스크의 명령어를 사실상 무시한다는 뜻이다.", "",
         "-" * 74, "δ 행렬 (행 = 스테이지 k, 열 = 태스크 j)", "-" * 74,
         "stage " + "".join(f"{j:>8d}" for j in range(N))]
    for k in range(N):
        if np.all(np.isnan(delta[k])):
            continue
        L.append(f"{k:>5d} " + "".join(
            f"{delta[k, j]:8.3f}" if not np.isnan(delta[k, j]) else "       ." for j in range(N)))
    L += ["", "-" * 74, "CLIP 명령어 코사인 유사도", "-" * 74,
          "task  " + "".join(f"{j:>7d}" for j in range(N))]
    for i in range(N):
        L.append(f"{i:>4d}  " + "".join(f"{sim[i, j]:7.3f}" for j in range(N)))
    L += ["", "-" * 74, "각 태스크의 최근접 이웃 (자기 제외)", "-" * 74]
    for i in range(N):
        o = [(sim[i, j], j) for j in range(N) if j != i]
        o.sort(reverse=True)
        L.append(f"  task {i}: 가장 가까운 것 task {o[0][1]} (cos {o[0][0]:.3f}), "
                 f"다음 task {o[1][1]} ({o[1][0]:.3f})")
    L += ["", "-" * 74, "명령어", "-" * 74]
    L += [f"  task {j}: {instr[j]}" for j in range(N)]

    rep = "\n".join(L)
    (out / "report.txt").write_text(rep)
    print("\n" + rep)
    print(f"\nsaved -> {out/'report.txt'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
