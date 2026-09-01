#!/usr/bin/env python
"""빈 자리 가설의 인과 검증 — 출발점이 얼마나 채워졌는가 × 페널티 유무.

가설 ④(사용자): 공간이 현재 태스크로 꽉 차 있어도, 페널티가 없으면 그냥 덮어쓰면
된다. 페널티가 있을 때만 '빈 자리가 없다'가 실제 손해로 나타난다.

설계
  출발점  task 0 만 {1000, 5000, 20000} 스텝 학습한 체크포인트 (채워진 정도가 다름)
  조건    lambda_anchor = 0 (페널티 없음)  vs  3 (있음).  p_drop 은 0.1 로 고정.
  학습    거기서 task 1 을 5000 스텝
  측정    task1 SR = 습득,  task0 SR = 보존

예측
  lambda=0 : 출발점과 무관하게 task1 을 잘 배운다 (습득 평평)
  lambda=3 : 출발점이 채워질수록 나빠진다 (습득 하락 또는 보존 붕괴)
"""
from __future__ import annotations
import argparse, json, logging, random, sys, time
from pathlib import Path
import torch
from torch.amp import GradScaler

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1

from lerobot.datasets.factory import make_dataset                    # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.sampler import EpisodeAwareSampler             # noqa: E402
from lerobot.optim.factory import make_optimizer_and_scheduler       # noqa: E402
from lerobot.policies.factory import make_policy                     # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging  # noqa: E402
from lerobot.utils.random_utils import set_seed                      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_steps", type=int, required=True, help="출발 체크포인트의 스텝")
    ap.add_argument("--lambda_anchor", type=float, required=True)
    ap.add_argument("--fill_root", default="outputs/B_fill/task0_s20000")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--p_drop", type=float, default=0.1)
    ap.add_argument("--episodes", type=int, default=45)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_batch_size", type=int, default=20)
    ap.add_argument("--guidance_w", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--mode", default="room")
    ap.add_argument("--out", default="results/B_room")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    init_logging()
    set_seed(a.seed)
    a.steps_per_task = a.steps          # B1.build_cfg 가 읽는 이름
    tag = f"s{a.start_steps}_lam{a.lambda_anchor:g}"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = get_safe_torch_device(a.device, log=True)
    ds_prefix, env_prefix = B1.suite_prefixes(a.suite)
    instr = [B1.task_instruction(f"{ds_prefix}{i}") for i in range(2)]

    start = (REPO / a.fill_root / "checkpoints" / f"{a.start_steps:06d}" / "pretrained_model")
    if not start.is_dir():
        raise SystemExit(f"출발 체크포인트 없음: {start}")
    logging.info(f"[room] {tag}  출발={start}")

    cfg = B1.build_cfg(a, 1, str(start), REPO / "outputs" / "B_room" / tag)
    meta = LeRobotDatasetMetadata(f"{ds_prefix}1")
    policy = make_policy(cfg=cfg.policy, ds_meta=meta)

    # 출발 체크포인트 자체가 teacher (= task 0 을 막 배운 시점의 모델)
    teacher = B1.snapshot(policy) if a.lambda_anchor > 0 else None

    # ── 출발점에서의 SR (학습 전) ────────────────────────────────────────────
    lang_dim = policy.dit_flow.language_embedding_projection.out_features
    sr_before = {f"task{j}": B1.rollout_sr(policy, cfg, f"{env_prefix}{j}", a, lang_dim)
                 for j in (0, 1)}
    logging.info(f"[room] {tag} 학습 전 SR {sr_before}")

    # ── task 1 학습 ──────────────────────────────────────────────────────────
    ds = make_dataset(cfg)
    train_eps = list(range(min(a.episodes, ds.meta.total_episodes)))
    sampler = EpisodeAwareSampler(
        ds.episode_data_index, episode_indices_to_use=train_eps,
        drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0), shuffle=True)
    loader = torch.utils.data.DataLoader(ds, num_workers=a.num_workers,
                                         batch_size=a.batch_size, sampler=sampler,
                                         pin_memory=True, drop_last=True)
    optimizer, sched = make_optimizer_and_scheduler(cfg, policy)
    scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    rng = random.Random(a.seed)
    policy.train()
    it = iter(loader)
    diag = []
    t0 = time.time()
    for step in range(a.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = B1.prep_batch(policy, B1.to_device(batch, device))
        x_t, t, tgt = B1.sample_fm(policy, batch)
        tail = B1.cond_tail(policy, batch)
        texts = [B1.NULL_TEXT if rng.random() < a.p_drop else instr[1]
                 for _ in range(x_t.shape[0])]
        cond = B1.make_cond(B1.encode_lang(policy, texts), tail)
        pred = policy.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond)
        l_fm = B1.fm_loss(pred, tgt)
        l_anc = (B1.anchor_against(policy, teacher, batch, tail, x_t, t, instr[0])
                 if teacher is not None else torch.zeros((), device=device))
        loss = l_fm + a.lambda_anchor * l_anc
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optimizer.grad_clip_norm,
                                       error_if_nonfinite=False)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        if sched is not None:
            sched.step()
        if step % a.log_every == 0 or step == a.steps - 1:
            d = {"step": step, "fm": float(l_fm), "anchor": float(l_anc),
                 "lr": optimizer.param_groups[0]["lr"]}
            diag.append(d)
            logging.info(f"[room] {tag} step {step} fm {d['fm']:.4f} anc {d['anchor']:.4f}")

    sr_after = {f"task{j}": B1.rollout_sr(policy, cfg, f"{env_prefix}{j}", a, lang_dim)
                for j in (0, 1)}
    res = {"tag": tag, "start_steps": a.start_steps, "lambda_anchor": a.lambda_anchor,
           "p_drop": a.p_drop, "steps": a.steps, "seed": a.seed,
           "sr_before": sr_before, "sr_after": sr_after,
           "minutes": round((time.time() - t0) / 60, 1), "diag": diag}
    with (out / "results.jsonl").open("a") as f:
        f.write(json.dumps(res) + "\n")
    logging.info(f"[room] {tag} 완료  학습전 {sr_before}  학습후 {sr_after}")
    print(json.dumps(res["sr_after"]))


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
