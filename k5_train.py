#!/usr/bin/env python
"""K5 — R13(가우시안 샘플 앵커) + witness 유도 manifold 정련.

R13 의 문제는 i.i.d. 가우시안 표본 b0 = μ_j[τ] + σ_j[τ]⊙ε 이 실제 임베딩 manifold
(얇은 곡면) 밖에 떨어진다는 것이다. K5 는 b0 를 **동결된 witness 의 블록별 활성
통계**를 에너지로 삼아 M 스텝 경사 정련해 manifold 위로 밀어 올린 뒤 앵커한다.

    b0 = μ_j[τ] + σ_j[τ] ⊙ ε                      ε ~ N(0,I), 매 스텝 새로  (R13 그대로)
    for m in 1..M:
        E(b) = Σ_l [‖μ̂_l(b) − μ_l[τ]‖² + ‖σ̂_l(b) − σ_l[τ]‖²] / d_h
        Δb   = −η ∇_b E,  좌표별 σ 단위로 ρ 클리핑
        b   += Δb
    b_final = b.detach()  ->  R13 의 level 앵커를 그 위에서 그대로 실행

K5 = R13 + 정련 모듈. --M 0 이면 정련이 통째로 꺼진다.

★ R13 과 하나 더 다른 점 — 과거 태스크를 **매 스텝 균등 1개만 뽑는다**.
  R13/R10 은 sum(lvl)/len(lvl) 로 (1/K)·Σ_j 를 쓰는데, j 를 균등 추출하면 그것의
  불편추정량이 되어 **기대 gradient 가 같다**. λ_level 은 R13 값 그대로 둔다.
  정련이 j 하나당 M 번의 witness forward/backward 를 요구하므로 비용 통제가 필요했다.
  즉 --M 0 은 R13 과 "기댓값 수준에서" 같지 그 자리에서 문자 그대로 같지는 않다.

모델 셋의 역할
    학생             학습 대상
    rolling teacher  앵커 target. R13 과 완전히 동일(직전 태스크 종료 스냅샷 1개).
    witness          사전학습 체크포인트, 전 구간 동결. 정련 에너지 계산에만.

저장물 (태스크 j 종료 시. 원시 프레임·행동은 어떤 형태로도 저장하지 않는다)
    μ_j(τ), σ_j(τ)                        R13 과 동일 형식
    witness 활성 통계 wstats/task{j}.pt   블록별 채널 μ_l[τ], σ_l[τ]
    eps_probe.pt                          고정 노이즈 1개, 전 태스크 공용
"""
from __future__ import annotations

import argparse
import difflib
import inspect
import json
import logging
import random
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
import R10
import R13
import k5_wstats as WS

OUT_DIR = REPO / "results" / "K5"
WITNESS_CKPT = "/home/sa090180/Models/dit_flow_mt_libero_90_pretrain"


class K5Anchor(R13.R13Anchor):
    """R13 의 합성/앵커를 그대로 쓰고, 그 사이에 정련 모듈을 끼운다."""

    name = "K5"

    def __init__(self, args):
        super().__init__(args)
        self.witness = None
        self.wstats: dict[int, dict] = {}      # j -> {block: {mu, sigma, ...}}
        self.wcur: dict | None = None          # 현재 태스크 것 (태스크 종료 시 저장)
        self.blocks: list[int] | None = None
        self.eps_probe = None
        self.eta = None                        # 첫 정련 배치에서 1회 자동 설정
        self.rng_j = random.Random(args.seed if hasattr(args, "seed") else 42)
        (self.out / "wstats").mkdir(exist_ok=True)
        self.klog = (self.out / "k5.jsonl").open("a")

    def describe(self):
        return (f"K5 — 가우시안 샘플 + witness 정련(M={self.a.M}, ρ={self.a.rho_clip}, "
                f"blocks_every={self.a.blocks_every}), rolling teacher 1개 + 통계 "
                f"{len(self.stats)}개, λ_lvl={self.lam_lvl}, bins={self.n_bins}")

    # ── witness 준비 ────────────────────────────────────────────────────────
    def _ensure_witness(self, policy, device):
        if self.witness is not None:
            return
        from lerobot.policies.factory import make_policy
        from lerobot.configs.policies import PreTrainedConfig
        cfg = PreTrainedConfig.from_pretrained(self.a.witness_ckpt)
        cfg.pretrained_path = self.a.witness_ckpt
        cfg.device = str(device)
        cfg.push_to_hub = False
        cfg.input_features = dict(policy.config.input_features)
        cfg.output_features = dict(policy.config.output_features)
        w = make_policy(cfg=cfg, ds_meta=self._ds_meta)
        w.eval()
        w.requires_grad_(False)                 # 전 구간 동결
        if self.a.witness_bf16:
            w = w.to(torch.bfloat16)
        self.witness = w
        self.blocks = WS.select_blocks(w, self.a.blocks_every)
        n_all = len(WS.block_list(w))
        # eps_probe — 고정 시드 노이즈 1개. 전 태스크·전 스텝 공용.
        p = self.out / "eps_probe.pt"
        if p.exists():
            self.eps_probe = torch.load(p).to(device)
        else:
            g = torch.Generator().manual_seed(20260829)
            H = policy.config.horizon
            A = policy.config.action_feature.shape[0]
            self.eps_probe = torch.randn(1, H, A, generator=g)
            torch.save(self.eps_probe, p)
            self.eps_probe = self.eps_probe.to(device)
        logging.info(f"[K5] witness={self.a.witness_ckpt}  동결  "
                     f"블록 {n_all}개 중 {self.blocks} 사용(every={self.a.blocks_every})  "
                     f"bf16={self.a.witness_bf16}  eps_probe{tuple(self.eps_probe.shape)}  "
                     f"state={self.a.witness_state}")

    # ── 태스크 시작 ─────────────────────────────────────────────────────────
    def on_task_start(self, policy, k, args, instructions, device, **kw):
        self._ds_meta = kw["dataset"].meta
        super().on_task_start(policy, k, args, instructions, device, **kw)
        self._ensure_witness(policy, device)
        self._kw = kw                       # 태스크 종료 시 통계 수집에 재사용
        self._instr_cur = instructions[f"task{k}"]
        self._k = k
        self._refine_logged = False

    # ── 정련 ────────────────────────────────────────────────────────────────
    def _refine(self, b0, tau, j, instruction, sigma_tau):
        """b0 -> manifold 쪽으로 M 스텝. 반환 (b_final, 진단 dict)."""
        M = self.a.M
        if M <= 0 or j not in self.wstats:
            return b0, {}
        n = b0.shape[0]
        b = b0.clone()
        st = self.wstats[j]
        clip_hits, moved = 0.0, None
        e0 = e1 = None
        with WS.BlockTap(self.witness, self.blocks) as tap:
            for m in range(M):
                b = b.detach().requires_grad_(True)
                tap.acts.clear()
                WS.witness_forward(self.witness, b.reshape(-1, b.shape[-1]),
                                   instruction, self.eps_probe, tap)
                E = WS.energy(tap.acts, tau, st, self.n_bins).float()
                if m == 0:
                    e0 = float(E.detach())
                g, = torch.autograd.grad(E, b)
                # η 자동 설정 — 첫 정련 배치에서 "절반의 표본이 클리핑에 걸리는" 크기
                if self.eta is None:
                    r = (g.abs() / sigma_tau.clamp_min(1e-6)).flatten(1).max(1).values
                    self.eta = float(self.a.rho_clip / r.median().clamp_min(1e-12))
                    logging.info(f"[K5] η 자동 설정 = {self.eta:.4g}  "
                                 f"(ρ={self.a.rho_clip}, |g/σ|max 중앙값 "
                                 f"{float(r.median()):.4g})")
                d = -self.eta * g
                # 좌표별 σ 단위 클리핑
                scale = (d.abs() / sigma_tau.clamp_min(1e-6)).flatten(1).max(1).values
                fac = (self.a.rho_clip / scale.clamp_min(1e-12)).clamp(max=1.0)
                clip_hits += float((fac < 1.0).float().mean()) / M
                b = (b + d * fac[:, None, None]).detach()
            tap.acts.clear()
            with torch.no_grad():
                WS.witness_forward(self.witness, b.reshape(-1, b.shape[-1]),
                                   instruction, self.eps_probe, tap)
                e1 = float(WS.energy(tap.acts, tau, st, self.n_bins).float())
        moved = float(((b - b0).abs() / sigma_tau.clamp_min(1e-6))
                      .flatten(1).norm(dim=1).median())
        return b.detach(), {"E0": e0, "E1": e1, "clip": clip_hits, "move_sigma": moved}

    @staticmethod
    def _diversity(b0, b1):
        """med‖b1_i−b1_j‖ / med‖b0_i−b0_j‖. 1 미만이면 서로 뭉쳤다는 뜻."""
        with torch.no_grad():
            f0, f1 = b0.flatten(1), b1.flatten(1)
            d0 = torch.pdist(f0).median().clamp_min(1e-12)
            d1 = torch.pdist(f1).median()
            return float(d1 / d0)

    # ── 손실 — R10.loss 를 옮겨 오되 j 를 1개만 뽑고 정련을 끼운다 ───────────
    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        if k == 0 or self.teacher is None or args.lambda_anchor == 0:
            return torch.zeros((), device=device)
        cls = getattr(self, "cls", None)
        if cls is None:
            cls = B1.rgb_cls(policy, batch)
        n = batch["observation.state"].shape[0]
        o = cls.view(n, -1, cls.shape[-1]).float()
        tau = R10.phase_bins(batch, self.ep_len, self.n_bins).to(device)
        h = self.cur["h"]

        # ── 합성 (R13 그대로) ────────────────────────────────────────────
        z = torch.randn_like(o).clamp_(-3.0, 3.0).detach()
        u = self.direction(z).detach()

        # ── 과거 태스크 1개 균등 추출 ────────────────────────────────────
        j = self.rng_j.choice(sorted(self.stats))
        st = self.stats[j]
        sg_j = st["sigma"].to(device)[tau]
        b0 = (st["mu"].to(device)[tau] + sg_j * z).detach()

        # ── 정련 ─────────────────────────────────────────────────────────
        diag = {}
        if self.a.M > 0 and self.a.refine_frac > 0:
            nn_ = max(1, int(round(n * self.a.refine_frac)))
            sel = torch.arange(nn_, device=device)
            b_ref, diag = self._refine(b0[sel], tau[sel], j,
                                       instructions[f"task{j}"], sg_j[sel])
            b_j = b0.clone()
            b_j[sel] = b_ref
            if not self._refine_logged:
                diag["diversity"] = self._diversity(b0[sel], b_ref)
        else:
            b_j = b0

        past = [instructions[f"task{j}"]] * n
        teach = self.teacher

        def fwd(pol, c):
            flat = c.reshape(-1, c.shape[-1]).to(x_t.dtype)
            tl = B1.cond_tail(pol, batch, flat)
            return pol.dit_flow.velocity_net(
                noisy_actions=x_t, time=t,
                global_cond=B1.make_cond(B1.encode_lang(pol, past), tl))

        with torch.no_grad():
            vt0 = fwd(teach, b_j)
        vs0 = fwd(policy, b_j)
        r0 = vs0 - vt0.to(vs0.dtype)
        L_lvl = self.reduce_level(r0)
        L_str = torch.zeros((), device=device)
        self.lam_str = 0.0
        out = self.lam_lvl * L_lvl
        if self.a.chunk_backward:
            (args.lambda_anchor * out).backward()
            out = out.detach()

        if not self._refine_logged:
            d = diag
            logging.info(
                f"[K5][sanity] task{k} j={j}  M={self.a.M}  "
                f"E: {d.get('E0', float('nan')):.5g} -> {d.get('E1', float('nan')):.5g}  "
                f"({(1 - d['E1'] / max(d['E0'], 1e-12)) * 100:.1f}% 감소)"
                if d else f"[K5][sanity] task{k} j={j}  M=0 (정련 없음)")
            if d:
                dv = d.get("diversity", float("nan"))
                logging.info(
                    f"[K5][sanity] task{k}  클리핑 발동 {d['clip']*100:.1f}%  "
                    f"이동 {d['move_sigma']:.3f}σ  diversity {dv:.3f}"
                    f"{'  ⚠0.7 미만 — 붕괴 의심' if dv == dv and dv < 0.7 else ''}  "
                    f"η={self.eta}")
            logging.info(f"[K5][sanity] task{k} j={j}  ‖r0‖="
                         f"{float(r0.flatten(1).norm(dim=1).mean()):.4f}  "
                         f"‖u‖={float(u.flatten(1).norm(dim=1).mean()):.4f}  "
                         f"b 유한={bool(torch.isfinite(b_j).all())}  "
                         f"null 호출={self.null_calls}  p_drop={args.p_drop}")
            self._refine_logged = True

        self.step += 1
        if self.step % self.a.log_every_anchor == 0:
            self.klog.write(json.dumps({
                "task": k, "step": self.step, "j": j, "L_level": float(L_lvl.detach()),
                "eta": self.eta, "n_past": len(self.stats), **diag}) + "\n")
            self.klog.flush()
        return out

    # ── 태스크 종료: teacher + μ/σ (R13) + witness 활성 통계 ────────────────
    def on_task_end(self, policy, k, args, instructions, device, **kw):
        super().on_task_end(policy, k, args, instructions, device, **kw)
        self._ensure_witness(policy, device)
        t0 = time.perf_counter()
        was = policy.training; policy.eval()
        ws = WS.collect_wstats(
            policy, self.witness, self._kw["dataset"], self._kw["train_eps"],
            self._kw["cfg"], device, self.n_bins, args.batch_size, self._kw["prep"],
            self._instr_cur, self.eps_probe, self.blocks,
            workers=getattr(args, "stats_workers", 4),
            use_batch_state=(self.a.witness_state == "batch"))
        if was:
            policy.train()
        self.wstats[k] = {bi: {kk: vv for kk, vv in v.items()} for bi, v in ws.items()}
        torch.save({"block_idx": self.blocks,
                    "stats": {str(bi): v for bi, v in ws.items()},
                    "d_h": {str(bi): v["d_h"] for bi, v in ws.items()},
                    "n_frames": int(sum(float(v["count"].sum()) for v in ws.values())
                                    / max(1, len(ws)) / policy.config.horizon)},
                   self.out / "wstats" / f"task{k}.pt")

        # ── sanity 1: 판별력 E(가우시안) / E(실제) ─────────────────────────
        ratio = self._discriminability(policy, device, k)
        logging.info(f"[K5] task {k} witness 통계 {time.perf_counter()-t0:.1f}s  "
                     f"블록 {self.blocks}  판별력 E(gauss)/E(real) = {ratio:.2f}"
                     f"{'  ⚠2 미만 — witness 판별력 부족, 정련 무력 가능' if ratio < 2 else ''}")
        self.klog.write(json.dumps({"task": k, "event": "task_end",
                                    "discriminability": ratio}) + "\n")
        self.klog.flush()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _discriminability(self, policy, device, k):
        """실제 태스크 k 프레임 배치 vs 가우시안 표본 배치의 E 비율."""
        from lerobot.datasets.sampler import EpisodeAwareSampler
        ds, cfg, eps = self._kw["dataset"], self._kw["cfg"], self._kw["train_eps"]
        sp = EpisodeAwareSampler(ds.episode_data_index, episode_indices_to_use=eps,
                                 drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
                                 shuffle=True)
        dl = torch.utils.data.DataLoader(ds, num_workers=0, batch_size=64, sampler=sp)
        raw = next(iter(dl))
        b = self._kw["prep"](policy, B1.to_device(raw, device))
        cls = B1.rgb_cls(policy, b).float()
        n = b["observation.state"].shape[0]
        o = cls.view(n, -1, cls.shape[-1])
        tau = R10.phase_bins(raw, R10.episode_lengths(ds), self.n_bins).to(device)
        st = self.wstats[k]

        def E(x):
            with WS.BlockTap(self.witness, self.blocks) as tap:
                WS.witness_forward(self.witness, x.reshape(-1, x.shape[-1]),
                                   self._instr_cur, self.eps_probe, tap)
                return float(WS.energy(tap.acts, tau, st, self.n_bins).float())

        e_real = E(o)
        mu, sg = self.cur["mu"].to(device), self.cur["sigma"].to(device)
        z = torch.randn_like(o).clamp_(-3.0, 3.0)
        e_gauss = E(mu[tau] + sg[tau] * z)
        return e_gauss / max(e_real, 1e-12)


# ═════════════════════════════════════════════════════════════════════════════
def loss_diff() -> str:
    a = inspect.getsource(R10.R10Anchor.loss).splitlines()
    b = inspect.getsource(K5Anchor.loss).splitlines()
    return "\n".join(l for l in difflib.unified_diff(a, b, "R13(R10).loss", "K5.loss", n=0)
                     if l.startswith(("+", "-", "@")) and not l.startswith(("+++", "---")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # R13 상속 (기본값 동일)
    ap.add_argument("--lambda_level", type=float, default=3.0)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--anchor_norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--stats_batches", type=int, default=0)
    ap.add_argument("--stats_workers", type=int, default=4)
    ap.add_argument("--chunk_backward", action="store_true")
    ap.add_argument("--log_every_anchor", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    # K5 고유
    ap.add_argument("--M", type=int, default=8, help="정련 스텝 수. 0 이면 R13(기댓값 동일)")
    ap.add_argument("--rho_clip", type=float, default=0.3, help="좌표별 σ 단위 이동 상한 ρ")
    ap.add_argument("--blocks_every", type=int, default=4, help="몇 블록마다 하나 tap 할지")
    ap.add_argument("--refine_frac", type=float, default=1.0, help="배치 중 정련 비율")
    ap.add_argument("--witness_ckpt", default=WITNESS_CKPT)
    ap.add_argument("--witness_bf16", action="store_true",
                    help="witness 를 bf16 으로. 기본은 float32 — 에너지 정밀도 우선.")
    ap.add_argument("--witness_state", choices=["zero", "batch"], default="zero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    args.rho = 0.0
    args.warmup_steps = 0
    args.n_white = 0
    args.use_ghat_weight = False
    args.lambda_swap = 0.0

    out_dir = Path(args.out) if args.out else OUT_DIR
    args.out_dir = str(out_dir)
    args.batch_size = 32
    args.p_drop = 0.0
    out_dir.mkdir(parents=True, exist_ok=True)

    B1.ANCHOR = K5Anchor(args)

    argv = ["B1.py", "--p_drop", "0", "--guidance_w", "1.0", "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir),
            "--ckpt_root", str(REPO / "outputs" / out_dir.name)]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru
    if "--suite" in argv:
        args.suite = argv[argv.index("--suite") + 1]

    json.dump({"arm": "K5", "base": "R13", "M": args.M, "rho_clip": args.rho_clip,
               "blocks_every": args.blocks_every, "refine_frac": args.refine_frac,
               "witness_ckpt": args.witness_ckpt, "witness_bf16": args.witness_bf16,
               "witness_state": args.witness_state,
               "past_task_selection": "uniform 1 per step (unbiased estimator of (1/K)sum)",
               "eta": "auto (first refine batch, rho-clip hits ~half of samples)",
               "lambda_level": args.lambda_level, "n_bins": args.n_bins,
               "anchor_norm": args.anchor_norm, "chunk_backward": args.chunk_backward,
               "structure_term": False, "sample_z": True,
               "teacher": "rolling (1 snapshot)", "embedding": "dinov2_cls_768_frozen",
               "p_drop": 0.0, "guidance_w": 1.0, "batch_size": 32,
               "suite": args.suite, "argv": argv},
              (out_dir / "k5_config.json").open("w"), indent=2, ensure_ascii=False)
    (out_dir / "k5_loss_diff.txt").write_text(loss_diff())
    print(f"[K5] R13(R10).loss 대비 차이 -> {out_dir/'k5_loss_diff.txt'}")

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old
    R10.write_table(out_dir, arm="K5",
                    subtitle=f"가우시안 샘플 + witness 정련 (M={args.M}, ρ={args.rho_clip})")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
