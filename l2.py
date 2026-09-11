#!/usr/bin/env python
"""L2 — teacher-부트스트랩 x_t: 앵커 보간의 행동 성분을 task-j 것으로.

문제
    R13 앵커는 B1 이 FM 본손실용으로 만든 (x_t, t) 를 **그대로 재사용**한다.
    B1.py:798  x_t, t, target = sample_fm(policy, batch)
    B1.py:817  ANCHOR.loss(policy, batch, tail, x_t, t, k, ...)
    그런데 sample_fm 의 x_t 는  (1−t)ε + t·a_cur  이고 a_cur = batch["action"] —
    **현재 태스크의 행동**이다. 즉 "task j 를 지키는 자리"가 x 축에서만 현재
    태스크 쪽에 있다. 추론 때 task j 질문은 ODE 가 task-j 행동으로 적분해 가는
    경로 위에서 발생하므로, 지키는 자리와 묻는 자리가 어긋나 있다.

수정 (앵커 브랜치의 x_t 생성부 단 한 곳)
    ε₀ ~ N(0,I)                                   부트스트랩용, 매 스텝 fresh
    â_j = ε₀ + v_Tj(ε₀, t=0, b_j, ℓ_j)            teacher 1-스텝 행동 추정 (no_grad)
    ε′ ~ N(0,I)                                   보간용, ε₀ 와 독립
    x_t = (1−t)·ε′ + t·â_j
    t 는 B1 이 넘겨준 것을 그대로 쓴다 — R13 과 같은 분포(U(0,1), 샘플별)이고
    이렇게 해야 diff 가 x_t 생성부 한 곳으로 국한된다.

    â_j 는 flow matching 의 정의상 자연스럽다: target = a − ε 이므로 a = ε + v.
    t=0 은 추론 루프의 첫 스텝 값이고(modeling:772 t_all[:,0]=0), 시간 임베딩도
    cos(0)=1, sin(0)=0 으로 잘 정의된다.

내장 진단 (본 실행 안에서, no_grad)
    각 stage 첫 앵커 스텝 + 매 500 step, t ∈ {0.1,0.5,0.9} 고정으로
        r_A = ‖v_S − v_Tj‖   x_t 를 current 방식((1−t)ε + t·a_cur)으로
        r_B = ‖v_S − v_Tj‖   x_t 를 teacher 방식((1−t)ε + t·â_j)으로
    ε 은 A/B 에 **같은 것**을 써서 차이가 행동 성분에서만 오게 한다.
    gap = r_B − r_A 가 t=0.9 에서 크고 t=0.1 에서 ≈0 이면 x축 어긋남이 실재한다는
    직접 증거다. 전 구간 gap≈0 이면 x축은 무관 — 소거 기록(실행은 계속).

금지 준수
    R10.py / R13.py / B1.py 미수정. 훅이 없어 loss 본문을 복사한 뒤 ★L2★ 구간만
    바꿨다. â_j·b·x_t 캐시 없음(매 스텝 fresh). 행동 데이터 저장 없음 — â_j 는
    즉석 생성·즉시 폐기. L0 의 조건응답 항은 넣지 않는다(단독 효과 분리).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
import R10
import R13

OUT_DIR = REPO / "results" / "L2"
DIAG_T = (0.1, 0.5, 0.9)


class L2Anchor(R13.R13Anchor):
    """R13 과 같되 앵커의 x_t 를 teacher 부트스트랩 행동으로 보간한다."""

    name = "L2"

    def __init__(self, args):
        super().__init__(args)
        self.xt_mode = args.xt_mode
        self.diag_every = args.diag_every
        self.dg = (self.out / "xt_diag.jsonl").open("a")
        self.t_anchor = 0.0
        self._xt_sanity = False

    def describe(self):
        return (f"L2 — R13 + teacher-부트스트랩 x_t (mode={self.xt_mode}), "
                f"통계 {len(self.stats)}개, λ_lvl={self.lam_lvl}, bins={self.n_bins}")

    def on_task_start(self, policy, k, args, instructions, device, **kw):
        super().on_task_start(policy, k, args, instructions, device, **kw)
        self._xt_sanity = k > 0

    # ── 손실 ────────────────────────────────────────────────────────────────
    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        # ══ R10.R10Anchor.loss 의 복사본. ★L2★ 로 표시한 곳만 다르다.
        if k == 0 or self.teacher is None or args.lambda_anchor == 0:
            return torch.zeros((), device=device)
        t0w = time.perf_counter()
        cls = getattr(self, "cls", None)
        if cls is None:
            cls = B1.rgb_cls(policy, batch)
        n = batch["observation.state"].shape[0]
        o = cls.view(n, -1, cls.shape[-1]).float()
        tau = R10.phase_bins(batch, self.ep_len, self.n_bins).to(device)

        mu_n, sg_n = self.cur["mu"].to(device), self.cur["sigma"].to(device)
        floor = self.cur["sigma_floor"]
        h = self.cur["h"]
        if self.sample_z:
            z = torch.randn_like(o).clamp_(-3.0, 3.0)
        else:
            z = ((o - mu_n[tau]) / sg_n[tau].clamp_min(floor)).clamp_(-3.0, 3.0)
        z = z.detach()
        u = self.direction(z).detach()

        chunk = self.a.chunk_backward
        if chunk and getattr(policy.config, "use_amp", False):
            raise RuntimeError("chunk_backward 는 use_amp=True 와 함께 쓸 수 없다")

        a_cur = batch["action"]                    # ★L2★ 진단용 (손실에는 안 쓴다)
        tcol = t[:, None, None]

        lvl, stc = [], []
        teach = self.teacher

        # forward 를 tail / velocity 로 쪼갠다 (R10 의 fwd 와 수치 동치)
        def tail_of(pol, c):
            flat = c.reshape(-1, c.shape[-1]).to(x_t.dtype)
            return B1.cond_tail(pol, batch, flat)

        def vel(pol, xx, tt, cond):
            return pol.dit_flow.velocity_net(noisy_actions=xx, time=tt, global_cond=cond)

        for j in sorted(self.stats):
            st = self.stats[j]
            b_j = (st["mu"].to(device)[tau] + st["sigma"].to(device)[tau] * z).detach()
            past = [instructions[f"task{j}"]] * n

            tl_S = tail_of(policy, b_j)
            with torch.no_grad():
                tl_T = tail_of(teach, b_j)
                cond_T = B1.make_cond(B1.encode_lang(teach, past), tl_T)
            cond_S = B1.make_cond(B1.encode_lang(policy, past), tl_S)

            # ══ ★L2★ x_t 생성 — 여기가 R13 과의 유일한 차이 ═══════════════
            if self.xt_mode == "current":
                x_j = x_t                                   # R13 재현
                a_hat = None
            else:
                with torch.no_grad():
                    eps0 = torch.randn_like(x_t)
                    v0 = vel(teach, eps0, torch.zeros_like(t), cond_T)
                    a_hat = (eps0 + v0).detach()            # task-j 행동 1-스텝 추정
                eps1 = torch.randn_like(x_t)                # 보간용, eps0 와 독립
                x_new = ((1 - tcol) * eps1 + tcol * a_hat).detach()
                if self.xt_mode == "mix":
                    half = n // 2
                    x_j = torch.cat([x_t[:half], x_new[half:]], dim=0)
                else:
                    x_j = x_new
            # ═══════════════════════════════════════════════════════════════

            with torch.no_grad():
                vt0 = vel(teach, x_j, t, cond_T)
            vs0 = vel(policy, x_j, t, cond_S)
            r0 = vs0 - vt0.to(vs0.dtype)
            if self.use_struct:
                b_h = (b_j + h * u).detach()
                tl_Sh = tail_of(policy, b_h)
                with torch.no_grad():
                    tl_Th = tail_of(teach, b_h)
                    vt1 = vel(teach, x_j, t, B1.make_cond(B1.encode_lang(teach, past), tl_Th))
                vs1 = vel(policy, x_j, t,
                          B1.make_cond(B1.encode_lang(policy, past), tl_Sh))
                r1 = vs1 - vt1.to(vs1.dtype)

            L_j = self.reduce_level(r0)
            S_j = (self.reduce_struct((r1 - r0) / h) if self.use_struct
                   else torch.zeros((), device=device))
            if chunk:
                term = (self.lam_lvl * L_j if self.lam_str is None
                        else self.lam_lvl * L_j + self.lam_str * S_j)
                (args.lambda_anchor * term).backward()
                lvl.append(L_j.detach()); stc.append(S_j.detach())
            else:
                lvl.append(L_j); stc.append(S_j)

            # ══ ★L2★ sanity + 내장 진단 (첫 j 에서만) ═════════════════════
            if j == min(self.stats):
                if self._xt_sanity:
                    self._sanity_xt(k, j, a_hat, a_cur, r0, u, b_j, st, tau, device)
                    self._xt_sanity = False
                if (self.diag_every > 0 and self.step % self.diag_every == 0
                        and a_hat is not None):
                    self._diag(policy, teach, k, j, cond_S, cond_T, a_hat, a_cur, vel)
            # ═══════════════════════════════════════════════════════════════

        L_lvl = sum(lvl) / len(lvl)
        L_str = sum(stc) / len(stc)
        if not self.use_struct:
            self.lam_str = 0.0
        out = (self.lam_lvl * L_lvl if self.lam_str is None
               else self.lam_lvl * L_lvl + self.lam_str * L_str)
        if chunk:
            out = out.detach()

        self.step += 1
        self.t_anchor += time.perf_counter() - t0w
        if self.step % self.a.log_every_anchor == 0:
            self.log.write(json.dumps({
                "task": k, "step": self.step, "L_level": float(L_lvl.detach()),
                "L_struct": float(L_str.detach()), "lambda_struct": self.lam_str,
                "h": h, "n_past": len(self.stats),
                "ms_per_step": 1000 * self.t_anchor / max(self.step, 1)}) + "\n")
            self.log.flush()
            logging.info(f"[L2] k={k} step={self.step:5d} L_lvl={float(L_lvl.detach()):.4f} "
                         f"mode={self.xt_mode} "
                         f"ms/step={1000*self.t_anchor/max(self.step,1):.1f}")
        return out

    # ── sanity ──────────────────────────────────────────────────────────────
    def _sanity_xt(self, k, j, a_hat, a_cur, r0, u, b_j, st, tau, device):
        m = [f"mode={self.xt_mode}"]
        if a_hat is None:
            m.append("â 없음(current 모드)")
        else:
            with torch.no_grad():
                fin = bool(torch.isfinite(a_hat).all())
                na = float(a_hat.flatten(1).norm(dim=1).mean())
                nc = float(a_cur.flatten(1).norm(dim=1).mean())
                out_rng = float((a_hat.abs() > 1.0).float().mean())
                d = float((a_hat - a_cur).flatten(1).norm(dim=1).mean())
            m += [f"â 유한={fin}", f"‖â‖={na:.3f}", f"‖a_cur‖={nc:.3f}",
                  f"자릿수비={na/max(nc,1e-8):.2f}", f"|â|>1 비율={out_rng:.3f}",
                  f"‖â−a_cur‖={d:.3f}", f"â.grad={'ON★위반★' if a_hat.requires_grad else 'OFF'}"]
            if not (0.2 <= na / max(nc, 1e-8) <= 5.0):
                m.append("★행동 스케일 이탈 경고★")
        with torch.no_grad():
            mj = st["mu"].to(device)[tau]
            rel = float((b_j.mean(0) - mj.mean(0)).norm() / mj.mean(0).norm().clamp_min(1e-8))
        m += [f"‖b̄−μ̄_j‖/‖μ̄_j‖={rel:.4f}",
              f"‖r0‖={float(r0.detach().flatten(1).norm(dim=1).mean()):.4f}"]
        logging.info(f"[L2][sanity] task{k} j={j}  " + "  ".join(m))

    # ── 내장 진단: t 별 gap ─────────────────────────────────────────────────
    def _diag(self, policy, teach, k, j, cond_S, cond_T, a_hat, a_cur, vel):
        was = policy.training
        policy.eval()                      # dropout 이 r 을 흔들지 않게
        try:
            with torch.no_grad():
                eps = torch.randn_like(a_hat)           # A/B 에 **같은** 노이즈
                d_act = float((a_hat - a_cur).flatten(1).norm(dim=1).mean())
                rows = []
                for tv in DIAG_T:
                    tt = torch.full((a_hat.shape[0],), tv, device=a_hat.device,
                                    dtype=a_hat.dtype)
                    tc = tt[:, None, None]
                    xA = (1 - tc) * eps + tc * a_cur     # current 방식
                    xB = (1 - tc) * eps + tc * a_hat     # teacher 방식
                    rA = float((vel(policy, xA, tt, cond_S)
                                - vel(teach, xA, tt, cond_T)).flatten(1).norm(dim=1).mean())
                    rB = float((vel(policy, xB, tt, cond_S)
                                - vel(teach, xB, tt, cond_T)).flatten(1).norm(dim=1).mean())
                    rows.append({"t": tv, "r_A": rA, "r_B": rB, "gap": rB - rA})
        finally:
            if was:
                policy.train()
        rec = {"task": k, "step": self.step, "j": j, "d_action": d_act, "rows": rows}
        self.dg.write(json.dumps(rec) + "\n"); self.dg.flush()
        logging.info(f"[L2][diag] k={k} step={self.step} ‖â−a_cur‖={d_act:.3f}  " +
                     "  ".join(f"t={r['t']}: A={r['r_A']:.4f} B={r['r_B']:.4f} "
                               f"gap={r['gap']:+.4f}" for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambda_level", type=float, default=3.0)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--anchor_norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--stats_batches", type=int, default=0)
    ap.add_argument("--chunk_backward", action="store_true")
    ap.add_argument("--log_every_anchor", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--xt_mode", choices=["teacher", "current", "mix"], default="teacher")
    ap.add_argument("--diag_every", type=int, default=500)
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

    B1.ANCHOR = L2Anchor(args)

    argv = ["B1.py", "--p_drop", "0", "--guidance_w", "1.0", "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir), "--ckpt_root", str(REPO / "outputs" / "L2")]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru

    json.dump({
        "arm": "L2", "base": "R13", "xt_mode": args.xt_mode,
        "base_diff": [
            "1. R13 앵커는 B1.sample_fm 의 x_t=(1−t)ε+t·a_cur 를 재사용한다 — 행동 성분이 현재 task 것.",
            "2. L2 는 과거 j 마다 â_j = ε₀ + v_Tj(ε₀, t=0, b_j, ℓ_j) 로 task-j 행동을 1-스텝 추정한다.",
            "3. x_t = (1−t)ε′ + t·â_j 로 다시 만든다. t 는 B1 이 준 것을 그대로(같은 분포).",
            "4. 그 외 level 항·λ·reduction·b_j 생성·teacher 운용은 R13 과 동일.",
            "5. FM 본손실의 x_t 는 건드리지 않는다. â_j 는 no_grad·detach·매 스텝 폐기.",
        ],
        "diag_every": args.diag_every, "diag_t": list(DIAG_T),
        "lambda_level": args.lambda_level, "n_bins": args.n_bins,
        "anchor_norm": args.anchor_norm, "chunk_backward": args.chunk_backward,
        "teacher": "rolling (1 snapshot)", "embedding": "dinov2_cls_768_frozen",
        "p_drop": 0.0, "guidance_w": 1.0, "argv": argv,
    }, (out_dir / "l2_config.json").open("w"), indent=2, ensure_ascii=False)

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
