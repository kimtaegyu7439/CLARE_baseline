#!/usr/bin/env python
"""B4 — Query-Point Conditional Anchoring (+ 선택적 이득 보정 target).

측정이 이끈 설계다. 세 가지 사실 위에 서 있다.

  (1) 태스크 j 를 막 배워 SR 90~100 인 시점의 ‖Δ_j‖ = 0.09~0.13 이다
      (results/B_rotation/report.txt). 즉 그 시점 지식은 Δ 가 아니라 null(marginal)
      에 있다 — condition blindness 진단 그 자체다.
      => teacher 의 Δ 를 복사하는 처방은 "0 을 복사"하는 것이라 이주를 금지한다.
         B3 가 실제로 ‖Δ‖ 를 0.89 로 눌렀고 SR 60 으로 가장 나빴다.
  (2) 이후 스테이지에서 Δ 가 40~60배 자란다. 이것은 오염이 아니라 **이주**다 —
      null 이 새 marginal 로 이사한 만큼 Δ 가 반대로 자라 합 v(ℓ_j) 를 지탱한다.
      그래서 보존해야 할 불변량은 Δ 도 상대좌표도 아니라 **조건부 절대 좌표 v(x,ℓ_j)** 다.
  (3) B1 이 실패한 진짜 지점은 그 합을 **엉뚱한 영역**에서만 지킨 것이다.
      1세대 drift 가 앵커가 붙잡는 o_k 에서 0.042, SR 이 결정되는 o_j 에서 0.307 —
      7.2배 격차다 (results/B1_coverage/report.txt).

그래서 B4 의 주축은 coverage 다.

  L = L_FM + λ_a · ‖ v_θ(x_t, t, o_q^{(j)}, ℓ_j) − sg v_tgt ‖²

  o_q^{(j)} : 태스크 j 종료 시 저장한 **관측 임베딩**(tail 벡터) 수십~수백 개.
              이미지·액션·정답은 저장하지 않는다 — replay the questions, not the answers.
              선별은 무작위가 아니라 클러스터 중심 + 저밀도 경계점을 섞는다
              (전형적인 곳과 취약한 곳을 함께 고정).
  v_tgt     : 기본은 rolling teacher 의 v_T(ℓ_j). --gain 을 켜면 CFG 외삽으로
              teacher 에 남은 blindness 를 풀어 준다:
                  v_tgt = v_T(∅) + w_j · (v_T(ℓ_j) − v_T(∅))
              w_j 는 프로브에서 측정한 수축률로 갱신한다(w ← w/ĉ, clip). Δ 를 보존하는
              게 아니라 **target 생성 시점에** 역보정하는 것이다.
              teacher 의 Δ 가 아직 0 인 stage 1 에서는 외삽할 방향이 없으므로 w=1 로 둔다.

  dropout(p_drop) 과 null 스트림은 유지한다. target 보정의 기준점이자, 이주가
  일어났는지 보는 계기판이다. Δ 는 loss 의 대상이 아니라 **관측량**으로 강등된다.

B3 와의 차분이 가설을 가른다
  B3 = 얼린 타깃(ckpt_j 고정) + cond 전체 캐시 2048
  B4 = rolling teacher 타깃 + 관측 임베딩만 N개
  B4 > B3 이면 B3 의 60 은 "Δ 억압" 탓, 비슷하면 "고정점 과적합" 탓이다.

사용법
    python B4.py --smoke
    python B4.py                            # 질의점 128, 이득 off
    python B4.py --gain --w_max 2.0         # 이득 보정 켜기
    python B4.py --xt_mode prior            # 질의점에서 x_t 를 prior 노이즈로 (ablation)
"""
from __future__ import annotations

import argparse, copy, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1

OUT_DIR = REPO / "results" / "B4"


def select_query_points(tails: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """클러스터 중심 절반 + 저밀도 경계점 절반.

    보호가 필요한 곳 = 이후 태스크 분포가 덮지 않을 영역. 미래 분포는 모르니
    대리 기준으로 '전형(중심)'과 '취약(저밀도)'을 함께 잡는다.
    """
    x = tails.float()
    m = x.shape[0]
    if m <= n:
        return x
    n_c = n // 2
    g = torch.Generator().manual_seed(seed)

    # k-means++ 유사 초기화 + 소수 반복 (질의점 선별용이라 정밀할 필요 없다)
    idx = [int(torch.randint(m, (1,), generator=g))]
    d2 = ((x - x[idx[0]]) ** 2).sum(1)
    for _ in range(n_c - 1):
        p = d2.clamp_min(0) / d2.clamp_min(0).sum().clamp_min(1e-12)
        idx.append(int(torch.multinomial(p, 1, generator=g)))
        d2 = torch.minimum(d2, ((x - x[idx[-1]]) ** 2).sum(1))
    C = x[idx].clone()
    for _ in range(8):
        a = torch.cdist(x, C).argmin(1)
        for c in range(C.shape[0]):
            sel = x[a == c]
            if sel.shape[0]:
                C[c] = sel.mean(0)
    # 중심에 가장 가까운 실제 샘플을 대표로 (합성점이 아니라 실제 관측을 쓴다)
    center_idx = torch.cdist(C, x).argmin(1).unique()

    # 저밀도: k 최근접 평균거리가 큰 점
    D = torch.cdist(x, x)
    kk = min(10, m - 1)
    dens = D.topk(kk + 1, largest=False).values[:, 1:].mean(1)
    order = dens.argsort(descending=True)
    low = [int(i) for i in order if int(i) not in set(center_idx.tolist())][: n - len(center_idx)]
    return x[torch.cat([center_idx, torch.tensor(low, dtype=torch.long)])]


class QueryPointAnchor:
    name = "query-point-conditional"

    def __init__(self, lambda_a=1.0, n_query=128, gain=False, w_max=2.0, ema=0.9,
                 ctrl_every=100, xt_mode="current", pool=2048, seed=42,
                 out_dir: Path = OUT_DIR):
        self.teacher = None
        self.lambda_a = lambda_a
        self.n_query, self.pool, self.seed = n_query, pool, seed
        self.gain, self.w_max, self.ema, self.ctrl_every = gain, w_max, ema, ctrl_every
        self.xt_mode = xt_mode
        self.qbank: dict[int, torch.Tensor] = {}
        self.w: dict[int, float] = {}
        self.probe = None
        self.step = 0
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = (self.out_dir / "controller.jsonl").open("a")

    # ── 한 스텝 ──────────────────────────────────────────────────────────────
    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        if k == 0 or self.teacher is None or not self.qbank:
            return torch.zeros((), device=device)
        j = rng.randrange(k)
        q = self.qbank.get(j)
        if q is None:
            return torch.zeros((), device=device)

        net = policy.dit_flow.velocity_net
        t_net = self.teacher.dit_flow.velocity_net
        bsz = x_t.shape[0]

        idx = torch.randint(0, q.shape[0], (bsz,), device=q.device)
        qt = q[idx].to(device=device, dtype=tail.dtype)

        if self.xt_mode == "prior":
            xq = policy.dit_flow.velocity_net.sample_noise(bsz, device)
            tq = policy.dit_flow.noise_distribution.sample((bsz,)).to(device)
        else:
            xq, tq = x_t, t

        past = [instructions[f"task{j}"]] * bsz
        pred = net(noisy_actions=xq, time=tq,
                   global_cond=B1.make_cond(B1.encode_lang(policy, past), qt))
        with torch.no_grad():
            v_c = t_net(noisy_actions=xq, time=tq,
                        global_cond=B1.make_cond(B1.encode_lang(self.teacher, past), qt))
            if self.gain:
                v_u = t_net(noisy_actions=xq, time=tq,
                            global_cond=B1.make_cond(
                                B1.encode_lang(self.teacher, [B1.NULL_TEXT] * bsz), qt))
                v_tgt = v_u + self.w.get(j, 1.0) * (v_c - v_u)
            else:
                v_tgt = v_c

        self.step += 1
        if self.probe is None:
            self.probe = (qt.detach().clone(), xq.detach().clone(), tq.detach().clone())
        if self.gain and self.step % self.ctrl_every == 0:
            self._update_gains(policy, k, instructions, device)

        return self.lambda_a * F.mse_loss(pred, v_tgt)

    @torch.no_grad()
    def _update_gains(self, policy, k, instructions, device):
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = policy.training
        policy.eval()
        try:
            qt, xq, tq = self.probe
            bsz = xq.shape[0]
            net = policy.dit_flow.velocity_net
            t_net = self.teacher.dit_flow.velocity_net
            u_s = net(noisy_actions=xq, time=tq,
                      global_cond=B1.make_cond(B1.encode_lang(policy, [B1.NULL_TEXT] * bsz), qt))
            u_T = t_net(noisy_actions=xq, time=tq,
                        global_cond=B1.make_cond(
                            B1.encode_lang(self.teacher, [B1.NULL_TEXT] * bsz), qt))
            row = {"step": self.step, "task": k}
            for j in range(k):
                past = [instructions[f"task{j}"]] * bsz
                d_s = net(noisy_actions=xq, time=tq,
                          global_cond=B1.make_cond(B1.encode_lang(policy, past), qt)) - u_s
                d_T = t_net(noisy_actions=xq, time=tq,
                            global_cond=B1.make_cond(
                                B1.encode_lang(self.teacher, past), qt)) - u_T
                den = float(d_T.pow(2).sum())
                row[f"normDT_{j}"] = den ** 0.5
                if den <= 1e-8:
                    row[f"w_{j}"] = self.w.get(j, 1.0)
                    continue                      # teacher Δ 가 0 이면 외삽 방향이 없다
                c = float((d_s * d_T).sum()) / den
                w_old = self.w.get(j, 1.0)
                w_new = w_old / c if c > 1e-6 else self.w_max
                self.w[j] = float(min(max(self.ema * w_old + (1 - self.ema) * w_new, 1.0), self.w_max))
                row[f"c_{j}"] = c
                row[f"cos_{j}"] = float(F.cosine_similarity(
                    d_s.flatten(1), d_T.flatten(1), dim=1).mean())
                row[f"w_{j}"] = self.w[j]
            self.log.write(json.dumps(row) + "\n"); self.log.flush()
        finally:
            if was_training:
                policy.train()
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    # ── 태스크 종료: 질의점 수집 + teacher 교체 ─────────────────────────────
    def on_task_end(self, policy, k, args, instructions, device, **kw):
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self._collect(policy, k, device, kw.get("dl_iter"))
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        del self.teacher
        self.teacher = B1.snapshot(policy, args.teacher_bf16)
        self.w[k] = 1.0
        self.probe = None
        self.step = 0
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _collect(self, policy, k, device, dl_iter):
        if dl_iter is None:
            raise RuntimeError("질의점을 모으려면 dl_iter 가 필요하다")
        was_training = policy.training
        policy.eval()
        buf, got = [], 0
        while got < self.pool:
            b = B1.prep_batch(policy, B1.to_device(next(dl_iter), device))
            tl = B1.cond_tail(policy, b)
            buf.append(tl.float().cpu()); got += tl.shape[0]
        if was_training:
            policy.train()
        pool = torch.cat(buf)[: self.pool]
        self.qbank[k] = select_query_points(pool, self.n_query, seed=self.seed + k)
        kb = self.qbank[k].numel() * 4 / 1e3
        print(f"[B4] task {k} 질의점 {self.qbank[k].shape[0]}개 "
              f"({pool.shape[0]} 후보에서 선별), {kb:.0f} KB")

    def describe(self):
        return (f"query-point v-anchor (n={self.n_query}, xt={self.xt_mode}"
                + (f", gain w_max={self.w_max}" if self.gain else ", gain off") + ")")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lambda_a", type=float, default=1.0)
    ap.add_argument("--n_query", type=int, default=128, help="태스크당 질의점 수")
    ap.add_argument("--pool", type=int, default=2048, help="선별 후보 풀 크기")
    ap.add_argument("--gain", action="store_true", help="CFG 외삽 target 보정 켜기")
    ap.add_argument("--w_max", type=float, default=2.0)
    ap.add_argument("--ema", type=float, default=0.9)
    ap.add_argument("--ctrl_every", type=int, default=100)
    ap.add_argument("--xt_mode", choices=["current", "prior"], default="current",
                    help="질의점에서 쓸 (x_t,t). current=현재 배치 재사용(B1 과 최소 차분), "
                         "prior=순수 노이즈에서 새로 샘플(반사실 갭 축소, ablation)")
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    out_dir = OUT_DIR; out_dir.mkdir(parents=True, exist_ok=True)
    n_q = min(args.n_query, 32) if args.smoke else args.n_query
    pool = min(args.pool, 128) if args.smoke else args.pool

    B1.ANCHOR = QueryPointAnchor(
        lambda_a=args.lambda_a, n_query=n_q, gain=args.gain, w_max=args.w_max,
        ema=args.ema, ctrl_every=args.ctrl_every, xt_mode=args.xt_mode,
        pool=pool, out_dir=out_dir)

    argv = ["B1.py", "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir), "--ckpt_root", str(REPO / "outputs" / "B4")]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru

    json.dump({"arm": "B4", "anchor": "query_point_conditional",
               "lambda_a": args.lambda_a, "n_query": n_q, "pool": pool,
               "gain": args.gain, "w_max": args.w_max, "xt_mode": args.xt_mode,
               "passthru": args.passthru},
              (out_dir / "arm.json").open("w"), indent=2, ensure_ascii=False)

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
