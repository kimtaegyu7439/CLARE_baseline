#!/usr/bin/env python
"""B5 — Self-Rollout Anchoring. 정책이 자기 자신의 replay buffer가 된다.

측정이 이끈 설계다. B1~B4 가 남긴 사실:

  (1) B1 의 앵커 손실은 0.004 로 낮은데 SR 은 0 이다. 앵커가 붙잡는 영역(o_k)에서
      드리프트 0.042, SR 이 결정되는 영역(o_j)에서 0.307 — 7.2배 격차
      (results/B1_coverage/report.txt). **앵커가 자기 홈그라운드에서 자기를 채점했다.**
  (2) 그런데 관측만 과거 것으로 바꾼 B4 는 더 나빴다(stage2 task0: B1 65 -> B4 15).
      B4 의 앵커점은 (o_j, ℓ_j, x_t) 인데 x_t = (1−t)ε + t·a_k 로 **현재 태스크 액션**
      에서 만들어진다. t~U(0,1) 이라 절반가량이 a_k 근처이고, 이는 task j 롤아웃이
      결코 지나지 않는 좌표다. teacher 도 거기서 학습된 적이 없어 답이 임의값이다.
  (3) 즉 문제는 관측 축 하나가 아니라 **flow-상태 축에도 같은 병**이 있었다.

일반화하면: 망각은 함수 전체에 균일하지 않다. **추론이 실제로 밟는 궤적(튜브) 위의
드리프트만 SR 을 결정하고, 튜브 밖 앵커는 낮은 loss 라는 가짜 안심만 준다.**
모방학습의 covariate shift 와 동형이다 — 보존의 covariate shift.

그러면 과거 태스크의 튜브 위 점을 어디서 얻는가. 액션을 저장하면 ER 이 된다.
저장할 필요가 없다 — **flow policy 는 생성 모델이라, task j 를 막 배운 시점(SR~100)의
자신이 자기 튜브를 직접 만들 수 있다.**

  태스크 j 종료 시 한 번만:
    질의 임베딩 o_m 마다 노이즈 시드 K개로 자기 ODE 를 적분(100 step, 추론과 동일)
    -> 궤적 위 W개 waypoint 에서 (x_t, t, 자기 velocity v) 를 캐시
       "전성기의 모델이 자기 시험지와 모범답안을 직접 써 둔다"

  이후 학습 내내:
    L_anchor = ‖ v_θ(x_t, t, o_m^(j), ℓ_j) − v_cached ‖²

이 한 수가 열린 구멍을 동시에 막는다
  · 튜브 위 보호      앵커점이 정의상 추론 경로 위에 있다
  · teacher OOD 소멸  생성 시점에 모든 입력이 in-distribution 이었다
  · 세대 오염 소멸    타깃이 항상 ckpt_j 원산 — rolling 증류 사슬 자체가 없다
  · teacher 상주 불필요  학습 중 모델을 하나도 더 들고 있지 않는다

비대칭 원리: 관측은 환경의 몫이라 생성할 수 없지만 **액션은 정책이 생성하는 대상 그
자체**다. 그래서 메모리는 조건 쪽(임베딩)만 담고 감독 쪽은 정책이 자가 재생한다.
ER = 답안지 저장, latent replay = 문제+답안지 저장, B5 = **문제만 저장하고 답은
전성기의 자신이 써 둔다.**

정직한 한계
  · 자가 타깃의 상한은 자기 전성기 실력이다. 그 시점의 오류도 함께 보존된다.
    expert action 을 쓰는 oracle 과의 갭이 supervision-free 의 가격이고 숨기지 않는다.
  · 튜브 '위'만 고정하므로 student 궤적이 초반에 이탈하면 앵커 밖으로 샐 수 있다.
    K 시드가 다발을 이뤄 두께를 주지만, tube adherence 진단을 함께 로깅한다.
  · 저장 임베딩의 신선도: DINOv2/CLIP 백본은 동결이지만 projection 은 학습된다.
    저장은 projection 이후 값이므로 낡을 수 있다 — 이것도 진단으로 관찰한다.

사용법
    python B5.py --smoke
    python B5.py                      # B1 기본 세팅 (5000 steps/task)
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
from B4 import select_query_points          # 질의점 선별(클러스터 중심 + 저밀도 경계)

OUT_DIR = REPO / "results" / "B5"


class SelfRolloutAnchor:
    name = "self-rollout"

    def __init__(self, lambda_a=1.0, n_query=32, n_seeds=4, n_waypoints=8,
                 ode_steps=100, pool=2048, seed=42, out_dir: Path = OUT_DIR):
        self.lambda_a = lambda_a
        self.n_query, self.n_seeds, self.n_waypoints = n_query, n_seeds, n_waypoints
        self.ode_steps, self.pool, self.seed = ode_steps, pool, seed
        self.qbank: dict[int, torch.Tensor] = {}      # (N, 2064) 관측 임베딩
        self.wp: dict[int, dict] = {}                 # 튜브 waypoint 캐시
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = (self.out_dir / "tube.jsonl").open("a")
        self.step = 0

    # ── 학습 중 앵커 ─────────────────────────────────────────────────────────
    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        if k == 0 or not self.wp:
            return torch.zeros((), device=device)
        j = rng.randrange(k)
        c = self.wp.get(j)
        if c is None:
            return torch.zeros((), device=device)

        bsz = x_t.shape[0]
        idx = torch.randint(0, c["x"].shape[0], (bsz,), device=c["x"].device)
        xq = c["x"][idx].to(device=device, dtype=x_t.dtype)
        tq = c["t"][idx].to(device=device, dtype=t.dtype)
        vq = c["v"][idx].to(device=device, dtype=x_t.dtype)
        qt = self.qbank[j][c["qidx"][idx]].to(device=device, dtype=tail.dtype)

        # 명령어는 student 자신의 인코딩을 쓴다 — "지금의 네 ℓ_j 표현으로 그때의 답을 내라"
        past = [instructions[f"task{j}"]] * bsz
        cond = B1.make_cond(B1.encode_lang(policy, past), qt)
        pred = policy.dit_flow.velocity_net(noisy_actions=xq, time=tq, global_cond=cond)

        self.step += 1
        if self.step % 500 == 0:                       # tube adherence 진단
            with torch.no_grad():
                rel = float((pred - vq).flatten(1).norm(dim=1).mean()
                            / vq.flatten(1).norm(dim=1).mean().clamp_min(1e-8))
                cos = float(F.cosine_similarity(pred.flatten(1), vq.flatten(1), dim=1).mean())
            self.log.write(json.dumps({"step": self.step, "task": k, "j": j,
                                       "tube_rel_err": rel, "tube_cos": cos}) + "\n")
            self.log.flush()
        return self.lambda_a * F.mse_loss(pred, vq)

    # ── 태스크 종료: 질의점 선별 + 자기 롤아웃 캐시 ─────────────────────────
    def on_task_end(self, policy, k, args, instructions, device, **kw):
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self._collect(policy, k, instructions, device, kw.get("dl_iter"))
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        torch.cuda.empty_cache()
        # ★ teacher 를 들고 있지 않는다. 타깃은 전부 캐시에서 나온다.

    @torch.no_grad()
    def _collect(self, policy, k, instructions, device, dl_iter):
        if dl_iter is None:
            raise RuntimeError("질의점을 모으려면 dl_iter 가 필요하다")
        was_training = policy.training
        policy.eval()

        # (a) 질의 관측 임베딩 선별
        buf, got = [], 0
        while got < self.pool:
            b = B1.prep_batch(policy, B1.to_device(next(dl_iter), device))
            tl = B1.cond_tail(policy, b)
            buf.append(tl.float().cpu()); got += tl.shape[0]
        pool = torch.cat(buf)[: self.pool]
        q = select_query_points(pool, self.n_query, seed=self.seed + k)
        self.qbank[k] = q.half()
        N = q.shape[0]

        # (b) 자기 ODE 적분 — 추론과 같은 100스텝 Euler (modeling:753-783)
        net = policy.dit_flow.velocity_net
        instr = instructions[f"task{k}"]
        wp_idx = sorted({int(round(i * (self.ode_steps - 1) / max(1, self.n_waypoints - 1)))
                         for i in range(self.n_waypoints)})
        X, T, V, QI = [], [], [], []
        dt = 1.0 / self.ode_steps
        for s in range(self.n_seeds):
            cond = B1.make_cond(B1.encode_lang(policy, [instr] * N),
                                q.to(device=device, dtype=torch.float32))
            x = net.sample_noise(N, device)
            for step in range(self.ode_steps):
                tt = torch.full((N,), step / self.ode_steps, device=device)
                v = net(noisy_actions=x, time=tt, global_cond=cond)
                if step in wp_idx:
                    X.append(x.half().cpu()); T.append(tt.half().cpu())
                    V.append(v.half().cpu()); QI.append(torch.arange(N))
                x = x + dt * v
                if net.clip_sample:
                    x = torch.clamp(x, -net.clip_sample_range, net.clip_sample_range)

        self.wp[k] = {"x": torch.cat(X), "t": torch.cat(T),
                      "v": torch.cat(V), "qidx": torch.cat(QI)}
        if was_training:
            policy.train()

        mb = (sum(v.numel() * v.element_size() for v in self.wp[k].values())
              + self.qbank[k].numel() * 2) / 1e6
        print(f"[B5] task {k} 자기튜브 캐시: 질의 {N}개 x 시드 {self.n_seeds} x "
              f"waypoint {len(wp_idx)} = {self.wp[k]['x'].shape[0]} 점, {mb:.2f} MB")

    def describe(self):
        return (f"self-rollout anchor (N={self.n_query} K={self.n_seeds} "
                f"W={self.n_waypoints} ODE={self.ode_steps}, teacher 없음)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lambda_a", type=float, default=1.0)
    ap.add_argument("--n_query", type=int, default=32, help="태스크당 질의 관측 수 N")
    ap.add_argument("--n_seeds", type=int, default=4, help="질의점당 노이즈 시드 K (튜브 두께)")
    ap.add_argument("--n_waypoints", type=int, default=8, help="궤적당 저장 waypoint W")
    ap.add_argument("--ode_steps", type=int, default=100, help="추론과 같은 적분 스텝")
    ap.add_argument("--pool", type=int, default=2048)
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    out_dir = OUT_DIR; out_dir.mkdir(parents=True, exist_ok=True)
    n_q = min(args.n_query, 8) if args.smoke else args.n_query
    pool = min(args.pool, 128) if args.smoke else args.pool
    ode = min(args.ode_steps, 20) if args.smoke else args.ode_steps

    B1.ANCHOR = SelfRolloutAnchor(
        lambda_a=args.lambda_a, n_query=n_q, n_seeds=args.n_seeds,
        n_waypoints=args.n_waypoints, ode_steps=ode, pool=pool, out_dir=out_dir)

    argv = ["B1.py", "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir), "--ckpt_root", str(REPO / "outputs" / "B5")]
    if args.smoke:
        argv.append("--smoke")
    argv += args.passthru

    json.dump({"arm": "B5", "anchor": "self_rollout", "lambda_a": args.lambda_a,
               "n_query": n_q, "n_seeds": args.n_seeds, "n_waypoints": args.n_waypoints,
               "ode_steps": ode, "pool": pool, "passthru": args.passthru},
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
