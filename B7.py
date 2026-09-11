#!/usr/bin/env python
"""B7 — Fresh Inversion Anchoring. 앵커 좌표를 teacher 의 홈그라운드로 순간이동.

다섯 팔이 확립한 규칙 위에 세운다.
  · 고정 삼중항 (x_t, t, v) 를 얼리면 죽는다      B3 32.5 / B4 37.5 / B5 (stage2 31.7)
  · 매 스텝 열린 좌표에서 채점해야 함수가 구속된다 B1 62.5 / B2 76.2
  · target 세대 표류는 실재하는 부차 요인          B2 − B1 = +13.7

그런데 B1 의 앵커점 x_t = (1−t)ε + t·a_k 는 **현재 태스크를 향한 직선 위**다.
teacher 의 ℓ_j-흐름이 지나가지 않는 좌표이고, 특히 t→1 구간에서는 teacher 가
감독받은 적 없는 허공이다. 거기서 나온 target 에 student 를 묶으면
(a) 앵커 예산이 추론과 무관한 영역에 쓰이고 (b) FM 학습과 불필요하게 충돌한다.

B7 은 좌표만 옮긴다. 현재 배치의 실제 액션 a_k 를 **teacher 의 ℓ_j-field 로
t=1 에서 t* 까지 역적분**해 얻은 점 x̃ 에서 채점한다.

    x = a_k
    for t in 1 → t*:   x ← x − η · v_T(x, t, o, ℓ_j)      (편도, no_grad)
    L_anchor = ‖ v_θ(x̃, t*, o, ℓ_j) − v_T(x̃, t*, o, ℓ_j) ‖²

x̃ 가 on-tube 인 이유는 구성적이다 — teacher 자신의 field 를 적분해 도달한 점이므로
정의상 그 field 의 궤적 위에 있다. 왕복(노이즈까지 갔다 되돌아오기)은 결정론적
ODE 에서 같은 점을 주면서 오차만 두 배라 채택하지 않았다.

저장: 과거 명령어(수 KB)와 rolling teacher 1개뿐. 관측·액션 캐시 없음.
좌표 개방성: a_k 가 매 스텝 새 배치이고 t*, j 를 매 스텝 재추첨하므로 B1 과 동일하게
             앵커점이 전부 다르다.

B1 과의 차분은 **앵커점의 좌표 하나**다. 2x2 의 빈칸을 채운다:
                 고정 좌표        열린 좌표
    off-tube     B3/B4 (실패)     B1/B2 (성공)
    on-tube      B5 (실패)        B7 = ?

안전망: teacher 가 아직 blind 하면(Δ_T≈0) ℓ_j-field ≈ marginal 이라 x̃ ≈ 직선점이
되어 B7 이 연속적으로 B1 로 퇴화한다. 최악이 현재 최선과 같다.

사용법
    python B7.py --smoke
    python B7.py                        # rolling teacher, n_steps=5, t_min=0.3
    python B7.py --n_steps 8 --t_min 0.5
"""
from __future__ import annotations

import argparse, copy, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1

OUT_DIR = REPO / "results" / "B7"


class FreshInversionAnchor:
    name = "fresh-inversion"

    def __init__(self, lambda_a=1.0, n_steps=5, t_min=0.3, guard=5.0,
                 log_every=500, out_dir: Path = OUT_DIR):
        self.teacher = None
        self.lambda_a, self.n_steps, self.t_min, self.guard = lambda_a, n_steps, t_min, guard
        self.log_every = log_every
        self.step = 0
        self.n_fallback = 0
        self.n_seen = 0
        self.tbin = [[0.0, 0] for _ in range(3)]   # t* 3구간별 anchor loss 누적
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = (self.out_dir / "inversion.jsonl").open("a")

    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        """과거 태스크들을 돌며 각각 역적분 앵커를 걸고 B1 의 규칙대로 모은다."""
        if k == 0 or self.teacher is None or args.lambda_anchor == 0:
            return torch.zeros((), device=device)
        terms = [self._loss_one(policy, batch, tail, x_t, t, k, j, instructions, device)
                 for j in B1.past_tasks(k, rng, getattr(args, "anchor_agg", "sum"))]
        return B1.reduce_anchor(terms, getattr(args, "anchor_agg", "sum"), device)

    def _loss_one(self, policy, batch, tail, x_t, t, k, j, instructions, device):
        """과거 스테이지 j 하나에 대한 역적분 앵커. 역적분은 j 마다 따로 해야 한다
        — teacher 의 ℓ_j 속도장을 따라 거슬러 올라가는 것이므로 j 가 바뀌면 경로가
        통째로 바뀐다. 그래서 여기서는 CLS 재사용 말고는 아낄 것이 없다."""
        bsz = x_t.shape[0]
        past = [instructions[f"task{j}"]] * bsz
        n = max(1, self.n_steps)

        # ★ t* 는 샘플마다 다르게 뽑는다. B1 의 sample_fm 이 per-sample t 를 쓰므로
        #   (B1.py:234) 여기서 배치당 스칼라 하나를 쓰면 좌표 다양성이 B1 보다 줄어
        #   "on-tube 이득"과 "다양성 손해"가 뒤섞여 해석이 불가능해진다.
        #   샘플별 step size eta_i = (1−t*_i)/n 를 쓰면 마스킹 없이 각자 자기 t*_i 에
        #   정확히 도착한다.
        t_star = (torch.rand(bsz, device=device, dtype=t.dtype)
                  * (1.0 - self.t_min) + self.t_min)
        eta = (1.0 - t_star) / n                               # (bsz,)

        t_net = self.teacher.dit_flow.velocity_net
        with torch.no_grad():
            # DINOv2 CLS 는 B1 이 스텝당 한 번 뽑아 self.cls 에 넣어 준다.
            t_tail = B1.teacher_tail(policy, self.teacher, batch, getattr(self, "cls", None))
            t_cond = B1.make_cond(B1.encode_lang(self.teacher, past), t_tail)
            x = batch["action"].to(t_cond.dtype).clone()        # t=1 에서 출발
            for i in range(n):
                tt = (1.0 - i * eta).to(t.dtype)                # (bsz,) 샘플별 시각
                x = x - eta[:, None, None].to(x.dtype) * t_net(
                    noisy_actions=x, time=tt, global_cond=t_cond)

            # ★ 폴백은 샘플 단위로. 한 샘플이 튀었다고 배치 전체를 직선점으로 돌리면
            #   on-tube 앵커 비율이 조용히 떨어져 B7 이 B1 로 퇴화한 채 돌아간다.
            bad = (~torch.isfinite(x).flatten(1).all(1)
                   | (x.flatten(1).abs().max(1).values > self.guard))
            n_bad = int(bad.sum())
            self.n_fallback += n_bad
            self.n_seen += bsz
            x_tilde = torch.where(bad[:, None, None], x_t.to(x.dtype), x)
            tt_star = torch.where(bad, t, t_star)
            v_tgt = t_net(noisy_actions=x_tilde.to(t_cond.dtype), time=tt_star,
                          global_cond=t_cond).to(x_t.dtype)

        cond_j = B1.make_cond(B1.encode_lang(policy, past), tail)
        pred = policy.dit_flow.velocity_net(
            noisy_actions=x_tilde.to(x_t.dtype), time=tt_star, global_cond=cond_j)

        self.step += 1
        if self.step % self.log_every == 0:
            with torch.no_grad():
                # displacement 만으로는 "튜브로 갔다"와 "허공으로 튀었다"를 구분 못 한다
                # (둘 다 크다). ‖x̃‖ 를 함께 남겨 x_t 와 같은 오더인지 본다.
                disp = float((x_tilde - x_t).flatten(1).norm(dim=1).mean()
                             / x_t.flatten(1).norm(dim=1).mean().clamp_min(1e-8))
                nx = float(x_tilde.flatten(1).norm(dim=1).mean())
                nxt = float(x_t.flatten(1).norm(dim=1).mean())
            self.log.write(json.dumps({
                "step": self.step, "task": k, "j": j,
                "t_star_mean": float(t_star.mean()),
                "displacement": disp, "norm_x_tilde": nx, "norm_x_t": nxt,
                "fallback_rate": self.n_fallback / max(1, self.n_seen),
                "anchor_by_t": [round(a / n, 6) if n else None for a, n in self.tbin]}) + "\n")
            self.log.flush()
        loss = F.mse_loss(pred, v_tgt)
        with torch.no_grad():   # t* 구간별 기여 — 이득이 후반 t 에 몰리는지 본다
            per = (pred - v_tgt).flatten(1).pow(2).mean(1)
            for b_i, (lo, hi) in enumerate(((0.0, 0.55), (0.55, 0.8), (0.8, 1.01))):
                m = (tt_star >= lo) & (tt_star < hi)
                if m.any():
                    self.tbin[b_i][0] += float(per[m].sum()); self.tbin[b_i][1] += int(m.sum())
        return self.lambda_a * loss   # lambda_a 는 B7 전용 배율. B1 의 λ 와 곱해진다.

    def on_task_end(self, policy, k, args, instructions, device, **kw):
        del self.teacher
        self.teacher = B1.snapshot(policy, args.teacher_bf16)
        self.step = 0
        self.n_fallback = 0
        self.n_seen = 0
        self.tbin = [[0.0, 0] for _ in range(3)]
        torch.cuda.empty_cache()

    def describe(self):
        return f"fresh inversion (n_steps={self.n_steps} t_min={self.t_min}, 저장 0)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lambda_a", type=float, default=1.0)
    ap.add_argument("--n_steps", type=int, default=5, help="역적분 Euler 스텝 수")
    ap.add_argument("--t_min", type=float, default=0.3, help="t* 하한. 후반부에 집중한다.")
    ap.add_argument("--guard", type=float, default=5.0,
                    help="|x| 폭주 임계(샘플 단위). 액션은 MIN_MAX 로 [-1,1] 이므로 5 면 충분히 느슨하다.")
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    out_dir = OUT_DIR; out_dir.mkdir(parents=True, exist_ok=True)
    B1.ANCHOR = FreshInversionAnchor(
        lambda_a=args.lambda_a, n_steps=args.n_steps, t_min=args.t_min,
        guard=args.guard, log_every=20 if args.smoke else 500, out_dir=out_dir)

    argv = ["B1.py", "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir), "--ckpt_root", str(REPO / "outputs" / "B7")]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru

    json.dump({"arm": "B7", "anchor": "fresh_inversion", "lambda_a": args.lambda_a,
               "n_steps": args.n_steps, "t_min": args.t_min, "guard": args.guard,
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
