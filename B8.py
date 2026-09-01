#!/usr/bin/env python
"""B8 — 정답 충돌량 Ĝ 로 앵커를 가중한다. B1 + 가중 하나.

동기
  B1 의 앵커는 모든 좌표를 동등하게 보존한다. 그런데 보존이 SR 로 이어지는 정도는
  좌표마다 다르다. B5 가 그 비정렬의 극단 사례였다 — 캐시 지점에서 velocity 를
  rel_err 0.054 / cos 0.997 로 거의 완벽히 보존했는데 SR 은 0 이었다.
  보존 충실도(prediction)와 성공률(decision)이 정렬돼 있지 않다.

원리
  같은 (x_t, t, o) 에서 과거 태스크 j 의 정답과 현재 태스크 k 의 정답이 얼마나
  충돌하는지를 G^data = ‖v*_j − v*_k‖ 라 하자. 조건을 무시하는 해(blind)의 손실 바닥이
  (λμ/(λ+μ))·(G^data)² 이므로, **정답이 실제로 충돌하는 좌표에서만 조건 분화 압력이
  걸린다**. 충돌이 0 인 공유 스킬 구간은 갈라질 이유가 없다.
  그렇다면 보존도 같은 좌표에 집중하는 것이 자연스럽다 — 분화가 필요한 곳과
  보존이 걸리는 곳을 일치시킨다.

온라인 대리
  G^data 를 직접 못 재지만(과거 데이터가 없다) 두 대리가 있다:
      v*_j ≈ v_T(x_t, t, o, ℓ_j)      teacher 는 task j 를 막 배운 모델이다
      v*_k =  a_k − ε                  현재 배치의 FM 정답. B1.py 의 target 그 자체다
  따라서
      Ĝ = ‖ v_T(x_t,t,o,ℓ_j) − target ‖
  ★ 이 대리는 teacher 의 조건 분화(Δ_T)에 의존하지 않는다. teacher 가 아직 blind 해서
    Δ_T ≈ 0 인 stage 1 에서도 v_T(ℓ_j) ≈ v*_j 는 성립하므로 첫 스텝부터 유효하다.
    (‖Δ_T‖=0.09 라는 측정은 "ℓ_j 와 ∅ 의 출력이 같다"는 뜻이지 "출력이 v*_j 가
     아니다"라는 뜻이 아니다.)

가중
  w_i = clip( Ĝ_i / mean(Ĝ), w_min, w_max ) 를 배치 내에서 평균 1 로 재정규화한다.
  총 앵커 크기가 B1 과 같아지므로, B1 대비 차분이 **가중의 분포 하나**로 좁혀진다.
  (λ 를 키우는 것과 혼동되지 않는다.)

  L_anchor = mean_i [ w_i · ‖v_θ(x_t,t,o,ℓ_j)_i − v_T(x_t,t,o,ℓ_j)_i‖² ]

teacher 는 B1 과 같은 rolling 스냅샷 1개. 저장은 명령어뿐.

사용법
    python B8.py --smoke
    python B8.py                       # w_min 0.1  w_max 5.0
    python B8.py --w_max 3 --w_min 0.2
"""
from __future__ import annotations

import argparse, copy, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1

OUT_DIR = REPO / "results" / "B8"


class ConflictWeightedAnchor:
    name = "conflict-weighted"

    def __init__(self, lambda_a=1.0, w_min=0.1, w_max=5.0, log_every=500,
                 out_dir: Path = OUT_DIR):
        self.teacher = None
        self.lambda_a, self.w_min, self.w_max = lambda_a, w_min, w_max
        self.log_every = log_every
        self.step = 0
        self.fm_target = None          # B1.main 이 매 스텝 채워 준다
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = (self.out_dir / "conflict.jsonl").open("a")

    def _target(self, batch, x_t, t):
        """v*_k = a_k − ε. B1 이 넘겨 주면 그대로, 없으면 대수적으로 복원한다."""
        if self.fm_target is not None:
            return self.fm_target
        # x_t = (1−t)ε + t·a  =>  a − ε = (a − x_t)/(1−t)
        denom = (1.0 - t).clamp_min(0.05)[:, None, None]
        return (batch["action"] - x_t) / denom

    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        """과거 태스크들을 돌며 각각 Ĝ 가중 앵커를 걸고 B1 의 규칙대로 모은다.

        가중치 w 는 **과거 태스크별로 따로** 정규화한다. j 마다 충돌량 분포가 다른데
        한꺼번에 정규화하면 충돌이 큰 태스크가 배치 평균을 끌어올려 다른 태스크의
        가중이 통째로 눌린다.
        """
        if k == 0 or self.teacher is None or args.lambda_anchor == 0:
            return torch.zeros((), device=device)
        agg = getattr(args, "anchor_agg", "sum")
        terms = [self._loss_one(policy, batch, tail, x_t, t, k, j, instructions, device)
                 for j in B1.past_tasks(k, rng, agg)]
        return B1.reduce_anchor(terms, agg, device)

    def _loss_one(self, policy, batch, tail, x_t, t, k, j, instructions, device):
        bsz = x_t.shape[0]
        past = [instructions[f"task{j}"]] * bsz

        cond_j = B1.make_cond(B1.encode_lang(policy, past), tail)
        pred = policy.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond_j)

        t_net = self.teacher.dit_flow.velocity_net
        with torch.no_grad():
            t_tail = B1.teacher_tail(policy, self.teacher, batch, getattr(self, "cls", None))
            t_cond = B1.make_cond(B1.encode_lang(self.teacher, past), t_tail)
            v_tgt = t_net(noisy_actions=x_t.to(t_cond.dtype), time=t,
                          global_cond=t_cond).to(pred.dtype)

            # Ĝ — 과거 정답(teacher 의 ℓ_j 출력)과 현재 정답(FM target)의 충돌량
            v_star_k = self._target(batch, x_t, t).to(pred.dtype)
            g = (v_tgt - v_star_k).flatten(1).norm(dim=1)       # (bsz,)
            w = (g / g.mean().clamp_min(1e-8)).clamp(self.w_min, self.w_max)
            w = w / w.mean().clamp_min(1e-8)                    # 배치 평균 1 로 재정규화

        per = (pred - v_tgt).flatten(1).pow(2).mean(1)          # 샘플별 앵커 오차
        loss = (w * per).mean()

        self.step += 1
        if self.step % self.log_every == 0:
            with torch.no_grad():
                # 가중이 실제로 유효한 신호인지: Ĝ 와 앵커 오차의 상관
                gc = g - g.mean(); pc = per - per.mean()
                corr = float((gc * pc).sum()
                             / (gc.norm() * pc.norm()).clamp_min(1e-8))
            self.log.write(json.dumps({
                "step": self.step, "task": k, "j": j,
                "G_mean": float(g.mean()), "G_std": float(g.std()),
                "w_min": float(w.min()), "w_max": float(w.max()),
                "corr_G_anchorerr": corr,
                "clip_frac": float(((g / g.mean().clamp_min(1e-8) <= self.w_min)
                                    | (g / g.mean().clamp_min(1e-8) >= self.w_max))
                                   .float().mean())}) + "\n")
            self.log.flush()
        return self.lambda_a * loss

    def on_task_end(self, policy, k, args, instructions, device, **kw):
        del self.teacher
        self.teacher = B1.snapshot(policy, args.teacher_bf16)
        self.step = 0
        torch.cuda.empty_cache()

    def describe(self):
        return f"conflict-weighted anchor (w∈[{self.w_min},{self.w_max}], 평균 1 정규화)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lambda_a", type=float, default=1.0)
    ap.add_argument("--w_min", type=float, default=0.1)
    ap.add_argument("--w_max", type=float, default=5.0)
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--out", default=None,
                    help="산출물 디렉토리 이름. λ 스윕처럼 여러 개를 돌릴 때 지정한다. "
                         "conflict.jsonl 까지 함께 옮겨져 서로 덮어쓰지 않는다.")
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    out_dir = (REPO / "results" / args.out) if args.out else OUT_DIR
    ckpt_root = (REPO / "outputs" / args.out) if args.out else (REPO / "outputs" / "B8")
    out_dir.mkdir(parents=True, exist_ok=True)
    B1.ANCHOR = ConflictWeightedAnchor(
        lambda_a=args.lambda_a, w_min=args.w_min, w_max=args.w_max,
        log_every=20 if args.smoke else 500, out_dir=out_dir)

    argv = ["B1.py", "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir), "--ckpt_root", str(ckpt_root)]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru

    json.dump({"arm": "B8", "anchor": "conflict_weighted", "lambda_a": args.lambda_a,
               "w_min": args.w_min, "w_max": args.w_max, "passthru": args.passthru},
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
