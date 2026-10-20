#!/usr/bin/env python
"""L2_V8 — OMP 의 Directional Alignment 손실(L_DA)을 앵커에 붙인다.

    python L2_V8.py --anchor_form mse_da --lambda_da 1.0 ...   # 팔 A
    python L2_V8.py --anchor_form mag_da --lambda_da 1.0 ...   # 팔 B

구조는 L2_V7.py (= L2_V6.py) 와 **전부 같다**. 질의 생성(_query), teacher
부트스트랩, 코드북 샘플링, 태스크별 즉시 backward 까지 그대로 쓰고 **손실만**
바꾼다.

────────────────────────────────────────────────────────────────────────────
L_DA (논문 식 12)
────────────────────────────────────────────────────────────────────────────
    cos α = (v₀ · u) / (‖v₀‖ · ‖u‖)
    L_DA  = − log( (cos α + 1) / 2 )

    ‖v₀‖ 에 ε_dir ≈ 1e-6 을 더해 0 나눗셈을 막는다.

왜 MSE 가 아니라 이 형태인가. 크기 ρ 의 벡터 두 개를 ρ·u, ρ*·u* 라 두면

    ∂‖ρu − ρ*u*‖² / ∂α = 2·ρ·ρ*·sin α

이라 **ρ → 0 이면 방향 기울기도 0 으로 죽는다**. 반면

    ∂L_DA / ∂α = sin α / (cos α + 1)

는 ρ 에 아예 의존하지 않는다. 크기가 작은 구간(LIBERO 는 파지 전환 ±8 스텝에서
‖a[0:6]‖ 이 43~48% 떨어진다)에서도 방향 감독이 살아 있다는 뜻이다. 또 α → π
에서 발산하므로 정반대로 뒤집힌 예측에 강한 벌점을 준다.

★ 공간은 --dir_space 가 정한다. 기본 vel (= V8, 속도 각도). L2_V9.py 가
  act 로 고정해 들어오면 끝점 액션 a = x_j + (1−t)v 로 잰다.
  ★ 유지 항과 방향 항은 **항상 같은 공간**이다. 섞으면
    ‖·‖² = m + 2·r·r·e 항등식이 깨진다 (V7 에서 1.582 vs 1.471 로 확인).

────────────────────────────────────────────────────────────────────────────
두 팔
────────────────────────────────────────────────────────────────────────────
                                              (과거 태스크 j 마다, 1/N 평균)
    mse_da   L = L_MSE + ‖v_s − v_T‖²              + λ_DA · L_DA
    mag_da   L = L_MSE + (‖v_s‖₂ − ‖v_T‖₂)²        + λ_DA · L_DA

L_MSE 는 현재 태스크의 flow matching 손실(B1 의 loss_fm)이고 학습 루프가 낸다.
여기서 만드는 것은 그 뒤에 붙는 과거 항뿐이다.

mse_da 의 앵커 항은 V6 sqerr 과 **같은 양**이다 — 항등식

    ‖v_s − v_T‖² = (r_s − r_T)² + 2·r_s·r_T·(1 − cos α) = m + 2·r_s·r_T·e

이므로 mse_da = (V6 sqerr) + λ_DA·L_DA 이고, mag_da 는 그 중 **크기 항만**
남기고 방향 감독을 전부 L_DA 에 맡긴 설계다. 즉 두 팔의 차이는
"2·r_s·r_T·e 를 유지하느냐 버리느냐" 한 가지다.

★ 정규화 관례는 V6/V7 와 맞춘다: pose 6 축만 보고(gripper 축은 앵커 대상이
  아니다), (B,H) 평균, 과거 태스크 1/N 평균, 전체에 LAMBDA_ANCHOR·λ_lvl.
  L_DA 도 같은 앵커 묶음 안에 두므로 과거 수가 늘면 함께 세진다.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

import L2_V4_seq as B1
from L2_V6 import DEAAnchor, split_mag_dir

EPS_DIR = 1e-6      # 논문의 ε_dir — ‖v₀‖ 에 더한다
LOG_FLOOR = 1e-8    # cos → −1 에서 log 가 −inf 가 되는 것만 막는다


def da_loss(v_s, v_t, pose_dims, eps_dir: float = EPS_DIR):
    """논문 식 12.  L_DA = −log((cos α + 1)/2),  cos α = (v_T·v_s)/(‖v_T‖·‖v_s‖).

    반환 (L_DA, cos α, valid) 모두 (B,H).
    ε_dir 은 teacher 노름 쪽에 더한다 (논문: "added to ‖v₀‖").
    """
    P = v_s.shape[-1] if pose_dims is None else pose_dims
    a, b = v_s[..., :P], v_t[..., :P]
    r_s, r_t = a.norm(dim=-1), b.norm(dim=-1)
    cos = (a * b).sum(-1) / ((r_t + eps_dir) * r_s.clamp_min(eps_dir))
    cos = cos.clamp(-1.0, 1.0)
    L = -torch.log(((cos + 1.0) * 0.5).clamp_min(LOG_FLOOR))
    return L, cos, (r_t > 1e-8) & (r_s > 1e-8)


class DAAnchor(DEAAnchor):
    """DEAAnchor 의 질의·샘플링을 그대로 쓰고 loss 만 교체한다."""

    def describe(self) -> str:
        return (f"L2_V8 — {self.a.anchor_form} 앵커 (속도 공간)  "
                f"λ_DA={self.a.lambda_da}, ε_dir={self.a.eps_dir:g}, "
                f"K={self.a.codebook_k}, λ_lvl={self.lam_lvl}, "
                f"reduce={self.a.anchor_reduce}")

    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        cls = getattr(self, "cls", None)
        if cls is None:
            cls = B1.rgb_cls(policy, batch)
        self.cls = cls
        self._collect(batch, cls, device)

        if k == 0 or self.teacher is None or not self.books:
            return torch.zeros((), device=device)

        t0w = time.perf_counter()
        n = batch["observation.state"].shape[0]
        st_shape = batch["observation.state"].shape
        tcol = t[:, None, None]
        js = sorted(self.books)
        N = len(js)
        P = getattr(self.a, "pose_dims", None) or B1.POSE_DIMS
        form = self.a.anchor_form
        lam_da = self.a.lambda_da
        inv_N = (1.0 / N) if self.a.anchor_reduce == "mean" else 1.0

        logs = {"m": [], "e": [], "da": [], "cos": []}
        for j in js:
            q = self._query(policy, batch, x_t, t, k, instructions, j, n,
                            st_shape, tcol, device)
            v_s = self._student_v(policy, q, t)
            # --dir_space 가 공간을 정한다. vel 이면 (v_s, v_T), act 면
            # (a_s, a_T) = (x_j+(1−t)v_s, x_j+(1−t)v_T).  V9 는 act 고정이다.
            p_s, p_t = self._pair(v_s, q, t)

            # 유지 항 — 팔마다 다른 한 가지
            if form == "mse_da":
                keep = (p_s[..., :P] - p_t[..., :P]).pow(2).sum(-1)     # ‖·_s−·_T‖²
            elif form == "mag_da":
                keep, _, _, _, _ = split_mag_dir(p_s, p_t, P)           # (r_s−r_T)²
            else:
                raise ValueError(f"알 수 없는 anchor_form: {form}")

            # 방향 항 — 두 팔 공통. 유지 항과 **같은 공간**이다.
            L_da, cos, valid = da_loss(p_s, p_t, P, self.a.eps_dir)
            L_da = torch.where(valid, L_da, torch.zeros_like(L_da))

            L_j = inv_N * (keep.mean() + lam_da * L_da.mean())
            (B1.LAMBDA_ANCHOR * self.lam_lvl * L_j).backward()

            with torch.no_grad():
                m_j, e_j, _, _, va = split_mag_dir(p_s.detach(), p_t, P)
                logs["m"].append(float(m_j.mean()))
                logs["e"].append(float(e_j[va].mean()) if va.any() else 0.0)
                logs["da"].append(float(L_da.mean()))
                logs["cos"].append(float(cos[valid].mean()) if valid.any() else 0.0)
            del q, v_s, p_s, p_t, L_da, keep

        self.step += 1
        self.t_anchor += time.perf_counter() - t0w
        if self.step % self.a.log_every_anchor == 0:
            rec = {"task": k, "step": self.step, "form": form, "N": N,
                   "lambda_da": lam_da,
                   "m_mean": sum(logs["m"]) / N, "e_mean": sum(logs["e"]) / N,
                   "da_mean": sum(logs["da"]) / N, "cos_mean": sum(logs["cos"]) / N,
                   "ms_per_step": 1000 * self.t_anchor / max(self.step, 1)}
            self.log.write(json.dumps(rec) + "\n"); self.log.flush()
            logging.info(f"[V8][{form}] k={k} step={self.step:5d} "
                         f"m={rec['m_mean']:.4f} e={rec['e_mean']:.4f} "
                         f"L_DA={rec['da_mean']:.4f} cos={rec['cos_mean']:.4f} "
                         f"ms/step={rec['ms_per_step']:.1f}")
        # 이미 backward 했다. 반환은 로그용(그래프 없음).
        return torch.tensor(sum(logs["m"]) / N + sum(logs["da"]) / N, device=device)


# ═════════════════════════════════════════════════════════════════════════════
#  메인 — B1 의 CLI 를 그대로 쓰고 V8 인자만 더한다
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    orig = B1.parse_args

    def parse_v8():
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--anchor_form", choices=["mse_da", "mag_da"], default="mse_da",
                       help="mse_da = ‖v_s−v_T‖² + λ_DA·L_DA (팔 A). "
                            "mag_da = (‖v_s‖−‖v_T‖)² + λ_DA·L_DA (팔 B).")
        p.add_argument("--lambda_da", type=float, default=1.0,
                       help="L_DA 의 세기. 논문의 λ_DA.")
        p.add_argument("--eps_dir", type=float, default=EPS_DIR,
                       help="논문의 ε_dir — ‖v₀‖ 에 더해 0 나눗셈을 막는다.")
        p.add_argument("--anchor_reduce", choices=["mean", "sum"], default="mean",
                       help="mean = 과거 태스크에 1/N (V6/V7 와 동일). "
                            "sum = 기존 V4/V5 거동(과거 수에 비례).")
        p.add_argument("--dir_space", choices=["vel", "act"], default="vel",
                       help="vel = 속도 v 로 유지·방향 항을 낸다 (V8). "
                            "act = 끝점 액션 a = x_j + (1−t)v 로 낸다 (V9). "
                            "★ 두 항은 항상 같은 공간이다 — 섞으면 "
                            "‖·‖² = m + 2·r·r·e 항등식이 깨진다.")
        known, rest = p.parse_known_args()
        sys.argv = [sys.argv[0]] + rest
        args = orig()
        for kk, vv in vars(known).items():
            setattr(args, kk, vv)
        # V6 의 로그가 참조하는 필드. V8/V9 는 β·ε_w 를 쓰지 않는다.
        args.beta = 0.0
        args.eps_w = 0.0
        return args

    args = parse_v8()
    B1._forbid_time_bin()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor = DAAnchor(args)

    json.dump({
        "arm": f"L2_V8 — {args.anchor_form}",
        "anchor_form": args.anchor_form, "lambda_da": args.lambda_da,
        "eps_dir": args.eps_dir, "dir_space": args.dir_space,
        "anchor_reduce": args.anchor_reduce,
        "sampler_mode": args.sampler_mode, "codebook_k": args.codebook_k,
        "lambda_level": args.lambda_level, "seed": args.seed, "eval_seed": args.eval_seed,
        "eval_episodes": args.eval_episodes, "steps_per_task": args.steps_per_task,
        "note": "OMP Directional Alignment. L_DA = -log((cos a + 1)/2), "
                "cos a = (v_T.v_s)/((|v_T|+eps_dir)|v_s|). "
                "mse_da: |v_s-v_T|^2 + lam*L_DA. mag_da: (|v_s|-|v_T|)^2 + lam*L_DA.",
    }, (out_dir / "v8_config.json").open("w"), indent=2, ensure_ascii=False)

    logging.info(f"[V8] {anchor.describe()}")
    B1._train_loop(args, anchor)


if __name__ == "__main__":
    main()
