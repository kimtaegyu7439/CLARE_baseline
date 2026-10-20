#!/usr/bin/env python
"""L2_V10 — DEA 가중을 **크기 항에도** 붙인다 (액션 공간).

    python L2_V10.py --beta 2.0 --eps_w 1e-3 ...

V7 dea 와 모든 것이 같고, **크기 항의 태스크 배분**만 다르다.

────────────────────────────────────────────────────────────────────────────
무엇이 바뀌나
────────────────────────────────────────────────────────────────────────────
속도(또는 액션)를 크기와 방향으로 쪼개면

    ‖p_s − p_T‖² = (r_s − r_T)²  +  2·r_s·r_T·(1 − cos α)
                   └── m 크기 ──┘   └──── e 방향 ────┘

기존 dea 는 **방향 항에만** 오차 비례 가중을 준다. 크기 항은 전 태스크에 1/N
균등이다 (노트: "Keep magnitude retention over all N tasks").

        dea     m 계수 1/N          e 계수 β·w^e_j ,  w^e_j ∝ e_j + ε_w
        V10     m 계수 w^m_j        e 계수 β·w^e_j

        w^m_j = sg[ (m_j + ε_w) / (Σ_i m_i + N·ε_w) ] ,   Σ_j w^m_j = 1

즉 **크기 오차가 큰 과거 태스크에 크기 예산을 더 준다.** 방향 쪽 w^e 와 완전히
같은 꼴이고, 쓰는 통계만 각도 e 대신 크기 m 이다.

★ 두 가중 모두 **태스크 축에서만** 정규화된다 (각 (b,h) 에서 Σ_j w_j = 1).
  그래서 1/N 이 따로 붙지 않는다 — 가중 자체가 평균 역할을 한다.

★ Σ_j w_j = 1 이어도 균등 배분과 **총량이 같지 않다**. 손실에 들어가는 건
  E[w_j·m_j] = E[m_j]/N + Cov(w_j, m_j) 이고 w 는 m 과 양의 상관이므로
  공분산만큼 더 세진다 (방향 쪽 실측: 액션 공간에서 3.4 배). 즉 V10 은
  재배분일 뿐 아니라 **크기 감독을 강화**하는 쪽이기도 하다.

★ 공간은 액션 고정 (--dir_space act). V7 과 같다.
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
from L2_V6 import DEAAnchor, dea_weights, split_mag_dir


class MagDEAAnchor(DEAAnchor):
    """DEAAnchor 의 질의·샘플링을 그대로 쓰고 loss 만 교체한다."""

    def describe(self) -> str:
        return (f"L2_V10 — dea 앵커 (크기+방향 가중, {self.a.dir_space} 공간)  "
                f"β={self.a.beta}, ε_w={self.a.eps_w}, K={self.a.codebook_k}, "
                f"λ_lvl={self.lam_lvl}")

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
        beta = self.a.beta

        # ── pass 1: 질의 재료 + 모든 e_j, m_j (그래프 없이) ───────────────
        qs = [self._query(policy, batch, x_t, t, k, instructions, j, n,
                          st_shape, tcol, device) for j in js]
        es, ms, vs = [], [], []
        with torch.no_grad():
            for q in qs:
                v_s = self._student_v(policy, q, t)
                p_s, p_t = self._pair(v_s, q, t)
                m, e, _, _, va = split_mag_dir(p_s, p_t, P)
                es.append(e); ms.append(m); vs.append(va)
        V = torch.stack(vs)
        W_e = dea_weights(torch.stack(es), V, self.a.eps_w)          # (N,B,H)
        # ★ 크기 항은 r 가 0 이어도 정의된다(나눗셈이 없다). 그래서 valid 마스크를
        #   전부 True 로 둔다 — 방향 쪽처럼 일부 칸을 빼면 Σ_j w^m_j = 1 이 깨진다.
        W_m = dea_weights(torch.stack(ms), torch.ones_like(V), self.a.eps_w)

        # ── pass 2: 태스크마다 손실을 만들고 즉시 backward ────────────────
        logs = {"m": [], "e": [], "we": [], "wm": []}
        for i, (j, q) in enumerate(zip(js, qs)):
            v_s = self._student_v(policy, q, t)
            p_s, p_t = self._pair(v_s, q, t)
            m, e, r_s, r_t, valid = split_mag_dir(p_s, p_t, P)

            L_m = (W_m[i] * m).mean()                                # ★ V10 의 변경점
            L_e = torch.where(valid, beta * W_e[i] * e,
                              torch.zeros_like(e)).mean()
            (B1.LAMBDA_ANCHOR * self.lam_lvl * (L_m + L_e)).backward()

            logs["m"].append(float(m.mean().detach()))
            logs["e"].append(float(e[valid].mean().detach()) if valid.any() else 0.0)
            logs["we"].append(float(W_e[i].mean()))
            logs["wm"].append(float(W_m[i].mean()))
            del q, v_s, p_s, p_t

        self.step += 1
        self.t_anchor += time.perf_counter() - t0w
        if self.step % self.a.log_every_anchor == 0:
            rec = {"task": k, "step": self.step, "form": "dea_mag", "N": N,
                   "m_mean": sum(logs["m"]) / N, "e_mean": sum(logs["e"]) / N,
                   "we_mean": sum(logs["we"]) / N, "we_max": max(logs["we"]),
                   "wm_mean": sum(logs["wm"]) / N, "wm_max": max(logs["wm"]),
                   "ms_per_step": 1000 * self.t_anchor / max(self.step, 1)}
            self.log.write(json.dumps(rec) + "\n"); self.log.flush()
            logging.info(f"[V10] k={k} step={self.step:5d} "
                         f"m={rec['m_mean']:.5f} e={rec['e_mean']:.5f} "
                         f"w_e={rec['we_mean']:.3f}/max {rec['we_max']:.3f} "
                         f"w_m={rec['wm_mean']:.3f}/max {rec['wm_max']:.3f} "
                         f"ms/step={rec['ms_per_step']:.1f}")
        return torch.tensor(sum(logs["m"]) / N + sum(logs["e"]) / N, device=device)


# ═════════════════════════════════════════════════════════════════════════════
#  메인
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    orig = B1.parse_args

    def parse_v10():
        p = argparse.ArgumentParser(add_help=False)
        # 큐 스크립트가 --anchor_form 을 넘긴다. V10 은 형태가 하나뿐이라 받기만
        # 하고 무시한다(아래에서 "dea" 로 고정). 안 받으면 B1 파서가 모르는
        # 인자라며 죽는다.
        p.add_argument("--anchor_form", default="dea", help=argparse.SUPPRESS)
        p.add_argument("--beta", type=float, default=2.0,
                       help="방향 항의 세기. 크기 항에는 곱하지 않는다(계수가 w^m_j).")
        p.add_argument("--eps_w", type=float, default=1e-3,
                       help="w 의 0/0 방지 상수. 크면 균등 가중에 가까워진다. "
                            "크기·방향 두 가중에 같은 값을 쓴다.")
        p.add_argument("--dir_space", choices=["vel", "act"], default="act",
                       help="기본 act (V7 과 같은 공간). m 과 e 는 항상 같은 공간이다.")
        p.add_argument("--anchor_reduce", choices=["mean", "sum"], default="mean",
                       help="V10 은 두 항 모두 Σ_j w_j = 1 이라 실제로는 쓰이지 않는다. "
                            "config 기록과 B1 호환을 위해 받는다.")
        known, rest = p.parse_known_args()
        sys.argv = [sys.argv[0]] + rest
        args = orig()
        for kk, vv in vars(known).items():
            setattr(args, kk, vv)
        args.anchor_form = "dea"          # 상속한 로그·분기가 참조한다
        return args

    args = parse_v10()
    B1._forbid_time_bin()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor = MagDEAAnchor(args)

    json.dump({
        "arm": "L2_V10 — dea (크기+방향 가중)",
        "anchor_form": "dea_mag", "beta": args.beta, "eps_w": args.eps_w,
        "dir_space": args.dir_space, "anchor_reduce": args.anchor_reduce,
        "sampler_mode": args.sampler_mode, "codebook_k": args.codebook_k,
        "lambda_level": args.lambda_level, "seed": args.seed,
        "eval_seed": args.eval_seed, "eval_episodes": args.eval_episodes,
        "steps_per_task": args.steps_per_task,
        "note": "m 계수 w^m_j (dea 와 같은 꼴, 통계만 m), e 계수 beta*w^e_j. "
                "두 가중 모두 태스크 축에서 합 1 이라 1/N 이 없다.",
    }, (out_dir / "v6_config.json").open("w"), indent=2, ensure_ascii=False)

    logging.info(f"[V10] {anchor.describe()}")
    B1._train_loop(args, anchor)


if __name__ == "__main__":
    main()
