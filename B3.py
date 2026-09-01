#!/usr/bin/env python
"""B3 — 타깃 캐싱 (frozen cached anchor targets).

B2 와 같은 문제(rolling teacher 의 세대 오염)를 다르게 푼다. 모델을 통째로 들고 있는
대신, 태스크 j 를 막 배운 시점에 **앵커 타깃 자체를 미리 계산해 얼려 둔다.**

  태스크 j 종료 시:  데이터로더를 N_CACHE 샘플만큼 훑으며 그 시점 모델로
                     (x_t, t, cond_j) -> v_tgt 를 계산해 저장한다.
  이후 스테이지:     저장된 (x_t, t, cond_j) 를 그대로 student 에 넣고
                     저장된 v_tgt 에 붙인다. 목표가 표류할 수 없다.

B2 와의 차이
  메모리   샘플당 (cond 2576 + x_t 112 + t 1 + v_tgt 112) float ≈ 11KB.
           2048 샘플이면 태스크당 약 23MB — 모델 780MB 보다 훨씬 싸다.
  입력     B2 는 현재 배치의 관측을 쓰고, B3 는 캐시에 얼린 조건 벡터를 쓴다.
           즉 앵커가 현재 관측 분포에 끌려다니지 않는다.

★ 정직하게 밝혀 둘 것
  캐시에 담기는 cond 는 과거 태스크의 관측에서 뽑은 **파생 특징**이다. 원본 이미지나
  액션은 아니지만 과거 데이터에서 유래한 벡터이므로, B3 는 엄밀히 말해
  rehearsal-free 가 아니라 latent replay 에 가깝다. B2 는 모델만 보관하므로
  rehearsal-free 가 유지된다. 두 팔을 비교할 때 이 차이를 함께 보고해야 한다.

그 외 모든 것은 B1 과 같다.

사용법
    python B3.py --smoke
    python B3.py                 # B1 기본 세팅 (libero_spatial 4태스크, 5000 steps)
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1

OUT_DIR = REPO / "results" / "B3"


class CachedTargets:
    """태스크 종료 시점의 앵커 타깃을 얼려 보관한다."""

    name = "cached"

    def __init__(self, n_cache: int = 2048, store_dtype=torch.float32):
        self.cache: dict[int, dict[str, torch.Tensor]] = {}
        self.n_cache = n_cache
        self.dtype = store_dtype

    # ── 학습 중 ──────────────────────────────────────────────────────────────
    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        if k == 0 or not self.cache or args.lambda_anchor == 0:
            return torch.zeros((), device=device)
        j = rng.randrange(k)                       # B1 과 같은 추첨
        c = self.cache.get(j)
        if c is None:
            return torch.zeros((), device=device)

        bsz = x_t.shape[0]
        idx = torch.randint(0, c["x_t"].shape[0], (bsz,), device=c["x_t"].device)
        cx = c["x_t"][idx].to(device=device, dtype=x_t.dtype)
        ct = c["t"][idx].to(device=device, dtype=t.dtype)
        cc = c["cond"][idx].to(device=device, dtype=x_t.dtype)
        cv = c["v_tgt"][idx].to(device=device, dtype=x_t.dtype)

        pred = policy.dit_flow.velocity_net(noisy_actions=cx, time=ct, global_cond=cc)
        return F.mse_loss(pred, cv)

    # ── 태스크 종료 시 캐시 생성 ─────────────────────────────────────────────
    @torch.no_grad()
    def on_task_end(self, policy, k, args, instructions, device, **kw):
        """★ RNG 상태를 저장했다 복원한다.

        B1.main 은 on_task_end 를 SR 평가보다 먼저 부른다(B1.py:610 vs 615).
        캐시를 만들며 sample_fm 이 torch 전역 RNG 를 소비하면 뒤이은 롤아웃의
        액션 샘플링 노이즈가 달라져, 앵커와 무관하게 SR 이 갈린다. 실제로 그래서
        B3 의 task 0 SR 이 100 이 아니라 95 로 나왔다(B1/B2 는 100).
        저장·복원하면 세 팔이 앵커 방식을 빼고는 같은 난수 경로를 탄다.
        """
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self._build_cache(policy, k, args, instructions, device, **kw)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _build_cache(self, policy, k, args, instructions, device, **kw):
        dl_iter = kw.get("dl_iter")
        prep = kw.get("prep", B1.prep_batch)
        if dl_iter is None:
            raise RuntimeError("B3 캐시를 만들려면 dl_iter 가 필요하다")

        was_training = policy.training
        policy.eval()
        xs, ts, conds, vs = [], [], [], []
        got = 0
        instr = instructions[f"task{k}"]
        while got < self.n_cache:
            batch = B1.to_device(next(dl_iter), device)
            batch = prep(policy, batch)
            tail = B1.cond_tail(policy, batch)
            x_t, t, _ = B1.sample_fm(policy, batch)
            bsz = x_t.shape[0]
            cond = B1.make_cond(B1.encode_lang(policy, [instr] * bsz), tail)
            v = policy.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond)
            xs.append(x_t.to(self.dtype).cpu()); ts.append(t.to(self.dtype).cpu())
            conds.append(cond.to(self.dtype).cpu()); vs.append(v.to(self.dtype).cpu())
            got += bsz
        if was_training:
            policy.train()

        self.cache[k] = {
            "x_t": torch.cat(xs)[: self.n_cache],
            "t": torch.cat(ts)[: self.n_cache],
            "cond": torch.cat(conds)[: self.n_cache],
            "v_tgt": torch.cat(vs)[: self.n_cache],
        }
        mb = sum(v.numel() * v.element_size() for v in self.cache[k].values()) / 1e6
        print(f"[B3] task {k} 캐시 {self.cache[k]['x_t'].shape[0]} 샘플, {mb:.1f} MB")

    def describe(self):
        n = sum(v["x_t"].shape[0] for v in self.cache.values())
        return f"cached targets — {len(self.cache)}태스크 / {n} 샘플"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_cache", type=int, default=2048,
                    help="태스크당 얼려 둘 앵커 타깃 샘플 수")
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    n_cache = 128 if args.smoke else args.n_cache
    B1.ANCHOR = CachedTargets(n_cache=n_cache)

    argv = ["B1.py",
            "--out_dir", str(out_dir),
            "--ckpt_root", str(REPO / "outputs" / "B3")]
    if args.smoke:
        argv.append("--smoke")
    argv += args.passthru

    json.dump({"arm": "B3", "anchor": "frozen_cached_targets",
               "n_cache": n_cache, "passthru": args.passthru},
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
