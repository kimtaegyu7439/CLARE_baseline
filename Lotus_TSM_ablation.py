#!/usr/bin/env python
"""Lotus_TSM_ablation — TSM(태스크 통계 메모리 = 코드북) 구성요소 분해, LOTUS 셋팅.

    python Lotus_TSM_ablation.py --tsm_mode state_map --suite object --seed 7 --out results/...
    python Lotus_TSM_ablation.py --selftest            # 모델 없이 코드북/샘플러 검증

큐 전체는 bash/Lotus_TSM_h200_queue.sh, 집계는 LOTUS_TSM_ablation_report.py.

════════════════════════════════════════════════════════════════════════════
팔 (--tsm_mode) — 질의 조건 (s̃, õ) 를 **어떻게 만드느냐**만 바꾼다
════════════════════════════════════════════════════════════════════════════
    팔           셀 나누기          s̃                    õ                       앵커
    indep_gauss  없음 (셀 1개)      N(μ_s, σ_s)           N(m, σ_o)                sqerr
                                    (태스크 전체 대각 가우시안 — s̃, õ 서로 독립)
    time_bin     에피소드 진행률    N(μ_s,k, σ_s,k)       N(m_k, σ_o,k)            sqerr
                 K 구간 (K=lifelong_k)                     (대각 가우시안, A 없음)
    state_mean   state k-means K    N(μ_s,k, σ_s,k)       m_k  (셀 o 평균만)      sqerr
    state_gauss  state k-means K    N(μ_s,k, σ_s,k)       N(m_k, σ_o,k)            sqerr
                                    (셀 안 o 의 원시 표준편차 — 선형 잔차 아님, A 없음)
    state_map    state k-means K    N(μ_s,k, σ_s,k)       m_k + A_k(z̃ − z̄_k)      sqerr   ← TETHER 의 TSM
    tether       = state_map                                                      MAR (L2_V5)

  · indep_gauss → time_bin: "묶는 것" 자체가 보태는 것 (셀 없이 (s,o) 를 독립 가우시안으로 뽑으면)
  · time_bin → state_mean : "시간이 아니라 **상태**로 묶는 것"이 보태는 것
  · state_mean → state_gauss: 셀 o 평균 대신 셀 안 분산만큼 **독립 잡음**을 더하면 (A 없이) 어떻게 되나
  · state_gauss → state_map: 셀 안에서 õ 가 s̃ 를 따라가는 **국소 선형 매핑 A_k** 가 보태는 것
    (state_mean → state_map 도 같은 비교 — state_gauss 는 "잡음 vs 매핑" 을 가르는 중간 팔)
  · state_map → tether    : 방향 항 계수를 2·r_s·r_T (MSE) 에서 1/(r_T²+c) (MAR) 로 — 앵커만 바뀜

★ 네 팔이 공유하는 것
    base 체크포인트, 스테이지 순서/데모, stage 당 step, 배치, 과거 태스크당 질의 수 n=B,
    hard 셀 배정 k~Cat(π), 셀 수 K(=--lifelong_k, 기본 25), teacher 1-step â 로 x_j 를 깔고
    같은 x_j 에서 student/teacher 속도를 비교하는 절차, λ(--lambda_level), 평가 프로토콜.
★ sqerr 세 팔은 Lotus_L3_V7dea 의 anchor_form=sqerr (액션 공간, 1/N, MSE 와 항등).
  tether 는 L2_V5 --setting lotus (λ, β, c). λ 는 네 팔에 같은 값을 준다.
★ time_bin 의 "진행률" 은 frame_index / 에피소드 길이 ∈ [0,1). 에피소드 길이는 메타에
  의존하지 않고 수집한 프레임에서 에피소드별 max(frame_index)+1 로 센다 (전 프레임이
  정확히 한 번씩 들어오므로 참값이다).
★ 코드북 dict 스키마는 기존과 같다 (make_books 의 로그/to_device_cb 가 그대로 읽는다).
  time_bin/state_mean 은 A=0 이라 TSM 저장량이 state_map 보다 작다 (A 는 K·3072·16 fp16).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

MODES = ("indep_gauss", "time_bin", "state_mean", "state_gauss", "state_map", "tether")
SQERR_MODES = ("indep_gauss", "time_bin", "state_mean", "state_gauss", "state_map")
GAUSS_O_MODES = ("indep_gauss", "time_bin", "state_gauss")   # õ 에 σ_o 잡음을 더하는 팔


# ════════════════════════════════════════════════════════════════════════════
#  time_bin 코드북 — LC 에 의존하지 않는다 (selftest 가 모델 없이 돈다)
# ════════════════════════════════════════════════════════════════════════════
def _per_cell(v: torch.Tensor, lab: torch.Tensor, K: int):
    """v (N,D), lab (N,) -> (평균, 표준편차) 각 (K,D). float64 누적."""
    n_k = torch.bincount(lab, minlength=K).double().clamp_min(1.0)[:, None]
    m = torch.zeros(K, v.shape[1], device=v.device, dtype=torch.float64)
    sq = torch.zeros_like(m)
    m.index_add_(0, lab, v.double())
    sq.index_add_(0, lab, v.double() ** 2)
    mu = m / n_k
    sd = (sq / n_k - mu ** 2).clamp_min(0.0).sqrt()
    return mu.float(), sd.float()


def build_codebook_timebin(s: torch.Tensor, o: torch.Tensor, K: int,
                           prog: torch.Tensor, h_scale: float = 1.0) -> dict:
    """진행률 prog∈[0,1) 를 K 등분한 구간을 셀로 쓴다. 셀마다 (s, o) 대각 가우시안.

    s (N,16), o (N,3072), prog (N,). 반환 dict 는 build_codebook 과 같은 키를 가진다.
    A=0 (매핑 없음), gauss_o=True (샘플러가 õ 에 σ_o 잡음을 더한다).
    """
    N = s.shape[0]
    mean_s, std_s = s.mean(0), s.std(0).clamp_min(1e-8)
    zs = (s - mean_s) / std_s
    lab = (prog.clamp(0.0, 1.0 - 1e-6) * K).long()                  # (N,) 구간 번호
    uniq, lab = torch.unique(lab, return_inverse=True)              # 빈 구간은 압축
    K_eff = int(uniq.numel())
    n_k = torch.bincount(lab, minlength=K_eff).float()
    pi = n_k / n_k.sum()
    mu_s, sig_s = _per_cell(s, lab, K_eff)
    m_o, sig_o = _per_cell(o, lab, K_eff)
    c, _ = _per_cell(zs, lab, K_eff)                                # 표준화 s 공간의 셀 중심
    sig_s = sig_s.clamp_min(1e-3 * float(s.std()))
    sig_o = sig_o.clamp_min(1e-3 * float(o.std()))
    h = float(torch.cdist(zs, c).min(1).values.median().clamp_min(1e-6)) * h_scale
    D, Ds = o.shape[1], s.shape[1]
    out = {"h_scale": h_scale, "pi": pi.cpu(), "mu_s": mu_s.cpu(), "sig_s": sig_s.cpu(),
           "c": c.cpu(), "m": m_o.cpu(), "sig_o": sig_o.cpu(),
           "mean_s": mean_s.cpu(), "std_s": std_s.cpu(), "h": h,
           "K_eff": K_eff, "N": N, "n_k": n_k.cpu(),
           "inertia": float((zs - c[lab]).pow(2).sum()), "n_merged": K - K_eff,
           "A": torch.zeros(K_eff, D, Ds, dtype=torch.float16),
           "zbar": c.cpu(), "ridge_rho": 0.0, "n_demoted": K_eff,
           "var_ratio_cells": torch.ones(K_eff), "A_fro": torch.zeros(K_eff),
           "tsm_mode": "time_bin", "gauss_o": True,
           "t_edges": (uniq.float() / K).cpu()}                     # 살아남은 구간의 시작점
    return out


def sample_hard_gauss(cb: dict, n: int, gen=None):
    """k~Cat(π), s̃~N(μ_s,k, σ_s,k), õ~N(m_k, σ_o,k). time_bin 팔의 샘플러."""
    pi, mu_s, sig_s = cb["pi"], cb["mu_s"], cb["sig_s"]
    k = torch.multinomial(pi, n, replacement=True, generator=gen)
    s = mu_s[k] + sig_s[k] * torch.randn(n, mu_s.shape[1], device=mu_s.device, generator=gen)
    m, sig_o = cb["m"], cb["sig_o"]
    o = m[k] + sig_o[k] * torch.randn(n, m.shape[1], device=m.device, generator=gen)
    return s, o


# ════════════════════════════════════════════════════════════════════════════
#  드라이버 패치 — Lotus_L3_V7dea 의 세 함수만 갈아끼운다
# ════════════════════════════════════════════════════════════════════════════
def install_patches(mode: str) -> None:
    import Lotus_L3_V7dea as LC

    orig_build = LC.build_codebook
    orig_hard = LC.sample_codebook_hard

    # ── 샘플러: gauss_o 코드북이면 잡음 샘플러, 아니면 기존 hard_linear ─────
    #   anchor_loss 는 모듈 전역 sample_codebook_hard 를 호출 시점에 찾으므로
    #   (V7dea 자체·L2_V5 모두) 모듈 속성을 바꾸면 둘 다 이 경로를 탄다.
    def sample_hard_tsm(cb, n, gen=None):
        if cb.get("gauss_o"):
            return sample_hard_gauss(cb, n, gen)
        return orig_hard(cb, n, gen)
    LC.sample_codebook_hard = sample_hard_tsm

    if mode == "indep_gauss":
        # 셀 없음: 진행률을 전부 0 으로 주면 time_bin 빌더가 구간 하나(K_eff=1)를 만든다 →
        # s̃~N(μ_s,σ_s), õ~N(m,σ_o) 가 태스크 전체 대각 가우시안이 되고 서로 독립이다.
        def build_indep(s, o, K, seed, h_scale=1.0, **kw):
            cb = build_codebook_timebin(s, o, K, torch.zeros(s.shape[0], device=s.device), h_scale=h_scale)
            cb["tsm_mode"] = "indep_gauss"
            return cb
        LC.build_codebook = build_indep
        return

    if mode == "state_gauss":
        # state k-means 셀은 그대로, õ 는 m_k 에 셀 안 o 의 **원시** 표준편차 잡음을 더한다 (A 없음).
        # 드라이버의 sig_o 는 A 를 뺀 선형 잔차 표준편차라 더 얇다 — 여기서는 원시 분산을 다시 센다.
        # 셀 라벨은 표준화 s 의 최근접 중심으로 복원한다 (k-means 하드 배정과 같다; 병합된 셀의
        # 점만 중심 기준으로 붙으므로 극소수 점이 다른 셀로 갈 수 있다).
        def build_state_gauss(s, o, K, seed, **kw):
            cb = orig_build(s, o, K, seed, **kw)
            zs = ((s - cb["mean_s"].to(s.device)) / cb["std_s"].to(s.device))
            lab = torch.cdist(zs, cb["c"].to(s.device)).argmin(1)
            _, sig_raw = _per_cell(o, lab, cb["K_eff"])
            cb["sig_o"] = sig_raw.clamp_min(1e-3 * float(o.std())).cpu()
            cb["A"] = torch.zeros_like(cb["A"])
            cb["A_fro"] = torch.zeros_like(cb["A_fro"])
            cb["n_demoted"] = cb["K_eff"]
            cb["gauss_o"] = True
            cb["tsm_mode"] = "state_gauss"
            return cb
        LC.build_codebook = build_state_gauss
        return

    if mode == "state_mean":
        def build_state_mean(s, o, K, seed, **kw):
            cb = orig_build(s, o, K, seed, **kw)
            cb["A"] = torch.zeros_like(cb["A"])          # õ = m_k 만 남긴다
            cb["A_fro"] = torch.zeros_like(cb["A_fro"])
            cb["n_demoted"] = cb["K_eff"]
            cb["tsm_mode"] = "state_mean"
            return cb
        LC.build_codebook = build_state_mean
        return

    if mode == "time_bin":
        # ── collect_pairs: s 의 마지막 열에 진행률을 실어 보낸다 ─────────────
        #   make_books 는 pairs 의 s 를 build_codebook 에만 넘기므로 거기서 떼면 된다.
        @torch.no_grad()
        def collect_pairs_time(policy, dataset, device, batch_size, num_workers, n_demos):
            loader = torch.utils.data.DataLoader(
                dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False,
                pin_memory=device.type == "cuda", drop_last=False,
                multiprocessing_context="spawn" if num_workers > 0 else None)
            buf, act, eps = {}, {}, {}
            fidx, eidx = {}, {}
            policy.eval()
            for raw in loader:
                batch = LC.prep_batch(policy, LC.to_device(raw, device))
                cls = LC.rgb_cls(policy, batch)
                n = batch["observation.state"].shape[0]
                s = batch["observation.state"].detach().flatten(1).float().cpu()
                o = cls.detach().reshape(n, -1).float().cpu()
                a = batch["action"].detach().float().cpu()
                ei = raw["episode_index"].cpu().reshape(-1)
                if "frame_index" not in raw:
                    raise SystemExit("time_bin 팔은 데이터셋의 frame_index 열이 필요하다 "
                                     "(LeRobot v2 스키마). 이 스플릿에는 없다.")
                fi = raw["frame_index"].cpu().reshape(-1).float()
                ti = ei // n_demos
                for t in ti.unique().tolist():
                    msk = ti == t
                    buf.setdefault(int(t), []).append((s[msk], o[msk]))
                    act.setdefault(int(t), []).append(a[msk])
                    eps.setdefault(int(t), set()).update(ei[msk].tolist())
                    fidx.setdefault(int(t), []).append(fi[msk])
                    eidx.setdefault(int(t), []).append(ei[msk])
            policy.train()
            pairs = {}
            for t, v in buf.items():
                s = torch.cat([x for x, _ in v]); o = torch.cat([y for _, y in v])
                fi = torch.cat(fidx[t]); ei = torch.cat(eidx[t])
                # 에피소드 길이 = max(frame_index)+1 (전 프레임이 한 번씩 들어왔다)
                u, inv = torch.unique(ei, return_inverse=True)
                ln = torch.zeros(u.numel()).scatter_reduce(0, inv, fi, reduce="amax") + 1.0
                prog = (fi / ln[inv]).clamp(0.0, 1.0 - 1e-6)
                pairs[t] = (torch.cat([s, prog[:, None]], 1), o)        # (N,17)
            return (pairs, {t: len(v) for t, v in eps.items()},
                    {t: torch.cat(v) for t, v in act.items()})
        LC.collect_pairs = collect_pairs_time

        def build_time_bin(s, o, K, seed, h_scale=1.0, **kw):
            prog, s = s[:, -1], s[:, :-1]
            return build_codebook_timebin(s, o, K, prog, h_scale=h_scale)
        LC.build_codebook = build_time_bin
        return
    # state_map / tether: 코드북 그대로 (샘플러 패치는 gauss_o 가 없어 원본과 같다)


# ════════════════════════════════════════════════════════════════════════════
#  selftest — 모델 없이 코드북/샘플러만
# ════════════════════════════════════════════════════════════════════════════
def selftest() -> int:
    torch.manual_seed(0)
    N, K = 1000, 25
    prog = torch.rand(N)
    s = torch.randn(N, 16) + 3 * prog[:, None]            # 시간에 따라 상태가 흐른다
    o = torch.randn(N, 3072) * 0.1 + s[:, :1]             # o 가 s 의 첫 축을 따른다
    ok = True
    cb = build_codebook_timebin(s, o, K, prog)
    bins = (prog * K).long()
    ok &= cb["K_eff"] == K and int(cb["n_k"].sum()) == N
    # 구간 평균이 맞는가 (0번 구간)
    ok &= torch.allclose(cb["mu_s"][0], s[bins == 0].mean(0), atol=1e-4)
    ok &= torch.allclose(cb["m"][0], o[bins == 0].mean(0), atol=1e-4)
    ss, oo = sample_hard_gauss(cb, 64)
    ok &= ss.shape == (64, 16) and oo.shape == (64, 3072)
    ok &= bool((oo.std(0).mean() > 0))                    # 잡음이 붙었다
    print(f"  ① time_bin: K_eff={cb['K_eff']} N={cb['N']} 샘플 shape {tuple(ss.shape)},{tuple(oo.shape)}  "
          f"{'OK' if ok else 'FAIL'}")
    # indep_gauss: 셀 하나, 전체 평균/표준편차
    cbi = build_codebook_timebin(s, o, K, torch.zeros(N))
    oki = cbi["K_eff"] == 1 and torch.allclose(cbi["mu_s"][0], s.mean(0), atol=1e-4) \
        and torch.allclose(cbi["m"][0], o.mean(0), atol=1e-4)
    si, oi = sample_hard_gauss(cbi, 64)
    oki &= si.shape == (64, 16) and oi.shape == (64, 3072) and bool(oi.std(0).mean() > 0)
    print(f"  ⓪ indep_gauss: K_eff={cbi['K_eff']} 전체 평균 일치, 샘플 shape OK  {'OK' if oki else 'FAIL'}")
    ok &= bool(oki)
    # state_mean/state_gauss/state_map 은 드라이버 함수를 쓴다 — 가능하면 같이 검증
    try:
        import Lotus_L3_V7dea as LC
        cb2 = LC.build_codebook(s, o, K, seed=0)
        # state_gauss: 원시 셀 표준편차 ≥ 선형 잔차 표준편차, A=0, 샘플이 m_k 에서 벗어난다
        zs2 = (s - cb2["mean_s"]) / cb2["std_s"]
        lab2 = torch.cdist(zs2, cb2["c"]).argmin(1)
        _, sig_raw = _per_cell(o, lab2, cb2["K_eff"])
        cbg = dict(cb2); cbg["sig_o"] = sig_raw.clamp_min(1e-3 * float(o.std()))
        cbg["A"] = torch.zeros_like(cb2["A"]); cbg["gauss_o"] = True
        okg = bool((cbg["sig_o"].pow(2).sum(1) >= cb2["sig_o"].pow(2).sum(1) * 0.999).all())
        sg, og = sample_hard_gauss(cbg, 8)
        okg &= float(torch.cdist(og, cbg["m"]).min(1).values.mean()) > 0 and sg.shape == (8, 16)
        print(f"  ④ state_gauss: 원시 σ_o ≥ 잔차 σ_o 모든 셀, 잡음 샘플 OK  {'OK' if okg else 'FAIL'}   "
              f"(총분산 비 원시/잔차 mean={float((cbg['sig_o'].pow(2).sum(1) / cb2['sig_o'].pow(2).sum(1)).mean()):.2f})")
        ok &= okg
        cb2m = dict(cb2); cb2m["A"] = torch.zeros_like(cb2["A"])
        s1, o1 = LC.sample_codebook_hard(cb2m, 8)
        k_hit = torch.cdist(o1, cb2m["m"]).argmin(1)
        ok2 = torch.allclose(o1, cb2m["m"][k_hit], atol=1e-5)
        s2, o2 = LC.sample_codebook_hard(cb2, 8)
        d2 = torch.cdist(o2, cb2["m"]).min(1).values
        print(f"  ② state_mean: õ == m_k 정확히  {'OK' if ok2 else 'FAIL'}   "
              f"③ state_map: õ 가 m_k 에서 벗어남 mean={float(d2.mean()):.4f} (A 작동, "
              f"‖A‖_F mean={float(cb2['A_fro'].mean()):.2f})")
        ok &= ok2 and float(d2.mean()) > 0
    except Exception as ex:                                   # lerobot 이 없는 환경
        print(f"  ②③ 드라이버 import 실패로 건너뜀: {type(ex).__name__}: {ex}")
    print("selftest", "OK" if ok else "FAIL")
    return 0 if ok else 1


# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--tsm_mode", choices=MODES, default=None)
    p.add_argument("--lambda_level", type=float, default=0.7,
                   help="앵커 세기 λ. 네 팔 공통 (sqerr 세 팔은 드라이버의 --lambda_level, "
                        "tether 는 L2_V5 의 --lambda_level 로 들어간다)")
    p.add_argument("--anchor_c", type=float, default=0.1, help="tether 만: MAR 의 c")
    p.add_argument("--beta", type=float, default=0.6, help="tether 만: MAR 방향 항 배율 β")
    p.add_argument("--selftest", action="store_true")
    opt, rest = p.parse_known_args()
    if opt.selftest:
        raise SystemExit(selftest())
    if opt.tsm_mode is None:
        raise SystemExit("--tsm_mode 가 필요하다: " + "|".join(MODES))

    q = argparse.ArgumentParser(add_help=False)
    q.add_argument("--out", default=None)
    q.add_argument("--lifelong_k", type=int, default=25)
    q2, _ = q.parse_known_args(rest)
    arm = {"tsm_mode": opt.tsm_mode, "lambda_level": opt.lambda_level,
           "anchor_form": "speedinv_dir(MAR)" if opt.tsm_mode == "tether" else "sqerr",
           "beta": opt.beta if opt.tsm_mode == "tether" else None,
           "anchor_c": opt.anchor_c if opt.tsm_mode == "tether" else None,
           "cells_per_task": q2.lifelong_k,
           "sampler": "hard_gauss" if opt.tsm_mode in GAUSS_O_MODES else "hard_linear",
           "local_map_A": opt.tsm_mode in ("state_map", "tether")}
    if q2.out:
        Path(q2.out).mkdir(parents=True, exist_ok=True)
        json.dump(arm, (Path(q2.out) / "tsm_arm.json").open("w"), indent=2, ensure_ascii=False)
    print(f"[TSM-ablation] {json.dumps(arm, ensure_ascii=False)}", flush=True)

    install_patches(opt.tsm_mode)
    common = ["--sampler_mode", "hard_linear", "--lambda_level", str(opt.lambda_level)]
    if opt.tsm_mode in SQERR_MODES:
        import Lotus_L3_V7dea as LC
        sys.argv = [LC.__file__] + rest + ["--anchor_form", "sqerr"] + common
        LC.main()
    else:
        import L2_V5
        sys.argv = [L2_V5.__file__, "--setting", "lotus",
                    "--anchor_c", str(opt.anchor_c), "--beta", str(opt.beta),
                    "--anchor_reduce", "mean", "--anchor_warmup", "0",
                    "--dir_space", "act"] + rest + common
        L2_V5.main()


if __name__ == "__main__":
    main()
