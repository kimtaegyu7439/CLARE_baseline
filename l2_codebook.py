#!/usr/bin/env python
"""L2_codebook — L2 + (s,o) 결합 코드북 앵커 샘플러.

무엇이 달라지는가
    앵커의 (observation, state) 공급원을 통째로 바꾼다. 시간 축은 어디에도 없다.

    기존 R13/L2                      L2_codebook
    o ~ N(μ_j(τ), σ_j(τ))            (s̃, õ) ~ task j 의 s-공간 코드북
    s = 현재 배치 것 그대로           s̃ = 코드북에서 뽑은 task j 의 state
    τ = 에피소드 진행도 bin           없음

    (i) 시간-bin 은 o 의 좌표계가 아니고, (ii) state 가 현재 태스크 것이라
    "task j 장면인데 팔은 현재 태스크 자세"인 조건이 만들어지던 두 결함을 함께 없앤다.

파이프라인
    학습 중        같은 프레임의 (s 16-d, o 3072-d) 쌍을 저수지 표본으로 모은다
    task 종료      s 표준화 -> k-means(K=96) -> 셀별 (π, μ_s, σ_s, c, m, σ_o) 저장, 버퍼 폐기
    이후 task      k ~ Cat(π);  s̃ = μ_s[k]+σ_s[k]ε;  w = softmax(−‖z(s̃)−c‖²/h²);
                   õ = w·m + sqrt(w·σ_o²)⊙ε′        <- s̃ 가 õ 를 고르므로 짝이 맞는다

    그 외(teacher 부트스트랩 x_t, δℓ 없음, t~U(0,1), λ, reduction, teacher 운용,
    eval 프로토콜)는 L2 와 동일하다.

구 시간-bin 경로
    이 팔은 R10.compute_stats / phase_bins 를 **한 번도 부르지 않는다**.
    R13Anchor 를 상속하지 않고 앵커 프로토콜을 직접 구현했고, 방어적으로
    _forbid_time_bin() 가 두 함수를 실행 시 예외로 바꿔 놓는다.
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

OUT_DIR = REPO / "results" / "L2_codebook"


# ═════════════════════════════════════════════════════════════════════════════
#  구 시간-bin 경로 차단 (§0.5)
# ═════════════════════════════════════════════════════════════════════════════
def _forbid_time_bin():
    """R10 의 시간-bin 통계 함수가 이 런에서 호출되면 즉시 터지게 한다."""
    import R10

    def _dead(*a, **kw):
        raise RuntimeError(
            "시간-bin 경로(R10.compute_stats/phase_bins)는 L2_codebook 에서 쓰지 않는다")

    R10.compute_stats = _dead
    R10.phase_bins = _dead


# ═════════════════════════════════════════════════════════════════════════════
#  k-means (표준화된 s 공간, 16-d) — torch, GPU
# ═════════════════════════════════════════════════════════════════════════════
def kmeans(x: torch.Tensor, K: int, n_init: int = 10, iters: int = 100,
           seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, float]:
    """k-means++ 초기화 x n_init. (centers(K,D), labels(N,), inertia) 를 돌려준다."""
    N, D = x.shape
    best = None
    for init in range(n_init):
        g = torch.Generator(device="cpu").manual_seed(seed * 1000 + init)
        # k-means++
        idx = [int(torch.randint(N, (1,), generator=g))]
        d2 = (x - x[idx[0]]).pow(2).sum(1)
        for _ in range(K - 1):
            p = (d2 / d2.sum().clamp_min(1e-12)).cpu()
            idx.append(int(torch.multinomial(p, 1, generator=g)))
            d2 = torch.minimum(d2, (x - x[idx[-1]]).pow(2).sum(1))
        c = x[idx].clone()
        prev = None
        for _ in range(iters):
            lab = torch.cdist(x, c).argmin(1)
            cnt = torch.bincount(lab, minlength=K).clamp_min(1)
            cnew = torch.zeros_like(c).index_add_(0, lab, x) / cnt[:, None]
            empty = torch.bincount(lab, minlength=K) == 0
            if empty.any():                       # 빈 셀은 가장 먼 점으로 재배치
                far = torch.cdist(x, c).min(1).values.argsort(descending=True)
                cnew[empty] = x[far[:int(empty.sum())]]
            inertia = float((x - cnew[lab]).pow(2).sum())
            if prev is not None and abs(prev - inertia) < 1e-7 * max(prev, 1.0):
                c = cnew
                break
            prev, c = inertia, cnew
        lab = torch.cdist(x, c).argmin(1)
        inertia = float((x - c[lab]).pow(2).sum())
        if best is None or inertia < best[2]:
            best = (c, lab, inertia)
    return best


# ═════════════════════════════════════════════════════════════════════════════
#  코드북
# ═════════════════════════════════════════════════════════════════════════════
def build_codebook(s: torch.Tensor, o: torch.Tensor, K: int, seed: int,
                   min_n: int = 5, h_scale: float = 1.0,
                   grad: bool = False, ridge_rho: float = 0.05,
                   grad_min_frames: int = 24) -> dict:
    """(s(N,16), o(N,3072)) -> 코드북 dict. §2 그대로."""
    N = s.shape[0]
    mean_s, std_s = s.mean(0), s.std(0).clamp_min(1e-8)
    zs = (s - mean_s) / std_s
    c, lab, inertia = kmeans(zs, K, n_init=10, seed=seed)

    # n_k < min_n 셀은 최근접 셀로 병합
    cnt = torch.bincount(lab, minlength=K)
    keep = (cnt >= min_n).nonzero().squeeze(1)
    drop = (cnt < min_n).nonzero().squeeze(1)
    n_merged = int(drop.numel())
    if n_merged and keep.numel():
        # 버려지는 셀의 소속 점들을 남는 셀 중 최근접으로
        remap = torch.arange(K, device=s.device)
        d = torch.cdist(c[drop], c[keep])          # (drop, keep)
        remap[drop] = keep[d.argmin(1)]
        lab = remap[lab]
        # 라벨을 0..K_eff-1 로 압축
        uniq, lab = torch.unique(lab, return_inverse=True)
        c = c[uniq]
    K_eff = int(c.shape[0])

    pi = torch.bincount(lab, minlength=K_eff).float()
    n_k = pi.clone()
    pi = pi / pi.sum()

    def per_cell(v, D):
        m = torch.zeros(K_eff, D, device=v.device, dtype=torch.float64)
        sq = torch.zeros_like(m)
        m.index_add_(0, lab, v.double())
        sq.index_add_(0, lab, v.double() ** 2)
        cnt = n_k.double().clamp_min(1.0)[:, None]
        mu = m / cnt
        sd = (sq / cnt - mu ** 2).clamp_min(0.0).sqrt()
        return mu.float(), sd.float()

    mu_s, sig_s = per_cell(s, s.shape[1])
    m_o, sig_o = per_cell(o, o.shape[1])

    # ── v2: 셀별 선형 기울기 A_k (3072x16) ──────────────────────────────────
    # v1 은 셀 안에서 o 가 s 를 따라 움직이는 성분을 m_k 하나로 뭉개고, 그 변동을
    # σ_o,k 에 분산으로 흡수한다(스미어). A_k 를 두면 õ 의 중심이 s̃ 를 서브셀
    # 스케일로 따라가고, σ_o,k 는 **선형 잔차** 기준이 되어 얇아진다.
    A = None
    n_demoted = 0
    stats_v1_var = sig_o.pow(2).sum(1).clone()      # 셀별 0차 총분산(로그용)
    if grad:
        D = o.shape[1]
        zbar = (mu_s - mean_s) / std_s                          # (K_eff,16)
        A = torch.zeros(K_eff, D, s.shape[1], device=o.device)
        sig_lin = sig_o.clone()
        for kk in range(K_eff):
            sel = (lab == kk).nonzero().squeeze(1)
            if sel.numel() == 0:
                continue
            dl = zs[sel] - zbar[kk]                             # (n_k,16)
            Y = o[sel] - m_o[kk]                                # (n_k,D)
            if int(sel.numel()) < grad_min_frames:
                n_demoted += 1                                  # A_k = 0 (0차 강등)
            else:
                S = dl.T @ dl                                   # (16,16)
                lam_r = ridge_rho * float(S.diagonal().sum()) / s.shape[1]
                G = Y.T @ dl                                    # (D,16)
                A[kk] = G @ torch.linalg.inv(
                    S + lam_r * torch.eye(s.shape[1], device=o.device))
                Y = Y - dl @ A[kk].T                            # 선형 잔차
            sig_lin[kk] = Y.std(0, unbiased=False)
        sig_o = sig_lin
    # σ 하한 (§2)
    sig_s = sig_s.clamp_min(1e-3 * float(s.std()))
    sig_o = sig_o.clamp_min(1e-3 * float(o.std()))
    # 대역폭 h = median(각 zs -> 최근접 중심 거리)
    # 스펙: h = median(각 zs -> 최근접 중심 거리).
    # 그런데 softmax(−d²/h²) 에 이 값을 그대로 쓰면 2등 셀도 거의 같은 가중을 받아
    # w 가 20여 개 셀에 퍼진다(합성 테스트 실측: 참여 셀 median 20.7/96, 최대가중 0.147).
    # 그러면 w@m 이 여러 중심의 평균이 되어 구조가 뭉개지고, 남은 분산은 대각 잡음이
    # 채워 d_eff 가 폭발한다 — R13 의 등방 실패 모드로 되돌아간다.
    # h_scale 로 대역폭을 좁힐 수 있게 열어 둔다(1.0 = 스펙 그대로).
    # ★ c 와 h 는 커널 가중(softmax(−‖zs−c‖²/h²))용이다. L2_codebook_bayes 는 가중을
    #   GMM 사후확률로 바꾸므로 그 팔에서는 **deprecated** — 파일에는 남긴다
    #   (스키마 호환 + sanity 의 구가중 대비용).
    h = float(torch.cdist(zs, c).min(1).values.median().clamp_min(1e-6)) * h_scale

    out = {"h_scale": h_scale,
            "pi": pi.cpu(), "mu_s": mu_s.cpu(), "sig_s": sig_s.cpu(),
            "c": c.cpu(), "m": m_o.cpu(), "sig_o": sig_o.cpu(),
            "mean_s": mean_s.cpu(), "std_s": std_s.cpu(), "h": h,
            "K_eff": K_eff, "N": N, "n_k": n_k.cpu(), "inertia": inertia,
            "n_merged": n_merged}
    if grad:
        out["A"] = A.half().cpu()                     # fp16 저장(~10MB), 연산은 fp32
        out["zbar"] = ((mu_s - mean_s) / std_s).cpu()
        out["ridge_rho"] = ridge_rho
        out["n_demoted"] = n_demoted
        out["var_ratio_cells"] = (stats_v1_var
                                  / sig_o.pow(2).sum(1).clamp_min(1e-12)).cpu()
        out["A_fro"] = A.flatten(1).norm(dim=1).cpu()
    return out


def sample_codebook(cb: dict, n: int, gen=None) -> tuple[torch.Tensor, torch.Tensor]:
    """§3. (s̃(n,16), õ(n,3072))."""
    pi, mu_s, sig_s = cb["pi"], cb["mu_s"], cb["sig_s"]
    c, m, sig_o = cb["c"], cb["m"], cb["sig_o"]
    k = torch.multinomial(pi, n, replacement=True, generator=gen)
    eps = torch.randn(n, mu_s.shape[1], device=mu_s.device, generator=gen)
    s = mu_s[k] + sig_s[k] * eps
    z = (s - cb["mean_s"]) / cb["std_s"]
    d2 = torch.cdist(z, c).pow(2)                          # (n, K)
    w = torch.softmax(-d2 / (cb["h"] ** 2), dim=1)
    o_mu = w @ m
    if cb.get("A") is not None:
        # õ_mean = Σ_j w_j (m_j + A_j δ_j)  —  δ_j = z − z̄_j
        # 두 번째 항을 (B,K,16) -> (B,K·16) @ (K·16, D) 한 번의 matmul 로 접는다.
        # (B,K,D) 중간 텐서를 만들지 않으므로 top-m 절단이 필요 없다.
        A = cb["A"].float()                                   # (K,D,16)
        d = z[:, None, :] - cb["zbar"][None]                  # (n,K,16)
        u = (w[:, :, None] * d).reshape(z.shape[0], -1)       # (n, K*16)
        o_mu = o_mu + u @ A.permute(0, 2, 1).reshape(-1, A.shape[1])
    o_sd = (w @ sig_o.pow(2)).clamp_min(0.0).sqrt()
    o = o_mu + o_sd * torch.randn(n, m.shape[1], device=m.device, generator=gen)
    return s, o


def to_device_cb(cb: dict, device) -> dict:
    return {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in cb.items()}


# ═════════════════════════════════════════════════════════════════════════════
#  앵커
# ═════════════════════════════════════════════════════════════════════════════
class L2CodebookAnchor:
    """(s,o) 코드북 좌표 위의 level 앵커 + teacher 부트스트랩 x_t.

    R13Anchor 를 상속하지 않는다 — 시간-bin 통계를 아예 만들지 않기 위해서다.
    B1 이 요구하는 프로토콜(describe/on_task_start/loss/on_task_end)만 구현한다.
    """

    name = "L2_codebook"

    def __init__(self, args):
        self.a = args
        self.teacher = None
        self.books: dict[int, dict] = {}          # j -> 코드북(GPU 상주)
        self.K = args.codebook_k
        self.n_pairs = args.n_pairs
        self.h_scale = args.h_scale
        self.grad_enable = args.grad_enable
        self.ridge_rho = args.ridge_rho
        self.grad_min_frames = args.grad_min_frames
        self.lam_lvl = args.lambda_level
        self.xt_mode = args.xt_mode
        self.step = 0
        self.t_anchor = 0.0
        self.out = Path(args.out_dir); self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "codebooks").mkdir(exist_ok=True)
        self.log = (self.out / "l2cb.jsonl").open("a")
        # 수집 저수지
        self._res_s = self._res_o = self._res_id = None
        self._seen = 0
        self._dataset = None
        self._prep = None
        self._sanity_done = False
        self._xt_sanity = False

    def describe(self):
        return (f"L2_codebook{'+grad' if self.grad_enable else ''} — (s,o) 코드북 앵커 "
                f"K={self.K}, 수집 {self.n_pairs}쌍, "
                f"xt={self.xt_mode}, λ_lvl={self.lam_lvl}, 코드북 {len(self.books)}개")

    def _sample(self, cb, n, gen=None):
        """(s̃, õ). 파생 팔(L2_codebook_bayes)이 가중치 원천을 갈아 끼우는 지점."""
        return sample_codebook(cb, n, gen)

    def reduce_level(self, x):
        if self.a.anchor_norm == "sum":
            return x.flatten(1).pow(2).sum(1).mean()
        return x.flatten(1).pow(2).mean(1).mean()

    # ── 태스크 시작 ─────────────────────────────────────────────────────────
    def on_task_start(self, policy, k, args, instructions, device, **kw):
        self._dataset = kw.get("dataset")
        self._prep = kw.get("prep")
        self._res_s = self._res_o = self._res_id = None
        self._seen = 0
        self.step = 0
        self._xt_sanity = k > 0
        logging.info(f"[L2CB] task {k} 시작 — 수집 목표 {self.n_pairs}쌍, "
                     f"보유 코드북 {sorted(self.books)}")

    # ── (s,o) 쌍 수집 (§1) ──────────────────────────────────────────────────
    @torch.no_grad()
    def _collect(self, batch, cls, device):
        """저수지 표본. s 와 o 는 **같은 배치 행**에서 온 짝이다."""
        s = batch["observation.state"].detach().flatten(1).float()      # (B,16)
        n = s.shape[0]
        o = cls.detach().reshape(n, -1).float()                          # (B,3072)
        ident = torch.stack([batch["episode_index"].detach().long(),
                             batch["frame_index"].detach().long()], 1)   # (B,2)
        if self._res_s is None:
            self._res_s = torch.zeros(self.n_pairs, s.shape[1])
            self._res_o = torch.zeros(self.n_pairs, o.shape[1])
            self._res_id = torch.zeros(self.n_pairs, 2, dtype=torch.long)
        s, o, ident = s.cpu(), o.cpu(), ident.cpu()
        # 배치 단위 벡터화 저수지 — 행 루프를 돌면 스테이지당 16만 회가 되어 느리다.
        base = self._seen
        pos = torch.arange(base, base + n)
        R = self.n_pairs
        fill = pos < R                                   # 아직 안 찬 자리
        if fill.any():
            dst = pos[fill]
            self._res_s[dst], self._res_o[dst], self._res_id[dst] = s[fill], o[fill], ident[fill]
        rest = ~fill
        if rest.any():
            # j 번째 원소는 확률 R/(j+1) 로 채택, 채택 시 [0,R) 균등 위치를 덮어쓴다
            j = pos[rest]
            r = (torch.rand(int(rest.sum())) * (j + 1).float()).long()
            take = r < R
            if take.any():
                idx_src = rest.nonzero().squeeze(1)[take]
                dst = r[take]
                # 같은 자리에 여러 번 쓰이면 마지막 것이 남는다(저수지 정의와 동치)
                self._res_s[dst], self._res_o[dst], self._res_id[dst] = (
                    s[idx_src], o[idx_src], ident[idx_src])
        self._seen += n

    # ── 손실 ────────────────────────────────────────────────────────────────
    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        cls = getattr(self, "cls", None)
        if cls is None:
            cls = B1.rgb_cls(policy, batch)
        self._collect(batch, cls, device)          # k==0 포함 — 매 스텝 수집

        if k == 0 or self.teacher is None or args.lambda_anchor == 0 or not self.books:
            return torch.zeros((), device=device)
        t0w = time.perf_counter()
        n = batch["observation.state"].shape[0]
        st_shape = batch["observation.state"].shape          # (B, n_obs, state_dim)
        tcol = t[:, None, None]
        chunk = self.a.chunk_backward
        if chunk and getattr(policy.config, "use_amp", False):
            raise RuntimeError("chunk_backward 는 use_amp=True 와 함께 쓸 수 없다")

        lvl = []
        teach = self.teacher
        for j in sorted(self.books):
            cb = self.books[j]
            s_t, o_t = self._sample(cb, n)                          # (n,16), (n,3072)
            s_t = s_t.detach(); o_t = o_t.detach()
            bb = dict(batch)
            bb["observation.state"] = s_t.view(st_shape).to(batch["observation.state"].dtype)
            cls_j = o_t.reshape(-1, cls.shape[-1]).to(x_t.dtype)     # (n*4, 768)
            past = [instructions[f"task{j}"]] * n

            tl_S = B1.cond_tail(policy, bb, cls_j)
            with torch.no_grad():
                tl_T = B1.cond_tail(teach, bb, cls_j)
                cond_T = B1.make_cond(B1.encode_lang(teach, past), tl_T)
            cond_S = B1.make_cond(B1.encode_lang(policy, past), tl_S)

            def vel(pol, xx, tt, cond):
                return pol.dit_flow.velocity_net(noisy_actions=xx, time=tt, global_cond=cond)

            # ── teacher 부트스트랩 x_t (L2 그대로) ──────────────────────────
            if self.xt_mode == "current":
                x_j = x_t
            else:
                with torch.no_grad():
                    eps0 = torch.randn_like(x_t)
                    a_hat = (eps0 + vel(teach, eps0, torch.zeros_like(t), cond_T)).detach()
                eps1 = torch.randn_like(x_t)
                x_j = ((1 - tcol) * eps1 + tcol * a_hat).detach()

            with torch.no_grad():
                vt0 = vel(teach, x_j, t, cond_T)
            vs0 = vel(policy, x_j, t, cond_S)
            L_j = self.reduce_level(vs0 - vt0.to(vs0.dtype))
            if chunk:
                (args.lambda_anchor * self.lam_lvl * L_j).backward()
                lvl.append(L_j.detach())
            else:
                lvl.append(L_j)

            if self._xt_sanity and j == min(self.books):
                with torch.no_grad():
                    logging.info(
                        f"[L2CB][sanity] task{k} j={j}  K_eff={cb['K_eff']}  "
                        f"‖s̃‖={float(s_t.norm(dim=1).mean()):.4f} "
                        f"(실배치 {float(batch['observation.state'].flatten(1).norm(dim=1).mean()):.4f})  "
                        f"‖õ‖={float(o_t.norm(dim=1).mean()):.2f} "
                        f"(실배치 {float(cls.reshape(n,-1).norm(dim=1).mean()):.2f})  "
                        f"â유한={bool(torch.isfinite(x_j).all())}  "
                        f"‖r0‖={float((vs0-vt0).flatten(1).norm(dim=1).mean()):.4f}")
                self._xt_sanity = False

        L_lvl = sum(lvl) / len(lvl)
        out = self.lam_lvl * L_lvl
        if chunk:
            out = out.detach()
        self.step += 1
        self.t_anchor += time.perf_counter() - t0w
        if self.step % self.a.log_every_anchor == 0:
            rec = {"task": k, "step": self.step, "L_level": float(L_lvl.detach()),
                   "n_past": len(self.books), "K": self.K,
                   "ms_per_step": 1000 * self.t_anchor / max(self.step, 1)}
            self.log.write(json.dumps(rec) + "\n"); self.log.flush()
            logging.info(f"[L2CB] k={k} step={self.step:5d} L_lvl={rec['L_level']:.4f} "
                         f"ms/step={rec['ms_per_step']:.1f}")
        return out

    # ── 태스크 종료: 코드북 빌드 + teacher ──────────────────────────────────
    def on_task_end(self, policy, k, args, instructions, device, **kw):
        n = min(self._seen, self.n_pairs)
        s = self._res_s[:n].to(device)
        o = self._res_o[:n].to(device)
        t0 = time.perf_counter()
        cb = build_codebook(s, o, self.K, seed=getattr(args, "seed", 42),
                            h_scale=self.h_scale, grad=self.grad_enable,
                            ridge_rho=self.ridge_rho,
                            grad_min_frames=self.grad_min_frames)
        dt = time.perf_counter() - t0
        p = self.out / "codebooks" / f"codebook_task{k}.pt"
        torch.save({kk: vv for kk, vv in cb.items()}, p)
        nk = cb["n_k"]
        logging.info(
            f"[L2CB] task {k} 코드북  N={cb['N']} (본 프레임 {self._seen})  "
            f"K_eff={cb['K_eff']} (병합 {cb['n_merged']})  "
            f"n_k min/med/max={int(nk.min())}/{int(nk.median())}/{int(nk.max())}  "
            f"h={cb['h']:.4f}  {dt:.1f}s  {p.stat().st_size/1e6:.2f}MB"
            + ("" if not self.grad_enable else
               f"  |  grad: 강등 {cb['n_demoted']}셀  "
               f"‖A‖_F med/max={float(cb['A_fro'].median()):.2f}/"
               f"{float(cb['A_fro'].max()):.2f}  "
               f"분산비(0차/1차) med={float(cb['var_ratio_cells'].median()):.2f}"))
        self.books[k] = to_device_cb(cb, device)

        if k == 0 and not self._sanity_done:
            self._sanity_done = True
            try:
                self._sanity_suite(policy, cb, s, o, k, device, instructions, args)
            except Exception as e:
                logging.warning(f"[L2CB][sanity] 실패(실행은 계속): {type(e).__name__}: {e}")

        del self.teacher
        self.teacher = B1.snapshot(policy, args.teacher_bf16)
        self._res_s = self._res_o = self._res_id = None
        torch.cuda.empty_cache()
        logging.info(f"[L2CB] task {k} teacher 갱신 (코드북 {len(self.books)}개)")

    def _extra_sanity(self, cb, cbd, s_real, o_real, device, W, X, r_real):
        """파생 팔이 sanity 를 덧붙이는 지점. 기본은 없음."""
        return []

    # ── §4 기울기 실재 진단 (v2 전용) ───────────────────────────────────────
    @torch.no_grad()
    def _grad_sanity(self, cb, cbd, s_real, o_real, device, W, X, r_real):
        L = ["", "── §4 기울기(A_k) 진단 ──"]
        mean_s, std_s = cbd["mean_s"], cbd["std_s"]
        zs = (s_real - mean_s) / std_s
        lab = torch.cdist(zs, cbd["c"]).argmin(1)          # hard 배정
        zbar, m, A = cbd["zbar"], cbd["m"], cbd["A"].float()
        K = m.shape[0]

        corrs, r2_0, r2_1, vr, anorm, resid0 = [], [], [], [], [], []
        ss0 = ss1 = sst = 0.0
        for kk in range(K):
            sel = (lab == kk).nonzero().squeeze(1)
            if sel.numel() < 20:
                continue
            d = zs[sel] - zbar[kk]
            Y = o_real[sel] - m[kk]
            # 1a. 셀 내 ‖Y‖ vs ‖δ‖ 상관
            a_, b_ = Y.norm(dim=1), d.norm(dim=1)
            corrs.append(float(((a_ - a_.mean()) * (b_ - b_.mean())).mean()
                               / (a_.std(unbiased=False) * b_.std(unbiased=False)).clamp_min(1e-12)))
            # 1b/1c. 셀 내 80/20
            pm = torch.randperm(sel.numel(), device=device)
            ntr = max(int(0.8 * sel.numel()), 16)
            tr, te = pm[:ntr], pm[ntr:]
            if te.numel() < 4:
                continue
            dt, Yt = d[tr], Y[tr]
            S = dt.T @ dt
            lam = cb.get("ridge_rho", 0.05) * float(S.diagonal().sum()) / d.shape[1]
            Ak = (Yt.T @ dt) @ torch.linalg.inv(S + lam * torch.eye(d.shape[1], device=device))
            m_tr = Yt.mean(0)
            e0 = Y[te] - m_tr                              # 0차 예측 잔차
            e1 = Y[te] - (m_tr + d[te] @ Ak.T)             # 1차 예측 잔차
            ss0 += float(e0.pow(2).sum()); ss1 += float(e1.pow(2).sum())
            sst += float((Y[te] - Yt.mean(0)).pow(2).sum())
            vr.append(float(e0.pow(2).sum() / e1.pow(2).sum().clamp_min(1e-12)))
            anorm.append(float((d[te] @ Ak.T).norm(dim=1).median()))
            resid0.append(float(e0.norm(dim=1).median()))
        import statistics as st
        q = lambda v, p: float(torch.tensor(v).quantile(p))
        R2_0 = 1 - ss0 / max(sst, 1e-12)
        R2_1 = 1 - ss1 / max(sst, 1e-12)
        gap = R2_1 - R2_0
        vr_med = st.median(vr) if vr else 0.0
        L.append(f"1a 셀내 corr(‖o−m‖,‖δ‖)  중앙값={st.median(corrs):.3f}  "
                 f"Q1/Q3={q(corrs,0.25):.3f}/{q(corrs,0.75):.3f}  (양수 = 기울기 실재)")
        L.append(f"1b held-out R²  0차={R2_0:.4f}  1차={R2_1:.4f}  gap={gap:+.4f}")
        L.append(f"1c 셀별 분산비(0차/1차) 중앙값={vr_med:.3f}  "
                 f"Q1/Q3={q(vr,0.25):.3f}/{q(vr,0.75):.3f}")
        gate = (gap >= 0.05) or (vr_med >= 1.2)
        L.append(f"   ★게이트: gap≥0.05 또는 분산비≥1.2 -> "
                 f"{'통과 — 본 런 진행' if gate else '미달 — 본 런 보류'}")

        # 2. d_eff 3열 (실측 / grad-õ / v1-õ)
        def deff(x):
            x = x - x.mean(0)
            lam = torch.linalg.svdvals(x.double()).pow(2) / (x.shape[0] - 1)
            return float(lam.sum() ** 2 / lam.pow(2).sum()), float(lam.sum())
        cb_v1 = {kk: vv for kk, vv in cbd.items() if kk != "A"}
        cb_v1["A"] = None
        g = torch.Generator(device=device).manual_seed(1234)
        s1, o1 = sample_codebook(cbd, 1000, g)
        g = torch.Generator(device=device).manual_seed(1234)
        s0, o0 = sample_codebook(cb_v1, 1000, g)
        sel = torch.randperm(o_real.shape[0], device=device)[:1000]
        dr, vrr = deff(o_real[sel]); dg, vg = deff(o1); dv, vv_ = deff(o0)
        L += ["2 d_eff / total variance",
              f"   {'':10}{'d_eff':>10}{'var':>12}{'d_eff 비':>10}",
              f"   {'실측':10}{dr:10.2f}{vrr:12.1f}{1.0:10.2f}",
              f"   {'grad-õ':10}{dg:10.2f}{vg:12.1f}{dg/dr:10.2f}"
              f"   {'OK' if abs(dg/dr-1) <= 0.20 else '★±20% 이탈★'}",
              f"   {'v1-õ':10}{dv:10.2f}{vv_:12.1f}{dv/dr:10.2f}"]

        # 3. 서브셀 정합 (v1 sanity 의 ridge f(s) 재사용)
        def rmse(ss, oo):
            Xs = torch.cat([ss, torch.ones(ss.shape[0], 1, device=device)], 1).double()
            return float((oo.double() - Xs @ W).pow(2).mean().sqrt())
        rg, rv = rmse(s1, o1), rmse(s0, o0)
        L += [f"3 서브셀 정합  RMSE(õ−f(s̃))  grad={rg:.4f}  v1={rv:.4f}  "
              f"실측 잔차={r_real:.4f}",
              f"   비(실측 대비)  grad={rg/max(r_real,1e-12):.2f}  v1={rv/max(r_real,1e-12):.2f}  "
              f"({'개선' if rg < rv else '개선 없음'})"]

        # 4. 외삽 안전
        L.append(f"4 외삽 안전  ‖A_k δ‖ 중앙값={st.median(anorm):.4f}  "
                 f"0차 잔차 중앙값={st.median(resid0):.4f}  "
                 f"비={st.median(anorm)/max(st.median(resid0),1e-12):.3f} (<1 정상)")
        self._grad_gate = gate
        return L

    # ── Sanity (§5) — task0 종료 시 1회, 로그만 ─────────────────────────────
    @torch.no_grad()
    def _sanity_suite(self, policy, cb, s_real, o_real, k, device, instructions, args):
        L = ["", "═" * 74, f"[L2CB][sanity] task{k} — 본 런 전 점검", "═" * 74]

        # 5.1 수집 짝 검증 — 저장한 (ep, frame) 을 데이터셋에서 다시 읽어 대조
        ok = bad = 0
        if self._dataset is not None and self._prep is not None:
            idx = torch.randperm(min(self._seen, self.n_pairs))[:20]
            ei = self._dataset.episode_data_index
            for i in idx.tolist():
                ep, fr = int(self._res_id[i, 0]), int(self._res_id[i, 1])
                gidx = int(ei["from"][ep]) + fr
                raw = self._dataset[gidx]
                bb = B1.to_device({kk: (vv.unsqueeze(0) if torch.is_tensor(vv) else [vv])
                                   for kk, vv in raw.items()}, device)
                bb = self._prep(policy, bb)
                s_ref = bb["observation.state"].flatten(1)[0].float().cpu()
                o_ref = B1.rgb_cls(policy, bb).reshape(1, -1)[0].float().cpu()
                ds = float((s_ref - self._res_s[i]).norm() / s_ref.norm().clamp_min(1e-8))
                do = float((o_ref - self._res_o[i]).norm() / o_ref.norm().clamp_min(1e-8))
                if ds < 1e-3 and do < 1e-3:
                    ok += 1
                else:
                    bad += 1
                    if bad <= 3:
                        L.append(f"  ✗ i={i} ep{ep} fr{fr}  Δs={ds:.2e}  Δo={do:.2e}")
            L.append(f"5.1 짝 검증  일치 {ok}/20" +
                     ("  ★짝 어긋남 — 치명★" if bad else "  OK"))
        else:
            L.append("5.1 짝 검증  건너뜀 (dataset/prep 없음)")

        # 5.2 빌드 지표
        nk = cb["n_k"]
        L.append(f"5.2 빌드  N={cb['N']}  K_eff={cb['K_eff']}(병합 {cb['n_merged']})  "
                 f"n_k {int(nk.min())}/{int(nk.median())}/{int(nk.max())}  h={cb['h']:.4f}")

        # 5.3 합성 1000개
        cbd = to_device_cb(cb, device)
        s_syn, o_syn = sample_codebook(cbd, 1000)
        # a. s̃ 주변통계
        rel_m = float(((s_syn.mean(0) - s_real.mean(0)).abs()
                       / s_real.std(0).clamp_min(1e-8)).mean())
        rel_s = float(((s_syn.std(0) - s_real.std(0)).abs()
                       / s_real.std(0).clamp_min(1e-8)).mean())
        L.append(f"5.3a s̃ 주변통계  |Δmean|/σ={rel_m:.4f}  |Δstd|/σ={rel_s:.4f} "
                 f"(<0.10 기대)  {'OK' if max(rel_m, rel_s) < 0.10 else '★이탈★'}")

        # b. d_eff = (Σλ)²/Σλ² , total variance
        def deff(x):
            x = x - x.mean(0)
            lam = torch.linalg.svdvals(x.double()).pow(2) / (x.shape[0] - 1)
            return float(lam.sum() ** 2 / lam.pow(2).sum()), float(lam.sum())
        m = min(1000, o_real.shape[0])
        sel = torch.randperm(o_real.shape[0])[:m]
        d_r, v_r = deff(o_real[sel])
        d_s, v_s = deff(o_syn[:m])
        ok_d = abs(d_s - d_r) / max(d_r, 1e-9) <= 0.20
        ok_v = 0.8 <= v_s / max(v_r, 1e-12) <= 1.2
        L.append(f"5.3b o 기하  d_eff 실측={d_r:.1f} 합성={d_s:.1f} (비 {d_s/d_r:.2f}, ±20% 게이트 "
                 f"{'OK' if ok_d else '★이탈★'})   var 비={v_s/v_r:.2f} "
                 f"({'OK' if ok_v else '★이탈★'})")

        # b2. w 의 확산 정도 + 분산 성분 분해 (대역폭 진단)
        zz = (s_syn - cbd["mean_s"]) / cbd["std_s"]
        wv = torch.softmax(-torch.cdist(zz, cbd["c"]).pow(2) / cbd["h"] ** 2, 1)
        pr = float((1.0 / wv.pow(2).sum(1)).median())
        omu = wv @ cbd["m"]
        osd2 = (wv @ cbd["sig_o"].pow(2))
        L.append(f"5.3b2 대역폭  h={cbd['h']:.4f}(scale {cb.get('h_scale',1.0)})  "
                 f"w 참여셀 median={pr:.1f}/{cb['K_eff']}  최대가중 median="
                 f"{float(wv.max(1).values.median()):.3f}")
        L.append(f"      분산 분해  평균부={float(omu.var(0).sum()):.1f}  "
                 f"잡음부={float(osd2.sum(1).mean()):.1f}  실측 total={float(o_real.var(0).sum()):.1f}  "
                 f"(평균부 비중 {float(omu.var(0).sum())/max(float(o_real.var(0).sum()),1e-9):.2f})")

        # c. ridge  o ≈ f(s)  (80/20)
        N = s_real.shape[0]
        pm = torch.randperm(N, device=device)
        ntr = int(0.8 * N)
        tr, te = pm[:ntr], pm[ntr:]
        X = torch.cat([s_real, torch.ones(N, 1, device=device)], 1).double()
        Y = o_real.double()
        Xm, Ym = X[tr], Y[tr]
        lam = 1e-3 * float(Xm.shape[0])
        A = Xm.T @ Xm + lam * torch.eye(X.shape[1], device=device, dtype=torch.float64)
        W = torch.linalg.solve(A, Xm.T @ Ym)
        res_te = Y[te] - X[te] @ W
        ss_res = float(res_te.pow(2).sum())
        ss_tot = float((Y[te] - Ym.mean(0)).pow(2).sum())
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        d_res, _ = deff(res_te.float())
        d_pool, _ = deff(Y[te].float())
        L.append(f"5.3c ridge o≈f(s)  held-out R²={r2:.4f}  "
                 f"잔차 d_eff={d_res:.1f} vs 풀드 {d_pool:.1f} (비 {d_res/max(d_pool,1e-9):.2f})")
        gate_fail = (r2 < 0.15) and (d_res / max(d_pool, 1e-9) > 0.9)
        L.append(f"     게이트: {'★전제 붕괴 — s 가 o 를 설명하지 못한다★' if gate_fail else 'OK'}")

        # d. 합성 정합
        Xs = torch.cat([s_syn, torch.ones(s_syn.shape[0], 1, device=device)], 1).double()
        r_syn = float((o_syn.double() - Xs @ W).pow(2).mean().sqrt())
        r_real = float(res_te.pow(2).mean().sqrt())
        L.append(f"5.3d 합성 정합  RMSE(õ−f(s̃))={r_syn:.4f}  실측 잔차 RMSE={r_real:.4f}  "
                 f"비={r_syn/max(r_real,1e-12):.2f} (1~2 기대)")

        # ── §4 v2 기울기 sanity ────────────────────────────────────────
        if self.grad_enable:
            L += self._grad_sanity(cb, cbd, s_real, o_real, device, W, X, r_real)

        L += self._extra_sanity(cb, cbd, s_real, o_real, device, W, X, r_real)

        # 5.4 teacher 스모크
        if self.teacher is not None:
            L.append("5.4 teacher 스모크  건너뜀 (teacher 는 이 뒤에 만들어진다)")
        else:
            L.append("5.4 teacher 스모크  task0 종료 시점엔 teacher 가 없다 — task1 sanity 로 대체")

        L.append("═" * 74)
        logging.info("\n".join(L))
        (self.out / "sanity_task0.txt").write_text("\n".join(L) + "\n")
        json.dump({"pair_ok": ok, "pair_bad": bad, "K_eff": cb["K_eff"], "h": cb["h"],
                   "rel_mean": rel_m, "rel_std": rel_s,
                   "d_eff_real": d_r, "d_eff_syn": d_s, "var_ratio": v_s / v_r,
                   "ridge_r2": r2, "resid_deff": d_res, "pooled_deff": d_pool,
                   "resid_ratio": d_res / max(d_pool, 1e-9),
                   "synth_rmse": r_syn, "real_rmse": r_real, "gate_fail": bool(gate_fail)},
                  (self.out / "sanity_task0.json").open("w"), indent=2)


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambda_level", type=float, default=3.0)
    ap.add_argument("--anchor_norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--chunk_backward", action="store_true")
    ap.add_argument("--log_every_anchor", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--codebook_k", type=int, default=96)
    ap.add_argument("--n_pairs", type=int, default=8000)
    ap.add_argument("--h_scale", type=float, default=1.0,
                    help="softmax 대역폭 배율. 1.0 = 스펙(최근접거리 중앙값)")
    ap.add_argument("--grad_enable", action="store_true",
                    help="셀별 선형 기울기 A_k 사용 (v2). 끄면 v1 과 동일 경로")
    ap.add_argument("--ridge_rho", type=float, default=0.05)
    ap.add_argument("--grad_min_frames", type=int, default=24)
    ap.add_argument("--xt_mode", choices=["teacher", "current"], default="teacher")
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    _forbid_time_bin()

    out_dir = Path(args.out) if args.out else OUT_DIR
    args.out_dir = str(out_dir)
    args.batch_size = 32
    args.p_drop = 0.0
    args.seed = 42
    out_dir.mkdir(parents=True, exist_ok=True)

    B1.ANCHOR = L2CodebookAnchor(args)

    argv = ["B1.py", "--p_drop", "0", "--guidance_w", "1.0", "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir), "--ckpt_root", str(REPO / "outputs" / "L2_codebook")]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru

    json.dump({
        "arm": "L2_codebook", "base": "L2",
        "base_diff": [
            "1. 앵커의 (o,s) 공급원을 시간-bin 가우시안에서 (s,o) 코드북으로 교체. 시간 축 제거.",
            "2. task j 학습 중 같은 프레임의 (state 16, CLS 3072) 쌍을 저수지 표본으로 수집.",
            "3. task 종료 시 표준화 s 공간 k-means(K=96) -> 셀별 (π, μ_s, σ_s, c, m, σ_o) 저장.",
            "4. 앵커에서 k~Cat(π) -> s̃ -> softmax(−‖z(s̃)−c‖²/h²) 가중으로 õ. s̃ 가 õ 를 고른다.",
            "5. teacher 부트스트랩 x_t·λ·reduction·teacher 운용·eval 은 L2 와 동일.",
        ],
        "codebook_k": args.codebook_k, "n_pairs": args.n_pairs, "h_scale": args.h_scale,
        "grad_enable": args.grad_enable, "ridge_rho": args.ridge_rho,
        "grad_min_frames": args.grad_min_frames,
        "xt_mode": args.xt_mode, "lambda_level": args.lambda_level,
        "anchor_norm": args.anchor_norm, "chunk_backward": args.chunk_backward,
        "time_bin": "제거 (R10.compute_stats/phase_bins 를 예외로 차단)",
        "teacher": "rolling (1 snapshot)", "embedding": "dinov2_cls_768_frozen",
        "p_drop": 0.0, "guidance_w": 1.0, "argv": argv,
    }, (out_dir / "l2_codebook_config.json").open("w"), indent=2, ensure_ascii=False)
    # l2_report.py 가 읽는 표준 이름으로도 남긴다
    import shutil
    shutil.copy(out_dir / "l2_codebook_config.json", out_dir / "l2_config.json")

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
