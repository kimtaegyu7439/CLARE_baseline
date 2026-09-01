#!/usr/bin/env python
"""K1 — R13 에서 **관측 합성 블록만** 공유기저 분위수 수송으로 교체한 팔.

R13 과 다른 곳은 딱 하나다: 앵커를 걸 좌표 b_j 를 어떻게 만드는가.

  R13   z ~ N(0,I)                          등방 가우시안 표본
        b_j = mu_j[τ] + sigma_j[τ] · z      태스크별 대각 가우시안으로 되채색

  K1    w   = W^T (o − c0)                  공유 PCA 기저(256d)로 회전
        p_i = F_new,τ,i(w_i)                현재 태스크 분위수표의 CDF
        w'_i= F_j,τ,i^{-1}(p_i)             과거 태스크 표의 역CDF
        b_j = c0 + W w' + res − m⊥_new[τ] + m⊥_j[τ]

무엇이 달라지는 것인가
  (1) 주변분포 충실도. mu/sigma 는 분포를 가우시안으로 못박지만 분위수 사상은
      실제 주변분포를 그대로 옮긴다. 비대칭·다봉이어도 형태가 보존된다.
  (2) 결합구조(copula). 좌표별 CDF 를 통과시켜도 **순위 구조**는 그대로다.
      현재 프레임의 좌표 간 상관이 과거 태스크 좌표계에서 유지된다.
      R13 은 등방 난수라 이것이 아예 없고, R12(수송)는 선형 스케일만 보존한다.
  (3) 좌표계. 공유 PCA 기저는 태스크 0 에서 한 번 정해 동결한다. 태스크마다
      다른 기저를 쓰면 표끼리 비교가 성립하지 않기 때문이다. 기저 밖 잔여
      성분은 모양을 유지한 채 평균만 옮긴다(m⊥).

  ★ results/R10_tsne/pca_tasks_bin5.png 실측: 태스크 간 중심거리가 태스크 내
    산포의 3.13 배다. 즉 지배적 변동 축이 태스크 정체성이라 location-scale
    가정(o = mu_j + sigma_j·ε, ε 공통)이 깨진다. 분위수+copula 는 그 가정을
    쓰지 않는다 — 그것이 이 팔의 동기다.

상속하는 것 (R13 = R10Anchor(use_struct=False, sample_z=True) 와 동일)
  rolling teacher 1개, level 앵커만(structure 없음), λ_level=3, anchor_norm=mean,
  n_bins=10, h = 0.1·median‖o−mu[τ]‖, λ_anchor=1, p_drop=0, guidance_w=1,
  5000 step/task, batch 32, 평가/스케줄/옵티마이저 전부 그대로.

  ★ R13 은 use_struct=False 라 방향 u 가 손실에 들어가지 않는다. K1 도 같다.
    스펙 D 의 u 정의는 구현해 두고 sanity 로그로만 확인한다(‖u‖=1). use_struct
    를 켜는 파생 팔이 생기면 그때 그대로 쓰인다.

금지사항 준수
  학습 중 과거 태스크의 원시 데이터를 로드하지 않는다. 저장물은
  공유기저 W 1개 + 태스크별 {qtab, m⊥} + rolling teacher 1개뿐이다.
"""
from __future__ import annotations

import argparse
import difflib
import inspect
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

OUT_DIR = REPO / "results" / "K1"


# ═════════════════════════════════════════════════════════════════════════════
#  분위수 보간 — 모두 detach 대상(앵커 좌표는 상수)
# ═════════════════════════════════════════════════════════════════════════════
def make_probs(Q: int, device) -> torch.Tensor:
    """p_k = (k − 0.5)/Q, k = 1..Q. 균일 격자라 역보간이 해석적으로 풀린다."""
    return (torch.arange(Q, device=device, dtype=torch.float32) + 0.5) / Q


def cdf_forward(qt: torch.Tensor, probs: torch.Tensor, w: torch.Tensor):
    """F(w). qt (M,Q) 오름차순, w (M,) -> (p (M,), 범위밖 마스크 (M,)).

    구간 밖은 [p_1, p_Q] 로 clamp. (q_Q − q_1) < 1e-6 인 퇴화 좌표는 p = 0.5.
    """
    Q = qt.shape[-1]
    i = torch.searchsorted(qt.contiguous(), w[:, None].contiguous()).squeeze(1)
    out = (i == 0) | (i == Q)
    i = i.clamp(1, Q - 1)
    lo = qt.gather(1, (i - 1)[:, None]).squeeze(1)
    hi = qt.gather(1, i[:, None]).squeeze(1)
    fr = ((w - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)
    p = (probs[i - 1] + fr * (probs[i] - probs[i - 1])).clamp(probs[0], probs[-1])
    deg = (qt[:, -1] - qt[:, 0]) < 1e-6
    p = torch.where(deg, torch.full_like(p, 0.5), p)
    return p, out & ~deg


def cdf_inverse(qt: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """F^{-1}(p). qt (M,Q), p (M,) -> (M,). probs 가 균일 격자라 위치가 닫힌 형태다."""
    Q = qt.shape[-1]
    t = (p * Q - 0.5).clamp(0.0, Q - 1.0)
    i0 = t.floor().long().clamp(0, Q - 2)
    fr = (t - i0.float())[:, None]
    lo = qt.gather(1, i0[:, None])
    hi = qt.gather(1, (i0 + 1)[:, None])
    return (lo + fr * (hi - lo)).squeeze(1)


def quant_at(qt: torch.Tensor, level: float) -> torch.Tensor:
    """표에서 확률수준 level 의 분위수를 보간해 뽑는다. qt (..., Q) -> (...)."""
    Q = qt.shape[-1]
    t = min(max(level * Q - 0.5, 0.0), Q - 1.0)
    i0 = min(int(t), Q - 2)
    fr = t - i0
    return qt[..., i0] + fr * (qt[..., i0 + 1] - qt[..., i0])


# ═════════════════════════════════════════════════════════════════════════════
#  통계 패스 — 공유기저 W + 태스크·단계별 분위수표 + 잔여 평균
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def collect_cls(policy, dataset, train_eps, cfg, device, n_bins, batch_size, prep,
                workers: int = 4):
    """학습 데모 전 프레임의 CLS 를 (N,3072) 로 모으고 phase bin 도 함께 반환.

    ★ R10.compute_stats 와 **같은 순회**다(EpisodeAwareSampler, shuffle=False,
      같은 drop_n_last_frames, 같은 batch_size). 워커 수만 다른데 sampler 순서가
      고정이라 내용은 동일하다. 이 한 패스로 mu/sigma/h 와 분위수표를 모두 내서
      데이터 패스를 태스크당 2회 -> 1회로 줄인다.
    """
    from lerobot.datasets.sampler import EpisodeAwareSampler

    sampler = EpisodeAwareSampler(
        dataset.episode_data_index, episode_indices_to_use=train_eps,
        drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0), shuffle=False)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=workers, batch_size=batch_size, sampler=sampler,
        drop_last=False, pin_memory=(device.type == "cuda"),
        multiprocessing_context="spawn" if workers > 0 else None,
        persistent_workers=False)
    ep_len = R10.episode_lengths(dataset)
    X, T = [], []
    for raw in loader:
        b = prep(policy, B1.to_device(raw, device))
        cls = B1.rgb_cls(policy, b).float()                 # (B*4, 768)
        n = b["observation.state"].shape[0]
        X.append(cls.reshape(n, -1).cpu())                  # (B, 3072) flatten
        T.append(R10.phase_bins(raw, ep_len, n_bins).cpu())
    return torch.cat(X), torch.cat(T)


@torch.no_grad()
def moments_from_cls(X, T, n_bins, device):
    """R10.compute_stats 와 **같은 식**으로 mu/sigma/sigma_floor/h 를 낸다.

        mu    = ssum/c                       (float64 누적 후 float32)
        var   = (ssq/c − mu²).clamp_min(0)
        floor = 0.1 · median(sigma)
        h     = 0.1 · median‖o − mu[τ]‖      (전 학습 프레임, 3072 원소 L2)

    반환 shape 는 R10 과 같은 [n_bins, 4, 768] 이다. 누적 순서만 다르고
    (배치 단위 -> 청크 단위) float64 라 차이는 1e-15 수준이다.
    """
    N, d = X.shape
    ssum = torch.zeros(n_bins, d, device=device, dtype=torch.float64)
    ssq = torch.zeros(n_bins, d, device=device, dtype=torch.float64)
    cnt = torch.zeros(n_bins, device=device, dtype=torch.float64)
    Td = T.to(device)
    for s_ in range(0, N, 2048):
        x = X[s_:s_ + 2048].to(device).double()
        t = Td[s_:s_ + 2048]
        ssum.index_add_(0, t, x)
        ssq.index_add_(0, t, x * x)
        cnt.index_add_(0, t, torch.ones(x.shape[0], device=device, dtype=torch.float64))
        del x
    c = cnt.clamp_min(1.0)[:, None]
    mu = (ssum / c).float()
    sigma = (ssq / c - (ssum / c) ** 2).clamp_min(0.0).sqrt().float()
    floor = float(0.1 * sigma.median())
    norms = []
    for s_ in range(0, N, 2048):
        dev_ = X[s_:s_ + 2048].to(device) - mu[Td[s_:s_ + 2048]]
        norms.append(dev_.norm(dim=1).cpu())
    h = float(0.1 * torch.cat(norms).median())
    return {"mu": mu.view(n_bins, -1, 768), "sigma": sigma.view(n_bins, -1, 768),
            "count": cnt.float(), "B": n_bins, "sigma_floor": floor, "h": h}


@torch.no_grad()
def fit_shared_basis(X: torch.Tensor, r: int, device):
    """태스크 0 데모 전 프레임으로 전역 평균 c0 + 상위 r 주성분 W (3072,r).

    공분산 고유분해(3072x3072)로 푼다 — SVD 보다 메모리가 훨씬 싸고 W 의
    정규직교성이 eigh 로 보장된다. float64 누적.
    """
    N, d = X.shape
    c0 = X.mean(0).to(device)
    C = torch.zeros(d, d, device=device, dtype=torch.float64)
    for s_ in range(0, N, 2048):                            # 청크 누적 — 전량 복사를 피한다
        xc = (X[s_:s_ + 2048].to(device) - c0).double()
        C += xc.T @ xc
        del xc
    C /= max(1, N - 1)                                      # (3072,3072) float64
    evals, evecs = torch.linalg.eigh(C)                     # 오름차순
    idx = torch.argsort(evals, descending=True)[:r]
    W = evecs[:, idx].float().contiguous()                  # (3072, r)
    frac = float(evals[idx].sum() / evals.clamp_min(0).sum())
    del C
    torch.cuda.empty_cache()
    return W, c0.float(), frac        # 둘 다 device 위. 저장할 때만 .cpu() 한다.


@torch.no_grad()
def build_tables(X, T, W, c0, n_bins, Q, device, chunk=2048):
    """태스크·단계별 분위수표 qtab (n_bins,r,Q) 과 잔여 평균 m⊥ (n_bins,3072).

    W 가 None 이면 identity 기저 — w = o − c0, 잔여는 0 이다.
    저장 전 좌표축을 따라 cummax 를 걸어 단조 비감소를 보장한다.
    """
    d = X.shape[1]
    r = d if W is None else W.shape[1]
    Wd = None if W is None else W.to(device)
    c0d = c0.to(device)
    Ws, Rs = [], []
    for s in range(0, X.shape[0], chunk):
        xc = X[s:s + chunk].to(device) - c0d
        if Wd is None:
            Ws.append(xc.cpu()); Rs.append(None)
        else:
            w = xc @ Wd
            Ws.append(w.cpu()); Rs.append((xc - w @ Wd.T).cpu())
        del xc
    Wm = torch.cat(Ws)                                       # (N, r)
    Rm = None if W is None else torch.cat(Rs)                # (N, 3072)
    del Ws, Rs

    probs = make_probs(Q, device)
    qtab = torch.zeros(n_bins, r, Q, device=device)
    mperp = torch.zeros(n_bins, d, device=device)
    counts = torch.zeros(n_bins, dtype=torch.long)
    for t in range(n_bins):
        m = (T == t)
        c = int(m.sum()); counts[t] = c
        if c < 2:                                            # 표본이 없으면 전역으로 대체
            m = torch.ones_like(T, dtype=torch.bool); c = int(m.sum())
        wt = Wm[m].to(device)
        qtab[t] = torch.quantile(wt, probs, dim=0).T         # (r, Q)
        if Rm is not None:
            mperp[t] = Rm[m].to(device).mean(0)
        del wt
    qtab = torch.cummax(qtab, dim=-1).values                 # 단조 비감소 보장
    viol = int((qtab[..., 1:] < qtab[..., :-1]).sum())       # cummax 이후 0 이어야 한다
    return qtab, mperp, counts, viol                         # device 상주


# ═════════════════════════════════════════════════════════════════════════════
#  앵커
# ═════════════════════════════════════════════════════════════════════════════
class K1Anchor(R10.R10Anchor):
    """R13 과 동일한 손실/teacher/스케줄. 관측 합성만 분위수 수송으로 교체."""

    name = "K1"
    use_struct = False        # R13 과 동일 — structure 항 없음
    sample_z = False          # z 경로를 쓰지 않는다(아래 loss 가 대체)

    def __init__(self, args):
        super().__init__(args)
        self.Q = args.quantiles
        self.r = args.rank
        self.marginal = args.marginal
        self.basis = args.basis
        self.iid_sample = args.iid_sample
        self.W = None                     # 공유기저 (3072,r). identity 면 None.
        self.c0 = None                    # 전역 평균 (3072,)
        self.clamp_frac = 0.0
        self.log = (self.out / "k1.jsonl").open("a")

    def describe(self):
        return (f"K1 — 공유기저 분위수 수송(level 만), rolling teacher 1개 + "
                f"표 {len(self.stats)}개, basis={self.basis}, marginal={self.marginal}, "
                f"iid={self.iid_sample}, Q={self.Q}, r={self.r}, λ_lvl={self.lam_lvl}, "
                f"bins={self.n_bins}")

    # ── 태스크 시작 ─────────────────────────────────────────────────────────
    def on_task_start(self, policy, k, args, instructions, device, **kw):
        # ★ R10.R10Anchor.on_task_start 를 부르지 않는다. 그쪽이 도는 데이터 패스와
        #   K1 의 표 패스가 완전히 겹쳐서 태스크당 전 프레임을 두 번 읽게 되기
        #   때문이다(실측: 4 프로세스 병렬에서 패스당 983s). mu/sigma/h 는
        #   moments_from_cls 가 R10 과 같은 식으로 낸다 — 정의는 그대로 상속한다.
        cfg, dataset, train_eps, prep = kw["cfg"], kw["dataset"], kw["train_eps"], kw["prep"]
        self.ep_len = R10.episode_lengths(dataset)
        t0 = time.perf_counter()
        was = policy.training; policy.eval()
        X, T = collect_cls(policy, dataset, train_eps, cfg, device,
                           self.n_bins, args.batch_size, prep, workers=self.a.stats_workers)
        if was:
            policy.train()
        self.cur = moments_from_cls(X, T, self.n_bins, device)

        if self.basis == "identity":
            if self.c0 is None:
                self.c0 = X.mean(0).to(device)
                self.r = X.shape[1]
                torch.save({"W": None, "c0": self.c0.cpu(), "r": self.r,
                            "basis": "identity"}, self.out / "shared_basis.pt")
            wtw = 0.0
        else:
            if self.W is None:                            # 태스크 0 에서 한 번만
                self.W, self.c0, frac = fit_shared_basis(X, self.r, device)
                torch.save({"W": self.W.cpu(), "c0": self.c0.cpu(), "r": self.r,
                            "basis": "shared_pca"}, self.out / "shared_basis.pt")
                logging.info(f"[K1] 공유기저 확정  W{tuple(self.W.shape)}  "
                             f"설명분산 {frac*100:.1f}%  (이후 동결)")
            wtw = float((self.W.T @ self.W
                         - torch.eye(self.r, device=self.W.device)).norm())

        qtab, mperp, counts, viol = build_tables(
            X, T, self.W, self.c0, self.n_bins, self.Q, device)
        self.cur["qtab"] = qtab
        self.cur["m_perp"] = mperp
        med = quant_at(qtab, 0.5)                              # (n_bins, r)
        sc = (quant_at(qtab, 0.841) - quant_at(qtab, 0.159)) / 2
        self.cur["med"] = med
        self.cur["s"] = sc
        self.cur["s_floor"] = float(0.1 * sc.flatten().median())
        del X, T
        torch.cuda.empty_cache()

        # R10.on_task_start 가 하던 스테이지 초기화를 그대로
        self.step = 0
        self.lam_str = None if k > 0 else 0.0
        self.warm = []
        logging.info(
            f"[K1] task {k} 통계+표 {time.perf_counter()-t0:.1f}s  "
            f"qtab{tuple(qtab.shape)}  m⊥{tuple(mperp.shape)}  Q={self.Q}  r={self.r}  "
            f"basis={self.basis}  marginal={self.marginal}  iid={self.iid_sample}  "
            f"h={self.cur['h']:.4f}  sigma_floor={self.cur['sigma_floor']:.5f}  "
            f"‖WᵀW−I‖={wtw:.2e}  단조위반={viol}건  "
            f"s_floor={self.cur['s_floor']:.5f}  bin 표본 {counts.tolist()}")
        self._sanity = True

    # ── 관측 합성 (R13 의 z / u / b_j 를 대체하는 부분) ──────────────────────
    def rotate(self, o):
        """o (B,4,768) -> (w (B,r), res (B,3072)). identity 면 res = 0."""
        n = o.shape[0]
        xc = o.reshape(n, -1) - self.c0
        if self.W is None:
            return xc, torch.zeros_like(xc)
        w = xc @ self.W
        return w, xc - w @ self.W.T

    def _hhat(self, w, tau):
        """ĥ = clip((w − med)/max(s, s_floor), −3, 3). 현재 태스크 표만 쓴다."""
        med = self.cur["med"][tau]
        s = self.cur["s"][tau].clamp_min(self.cur["s_floor"])
        return ((w - med) / s).clamp_(-3.0, 3.0)

    def struct_dir(self, w, tau):
        """u = W ĥ / ‖ĥ‖. W 가 정규직교라 ‖u‖ = 1. (B,4,768) 로 복원."""
        h = self._hhat(w, tau)
        u = h if self.W is None else h @ self.W.T
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return u.reshape(w.shape[0], -1, 768)

    def transport(self, w, res, tau, j):
        """w -> 과거 태스크 j 좌표로. b_j = c0 + W w' + res − m⊥_new[τ] + m⊥_j[τ]."""
        dev = w.device
        B, r = w.shape
        qtab_j = self.stats[j]["qtab"]
        qj = qtab_j[tau].reshape(-1, self.Q)                             # (B*r, Q)

        if self.marginal == "zscore":
            # 표에서 med/s 만 써서 표준화 -> 과거 med/s 로 재채색. 주변분포 형태는 버린다.
            hh = self._hhat(w, tau)
            mj = quant_at(qtab_j, 0.5)[tau]
            sj = ((quant_at(qtab_j, 0.841) - quant_at(qtab_j, 0.159)) / 2)[tau]
            wp = mj + sj.clamp_min(self.cur["s_floor"]) * hh
            self.clamp_frac = 0.0
        else:
            probs = make_probs(self.Q, dev)
            if self.iid_sample:
                # copula 를 끊는 negative control — 좌표마다 독립 균등 p.
                p = probs[0] + (probs[-1] - probs[0]) * torch.rand(B * r, device=dev)
                self.clamp_frac = 0.0
            else:
                qn = self.cur["qtab"][tau].reshape(-1, self.Q)           # (B*r, Q)
                p, out = cdf_forward(qn, probs, w.reshape(-1))
                self.clamp_frac = float(out.float().mean())
            wp = cdf_inverse(qj, p).view(B, r)

        if self.W is None:
            b = self.c0 + wp
        else:
            b = (self.c0 + wp @ self.W.T + res
                 - self.cur["m_perp"][tau] + self.stats[j]["m_perp"][tau])
        return b.reshape(B, -1, 768)

    # ── 손실 — R10.loss 의 복사본에서 합성 블록만 교체 ───────────────────────
    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        if k == 0 or self.teacher is None or args.lambda_anchor == 0:
            return torch.zeros((), device=device)
        cls = getattr(self, "cls", None)
        if cls is None:
            cls = B1.rgb_cls(policy, batch)
        n = batch["observation.state"].shape[0]
        o = cls.view(n, -1, cls.shape[-1]).float()                  # (B,4,768)
        tau = R10.phase_bins(batch, self.ep_len, self.n_bins).to(device)

        # ★★★ 여기가 R13 과 다른 유일한 블록 ★★★
        h = self.cur["h"]                              # 보폭은 R13 정의 그대로 상속
        with torch.no_grad():
            w, res = self.rotate(o)
            w, res = w.detach(), res.detach()
            u = self.struct_dir(w, tau).detach()
        # ★★★ 여기까지 ★★★

        chunk = self.a.chunk_backward
        if chunk and getattr(policy.config, "use_amp", False):
            raise RuntimeError("chunk_backward 는 use_amp=True 와 함께 쓸 수 없다")

        lvl, stc = [], []
        teach = self.teacher                        # rolling — j 와 무관하게 하나
        for j in sorted(self.stats):
            past = [instructions[f"task{j}"]] * n
            with torch.no_grad():                                  # ★ 교체된 b_j
                b_j = self.transport(w, res, tau, j).detach()

            def fwd(pol, c):                       # 주입된 CLS 로 velocity 를 낸다
                flat = c.reshape(-1, c.shape[-1]).to(x_t.dtype)
                tl = B1.cond_tail(pol, batch, flat)
                return pol.dit_flow.velocity_net(
                    noisy_actions=x_t, time=t,
                    global_cond=B1.make_cond(B1.encode_lang(pol, past), tl))

            with torch.no_grad():
                vt0 = fwd(teach, b_j)
            vs0 = fwd(policy, b_j)
            r0 = vs0 - vt0.to(vs0.dtype)
            if self.use_struct:
                b_h = (b_j + h * u).detach()
                with torch.no_grad():
                    vt1 = fwd(teach, b_h)
                vs1 = fwd(policy, b_h)
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

            if getattr(self, "_sanity", False) and j == min(self.stats):
                with torch.no_grad():
                    st_msg = (
                        f"‖(r1−r0)/h‖={float(((r1-r0)/h).flatten(1).norm(dim=1).mean()):.4f}"
                        if self.use_struct else "structure 없음")
                    hh = self._hhat(w, tau)
                    logging.info(
                        f"[K1][sanity] task{k} j={j}  clamp={self.clamp_frac*100:.2f}%"
                        f"{'  ⚠5%초과' if self.clamp_frac > 0.05 else ''}  "
                        f"ĥ 평균={float(hh.mean()):+.4f} 분산={float(hh.var()):.4f}  "
                        f"‖u‖={float(u.flatten(1).norm(dim=1).mean()):.4f}  "
                        f"b_j 유한={bool(torch.isfinite(b_j).all())}  "
                        f"teacher 유한={bool(torch.isfinite(vt0).all())}  "
                        f"‖r0‖={float(r0.flatten(1).norm(dim=1).mean()):.4f}  "
                        f"{st_msg}  null 호출={self.null_calls}")
                self._sanity = False

        L_lvl = sum(lvl) / len(lvl)
        L_str = sum(stc) / len(stc)

        if not self.use_struct:
            self.lam_str = 0.0
        if self.lam_str is None:
            self.warm.append((float(L_lvl.detach()), float(L_str.detach())))
            if len(self.warm) >= self.a.warmup_steps:
                ml = sum(x for x, _ in self.warm) / len(self.warm)
                ms = sum(y for _, y in self.warm) / len(self.warm)
                self.lam_str = self.a.rho * self.lam_lvl * ml / max(ms, 1e-12)
                logging.info(f"[K1] task {k} λ_struct = {self.lam_str:.6g}")
            out = self.lam_lvl * L_lvl
        else:
            out = self.lam_lvl * L_lvl + self.lam_str * L_str
        if chunk:
            out = out.detach()

        self.step += 1
        if self.step % self.a.log_every_anchor == 0:
            self.log.write(json.dumps({
                "task": k, "step": self.step, "L_level": float(L_lvl.detach()),
                "L_struct": float(L_str.detach()), "lambda_struct": self.lam_str,
                "h": h, "clamp_frac": self.clamp_frac,
                "n_past": len(self.stats)}) + "\n")
            self.log.flush()
        return out

    # ── 태스크 종료 — teacher rolling 은 R10 그대로, 저장물만 스펙 형식 ──────
    def on_task_end(self, policy, k, args, instructions, device, **kw):
        super().on_task_end(policy, k, args, instructions, device, **kw)
        # 저장물은 스펙 형식으로 다시 쓴다. 메모리에는 수송에 필요한 것만 GPU 로
        # 남긴다 — qtab 164KB + m⊥ 123KB 라 10 태스크에도 3 MB 다.
        torch.save({"qtab": self.cur["qtab"].cpu(), "m_perp": self.cur["m_perp"].cpu(),
                    "Q": self.Q, "r": self.r, "n_bins": self.n_bins,
                    "encoder": "dinov2_cls_frozen"},
                   self.out / "stats" / f"task{k}.pt")
        self.stats[k] = {"qtab": self.cur["qtab"], "m_perp": self.cur["m_perp"]}


# ═════════════════════════════════════════════════════════════════════════════
def loss_diff() -> str:
    """R10.loss 대비 실제로 달라진 줄. 완료 보고용이자 회귀 감시용이다."""
    a = inspect.getsource(R10.R10Anchor.loss).splitlines()
    b = inspect.getsource(K1Anchor.loss).splitlines()
    return "\n".join(l for l in difflib.unified_diff(a, b, "R10.loss", "K1.loss", n=0)
                     if l.startswith(("+", "-", "@")) and not l.startswith(("+++", "---")))


def write_table(out_dir: Path, suite: str, r13_ref: str) -> None:
    src = out_dir / "sr_matrix.csv"
    if not src.exists():
        print("[K1] sr_matrix.csv 없음 — 표 생략"); return
    cells, K = {}, 0
    for line in src.read_text().splitlines():
        if line.startswith("#"):
            continue
        f = line.split(",")
        if not f or not f[0].strip().isdigit():
            continue
        k = int(f[0]); K = max(K, len(f) - 1)
        for t, v in enumerate(f[1:]):
            if v.strip():
                cells[(k, t)] = float(v)
    with (out_dir / "sr_table.csv").open("w") as fp:
        fp.write("after_task," + ",".join(f"task{t}" for t in range(K)) + "\n")
        for k in range(K):
            fp.write(f"{k}," + ",".join(
                f"{cells[(k,t)]:.1f}" if (k, t) in cells else "" for t in range(K)) + "\n")
    last = [cells.get((K - 1, t)) for t in range(K)]
    got = [v for v in last if v is not None]
    avg = sum(got) / max(1, len(got))
    diag = [cells.get((k, k)) for k in range(K)]
    acq = [v for v in diag if v is not None]
    L = [f"# K1 — 공유기저 분위수 수송 level 앵커 ({suite}, {K} task, 20 rollout/칸)", "",
         "| after task | " + " | ".join(f"task{t}" for t in range(K)) + " |",
         "|---" * (K + 1) + "|"]
    for k in range(K):
        L.append(f"| {k} | " + " | ".join(
            f"{cells[(k,t)]:.0f}" if (k, t) in cells else "" for t in range(K)) + " |")
    L += ["", f"**AvgSR (마지막 행 평균) = {avg:.1f}**   "
              f"습득(대각 평균) = {sum(acq)/max(1,len(acq)):.1f}", "",
          f"참고값  {r13_ref}   ER(4task) = 93.8   ER(10task, spatial) = 86.0"]
    (out_dir / "sr_table.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nsaved -> {out_dir/'sr_table.csv'}, {out_dir/'sr_table.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # ── R13 에서 상속한 것 (기본값 동일) ────────────────────────────────────
    ap.add_argument("--lambda_level", type=float, default=3.0)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--anchor_norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--stats_batches", type=int, default=0)
    ap.add_argument("--stats_workers", type=int, default=4,
                    help="통계/표 패스의 DataLoader 워커 수. 알고리즘과 무관.")
    ap.add_argument("--chunk_backward", action="store_true")
    ap.add_argument("--log_every_anchor", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--out", default=None)
    # ── K1 고유 (ablation 축) ───────────────────────────────────────────────
    ap.add_argument("--quantiles", type=int, default=16, help="분위수 개수 Q")
    ap.add_argument("--rank", type=int, default=256, help="공유 PCA 기저 차원 r")
    ap.add_argument("--marginal", choices=["quantile", "zscore"], default="quantile",
                    help="주변분포 충실도 축. zscore 는 med/s 만 써서 표준화-재채색")
    ap.add_argument("--basis", choices=["shared_pca", "identity"], default="shared_pca",
                    help="좌표계 축. identity 는 3072 raw 축에서 수송(m⊥ 없음)")
    ap.add_argument("--iid_sample", action="store_true",
                    help="copula 를 끊는 negative control — 좌표마다 독립 균등 p")
    ap.add_argument("--suite", default="libero_spatial", help="표/설정 기록용(passthru 와 일치시킬 것)")
    ap.add_argument("--r13_ref", default="R13(4task) = 85.0, R13(10task) = 79.5")
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    # R13 이 고정하는 값들 — 그대로 상속한다
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

    B1.ANCHOR = K1Anchor(args)

    argv = ["B1.py",
            "--p_drop", "0",
            "--guidance_w", "1.0",
            "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir),
            "--ckpt_root", str(REPO / "outputs" / out_dir.name)]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru
    if "--suite" in argv:
        args.suite = argv[argv.index("--suite") + 1]

    json.dump({"arm": "K1", "base": "R13", "anchor_coord": "shared-basis quantile transport",
               "marginal": args.marginal, "basis": args.basis, "iid_sample": args.iid_sample,
               "Q": args.quantiles, "r": args.rank,
               "structure_term": False, "sample_z": False,
               "lambda_level": args.lambda_level, "n_bins": args.n_bins,
               "anchor_norm": args.anchor_norm, "chunk_backward": args.chunk_backward,
               "stats_batches": args.stats_batches, "stats_workers": args.stats_workers,
               "rho": args.rho,
               "warmup_steps": args.warmup_steps, "use_ghat_weight": args.use_ghat_weight,
               "lambda_swap": args.lambda_swap,
               "teacher": "rolling (1 snapshot)", "embedding": "dinov2_cls_768_frozen",
               "p_drop": 0.0, "guidance_w": 1.0, "batch_size": 32,
               "student_forward_per_step": "1+K", "suite": args.suite, "argv": argv},
              (out_dir / "k1_config.json").open("w"), indent=2, ensure_ascii=False)

    (out_dir / "k1_loss_diff.txt").write_text(loss_diff())
    print(f"[K1] R10.loss 대비 코드 차이 -> {out_dir/'k1_loss_diff.txt'}")

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old
    write_table(out_dir, args.suite, args.r13_ref)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
