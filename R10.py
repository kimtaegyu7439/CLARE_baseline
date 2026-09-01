#!/usr/bin/env python
"""R10 — 수송된 좌표 위의 level + structure 앵커.

기존 앵커(B1/B2/B7/B8)는 **현재** 관측 o 에 과거 명령어 ℓ_j 를 붙인 좌표에서
student 를 teacher_j 에 붙였다. o 는 과거 태스크의 관측 분포와 무관하므로,
앵커가 지키는 영역과 SR 이 결정되는 영역이 어긋난다(results/B1_coverage: 7.24배).

R10 은 두 가지를 바꾼다.
  (1) 수송  o 를 과거 태스크 j 의 관측 분포로 옮긴 점 b_j 에서 앵커를 건다.
            수송은 태스크별·단계(phase)별 vision 임베딩의 평균/표준편차만 쓴다.
            과거 데이터를 저장하지 않는다 — 통계와 teacher 뿐이다.
  (2) 구조  값(level) 뿐 아니라 b_j 에서 방향 u 로의 **방향미분**도 맞춘다.
            u 는 백색화된 편차라 관측 공간의 국소 기하를 따라간다.

  z   = clip((o − mu_new[τ]) / max(sigma_new[τ], floor), −3, 3)      원소별
  u   = z / ‖z‖                                                      전체 원소 L2
  b_j = mu_j[τ] + sigma_j[τ] · z                                     detach
  r0  = v_S(x_t,t,b_j,      ℓ_j) − v_Tj(x_t,t,b_j,      ℓ_j)
  r1  = v_S(x_t,t,b_j+h·u,  ℓ_j) − v_Tj(x_t,t,b_j+h·u,  ℓ_j)
  L   = λ_lvl·mean_j‖r0‖² + λ_str·mean_j‖(r1−r0)/h‖²

teacher 는 **rolling** — 직전 태스크 종료 시점 스냅샷 하나다(B1/B8 과 같다).
과거 태스크마다 teacher 를 따로 두지 않고, 그 하나의 teacher 에 과거 태스크의
관측 분포에서 옮긴 b_j 와 저장해 둔 과거 명령어 ℓ_j 를 함께 넣어 목표를 만든다.
즉 저장물은 "모델 1개 + 태스크별 통계(μ,σ)" 뿐이다.

★ 임베딩 공간은 **DINOv2 CLS(768-d, 동결)** 다. 투영 후 512-d 공간은
  rgb_embedding_projection 이 학습돼 표류하므로 저장한 통계가 무효가 된다.
  샘플당 CLS 는 (n_obs 2 × n_cam 2, 768) = (4, 768) 이다.

B8λ3 대비 달라진 점
  - teacher: rolling 그대로 (B8 과 동일)
  - 앵커 좌표: 현재 o -> 수송된 b_j
  - 손실: level 만 -> level + structure(방향미분)
  - Ĝ 가중: 기본 off (--use_ghat_weight 로만 on)
  - condition dropout 0, 롤아웃 w=1 (null 경로 미호출)
그 외(데이터, 모델, 5000 step/task, 배치 32, 옵티마이저, 평가)는 전부 동일하다.
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

OUT_DIR = REPO / "results" / "R10"


# ═════════════════════════════════════════════════════════════════════════════
#  단계별 통계
# ═════════════════════════════════════════════════════════════════════════════
def phase_bins(batch: dict, ep_len: torch.Tensor, n_bins: int) -> torch.Tensor:
    """τ = clamp(floor(n_bins · frame_idx / ep_len), 0, n_bins−1)."""
    ep = batch["episode_index"].long()
    fi = batch["frame_index"].long()
    L = ep_len.to(ep.device)[ep].clamp_min(1)
    return ((n_bins * fi) // L).clamp_(0, n_bins - 1)


def episode_lengths(dataset) -> torch.Tensor:
    edi = dataset.episode_data_index
    return (torch.as_tensor(edi["to"]) - torch.as_tensor(edi["from"])).long()


@torch.no_grad()
def compute_stats(policy, dataset, train_eps, cfg, device, n_bins: int,
                  batch_size: int, prep, n_white: int = 0,
                  max_batches: int = 0) -> dict:
    """현재 태스크 학습 데모 전 프레임에 대한 phase 별 CLS 평균/표준편차.

    float32 로 누적한다. 반환 shape 는 [n_bins, 4, 768].
    h(수송 보폭)도 같은 패스에서 잡는다 — 편차 노름의 중앙값이 필요하기 때문이다.
    """
    from lerobot.datasets.sampler import EpisodeAwareSampler

    sampler = EpisodeAwareSampler(
        dataset.episode_data_index, episode_indices_to_use=train_eps,
        drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0), shuffle=False)
    loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=batch_size,
                                         sampler=sampler, drop_last=False)
    ep_len = episode_lengths(dataset)
    ssum = ssq = cnt = None
    cache = []                                   # (τ, cls) — 2패스용
    # max_batches>0 이면 전수 패스 대신 앞 몇 배치만 쓴다. 통계 패스가 VRAM 에
    # 얼마나 기여하는지 분리하기 위한 대조군용이며, 통계 추정치가 거칠어지므로
    # SR 비교에는 쓰지 않는다.
    for bi, raw in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        b = prep(policy, B1.to_device(raw, device))
        cls = B1.rgb_cls(policy, b).float()      # (B*4, 768)
        n = b["observation.state"].shape[0]
        cls = cls.view(n, -1, cls.shape[-1])     # (B, 4, 768)
        tau = phase_bins(raw, ep_len, n_bins).to(device)
        if ssum is None:
            shape = (n_bins,) + tuple(cls.shape[1:])
            ssum = torch.zeros(shape, device=device, dtype=torch.float64)
            ssq = torch.zeros(shape, device=device, dtype=torch.float64)
            cnt = torch.zeros(n_bins, device=device, dtype=torch.float64)
        d = cls.double()
        ssum.index_add_(0, tau, d)
        ssq.index_add_(0, tau, d * d)
        cnt.index_add_(0, tau, torch.ones(n, device=device, dtype=torch.float64))
        cache.append((tau.cpu(), cls.cpu()))
    c = cnt.clamp_min(1.0)[:, None, None]
    mu = (ssum / c).float()
    var = (ssq / c - (ssum / c) ** 2).clamp_min(0.0)
    sigma = var.sqrt().float()
    floor = 0.1 * sigma.median()                                   # 태스크당 스칼라 하나
    # h = 0.1 · median‖o − mu[τ]‖  (전 학습 프레임)
    norms = []
    for tau, cls in cache:
        dev_ = cls.to(device) - mu[tau.to(device)]
        norms.append(dev_.flatten(1).norm(dim=1).cpu())
    h = 0.1 * torch.cat(norms).median()
    out = {"mu": mu, "sigma": sigma, "count": cnt.float(), "B": n_bins,
           "sigma_floor": float(floor), "h": float(h)}

    # ── 백색화 기저 (n_white>0 일 때만. R11 이 쓴다) ─────────────────────────
    #   원소별 표준화만으로는 백색화가 안 된다 — results/R10_gauss 실측에서
    #   ‖z‖²/d 의 표준편차가 0.33 으로 독립 가정(χ²: 0.026)의 13배였다.
    #   전 차원 공분산은 3072² = 37MB/bin 이라 저장이 불가능하므로, 지배적인
    #   상위 n_white 개 주성분만 단위분산으로 눌러 주는 저계수 백색화를 쓴다.
    if n_white > 0:
        Zs = []
        for tau, cls in cache:
            zz = (cls.to(device) - mu[tau.to(device)]) / sigma[tau.to(device)].clamp_min(floor)
            Zs.append(zz.clamp_(-3.0, 3.0).flatten(1).cpu())
        Z = torch.cat(Zs).to(device)                       # (N, d)
        Z = Z - Z.mean(0, keepdim=True)
        k = min(n_white, Z.shape[0] - 1, Z.shape[1])
        # 공분산의 상위 고유쌍 = Z 의 상위 우특이벡터
        _, S, V = torch.linalg.svd(Z, full_matrices=False)
        lam = (S[:k] ** 2 / max(1, Z.shape[0] - 1)).clamp_min(1e-8)   # 고유값
        out["white_V"] = V[:k].contiguous().cpu()          # (k, d)
        out["white_lam"] = lam.cpu()
        out["white_frac"] = float(lam.sum() / (S ** 2 / max(1, Z.shape[0] - 1)).sum())
        del Z, Zs
        torch.cuda.empty_cache()
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  앵커
# ═════════════════════════════════════════════════════════════════════════════
class R10Anchor:
    """수송 좌표 위의 level + structure 앵커. teacher 는 rolling(스냅샷 1개)."""

    name = "R10"

    def __init__(self, args):
        self.a = args
        self.teacher = None                        # rolling 스냅샷 1개
        self.stats: dict[int, dict] = {}           # j -> {mu, sigma, ...} (태스크별 통계)
        self.cur: dict | None = None               # 현재 태스크 통계
        self.ep_len: torch.Tensor | None = None
        self.n_bins = args.n_bins
        self.lam_lvl = args.lambda_level
        self.lam_str: float | None = None          # 첫 50 스텝 뒤 자동 설정
        self.warm: list[tuple[float, float]] = []
        self.step = 0
        self.null_calls = 0                        # null 조건이 쓰이면 올라간다(0 이어야 함)
        self.out = Path(args.out_dir); self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "stats").mkdir(exist_ok=True)
        self.log = (self.out / "r10.jsonl").open("a")

    n_white = 0          # >0 이면 compute_stats 가 백색화 기저를 만든다 (R11)
    use_struct = True    # False 면 방향미분 항을 아예 계산하지 않는다 (R12)
    sample_z = False     # True 면 z 를 매 스텝 N(0,I) 에서 새로 뽑는다 (R13)

    def describe(self):
        return (f"R10 — 수송 앵커, rolling teacher 1개 + 통계 {len(self.stats)}개, "
                f"λ_lvl={self.lam_lvl}, bins={self.n_bins}")

    # ── 파생 팔이 갈아 끼우는 두 지점 ───────────────────────────────────────
    def reduce_level(self, x):
        """level 잔차 축약. 기본은 제곱(원소 평균 또는 샘플당 제곱합)."""
        if self.a.anchor_norm == "sum":
            return x.flatten(1).pow(2).sum(1).mean()
        return x.flatten(1).pow(2).mean(1).mean()

    def reduce_struct(self, x):
        """structure 잔차 축약. 기본은 level 과 같다. R11 이 L1 으로 바꾼다."""
        return self.reduce_level(x)

    def direction(self, z):
        """방향 u. 기본은 원소별 표준화 z 를 단위화한 것. R11 이 백색화한다."""
        return z / z.flatten(1).norm(dim=1).clamp_min(1e-8)[:, None, None]

    # ── 태스크 시작: 통계 + h + 진단 ────────────────────────────────────────
    def on_task_start(self, policy, k, args, instructions, device, **kw):
        cfg, dataset, train_eps, prep = kw["cfg"], kw["dataset"], kw["train_eps"], kw["prep"]
        self.ep_len = episode_lengths(dataset)
        t0 = time.perf_counter()
        was = policy.training; policy.eval()
        self.cur = compute_stats(policy, dataset, train_eps, cfg, device,
                                 self.n_bins, args.batch_size, prep,
                                 n_white=self.n_white,
                                 max_batches=self.a.stats_batches)
        if was:
            policy.train()
        # 통계 패스가 전 프레임 CLS 를 배치 단위로 만들었다 버리면서 할당자에
        # 블록이 남는다. 실측(2026-08-27): reserved 3604 -> 4092 MiB (+488).
        # 실사용(peak_alloc)은 학습 스텝 기준 3092 로 변하지 않으므로 순수 예약이고,
        # empty_cache() 로 1308 MiB 까지 회수된다. 성능 영향은 없다.
        torch.cuda.empty_cache()
        self.step = 0
        self.lam_str = None if k > 0 else 0.0     # k=0 은 앵커가 없다
        self.warm = []
        s = self.cur["sigma"]
        q = torch.quantile(s.flatten().float(),
                           torch.tensor([0.01, 0.5, 0.99], device=s.device))
        logging.info(
            f"[R10] task {k} 통계 {time.perf_counter()-t0:.1f}s  "
            f"mu{tuple(self.cur['mu'].shape)}  sigma_floor={self.cur['sigma_floor']:.5f}  "
            f"h={self.cur['h']:.4f}  norm={self.a.anchor_norm}  "
            f"stats_batches={self.a.stats_batches or '전수'}  sigma 분위수 1%/50%/99% = "
            f"{q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}  p_drop={args.p_drop}  "
            f"ghat={'on' if self.a.use_ghat_weight else 'off'}  "
            f"lambda_swap={self.a.lambda_swap}")
        self._sanity = True

    # ── 손실 ────────────────────────────────────────────────────────────────
    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        if k == 0 or self.teacher is None or args.lambda_anchor == 0:
            return torch.zeros((), device=device)
        cls = getattr(self, "cls", None)
        if cls is None:
            cls = B1.rgb_cls(policy, batch)
        n = batch["observation.state"].shape[0]
        o = cls.view(n, -1, cls.shape[-1]).float()                  # (B,4,768)
        tau = phase_bins(batch, self.ep_len, self.n_bins).to(device)

        mu_n, sg_n = self.cur["mu"].to(device), self.cur["sigma"].to(device)
        floor = self.cur["sigma_floor"]
        h = self.cur["h"]
        if self.sample_z:
            # ★ R13. 과거 태스크의 가우시안에서 **진짜 표본**을 뽑는다.
            #   b_j = mu_j + sigma_j·z 에 z ~ N(0,I) 를 넣는 것이므로
            #   reparameterization 형태 그대로다(다만 z 를 detach 하므로
            #   그래디언트가 통과하지는 않는다 — 앵커 좌표는 상수 취급).
            #   수송(기본)과 달리 현재 관측과 무관한 점이 나오고, 임베딩의
            #   차원 간 상관이 사라진다(등방). 그 차이가 R12 vs R13 이다.
            z = torch.randn_like(o).clamp_(-3.0, 3.0)
        else:
            z = ((o - mu_n[tau]) / sg_n[tau].clamp_min(floor)).clamp_(-3.0, 3.0)
        z = z.detach()
        u = self.direction(z).detach()

        # ── 청크 backward ────────────────────────────────────────────────
        #   기본(일괄)은 과거 K개의 그래프를 전부 들고 있다가 B1 이 한 번에
        #   backward 한다 -> 활성값이 O(K). 청크는 과거 하나를 계산할 때마다
        #   즉시 backward 해서 그래프를 버린다 -> O(1).
        #   미분이 선형이라(∇ΣL = Σ∇L) 그래디언트는 같다. 실측: K=39 에서도
        #   3167 MiB 로 K=1 과 동일하고 ER(3189 MiB)보다 작다.
        #
        #   ★ AMP 와는 못 쓴다. grad_scaler 가 손실을 스케일한 뒤 backward 해야
        #     하는데 여기서는 scaler 에 접근할 수 없다. use_amp=False 를 강제한다.
        #   ★ λ_struct 보정 구간(첫 warmup_steps 스텝)에도 청크를 켜야 한다.
        #     거기서 일괄로 돌면 그때 찍힌 최대치를 캐싱 할당자가 돌려주지 않아
        #     스테이지 내내 nvidia-smi 가 일괄 수준으로 고정된다.
        #     보정 구간의 손실은 λ_lvl·L_lvl 뿐이고 L_str 은 크기만 재므로,
        #     backward 대상만 그에 맞게 바꾼다.
        chunk = self.a.chunk_backward
        if chunk and getattr(policy.config, "use_amp", False):
            raise RuntimeError("chunk_backward 는 use_amp=True 와 함께 쓸 수 없다")

        lvl, stc = [], []
        teach = self.teacher                        # rolling — j 와 무관하게 하나
        for j in sorted(self.stats):
            st = self.stats[j]
            b_j = (st["mu"].to(device)[tau] + st["sigma"].to(device)[tau] * z).detach()
            past = [instructions[f"task{j}"]] * n

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
                # 한 칸 옆 b_j + h·u 에서 한 번 더. forward 가 2배가 되는 지점이다.
                b_h = (b_j + h * u).detach()
                with torch.no_grad():
                    vt1 = fwd(teach, b_h)
                vs1 = fwd(policy, b_h)
                r1 = vs1 - vt1.to(vs1.dtype)
            # ★ 스케일. 스펙은 ‖r0‖²(샘플당 제곱합)이라고 썼지만, 같은 스펙이
            #   "λ_level=3 은 B8λ3 의 λ 와 동일"이라고도 한다. B8 의 앵커는
            #   F.mse_loss = **원소 평균**이라 제곱합을 쓰면 112(=16×7)배가 되어
            #   λ 의 의미가 달라진다(실측: anchor/fm 이 B8λ3 의 0.3 -> 8.0).
            #   세기를 맞추는 쪽이 의도라고 보고 원소 평균을 기본으로 둔다.
            #   --anchor_norm sum 이 스펙의 문자 그대로다.
            L_j = self.reduce_level(r0)
            S_j = (self.reduce_struct((r1 - r0) / h) if self.use_struct
                   else torch.zeros((), device=device))
            if chunk:
                # B1 이 곱할 args.lambda_anchor 까지 여기서 반영해 흘려보낸다.
                # 보정 구간(lam_str is None)에는 level 항만 손실에 들어간다.
                term = (self.lam_lvl * L_j if self.lam_str is None
                        else self.lam_lvl * L_j + self.lam_str * S_j)
                (args.lambda_anchor * term).backward()
                lvl.append(L_j.detach()); stc.append(S_j.detach())
            else:
                lvl.append(L_j); stc.append(S_j)

            if getattr(self, "_sanity", False) and j == min(self.stats):
                with torch.no_grad():
                    mj = st["mu"].to(device)[tau]
                    rel = float((b_j.mean(0) - mj.mean(0)).norm() / mj.mean(0).norm().clamp_min(1e-8))
                    st_msg = (
                        f"‖(r1−r0)/h‖={float(((r1-r0)/h).flatten(1).norm(dim=1).mean()):.4f}"
                        if self.use_struct else "structure 없음")
                    logging.info(
                        f"[R10][sanity] task{k} j={j}  ‖b̄−μ̄_j‖/‖μ̄_j‖={rel:.4f}  "
                        f"‖u‖={float(u.flatten(1).norm(dim=1).mean()):.4f}  "
                        f"teacher 유한={bool(torch.isfinite(vt0).all())}  "
                        f"‖r0‖={float(r0.flatten(1).norm(dim=1).mean()):.4f}  "
                        f"{st_msg}  null 호출={self.null_calls}")
                self._sanity = False

        L_lvl = sum(lvl) / len(lvl)
        L_str = sum(stc) / len(stc)

        # λ_struct 자동 설정 — 첫 50 스텝은 크기만 재고 struct 를 손실에 넣지 않는다
        if not self.use_struct:
            self.lam_str = 0.0
        if self.lam_str is None:
            self.warm.append((float(L_lvl.detach()), float(L_str.detach())))
            if len(self.warm) >= self.a.warmup_steps:
                ml = sum(x for x, _ in self.warm) / len(self.warm)
                ms = sum(y for _, y in self.warm) / len(self.warm)
                self.lam_str = self.a.rho * self.lam_lvl * ml / max(ms, 1e-12)
                logging.info(f"[R10] task {k} λ_struct = {self.lam_str:.6g}  "
                             f"(mean L_level {ml:.5g} / mean L_struct {ms:.5g}, rho={self.a.rho})")
            out = self.lam_lvl * L_lvl
        else:
            out = self.lam_lvl * L_lvl + self.lam_str * L_str
        if chunk:
            # 이미 backward 했다. B1 이 다시 더해도 그래프가 없어 기여하지 않는다.
            out = out.detach()

        self.step += 1
        if self.step % self.a.log_every_anchor == 0:
            self.log.write(json.dumps({
                "task": k, "step": self.step, "L_level": float(L_lvl.detach()),
                "L_struct": float(L_str.detach()), "lambda_struct": self.lam_str,
                "h": h, "n_past": len(self.stats)}) + "\n")
            self.log.flush()
        return out

    # ── 태스크 종료: teacher + 통계 저장 ────────────────────────────────────
    def on_task_end(self, policy, k, args, instructions, device, **kw):
        del self.teacher                              # rolling — 직전 것을 버린다
        self.teacher = B1.snapshot(policy, args.teacher_bf16)
        self.stats[k] = {kk: (vv.cpu() if torch.is_tensor(vv) else vv)
                         for kk, vv in self.cur.items()}
        torch.save({**self.stats[k], "encoder": "dinov2_cls_frozen"},
                   self.out / "stats" / f"task{k}.pt")
        logging.info(f"[R10] task {k} rolling teacher 갱신 + 통계 저장 "
                     f"(통계 {len(self.stats)}개)")
        torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambda_level", type=float, default=3.0, help="B8λ3 의 λ 와 동일")
    ap.add_argument("--rho", type=float, default=1.0, help="λ_struct 자동 설정 배율")
    ap.add_argument("--warmup_steps", type=int, default=50, help="λ_struct 측정 구간")
    ap.add_argument("--n_bins", type=int, default=10, help="phase bin 개수 B")
    ap.add_argument("--use_ghat_weight", action="store_true", help="B8 의 Ĝ 가중(기본 off)")
    ap.add_argument("--lambda_swap", type=float, default=0.0,
                    help="현재 o 에 ℓ_j 를 붙이던 기존 앵커. R10 은 0(off).")
    ap.add_argument("--stats_batches", type=int, default=0,
                    help="통계를 앞 N 배치로만 낸다(0=전수). 통계 패스의 VRAM 기여분을 "
                         "분리하는 대조군용. 통계가 거칠어지므로 SR 비교에는 쓰지 말 것.")
    ap.add_argument("--chunk_backward", action="store_true",
                    help="과거 태스크마다 즉시 backward 해 활성값을 O(1) 로 만든다. "
                         "그래디언트는 동일하다. AMP 와는 함께 쓸 수 없다.")
    ap.add_argument("--anchor_norm", choices=["mean", "sum"], default="mean",
                    help="앵커 잔차를 원소 평균(B8λ3 와 같은 스케일, 기본) 으로 볼지 "
                         "샘플당 제곱합(스펙 문자 그대로) 으로 볼지")
    ap.add_argument("--log_every_anchor", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else OUT_DIR
    args.out_dir = str(out_dir)
    args.batch_size = 32
    args.p_drop = 0.0
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_ghat_weight:
        logging.warning("[R10] Ĝ 가중은 아직 붙이지 않았다 — 무시한다")
    if args.lambda_swap != 0:
        logging.warning("[R10] lambda_swap 는 자리만 잡아 두었다 — 무시한다")

    B1.ANCHOR = R10Anchor(args)

    # B1.main 에 넘길 인자. p_drop 0 과 guidance_w 1 을 강제한다.
    argv = ["B1.py",
            "--p_drop", "0",
            "--guidance_w", "1.0",
            "--lambda_anchor", "1.0",       # 세기는 R10 내부의 λ_level/λ_struct 가 정한다
            "--out_dir", str(out_dir),
            "--ckpt_root", str(REPO / "outputs" / "R10")]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru

    json.dump({"arm": "R10", "lambda_level": args.lambda_level, "rho": args.rho,
               "anchor_norm": args.anchor_norm, "chunk_backward": args.chunk_backward,
               "stats_batches": args.stats_batches,
               "n_bins": args.n_bins, "warmup_steps": args.warmup_steps,
               "use_ghat_weight": args.use_ghat_weight, "lambda_swap": args.lambda_swap,
               "teacher": "rolling (1 snapshot)", "embedding": "dinov2_cls_768_frozen",
               "p_drop": 0.0, "guidance_w": 1.0, "argv": argv},
              (out_dir / "r10_config.json").open("w"), indent=2, ensure_ascii=False)

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old
    write_table(out_dir)


def write_table(out_dir: Path, arm: str = "R10", subtitle: str = "수송 좌표 level+structure 앵커") -> None:
    """4x4 표를 csv / md 로. B1 이 쓴 sr_matrix.csv 를 읽는다."""
    src = out_dir / "sr_matrix.csv"
    if not src.exists():
        print("[R10] sr_matrix.csv 없음 — 표 생략"); return
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
    avg = sum(v for v in last if v is not None) / max(1, sum(v is not None for v in last))
    L = [f"# {arm} — {subtitle} (LIBERO-spatial, 4 task, 20 rollout/칸)", "",
         "| after task | " + " | ".join(f"task{t}" for t in range(K)) + " |",
         "|---" * (K + 1) + "|"]
    for k in range(K):
        L.append(f"| {k} | " + " | ".join(
            f"{cells[(k,t)]:.0f}" if (k, t) in cells else "" for t in range(K)) + " |")
    L += ["", f"**AvgSR (마지막 행 평균) = {avg:.1f}**", "",
          "참고값  B8λ3 = 72.5   B2λ3 = 80.0   ER = 93.8   joint(상한) = 95.5",
          "(참고값은 최초 실행 기준이다. 앵커 집계를 합으로 고친 뒤에는",
          " B8λ3 = 83.8, B2λ3 = 88.8 이다 — results/B_mod.txt)"]
    (out_dir / "sr_table.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nsaved -> {out_dir/'sr_table.csv'}, {out_dir/'sr_table.md'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
