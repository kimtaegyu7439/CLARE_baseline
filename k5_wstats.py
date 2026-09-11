#!/usr/bin/env python
"""K5 witness — 동결된 사전학습망의 블록별 활성 통계와 그것을 쓰는 에너지.

witness 는 세 모델 중 하나다. 혼동하지 말 것.
    학생            학습 대상
    rolling teacher 앵커 target (R13 과 동일 운용)
    witness         사전학습 체크포인트, 전 구간 동결. **에너지 계산에만** 쓴다.

witness forward 규약 — 통계 수집과 정련에서 **반드시 동일**해야 한다.
    vision emb  주입 (수집: 해당 프레임 CLS / 정련: 정련 중인 b)
    language    ℓ_j
    t           0
    x_t         eps_probe (고정 시드 노이즈 1개, 전 태스크 공용)
    state       ★ 0 벡터

  state 를 0 으로 두는 이유: 스펙이 state 를 규약에 넣지 않았는데 cond_tail 은 state 를
  받는다. 수집 시점(태스크 j 의 상태)과 정련 시점(현재 태스크의 상태)이 다르면 에너지가
  vision emb 이외의 것으로도 움직여 비교가 성립하지 않는다. 0 으로 고정하면 양쪽 regime
  이 문자 그대로 같아지고, 에너지는 오직 (vision emb, ℓ_j) 의 함수가 된다.
  --witness_state batch 로 현재 배치의 실제 state 를 쓸 수도 있다(비권장, 위 이유).

에너지
    E(b) = Σ_l [ ‖μ̂_l(b) − μ_l[τ]‖² + ‖σ̂_l(b) − σ_l[τ]‖² ] / d_h

  μ̂, σ̂ 는 **배치 통계**다(채널 축 d_h=512, (T,B) 로 평균). 그래서 gradient 가 배치 안에서
  서로 얽힌다 — DeepInversion 계열의 feature-statistics matching 과 같은 구조다.
  배치 안에 여러 bin 이 섞이므로 bin 별로 묶어 그 bin 의 저장 통계와 비교하고 표본 수로
  가중한다. 표본이 1개뿐인 bin 은 σ̂ 가 정의되지 않아 μ 항만 쓴다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
import R10


# ═════════════════════════════════════════════════════════════════════════════
def block_list(policy):
    """velocity_net 의 디코더 블록들. modeling_dit_flow_mt.py:_TransformerDecoder."""
    return policy.dit_flow.velocity_net.decoder.layers


def select_blocks(policy, every: int) -> list[int]:
    n = len(block_list(policy))
    return list(range(0, n, max(1, every)))


class BlockTap:
    """선택 블록의 출력 (T, B, H) 를 붙잡는 forward hook 묶음."""

    def __init__(self, policy, idx: list[int]):
        self.layers = block_list(policy)
        self.idx = idx
        self.acts: dict[int, torch.Tensor] = {}
        self._h = []

    def __enter__(self):
        def mk(i):
            def fn(_m, _in, out):
                self.acts[i] = out
            return fn
        self._h = [self.layers[i].register_forward_hook(mk(i)) for i in self.idx]
        return self

    def __exit__(self, *a):
        for h in self._h:
            h.remove()
        self._h = []
        return False


def witness_forward(witness, cls_flat, instruction, eps_probe, tap,
                    state=None, n_obs=2, state_dim=8):
    """규약대로 witness 를 한 번 통과시키고 tap.acts 를 채운다.

    cls_flat  (B*4, 768)  주입할 비전 임베딩. 정련 중에는 grad 가 흐른다.
    반환은 velocity 출력(쓰지 않음). 활성값은 tap.acts 에 담긴다.
    """
    n = cls_flat.shape[0] // 4
    dev = cls_flat.device
    # witness 파라미터 dtype 에 맞춘다. 호출부가 캐스팅을 신경 쓰지 않도록 여기서 처리.
    wdt = next(witness.parameters()).dtype
    cls_flat = cls_flat.to(wdt)
    if state is None:
        state = torch.zeros(n, n_obs, state_dim, device=dev, dtype=wdt)
    batch = {"observation.state": state.to(wdt)}
    tail = B1.cond_tail(witness, batch, cls_flat)
    lang = B1.encode_lang(witness, [instruction] * n)
    cond = B1.make_cond(lang.to(wdt), tail.to(wdt))
    x_t = eps_probe.to(dev, wdt).expand(n, -1, -1)
    t = torch.zeros(n, device=dev, dtype=wdt)
    return witness.dit_flow.velocity_net(noisy_actions=x_t, time=t, global_cond=cond)


def batch_stats(acts: dict[int, torch.Tensor], tau: torch.Tensor, n_bins: int):
    """bin 별 채널 통계. {block: (mu (n_bins,H), sd (n_bins,H), cnt (n_bins,))}."""
    out = {}
    for bi, a in acts.items():
        x = a.permute(1, 0, 2)                        # (T,B,H) -> (B,T,H)
        H = x.shape[-1]
        mu = x.new_zeros(n_bins, H)
        sd = x.new_zeros(n_bins, H)
        cnt = torch.zeros(n_bins, device=x.device)
        for t in range(n_bins):
            m = tau == t
            c = int(m.sum())
            if c == 0:
                continue
            xm = x[m].reshape(-1, H)                  # (c*T, H)
            mu[t] = xm.mean(0)
            if c > 1:
                sd[t] = xm.std(0, unbiased=False)
            cnt[t] = c
        out[bi] = (mu, sd, cnt)
    return out


def energy(acts, tau, stats, n_bins):
    """E(b). stats = {block: {"mu": (n_bins,H), "sigma": (n_bins,H)}}."""
    bs = batch_stats(acts, tau, n_bins)
    tot = None
    for bi, (mu, sd, cnt) in bs.items():
        ref = stats[bi]
        rm, rs = ref["mu"].to(mu.device, mu.dtype), ref["sigma"].to(mu.device, mu.dtype)
        H = mu.shape[-1]
        w = (cnt / cnt.sum().clamp_min(1)).to(mu.dtype)          # 표본 수 가중
        e_mu = ((mu - rm) ** 2).sum(-1) / H
        e_sd = ((sd - rs) ** 2).sum(-1) / H
        e_sd = torch.where(cnt > 1, e_sd, torch.zeros_like(e_sd))  # 표본 1개면 σ 항 제외
        e = (w * (e_mu + e_sd)).sum()
        tot = e if tot is None else tot + e
    return tot


# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def collect_wstats(policy, witness, dataset, train_eps, cfg, device, n_bins,
                   batch_size, prep, instruction, eps_probe, blocks,
                   workers: int = 4, use_batch_state: bool = False):
    """태스크 j 학습 데모 전 프레임을 witness 에 흘려 블록별 채널 통계를 낸다.

    float64 로 누적하고 float32 로 저장한다. **행동 데이터는 쓰지 않는다** —
    x_t 는 고정 eps_probe 이고 t=0 이라 batch["action"] 은 어디에도 안 들어간다.
    """
    from lerobot.datasets.sampler import EpisodeAwareSampler

    sampler = EpisodeAwareSampler(
        dataset.episode_data_index, episode_indices_to_use=train_eps,
        drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0), shuffle=False)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=workers, batch_size=batch_size, sampler=sampler,
        drop_last=False, pin_memory=(device.type == "cuda"),
        multiprocessing_context="spawn" if workers > 0 else None)
    ep_len = R10.episode_lengths(dataset)

    acc = None
    with BlockTap(witness, blocks) as tap:
        for raw in loader:
            b = prep(policy, B1.to_device(raw, device))
            cls = B1.rgb_cls(policy, b).float()                  # (B*4, 768)
            tau = R10.phase_bins(raw, ep_len, n_bins).to(device)
            st = b["observation.state"].float() if use_batch_state else None
            witness_forward(witness, cls, instruction, eps_probe, tap, state=st)
            for bi, a in tap.acts.items():
                x = a.detach().permute(1, 0, 2).double()          # (B,T,H)
                B_, T, H = x.shape
                if acc is None:
                    acc = {}
                if bi not in acc:
                    acc[bi] = [torch.zeros(n_bins, H, device=device, dtype=torch.float64),
                               torch.zeros(n_bins, H, device=device, dtype=torch.float64),
                               torch.zeros(n_bins, device=device, dtype=torch.float64)]
                s, sq, c = acc[bi]
                s.index_add_(0, tau, x.sum(1))
                sq.index_add_(0, tau, (x * x).sum(1))
                c.index_add_(0, tau, torch.full((B_,), float(T), device=device,
                                                dtype=torch.float64))
            tap.acts.clear()

    out = {}
    for bi, (s, sq, c) in acc.items():
        cc = c.clamp_min(1.0)[:, None]
        mu = (s / cc)
        var = (sq / cc - mu ** 2).clamp_min(0.0)
        out[bi] = {"mu": mu.float().cpu(), "sigma": var.sqrt().float().cpu(),
                   "count": c.float().cpu(), "d_h": int(mu.shape[-1])}
    return out
