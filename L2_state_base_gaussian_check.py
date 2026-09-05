#!/usr/bin/env python
"""L2_state_base_gaussian_check — 코드북 셀 **안**의 state 분포가 가우시안인가.

왜 이 그림이 필요한가
    l2_codebook 은 state 를 표준화 -> k-means(K) 로 자른 뒤 셀마다 p(s|j) 를
    가우시안으로 두고 s̃ 를 뽑는다. 그 가정이 참인지 아무도 확인한 적이 없다.
    R10_tsne 가 phase bin 안에서 관측이 가우시안인지 물었던 것과 같은 질문을,
    이번에는 **시간 bin 이 아니라 k-means 셀** 위에서 묻는다.

두 가지를 분리해서 본다 — 섞으면 답이 안 나온다.
    Q1  셀 안이 가우시안인가?            -> 완전 공분산 기준 PP/타원 커버리지
    Q2  그 가우시안이 대각인가?          -> 같은 점을 대각 공분산으로 다시 재고
                                           셀 안 상관계수 |r| 을 우연 수준과 비교
    Q1 이 참이고 Q2 가 거짓이면 --full_cov_s 가 옳은 수정이다. Q1 이 거짓이면
    공분산을 아무리 채워도 소용없고 셀을 더 잘게 잘라야 한다.

t-SNE 는 왼쪽 패널에서 **셀 구조를 보는 용도로만** 쓴다. 밀도를 의도적으로
왜곡하므로 가우시안 판정에는 절대 쓰지 않는다(R10_tsne 의 주의와 같다).

state 는 정책이 MIN_MAX 로 정규화해 쓰지만, 그것은 차원별 아핀 변환이고
l2_codebook 은 클러스터링 직전에 z = (s-mean)/std 로 다시 표준화한다. 아핀
변환은 z, k-means 라벨, 가우시안 여부, 상관계수를 모두 보존하므로 여기서는
데이터셋 원본 state 를 그대로 쓴다 — 정책(DINOv2)을 올릴 필요가 없어 CPU 로 돈다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Ellipse                # noqa: E402
from scipy.stats import beta as beta_dist             # noqa: E402
from sklearn.decomposition import PCA                 # noqa: E402
from sklearn.manifold import TSNE                     # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1                                             # noqa: E402
import l2_codebook                                    # noqa: E402  (kmeans 를 그대로 재사용)

from lerobot.datasets.factory import make_dataset     # noqa: E402
from lerobot.utils.utils import init_logging          # noqa: E402

CK_DEFAULT = "outputs/B2_lam3/libero_spatial_seed42_ours/task_0/checkpoints/005000/pretrained_model"
SIGMA_THEORY = (0.393, 0.865, 0.989)      # 2D 가우시안의 1σ/2σ/3σ 타원 안 비율


# ═════════════════════════════════════════════════════════════════════════════
#  state 수집 — 비디오/이미지를 건드리지 않는다
# ═════════════════════════════════════════════════════════════════════════════
def collect_state(ds, n_want: int, drop_n_last: int, seed: int):
    """(s(N,16), ep(N,), frame(N,), phase(N,)). l2_codebook._collect 와 같은 s 를 만든다.

    DataLoader 를 돌리면 프레임마다 PNG 두 장을 디코딩하는데, state 만 필요한
    여기서는 순수 낭비다. 대신 hf_dataset 의 state 열만 통째로 읽고
    LeRobotDataset._get_query_indices 의 클램프 규칙
    (idx+delta 를 [ep_start, ep_end-1] 에 가둔다) 을 그대로 벡터화한다.
    """
    # hf_dataset[...] 로 읽으면 포매터가 이미지 열까지 tensorize 하려 든다(비디오 백엔드
    # 의존성까지 끌고 온다). arrow 열을 직접 집으면 state 만 순수하게 나온다.
    hf = ds.hf_dataset
    col = lambda k, dt: np.asarray(hf.data.column(k).to_pylist(), dtype=dt)   # noqa: E731
    S = col("observation.state", np.float64)                 # (Nf,8)
    ep_of = col("episode_index", np.int64)                   # (Nf,)
    frm = col("frame_index", np.int64)
    ef = ds.episode_data_index["from"].numpy()
    et = ds.episode_data_index["to"].numpy()
    delta = np.asarray(ds.delta_indices["observation.state"], dtype=np.int64)        # [-1, 0]

    # EpisodeAwareSampler(drop_n_last_frames) 가 학습에서 쓰는 후보와 같은 집합
    cand = np.concatenate([np.arange(a, max(a, b - drop_n_last)) for a, b in zip(ef, et)])
    rng = np.random.default_rng(seed)
    if n_want > 0 and cand.size > n_want:
        cand = np.sort(rng.choice(cand, n_want, replace=False))

    e = ep_of[cand]
    rows = np.clip(cand[:, None] + delta[None, :], ef[e][:, None], et[e][:, None] - 1)
    s = S[rows].reshape(cand.size, -1)                       # (N, n_obs*8) = (N,16)
    ep_len = (et - ef).astype(np.float64)
    phase = frm[cand] / np.maximum(ep_len[e] - 1.0, 1.0)     # 0=시작, 1=끝 (참고용 색)
    return s, e, frm[cand], phase


# ═════════════════════════════════════════════════════════════════════════════
#  셀 나누기 — l2_codebook.build_codebook 과 같은 규칙
# ═════════════════════════════════════════════════════════════════════════════
def build_cells(s: torch.Tensor, K: int, seed: int, min_n: int):
    """(labels(N,), centers(K_eff,D), zs(N,D)). l2_codebook.py:110-128 과 동일한 절차."""
    mean_s, std_s = s.mean(0), s.std(0).clamp_min(1e-8)
    zs = (s - mean_s) / std_s
    c, lab, _ = l2_codebook.kmeans(zs, K, n_init=10, seed=seed)

    cnt = torch.bincount(lab, minlength=K)
    keep = (cnt >= min_n).nonzero().squeeze(1)
    drop = (cnt < min_n).nonzero().squeeze(1)
    if drop.numel() and keep.numel():
        remap = torch.arange(K, device=s.device)
        remap[drop] = keep[torch.cdist(c[drop], c[keep]).argmin(1)]
        lab = remap[lab]
        uniq, lab = torch.unique(lab, return_inverse=True)
        c = c[uniq]
    return lab, c, zs, int(drop.numel())


# ═════════════════════════════════════════════════════════════════════════════
#  셀 하나의 가우시안 진단
# ═════════════════════════════════════════════════════════════════════════════
def cell_stats(x: np.ndarray, base_dim: int) -> dict | None:
    """x(n,d) 한 셀. 완전 공분산/대각 두 기준의 마할라노비스 PIT 과 상관계수.

    표본 평균·공분산으로 잰 d² 는 카이제곱이 아니다. 정확히는
        d²·n/(n-1)² ~ Beta(d/2, (n-d-1)/2)                 (Gnanadesikan-Kettenring)
    이고 n 이 셀마다 다르므로, 이 Beta 로 PIT(u=F(d²)) 을 취해 셀들을 한데 모은다.
    셀 안이 가우시안이면 u 는 정확히 Uniform(0,1) 이다.

    상관 쌍은 두 무리로 나눠 센다. s 는 [s_{t-1}, s_t] 를 이어 붙인 것이라
    (j, j+base_dim) 쌍은 **같은 관절의 이웃 프레임**이고 거의 |r|=1 이다. 이것을
    나머지(관절 사이 상관)와 섞으면 히스토그램의 1.0 스파이크가 뭘 뜻하는지
    읽을 수 없다.
    """
    n, d = x.shape
    if n < d + 5:                       # 표본이 차원보다 충분히 크지 않으면 공분산이 못 선다
        return None
    xc = x - x.mean(0)
    S = xc.T @ xc / (n - 1)
    var = np.clip(np.diag(S), 1e-12, None)

    d2_full = np.einsum("ij,jk,ik->i", xc, np.linalg.inv(S), xc)
    d2_diag = ((xc ** 2) / var).sum(1)
    a, b = d / 2.0, (n - d - 1) / 2.0
    scale = (n - 1) ** 2 / n
    pit = lambda v: beta_dist.cdf(np.clip(v / scale, 0.0, 1.0), a, b)   # noqa: E731

    R = np.abs(S / np.sqrt(np.outer(var, var)))
    iu = np.triu_indices(d, 1)
    same = (iu[0] % base_dim) == (iu[1] % base_dim)      # 같은 관절, 다른 obs step
    return {"n": n, "u_full": pit(d2_full), "u_diag": pit(d2_diag),
            "r_time": R[iu][same], "r_joint": R[iu][~same],
            "chance_r": 0.6745 / np.sqrt(n)}


def ks(u: np.ndarray) -> float:
    """Uniform(0,1) 과의 KS 거리. 0 이면 완벽히 일치."""
    if u.size == 0:
        return float("nan")
    v = np.sort(u)
    m = np.arange(1, v.size + 1) / v.size
    return float(max(np.max(np.abs(v - m)), np.max(np.abs(v - (m - 1 / v.size)))))


def ellipse_cov(ax, M: np.ndarray, n_std: float, **kw):
    """공분산 M(2x2) 의 n_std 타원을 원점 중심으로 그린다.

    R10_tsne 의 ellipse 는 그려진 점들에 다시 공분산을 맞췄다. 그건 "이 점들의
    최적 타원"이지 **모델**이 아니다. 여기서는 M 을 밖에서 받는다 — l2_codebook 이
    실제로 쓰는 분포를 그 평면에 사영한 것이 들어온다.
    """
    v, w = np.linalg.eigh(M)
    v = np.clip(v, 1e-18, None)
    ang = np.degrees(np.arctan2(w[1, -1], w[0, -1]))
    ax.add_patch(Ellipse((0.0, 0.0), 2 * n_std * np.sqrt(v[-1]),
                         2 * n_std * np.sqrt(v[0]), angle=ang, fill=False, **kw))


def cov_inside(q: np.ndarray, M: np.ndarray) -> list[float]:
    """모델 공분산 M 의 1σ/2σ/3σ 타원 안에 실제 점 q 가 몇 % 들어오는가."""
    Mi = np.linalg.inv(M)
    return [float(np.mean(np.sum(q @ Mi * q, 1) <= k ** 2)) for k in (1, 2, 3)]


def cell_models(x: np.ndarray, ridge: float, plane: str):
    """한 셀의 (실제 점의 2D 사영 q, 완전공분산 모델, 대각 모델).

    l2_codebook.build_codebook 이 셀마다 만드는 두 가지를 그대로 재현한다.
        대각 (현행)      Σ = diag(σ_k²),  σ_k = per_cell 의 ML 표준편차
        완전공분산 (--full_cov_s)
                         Σ = C + ridge·diag(C) + 1e-8 I,  C = ML 공분산
    ridge 는 완전공분산 쪽에만 붙는다 — 코드가 그렇게 돼 있다(l2_codebook.py:166).

    가우시안은 선형 사영이 다시 가우시안이므로, 평면 W(16x2) 위에서 모델은
    정확히 N(0, Wᵀ Σ W) 이다. 근사가 아니다.
    """
    n = x.shape[0]
    xc = x - x.mean(0)
    C = xc.T @ xc / n                                  # ML — build_codebook 과 같은 분모
    var = np.clip(np.diag(C), 1e-12, None)
    C_full = C + ridge * np.diag(var) + 1e-8 * np.eye(x.shape[1])
    C_diag = np.diag(var)

    if plane == "diff":
        # 두 모델이 가장 크게 어긋나는 평면. "네가 고른 평면이 유리한 것" 이라는
        # 반론을 막는 쪽이 아니라, 최악을 보여주는 쪽이다.
        from scipy.linalg import eigh as geigh                       # noqa: PLC0415
        w_, V = geigh(C_full, C_diag)
        W = np.stack([V[:, -1], V[:, 0]], 1)
        W = W / np.linalg.norm(W, axis=0, keepdims=True)
    else:                                              # 셀 자신의 주성분 평면
        _, V = np.linalg.eigh(C)
        W = V[:, ::-1][:, :2]
    return xc @ W, W.T @ C_full @ W, W.T @ C_diag @ W


# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--codebook_k", type=int, default=96, help="l2_codebook 과 같은 K")
    ap.add_argument("--n_pairs", type=int, default=8000, help="l2_codebook 의 수집 표본 수")
    ap.add_argument("--min_n", type=int, default=5, help="셀 병합 임계 (build_codebook 과 동일)")
    ap.add_argument("--cell", type=int, default=-1, help="-1 = --pick 규칙으로 고른다")
    ap.add_argument("--pick", choices=("median", "worst", "best", "largest"), default="median",
                    help="초점 셀을 고르는 규칙. 기본은 KS_full 중앙값 셀(전형적인 셀). "
                         "largest 는 KS 가 n 과 음의 상관이라 가장 가우시안해 보이는 셀을 "
                         "고르는 셈이므로 기본값으로 쓰지 않는다.")
    ap.add_argument("--cov_ridge", type=float, default=0.05,
                    help="완전공분산 모델의 ridge. l2_codebook --cov_ridge 와 같게 둔다")
    ap.add_argument("--grid", type=int, default=6,
                    help="셀을 무작위로 N 개 뽑아 모델-vs-실제 타원만 따로 그린 그림도 "
                         "낸다. 0 이면 끔. 무작위이므로 예시 셀 선택 시비가 없다")
    ap.add_argument("--grid_seed", type=int, default=-1, help="-1 = --seed 를 쓴다")
    ap.add_argument("--plane", choices=("pca", "diff"), default="pca",
                    help="타원을 그릴 2D 평면. pca = 셀 자신의 주성분(기본), "
                         "diff = 두 모델이 가장 크게 어긋나는 평면")
    ap.add_argument("--color", choices=("cell", "phase"), default="cell",
                    help="왼쪽 패널 색: 코드북 셀 / 에피소드 진행도")
    ap.add_argument("--perplexity", type=float, default=40)
    ap.add_argument("--tsne_max", type=int, default=4000, help="임베딩에 넣을 점 수 상한")
    ap.add_argument("--embed", choices=("tsne", "umap", "pca"), default="tsne",
                    help="왼쪽 패널 좌표. umap 은 umap-learn 이 있을 때만 (없으면 t-SNE 로 "
                         "떨어진다). 어느 쪽이든 이 패널은 셀 배치를 보는 용도이고 "
                         "가우시안 판정에는 쓰지 않는다 — UMAP 은 밀도를 t-SNE 보다 "
                         "더 왜곡한다.")
    ap.add_argument("--umap_neighbors", type=int, default=15)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    ap.add_argument("--no_tsne", action="store_true", help="(구식) --embed pca 와 같다")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=CK_DEFAULT, help="delta_timestamps 를 읽을 정책 설정")
    ap.add_argument("--out", default="results/L2_state_gauss")
    ap.add_argument("--device", default="cpu", help="k-means 장치. state 는 CPU 로 충분하다")
    a = ap.parse_args()

    init_logging()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(suite=a.suite, device="cpu", seed=a.seed, num_workers=0,
                            batch_size=32, steps_per_task=1, log_every=100,
                            eval_episodes=1, eval_batch_size=1, mode="gausscheck",
                            p_drop=0.0, lambda_anchor=0.0)
    cfg = B1.build_cfg(ns, a.task, a.ckpt, Path("/tmp/l2sgauss"))
    ds = make_dataset(cfg)
    drop_n = int(getattr(cfg.policy, "drop_n_last_frames", 0))

    s_np, ep, frm, phase = collect_state(ds, a.n_pairs, drop_n, a.seed)
    N, D = s_np.shape
    n_obs = len(ds.delta_indices["observation.state"])
    base_dim = D // n_obs                      # 한 프레임의 state 차원 (LIBERO: 8)
    print(f"[gauss] task{a.task}  N={N}  state_dim={D}  에피소드 {len(np.unique(ep))}개  "
          f"drop_n_last={drop_n}", flush=True)

    s = torch.as_tensor(s_np, dtype=torch.float32, device=a.device)
    lab_t, cen, zs_t, n_merged = build_cells(s, a.codebook_k, a.seed, a.min_n)
    lab = lab_t.cpu().numpy()
    zs = zs_t.cpu().numpy()
    K_eff = int(cen.shape[0])
    cnt = np.bincount(lab, minlength=K_eff)
    print(f"[gauss] K={a.codebook_k} -> K_eff={K_eff} (병합 {n_merged})  "
          f"셀 표본 수 min/중앙/max = {cnt.min()}/{int(np.median(cnt))}/{cnt.max()}", flush=True)

    # ── 셀별 진단 ────────────────────────────────────────────────────────────
    per_cell, uf, ud, rj, rt, chance = {}, [], [], [], [], []
    for k in range(K_eff):
        st = cell_stats(s_np[lab == k], base_dim)
        if st is None:
            continue
        per_cell[k] = st
        uf.append(st["u_full"]); ud.append(st["u_diag"])
        rj.append(st["r_joint"]); rt.append(st["r_time"]); chance.append(st["chance_r"])
    if not per_cell:
        raise SystemExit(f"표본이 {D + 5} 개 이상인 셀이 없다. --codebook_k 를 줄이거나 "
                         f"--n_pairs 를 늘려라.")
    uf = np.concatenate(uf); ud = np.concatenate(ud)
    rj = np.concatenate(rj); rt = np.concatenate(rt)
    ks_full, ks_diag = ks(uf), ks(ud)
    med_j, med_t = float(np.median(rj)), float(np.median(rt))
    med_chance = float(np.median(chance))

    # ── 셀별 요약 ────────────────────────────────────────────────────────────
    # 셀 하나로 결론을 내면 "뭉친 셀만 골랐다"는 반론을 못 막는다. 모든 셀의
    # KS 와 커버리지를 먼저 낸 뒤, 초점 셀은 그 분포 위의 한 점으로만 쓴다.
    usable = sorted(per_cell)
    ksf_c = np.array([ks(per_cell[k]["u_full"]) for k in usable])
    ksd_c = np.array([ks(per_cell[k]["u_diag"]) for k in usable])
    cov_full, cov_diag = [], []
    for k in usable:
        qk, Mf, Md = cell_models(s_np[lab == k], a.cov_ridge, a.plane)
        cov_full.append(cov_inside(qk, Mf)); cov_diag.append(cov_inside(qk, Md))
    cov_full = np.array(cov_full); cov_diag = np.array(cov_diag)
    n_c = cnt[usable]
    diag_worse = int((ksd_c > ksf_c).sum())
    r_ks_n = float(np.corrcoef(ksf_c, n_c)[0, 1]) if len(usable) > 2 else float("nan")

    # ── 초점 셀 ──────────────────────────────────────────────────────────────
    if a.cell >= 0:
        cell, pick_note = int(a.cell), "explicit"
    else:
        order_ks = np.argsort(ksf_c)
        pos = {"median": len(usable) // 2, "worst": len(usable) - 1, "best": 0}
        if a.pick == "largest":
            cell, pick_note = int(usable[int(np.argmax(n_c))]), "largest cell"
        else:
            cell = int(usable[int(order_ks[pos[a.pick]])])
            pick_note = f"{a.pick}-KS cell"
    if cell not in per_cell:
        raise SystemExit(f"셀 {cell} 은 표본이 {cnt[cell] if cell < K_eff else 0} 개뿐이라 진단 불가")
    ci = usable.index(cell)
    cell_rank = int(np.searchsorted(np.sort(ksf_c), ksf_c[ci]) + 1)
    q, M_full, M_diag = cell_models(s_np[lab == cell], a.cov_ridge, a.plane)
    in_full, in_diag = cov_inside(q, M_full), cov_inside(q, M_diag)

    # ── 왼쪽 패널 좌표 ───────────────────────────────────────────────────────
    m_plot = np.arange(N)
    if a.tsne_max > 0 and N > a.tsne_max:
        keep = np.random.default_rng(a.seed).choice(N, a.tsne_max, replace=False)
        m_plot = np.sort(np.concatenate([keep, np.nonzero(lab == cell)[0]]))
        m_plot = np.unique(m_plot)
    mode = "pca" if a.no_tsne else a.embed
    if mode == "umap":
        try:
            import umap                                              # noqa: PLC0415
        except ImportError:
            print("[gauss] umap-learn 이 없다 -> t-SNE 로 대체 "
                  "(설치: pip install umap-learn, numba/llvmlite 를 끌고 온다)", flush=True)
            mode = "tsne"
    if mode == "pca":
        emb = PCA(n_components=2, random_state=a.seed).fit_transform(zs[m_plot])
        left_title = f"state PCA 2D — colored by {a.color}"
    elif mode == "umap":
        print("[gauss] UMAP 계산 중...", flush=True)
        emb = umap.UMAP(n_components=2, n_neighbors=a.umap_neighbors,
                        min_dist=a.umap_min_dist, random_state=a.seed
                        ).fit_transform(zs[m_plot])
        left_title = (f"UMAP of state (k={a.umap_neighbors}) — colored by {a.color}")
    else:
        print("[gauss] t-SNE 계산 중...", flush=True)
        emb = TSNE(n_components=2, perplexity=a.perplexity, init="pca",
                   random_state=a.seed, max_iter=1000).fit_transform(zs[m_plot])
        left_title = f"t-SNE of state — colored by {a.color}"
    # 셀 번호는 k-means 가 준 임의 순서다. 색이 뜻을 갖도록 중심의 PC1 순으로 다시 매긴다.
    order = np.argsort(PCA(n_components=1, random_state=a.seed)
                       .fit_transform(cen.cpu().numpy())[:, 0])
    rank = np.empty(K_eff, dtype=np.int64); rank[order] = np.arange(K_eff)

    # ═════════════════════ 그림 ══════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18.2, 10.2))
    fig.subplots_adjust(left=.045, right=.985, top=.865, bottom=.065, wspace=.26, hspace=.30)

    # A. 셀 구조 -------------------------------------------------------------
    ax = axes[0, 0]
    cval, clab = ((rank[lab[m_plot]], "codebook cell (ordered by center PC1)")
                  if a.color == "cell" else (phase[m_plot], "episode phase (0=start, 1=end)"))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=cval, cmap=plt.cm.viridis,
                    s=5, alpha=.70, edgecolors="none")
    hit = lab[m_plot] == cell
    ax.scatter(emb[hit, 0], emb[hit, 1], s=13, color="crimson", edgecolors="none",
               label=f"cell {cell} (n={cnt[cell]})")
    fig.colorbar(sc, ax=ax, fraction=.046, label=clab)
    ax.set_title(f"{left_title}\n(layout only — density here is distorted, not evidence)",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=.9)
    ax.set_xticks([]); ax.set_yticks([])

    # B. 초점 셀 안 — PCA 2D + 타원 (R10_tsne 오른쪽 패널과 같은 그림) --------
    # 이 패널은 **예시**다. 결론은 옆의 C/F 가 지고 간다.
    ax = axes[0, 1]
    ax.scatter(q[:, 0], q[:, 1], s=11, alpha=.55, color="#3b5f86", edgecolors="none",
               zorder=2, label="real frames in the cell")
    for k_, ls in ((1, "-"), (2, "--"), (3, ":")):
        ellipse_cov(ax, M_diag, k_, ec="#c2410c", lw=1.8, ls=ls, zorder=3)
        ellipse_cov(ax, M_full, k_, ec="#1a7f37", lw=1.8, ls=ls, zorder=3)
    ax.plot([], [], color="#c2410c", lw=1.8, label="diagonal model  N(mu, diag sig^2)  = what l2_codebook uses")
    ax.plot([], [], color="#1a7f37", lw=1.8, label="full covariance model  = --full_cov_s")
    ax.text(.02, .98,
            "inside 1σ / 2σ / 3σ      (theory 39 / 86 / 99)\n"
            f"diag  {in_diag[0]*100:5.1f} {in_diag[1]*100:5.1f} {in_diag[2]*100:5.1f}\n"
            f"full  {in_full[0]*100:5.1f} {in_full[1]*100:5.1f} {in_full[2]*100:5.1f}",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace",
            bbox=dict(fc="white", ec="0.7", alpha=.85, pad=3))
    ax.set_title(f"Example: cell {cell} — {pick_note}, KS rank {cell_rank}/{len(usable)}, "
                 f"n={cnt[cell]}\nmodelled Gaussian vs the real points "
                 f"({'cell PCA' if a.plane == 'pca' else 'max-discrepancy'} plane)",
                 fontsize=10.5)
    ax.legend(loc="lower right", fontsize=7.6, framealpha=.9)
    ax.set_xlabel("axis 1"); ax.set_ylabel("axis 2")
    ax.set_aspect("equal", adjustable="datalim")

    # C. 전 셀 커버리지 — 예시 셀이 특별하지 않다는 증거 ---------------------
    ax = axes[0, 2]
    rngj = np.random.default_rng(a.seed)
    for i, th in enumerate(SIGMA_THEORY):
        for dx, cov_m, col in ((-.19, cov_diag, "#c2410c"), (.19, cov_full, "#1a7f37")):
            xj = i + 1 + dx + rngj.uniform(-.11, .11, len(usable))
            ax.scatter(xj, cov_m[:, i] * 100, s=12, alpha=.45, color=col,
                       edgecolors="none")
            ax.scatter([i + 1 + dx], [cov_m[ci, i] * 100], s=62, color=col, zorder=4,
                       marker="D", edgecolors="white", linewidths=.9)
            ax.text(i + 1 + dx, 3, f"{np.median(cov_m[:, i]) * 100:.0f}", ha="center",
                    fontsize=8.5, color=col, fontweight="bold")
        ax.hlines(th * 100, i + .58, i + 1.42, color="0.2", lw=1.8, ls="--", zorder=5)
    ax.plot([], [], color="#c2410c", lw=6, alpha=.5, label="diagonal model")
    ax.plot([], [], color="#1a7f37", lw=6, alpha=.5, label="full covariance model")
    ax.plot([], [], color="0.2", lw=1.8, ls="--", label="Gaussian theory")
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["1σ", "2σ", "3σ"])
    ax.set_xlim(.5, 3.5); ax.set_ylim(0, 108)
    ax.set_ylabel("real points inside the model ellipse (%)")
    ax.set_title(f"Same check on every cell ({len(usable)} cells)\n"
                 f"◆ = the example cell,  number = median", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, framealpha=.9)
    ax.grid(alpha=.25, axis="y")

    # D. 전 셀 마할라노비스 PP — 완전 공분산 vs 대각 -------------------------
    ax = axes[1, 0]
    for k in usable:                                   # 셀별 곡선을 옅게 깔아 퍼짐을 보인다
        v = np.sort(per_cell[k]["u_full"])
        ax.plot(np.arange(1, v.size + 1) / v.size, v, lw=.5, color="#1a7f37", alpha=.16)
    for u, lb, col in ((uf, f"full covariance   KS={ks_full:.3f}", "#1a7f37"),
                       (ud, f"diagonal only     KS={ks_diag:.3f}", "#c2410c")):
        v = np.sort(u)
        ax.plot(np.arange(1, v.size + 1) / v.size, v, lw=2.2, color=col, label=lb)
    ax.plot([0, 1], [0, 1], color="0.35", lw=1.2, ls="--", label="Gaussian (exact)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("expected quantile (Gaussian cell)")
    ax.set_ylabel("observed Mahalanobis PIT")
    ax.set_title(f"Mahalanobis PP, pooled over {len(usable)} cells\n"
                 f"thin lines = individual cells (full cov)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=.9)
    ax.grid(alpha=.25)

    # E. 셀 안 상관 — 대각 가정이 버리는 것 ----------------------------------
    ax = axes[1, 1]
    bins = np.linspace(0, 1, 61)
    ax.hist([rj, rt], bins=bins, stacked=True, edgecolor="none",
            color=["#2f6db5", "#e08a1e"],
            label=[f"joint x joint  (median {med_j:.3f})",
                   f"same joint, t-1 vs t  (median {med_t:.3f})"])
    ax.axvline(med_j, color="crimson", lw=2.0)
    ax.axvline(med_chance, color="0.35", lw=1.6, ls="--",
               label=f"chance level  {med_chance:.3f}")
    ax.set_xlabel("|correlation| between state dims, within cell")
    ax.set_ylabel("count")
    ax.set_title(f"Within-cell correlations ({rj.size + rt.size:,} pairs, all cells)",
                 fontsize=11)
    ax.legend(fontsize=8.5, framealpha=.9)

    # F. 셀 하나하나에서 full 이 diag 를 이기는가 ----------------------------
    # 여기가 결론을 지는 패널이다. 대각선 위 = 그 셀에서는 대각이 더 나쁘다.
    ax = axes[1, 2]
    lim = float(max(ksf_c.max(), ksd_c.max())) * 1.08
    ax.plot([0, lim], [0, lim], color="0.35", lw=1.2, ls="--")
    sc = ax.scatter(ksf_c, ksd_c, s=8 + 0.55 * n_c, c=n_c, cmap="viridis",
                    alpha=.80, edgecolors="none")
    ax.scatter([ksf_c[ci]], [ksd_c[ci]], s=90, facecolors="none", edgecolors="crimson",
               linewidths=2.0, zorder=4, label=f"example cell {cell}")
    fig.colorbar(sc, ax=ax, fraction=.046, label="samples in cell")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("KS with full covariance"); ax.set_ylabel("KS with diagonal only")
    ax.set_title(f"Per cell: diagonal is worse in {diag_worse}/{len(usable)} cells\n"
                 f"(above the line = full covariance wins)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=.9)
    ax.grid(alpha=.25)

    verdict_g = ("close to Gaussian" if ks_full < 0.05 else
                 "mildly non-Gaussian" if ks_full < 0.12 else "clearly non-Gaussian")
    verdict_d = ("adequate" if diag_worse < len(usable) * 0.6 and med_j < 2 * med_chance
                 else "not adequate")
    fig.suptitle(
        f"Codebook cell interior (state, {D}-dim) — {a.suite} task {a.task}, "
        f"N={N}, K={a.codebook_k}->K_eff={K_eff}, {len(usable)} cells diagnosed\n"
        f"ellipses = the Gaussian l2_codebook actually samples from,  points = the real data\n"
        f"Q1 is a cell Gaussian: {verdict_g} (pooled KS {ks_full:.3f})   "
        f"Q2 is it diagonal: {verdict_d} "
        f"(diag worse in {diag_worse}/{len(usable)} cells, median |r| {med_j:.3f})",
        fontsize=12, y=.965)

    tag = f"task{a.task}_k{a.codebook_k}"
    fp = out / f"state_gauss_{tag}.png"
    fig.savefig(fp, dpi=140)
    plt.close(fig)

    # ═════════════════════ 셀 격자 그림 ══════════════════════════════════════
    # 위 그림의 패널 B 하나만 여러 셀에 대해 반복한다. 셀은 **무작위**로 뽑는다 —
    # 어떤 규칙으로 골랐냐는 질문 자체가 안 나오게 하는 것이 이 그림의 목적이다.
    fp_grid = None
    if a.grid > 0:
        gs = a.seed if a.grid_seed < 0 else a.grid_seed
        pick_n = min(a.grid, len(usable))
        gsel = sorted(np.random.default_rng(gs).choice(np.array(usable), pick_n,
                                                       replace=False).tolist())
        ncol = 3 if pick_n > 2 else pick_n
        nrow = int(np.ceil(pick_n / ncol))
        fh = 4.5 * nrow + 1.9                       # 제목/범례용 여백을 인치로 확보
        fg, gaxes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, fh), squeeze=False)
        fg.subplots_adjust(left=.06, right=.985, top=1 - 1.05 / fh,
                           bottom=1.05 / fh, wspace=.24, hspace=.30)
        gmed = []
        for i, k in enumerate(gsel):
            ax = gaxes[i // ncol][i % ncol]
            qk, Mf, Md = cell_models(s_np[lab == k], a.cov_ridge, a.plane)
            fk, dk = cov_inside(qk, Mf), cov_inside(qk, Md)
            gmed.append((k, fk, dk))
            ax.scatter(qk[:, 0], qk[:, 1], s=11, alpha=.55, color="#3b5f86",
                       edgecolors="none", zorder=2)
            for k_, ls in ((1, "-"), (2, "--"), (3, ":")):
                ellipse_cov(ax, Md, k_, ec="#c2410c", lw=1.6, ls=ls, zorder=3)
                ellipse_cov(ax, Mf, k_, ec="#1a7f37", lw=1.6, ls=ls, zorder=3)
            ax.text(.02, .98,
                    "in 1σ/2σ/3σ  (39/86/99)\n"
                    f"diag {dk[0]*100:5.1f}{dk[1]*100:6.1f}{dk[2]*100:6.1f}\n"
                    f"full {fk[0]*100:5.1f}{fk[1]*100:6.1f}{fk[2]*100:6.1f}",
                    transform=ax.transAxes, va="top", fontsize=8.2,
                    family="monospace",
                    bbox=dict(fc="white", ec="0.75", alpha=.88, pad=2.5))
            ax.set_title(f"cell {k}   n={cnt[k]}   KS_full={ksf_c[usable.index(k)]:.3f}",
                         fontsize=10)
            ax.set_aspect("equal", adjustable="datalim")
            ax.tick_params(labelsize=8)
            if i // ncol == nrow - 1:
                ax.set_xlabel("axis 1", fontsize=9)
            if i % ncol == 0:
                ax.set_ylabel("axis 2", fontsize=9)
        for j in range(pick_n, nrow * ncol):
            gaxes[j // ncol][j % ncol].axis("off")
        h = [plt.Line2D([], [], color="#c2410c", lw=1.8),
             plt.Line2D([], [], color="#1a7f37", lw=1.8),
             plt.Line2D([], [], ls="none", marker="o", ms=4, color="#3b5f86")]
        fg.legend(h, ["diagonal model  N(mu, diag sig^2)  = what l2_codebook uses",
                      "full covariance model  = --full_cov_s",
                      "real frames in the cell"],
                  loc="lower center", ncol=3, fontsize=9, frameon=False,
                  bbox_to_anchor=(.5, .16 / fh))
        n_beats = sum(1 for _, fk, dk in gmed if abs(fk[1] - .865) < abs(dk[1] - .865))
        fg.suptitle(
            f"Modelled Gaussian vs real points — {pick_n} cells drawn at random "
            f"(seed {gs}) out of {len(usable)}\n"
            f"{a.suite} task {a.task}, K_eff={K_eff}, "
            f"{'cell PCA' if a.plane == 'pca' else 'max-discrepancy'} plane"
            f"   ·   full covariance closer at 2σ in {n_beats}/{pick_n} of them",
            fontsize=12.5, y=1 - .22 / fh)
        fp_grid = out / f"state_gauss_cells_{tag}.png"
        fg.savefig(fp_grid, dpi=140)
        plt.close(fg)

    # ── 보고 ─────────────────────────────────────────────────────────────────
    L = ["=" * 78,
         f"코드북 셀 안의 state 분포 — {a.suite} task {a.task}", "=" * 78, "",
         f"  표본 N={N}  state {D}차원 ({n_obs} obs step x {base_dim})"
         f"  K={a.codebook_k} -> K_eff={K_eff} (병합 {n_merged})",
         f"  셀당 표본 min/중앙/max = {cnt.min()}/{int(np.median(cnt))}/{cnt.max()}"
         f"   진단 가능 셀 {len(per_cell)}/{K_eff} (n >= {D + 5})", "",
         "Q1  셀 안이 가우시안인가  (완전 공분산 기준 마할라노비스 PIT)",
         f"      전체 표본 KS(uniform) = {ks_full:.4f}   -> {verdict_g}",
         f"      셀별 KS  min/25%/중앙/75%/max = {ksf_c.min():.3f}/"
         f"{np.percentile(ksf_c, 25):.3f}/{np.median(ksf_c):.3f}/"
         f"{np.percentile(ksf_c, 75):.3f}/{ksf_c.max():.3f}",
         "",
         "모델 타원 안에 실제 점이 몇 % 드는가  (전 셀 중앙값, 이론 39.3 / 86.5 / 98.9)",
         f"      완전공분산 모델(--full_cov_s)   {np.median(cov_full[:, 0]) * 100:5.1f}% /"
         f"{np.median(cov_full[:, 1]) * 100:6.1f}% /{np.median(cov_full[:, 2]) * 100:6.1f}%",
         f"      대각 모델(현행 sample_codebook) {np.median(cov_diag[:, 0]) * 100:5.1f}% /"
         f"{np.median(cov_diag[:, 1]) * 100:6.1f}% /{np.median(cov_diag[:, 2]) * 100:6.1f}%",
         "",
         "Q2  그 가우시안이 대각인가  (같은 점을 대각 공분산으로 다시 잰다)",
         f"      전체 표본 KS(uniform) = {ks_diag:.4f}   (완전 공분산 {ks_full:.4f} 대비 "
         f"{ks_diag - ks_full:+.4f})",
         f"      ★ 셀 하나하나에서 대각이 더 나쁜 셀 = {diag_worse}/{len(usable)}"
         f" ({diag_worse / len(usable) * 100:.0f}%)",
         f"      셀 안 |r| 중앙값  관절x관절 {med_j:.3f}   같은 관절 t-1 vs t {med_t:.3f}"
         f"   (우연 수준 {med_chance:.3f})",
         f"      -> 대각 가정은 {verdict_d}", "",
         f"예시 셀 (셀 하나가 결론을 지지 않는다 — Q1/Q2 는 전 셀 통계다)",
         f"      셀 {cell}  규칙 '{pick_note}'  n={cnt[cell]}"
         f"  KS_full={ksf_c[ci]:.3f} (오름차순 {cell_rank}/{len(usable)})"
         f"  KS_diag={ksd_c[ci]:.3f}",
         f"      모델 타원 커버리지  완전공분산 {in_full[0] * 100:.1f}/"
         f"{in_full[1] * 100:.1f}/{in_full[2] * 100:.1f}%"
         f"   대각 {in_diag[0] * 100:.1f}/{in_diag[1] * 100:.1f}/{in_diag[2] * 100:.1f}%", "",
         "읽을 때 주의",
         "  · k-means 셀은 보로노이로 잘린 영역이라 꼬리가 원래 깎여 있다. KS 가 0 이",
         "    아닌 것 자체는 놀랄 일이 아니고, 의미 있는 것은 완전 공분산과 대각의 **차이**다.",
         f"  · KS 와 셀 표본 수의 상관 = {r_ks_n:+.3f}. 음수면 큰 셀일수록 KS 가 낮게 나오므로,",
         "    '표본이 가장 많은 셀'을 예시로 고르면 가장 가우시안해 보이는 셀을 고르는 것이",
         "    된다. 그래서 기본 --pick 은 largest 가 아니라 median 이다. --pick worst 로",
         "    최악의 셀도 같은 그림으로 뽑아 볼 수 있다.",
         "  · 한 셀 안의 프레임은 같은 궤적의 이웃 구간이라 독립이 아니다. 유효 표본 수가",
         "    n 보다 작으므로 KS 는 위로 편향된다.",
         f"  · s = [s_(t-1), s_t] 이므로 같은 관절의 두 obs step 쌍({base_dim}개/셀)은 거의 |r|=1 이다.",
         "    대각 가우시안은 이 둘을 독립으로 뽑으므로 물리적으로 불가능한 s̃ 를 만든다.", "",
         f"  그림 {fp}"] + ([f"  셀 격자 {fp_grid}"] if fp_grid else []) + ["=" * 78]
    rep = "\n".join(L)
    print(rep)
    (out / f"report_{tag}.txt").write_text(rep + "\n")
    json.dump({"suite": a.suite, "task": a.task, "N": N, "dim": D,
               "K": a.codebook_k, "K_eff": K_eff, "n_merged": n_merged,
               "cells_diagnosed": len(per_cell), "focus_cell": cell,
               "focus_cell_n": int(cnt[cell]), "plane": a.plane, "cov_ridge": a.cov_ridge,
               "focus_inside_full": in_full, "focus_inside_diag": in_diag,
               "median_inside_full": np.median(cov_full, 0).tolist(),
               "median_inside_diag": np.median(cov_diag, 0).tolist(),
               "theory_inside": list(SIGMA_THEORY),
               "embed": mode, "pick_rule": pick_note, "focus_cell_ks_rank": cell_rank,
               "ks_full": ks_full, "ks_diag": ks_diag,
               "cells_diag_worse": diag_worse, "corr_ks_vs_n": r_ks_n,
               "per_cell_ids": [int(k) for k in usable],
               "per_cell_ks_full": ksf_c.tolist(), "per_cell_ks_diag": ksd_c.tolist(),
               "per_cell_inside_full": cov_full.tolist(),
               "per_cell_inside_diag": cov_diag.tolist(),
               "median_abs_r_joint": med_j, "median_abs_r_time": med_t,
               "chance_abs_r": med_chance, "n_obs_steps": n_obs, "base_dim": base_dim,
               "cell_counts": cnt.tolist()},
              (out / f"summary_{tag}.json").open("w"), indent=2)


if __name__ == "__main__":
    main()
