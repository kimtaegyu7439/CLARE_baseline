#!/usr/bin/env python
"""L2_V12 — 앵커 표본을 **과거 태스크에 무작위 배분**한다 (ER 과 같은 꼴, 액션 공간).

    python L2_V12.py --beta 2.0 --batch_size 24 --anchor_rows 8 ...

V7(= V6 + --dir_space act) 의 dea 앵커와 수식이 같다. 바뀌는 것은
**앵커 표본을 어떻게 나누느냐** 하나다.

────────────────────────────────────────────────────────────────────────────
무엇이 바뀌나
────────────────────────────────────────────────────────────────────────────
기존(V7)
    과거 태스크 j **마다** 코드북에서 n=B 개를 뽑는다. 태스크가 3 개면 3B 개.
    태스크별로 손실을 만들어 1/N 로 평균한다. 즉 **모든 과거 태스크가 매 스텝
    같은 수의 표본을 받는다**.

V12
    ER 의 배치 분할을 그대로 따른다. 배치 32 = 새 태스크 24 + 앵커 8.
    앵커 8 개를 과거 태스크에 **무작위로 배분**한다.
        task 0,1,2 가 과거라면  예: task0 2 개, task1 3 개, task2 3 개
    그 8 개를 한 덩어리로 모아 손실을 만들고 평균한다.

    ER 이 버퍼에서 태스크 구분 없이 8 개를 뽑는 것과 같은 구조다
    (L2_er_cl.py:181 의 cur_bs = batch_size - replay_batch_size, 그리고
     build_replay 가 과거 스플릿을 ConcatDataset 으로 합쳐 shuffle 하는 것).

    배분은 과거 태스크에 **균일한 다항분포**다. 매 스텝 다시 뽑으므로 장기적으로는
    태스크마다 같은 기대 표본 수를 받지만, 한 스텝에서는 치우친다.

────────────────────────────────────────────────────────────────────────────
W (dea 가중) 가 어떻게 달라지나
────────────────────────────────────────────────────────────────────────────
V7 의 dea_weights 는 (N,B,H) 를 **태스크 축 N 에 걸쳐** 정규화한다 — 같은
(배치, 청크) 칸에서 태스크들이 방향 예산을 나눠 갖는 구조다.

V12 는 태스크 축이 없다. 8 개 행이 이미 섞인 덩어리이므로, 같은 action horizon
위치끼리 **행 축(8)에 걸쳐** 정규화한다.

        w_r = (e_r + ε_w) / (Σ_{r'} e_{r'} + n_valid·ε_w)      r = 0..7

Σ_r w_r = 1 이라 방향 항의 총량은 V7 과 같게 유지된다. 크기 항은 8 개 행 전체
평균이다(V7 의 1/N 평균에 대응).

────────────────────────────────────────────────────────────────────────────
읽을 때 주의
────────────────────────────────────────────────────────────────────────────
앵커 표본이 과거 태스크당 B 개에서 **전체 8 개**로 줄어든다. 과거가 9 개인
스테이지 9 에서는 태스크당 기대 0.9 개다. 앵커 추정량의 분산이 V7 보다 훨씬
크므로, V11(배치당 1 개)에서 본 것과 같은 방향의 손해가 예상된다.
목적은 성능 향상이 아니라 **ER 과 같은 표본 예산에서 TETHER 가 어떻게 되는가**
를 보는 것이다.
"""
from __future__ import annotations

import sys
import time

import torch

import L2_V6
from L2_V6 import split_mag_dir

B1 = L2_V6.B1


def pooled_dea_weights(e: torch.Tensor, valid: torch.Tensor,
                       eps_w: float) -> torch.Tensor:
    """(R,H) 방향오차 -> (R,H) 가중. **행 축 R 에 걸쳐** 합이 1.

    V6 의 dea_weights 는 태스크 축에 걸쳐 정규화하는데, 여기서는 8 개 행이
    이미 섞인 덩어리라 태스크 축이 없다. 같은 horizon 위치끼리 행들이
    방향 예산을 나눠 갖도록 축만 바꾼 것이다. 수식은 동일하다.
    """
    e = torch.where(valid, e, torch.zeros_like(e))               # (R,H)
    nv = valid.sum(0).clamp_min(1)                               # (H,)
    den = e.sum(0) + nv.to(e.dtype) * eps_w                      # (H,)
    num = torch.where(valid, e + eps_w, torch.zeros_like(e))     # (R,H)
    return (num / den.clamp_min(1e-12)).detach()                 # (R,H)


class PooledTaskAnchor(L2_V6.DEAAnchor):
    """앵커 표본 ANCHOR_ROWS 개를 과거 태스크에 무작위 배분한다."""

    name = "L2_V12_pooled"
    ANCHOR_ROWS = 8

    def describe(self):
        return (f"L2_V12 — 앵커 {self.ANCHOR_ROWS} 개를 과거 태스크에 무작위 배분 "
                f"(act), form={self.a.anchor_form}, β={self.a.beta}, "
                f"코드북 {len(self.books)}개")

    # ── 태스크 배분 ─────────────────────────────────────────────────────
    def _allot(self, js: list[int], rows: int) -> dict[int, int]:
        """과거 태스크 js 에 rows 개를 균일 다항분포로 나눈다.

        ER 이 버퍼 풀에서 무작위로 뽑는 것과 같은 효과다. 매 스텝 다시 뽑으므로
        장기 기대값은 태스크마다 rows/N 로 같고, 한 스텝에서는 치우친다.
        """
        idx = torch.randint(len(js), (rows,), device="cpu")
        cnt: dict[int, int] = {}
        for i in idx.tolist():
            cnt[js[i]] = cnt.get(js[i], 0) + 1
        return cnt

    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        cls = getattr(self, "cls", None)
        if cls is None:
            cls = B1.rgb_cls(policy, batch)
        self.cls = cls
        self._collect(batch, cls, device)                 # 수집은 전체 배치로

        if k == 0 or self.teacher is None or not self.books:
            return torch.zeros((), device=device)

        t0w = time.perf_counter()
        B = x_t.shape[0]
        R = min(self.ANCHOR_ROWS, B)                      # 앵커 행 수 (보통 8)
        js = sorted(self.books)
        alloc = self._allot(js, R)
        P = getattr(self.a, "pose_dims", None) or B1.POSE_DIMS
        beta = self.a.beta
        per_row = cls.shape[0] // B                       # 보통 4 (카메라2 x 프레임2)

        # ── 태스크별로 배정된 개수만큼 질의를 만들어 한 덩어리로 잇는다 ────
        # x_t / t 는 현재 배치의 앞 R 행을 쓴다. 태스크별로 자르는 구간이
        # 서로 겹치지 않게 offset 으로 배분한다.
        segs, off = [], 0
        for j in sorted(alloc):
            c = alloc[j]
            sl = slice(off, off + c)
            off += c
            sub = {}
            for key, v in batch.items():
                sub[key] = v[sl] if (torch.is_tensor(v) and v.dim() > 0
                                     and v.shape[0] == B) else v
            saved_cls, self.cls = self.cls, cls[sl.start * per_row:sl.stop * per_row]
            try:
                q = self._query(policy, sub, x_t[sl], t[sl], k, instructions, j, c,
                                sub["observation.state"].shape, t[sl][:, None, None],
                                device)
            finally:
                self.cls = saved_cls
            segs.append((j, c, sl, q))

        # ── pass 1: 모든 행의 e 를 모아 행 축으로 정규화한 W 를 만든다 ──────
        W = None
        if self.a.anchor_form == "dea":
            es, vs = [], []
            with torch.no_grad():
                for j, c, sl, q in segs:
                    v_s = self._student_v(policy, q, t[sl])
                    p_s, p_t = self._pair(v_s, q, t[sl])
                    _, e, _, _, va = split_mag_dir(p_s, p_t, P)      # (c,H)
                    es.append(e); vs.append(va)
            W = pooled_dea_weights(torch.cat(es), torch.cat(vs), self.a.eps_w)  # (R,H)

        # ── pass 2: 태스크별로 손실을 만들고 즉시 backward (chunk) ──────────
        # ★ 각 조각의 손실은 **전체 R 행 기준**으로 정규화한다(c/R 가중).
        #   그래야 조각을 나눠 backward 해도 "8 개의 평균" 과 같아진다.
        logs = {"m": [], "e": [], "w": []}
        total = 0.0
        pos = 0
        for j, c, sl, q in segs:
            v_s = self._student_v(policy, q, t[sl])
            p_s, p_t = self._pair(v_s, q, t[sl])
            m, e, r_s, r_t, valid = split_mag_dir(p_s, p_t, P)       # (c,H)

            if self.a.anchor_form == "sqerr":
                coef = (2.0 * r_s * r_t).detach()
            elif self.a.anchor_form == "unifmd":
                coef = torch.full_like(e, beta)
            elif self.a.anchor_form == "dea":
                coef = beta * W[pos:pos + c]                         # (c,H)
            else:
                raise ValueError(f"알 수 없는 anchor_form: {self.a.anchor_form}")
            pos += c

            e0 = torch.where(valid, e, torch.zeros_like(e))
            # c/R 가중: 조각 평균 x (c/R) 를 더하면 전체 R 행 평균이 된다
            w_seg = c / R
            L_m = m.mean() * w_seg
            L_e = (coef * e0).mean() * w_seg
            L_j = self.lam_lvl * (L_m + L_e)
            L_j.backward()
            total += float(L_j.detach())
            logs["m"].append(float(m.mean()))
            logs["e"].append(float(e0.mean()))
            logs["w"].append(float(coef.mean()))

        self.step += 1
        if self.step % max(1, self.a.log_every_anchor) == 0:
            import logging
            alloc_s = " ".join(f"t{j}:{alloc[j]}" for j in sorted(alloc))
            logging.info(f"[V12][{self.a.anchor_form}] k={k} step={self.step:5d} "
                         f"rows={R} [{alloc_s}] "
                         f"m={sum(logs['m'])/len(logs['m']):.4f} "
                         f"e={sum(logs['e'])/len(logs['e']):.4f} "
                         f"w={sum(logs['w'])/len(logs['w']):.3f} "
                         f"ms={1e3*(time.perf_counter()-t0w):.0f}")
            self.log.write(__import__("json").dumps(
                {"task": k, "step": self.step, "form": self.a.anchor_form,
                 "rows": R, "alloc": {str(a): b for a, b in alloc.items()},
                 "m_mean": sum(logs["m"]) / len(logs["m"]),
                 "e_mean": sum(logs["e"]) / len(logs["e"]),
                 "w_mean": sum(logs["w"]) / len(logs["w"]),
                 "ms_per_step": 1e3 * (time.perf_counter() - t0w)}) + "\n")
            self.log.flush()

        # 이미 backward 했으므로 상위에는 상수만 돌려준다(그래프 없음)
        return torch.tensor(total, device=device)


def main() -> None:
    argv, out, skip = sys.argv[1:], [], False
    rows = None
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--dir_space":
            skip = True; continue
        if a.startswith("--dir_space="):
            continue
        if a == "--anchor_rows":
            skip = True
            rows = argv[argv.index(a) + 1]; continue
        if a.startswith("--anchor_rows="):
            rows = a.split("=", 1)[1]; continue
        out.append(a)
    if rows is not None:
        PooledTaskAnchor.ANCHOR_ROWS = int(rows)
    sys.argv = [sys.argv[0]] + out + ["--dir_space", "act"]

    orig = L2_V6.DEAAnchor
    L2_V6.DEAAnchor = PooledTaskAnchor
    try:
        L2_V6.main()
    finally:
        L2_V6.DEAAnchor = orig


if __name__ == "__main__":
    main()
