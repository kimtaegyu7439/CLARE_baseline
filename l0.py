#!/usr/bin/env python
"""L0 — Implicit CARA: 명령어-앙상블 조건응답 앵커 (R13 + L_icara).

가설
    task1형 붕괴는 값(level)의 붕괴가 아니라 **routing(조건응답)의 붕괴**다.
    level 앵커 ‖v_S(b,ℓ_j) − v_T(b,ℓ_j)‖² 는 학생·teacher 공통 오차가 지배해서
    routing 을 못 지킨다. 명령어를 흔들었을 때의 **응답 차이**를 직접 앵커하면
    공통 성분이 뺄셈에서 상쇄되고 routing 만 남는다.

추가 항 (기존 b·(x_t,t)·j 를 그대로 재사용)
    seen  = {ℓ_0..ℓ_k}  (--no_include_current 로 현재 제외),  K_s = |seen|
    δℓ    = Σ_m α_m (ℓ_m − ℓ̄),   α_m ~ N(0, 1/K_s)
            ⇒ E[δℓ δℓᵀ] = (1/K_s) Σ_m (ℓ_m−ℓ̄)(ℓ_m−ℓ̄)ᵀ = 경험 공분산 Σ_ℓ
            ISDA 의 조건축 버전. ℓ_ref 를 고르는 자의성과 O(K) 비용이 사라진다.
    Δ_S   = v_S(b, ℓ_j+δℓ) − v_S(b, ℓ_j)      # v_S(b,ℓ_j) 는 level 항과 공유
    Δ_T   = v_T(b, ℓ_j+δℓ) − v_T(b, ℓ_j)      # no_grad, v_T(b,ℓ_j) 공유
    L_icara = ‖Δ_S − Δ_T‖²                    # reduction 은 level 과 동일
    L_anchor = R13 앵커 + λ_ic · L_icara

    새 파라미터 0, 저장 추가 0. forward 는 j 당 학생 1→2, teacher 1→2
    (R10 의 structure 항과 같은 비용 구조).

δℓ 을 더하는 공간 — 사전 조사 결과
    B1.encode_lang(pol, texts) = pol.dit_flow.language_embedding_projection(
                                     pol.dit_flow.language_encoder(texts))
    앞쪽 language_encoder(CLIP)는 **동결**이고 B1.snapshot 이 teacher 와 student 가
    같은 객체를 가리키게 한다 → 두 모델의 CLIP 출력이 문자 그대로 같다.
    뒤쪽 language_embedding_projection 은 **학습 대상**이라 모델마다 다르다.
    그래서 기본값은 CLIP 공간(--delta_space clip)이다. 그래야 δℓ 이 "명령어의
    섭동"이라는 모델 독립적 의미를 갖고, 학생/teacher 에 **같은 섭동**이 들어간다.
    투영 뒤 공간(--delta_space proj)은 학생의 투영으로 만든 방향을 teacher 에
    적용하게 되어 두 모델이 서로 다른 좌표계에서 흔들린다.

λ_ic 자동 설정
    각 스테이지 첫 --warmup_ic 스텝 동안 L_icara 를 **손실에 넣지 않고** 크기만 잰다
    (그 구간은 no_grad 로 계산해 메모리도 안 쓴다). 그 뒤
        λ_ic = ρ_ic · λ_level · mean(L_level) / mean(L_icara)
    ρ_ic=1.0 이면 두 항의 기여가 같은 크기에서 출발한다.

금지 사항 준수
    R10.py / R13.py / B1.py 를 수정하지 않는다. loss 를 통째로 override 하는 대신
    쓸 만한 훅이 없어서 R10.R10Anchor.loss 본문을 복사한 뒤 표시된 구간만 추가했다.
    b·δℓ 캐시 없음(매 스텝 fresh). 과거 원시 데이터·행동 데이터 미사용.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import B1
import R10
import R13

OUT_DIR = REPO / "results" / "L0"


class L0Anchor(R13.R13Anchor):
    """R13(가우시안 샘플 좌표 + level 앵커) + implicit CARA 조건응답 항."""

    name = "L0"

    def __init__(self, args):
        super().__init__(args)
        self.rho_ic = args.rho_ic
        self.include_current = args.include_current
        self.delta_space = args.delta_space
        self.warmup_ic = args.warmup_ic
        self.lam_ic: float | None = None        # 스테이지마다 리셋
        self.warm_ic: list[tuple[float, float]] = []
        self.lam_ic_by_stage: dict[int, float] = {}
        self._acc = None                        # stage 누적 (GPU 텐서)
        self.t_extra = 0.0                      # icara 전용 forward 누적 시간
        self.t_anchor = 0.0                     # 앵커 loss 전체 누적 시간
        self.ic = (self.out / "l0.jsonl").open("a")
        self._ic_sanity = False
        self.warned_blind = 0

    def describe(self):
        return (f"L0 — R13 + implicit CARA (ρ_ic={self.rho_ic}, "
                f"δℓ공간={self.delta_space}, 현재포함={self.include_current}), "
                f"통계 {len(self.stats)}개, λ_lvl={self.lam_lvl}")

    # ── 스테이지 시작 ───────────────────────────────────────────────────────
    def on_task_start(self, policy, k, args, instructions, device, **kw):
        super().on_task_start(policy, k, args, instructions, device, **kw)
        self.lam_ic = None if k > 0 else 0.0
        self.warm_ic = []
        self._acc = None
        self._ic_sanity = k > 0

    # ── δℓ 생성 (매 스텝 fresh, 캐시 없음) ──────────────────────────────────
    def _seen_js(self, k):
        js = sorted(self.stats)
        if self.include_current and k not in js:
            js = js + [k]
        return js

    def _clip(self, policy, texts, device):
        """CLIP 텍스트 임베딩 (문자열 캐시가 있어 사실상 공짜). (len,512) float32."""
        with torch.no_grad():
            return policy.dit_flow.language_encoder(texts).to(device).float()

    def _delta(self, policy, k, instructions, n, device):
        """δℓ (n,512). α_m ~ N(0,1/K_s) 를 **배치 샘플마다 독립으로** 뽑는다.

        스펙은 '매 스텝 새로'만 요구한다. 배치 안에서도 독립으로 뽑으면 한 스텝에
        B개의 방향을 보게 되어 같은 비용으로 추정 분산이 준다. j 사이에서는 공유해
        '같은 명령어 섭동에 대한 응답'을 과거 태스크들끼리 비교하게 둔다.
        """
        js = self._seen_js(k)
        texts = [instructions[f"task{j}"] for j in js]
        Ks = len(texts)
        if Ks < 2:
            return None, Ks
        if self.delta_space == "clip":
            L = self._clip(policy, texts, device)                     # (Ks,512) 동결·공유
        else:
            L = B1.encode_lang(policy, texts).detach().float()        # (Ks,512) 학생 투영
        C = L - L.mean(0, keepdim=True)                               # (Ks,512)
        alpha = torch.randn(n, Ks, device=device) / math.sqrt(Ks)     # N(0, 1/Ks)
        return (alpha @ C).detach(), Ks                               # (n,512)

    # ── 손실 ────────────────────────────────────────────────────────────────
    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        # ══ 아래는 R10.R10Anchor.loss 의 복사본이다. ★L0★ 로 표시한 구간만 추가했다.
        if k == 0 or self.teacher is None or args.lambda_anchor == 0:
            return torch.zeros((), device=device)
        t_loss0 = time.perf_counter()
        cls = getattr(self, "cls", None)
        if cls is None:
            cls = B1.rgb_cls(policy, batch)
        n = batch["observation.state"].shape[0]
        o = cls.view(n, -1, cls.shape[-1]).float()
        tau = R10.phase_bins(batch, self.ep_len, self.n_bins).to(device)

        mu_n, sg_n = self.cur["mu"].to(device), self.cur["sigma"].to(device)
        floor = self.cur["sigma_floor"]
        h = self.cur["h"]
        if self.sample_z:
            z = torch.randn_like(o).clamp_(-3.0, 3.0)
        else:
            z = ((o - mu_n[tau]) / sg_n[tau].clamp_min(floor)).clamp_(-3.0, 3.0)
        z = z.detach()
        u = self.direction(z).detach()

        chunk = self.a.chunk_backward
        if chunk and getattr(policy.config, "use_amp", False):
            raise RuntimeError("chunk_backward 는 use_amp=True 와 함께 쓸 수 없다")

        # ══ ★L0★ δℓ — 매 스텝 새로 뽑는다 ═══════════════════════════════════
        dl, Ks = self._delta(policy, k, instructions, n, device)
        ic_on = dl is not None
        ic_grad = ic_on and (self.lam_ic is not None) and (self.lam_ic > 0)
        # ═══════════════════════════════════════════════════════════════════

        lvl, stc, ics = [], [], []
        # ★L0★ 누적은 GPU 텐서로 한다. float() 로 매 스텝 꺼내면 j 마다 동기화가
        #      걸려(K=9 면 스텝당 18회) 학습이 눈에 띄게 느려진다.
        rS = torch.zeros((), device=device)
        rT = torch.zeros((), device=device)
        teach = self.teacher

        # ══ ★L0★ fwd 를 (tail 계산) / (velocity 호출) 로 쪼갠다.
        #    같은 b_j 에서 명령어만 바꿔 두 번 호출하므로 tail 을 재사용할 수 있다.
        #    쪼개기 전후 수치는 동일하다(R10 의 fwd 를 그대로 두 조각으로 나눈 것).
        def tail_of(pol, c):
            flat = c.reshape(-1, c.shape[-1]).to(x_t.dtype)
            return B1.cond_tail(pol, batch, flat)

        def vel(pol, lang, tl):
            return pol.dit_flow.velocity_net(
                noisy_actions=x_t, time=t, global_cond=B1.make_cond(lang, tl))

        def lang_of(pol, emb):
            m = pol.dit_flow.language_embedding_projection
            return m(emb.to(next(m.parameters()).dtype))

        # (R10 원본의 fwd(pol,c) 는 vel(pol, encode_lang(pol,past), tail_of(pol,c)) 와
        #  같다. 명령어만 바꿔 두 번 부르므로 위처럼 쪼갰다.)
        # ═══════════════════════════════════════════════════════════════════

        for j in sorted(self.stats):
            st = self.stats[j]
            b_j = (st["mu"].to(device)[tau] + st["sigma"].to(device)[tau] * z).detach()

            # ★L0★ CLIP 임베딩을 한 번만 뽑아 학생/teacher/섭동에 모두 재사용.
            #      language_encoder 는 동결·공유이므로 두 모델에서 값이 같다.
            emb0 = self._clip(policy, [instructions[f"task{j}"]], device).expand(n, -1)
            tl_S = tail_of(policy, b_j)
            with torch.no_grad():
                tl_T = tail_of(teach, b_j)

            with torch.no_grad():
                vt0 = vel(teach, lang_of(teach, emb0), tl_T)
            # ★L0★ 학생은 train() 이라 dropout(p=0.1)이 살아 있다. vs0 와 vs1c 가
            #      **다른 마스크**를 쓰면 Δ_S = vs1c−vs0 에 δℓ 과 무관한 dropout
            #      잡음이 섞인다. 실측(스모크): resp_S 0.892 vs resp_T 0.278 로
            #      학생 쪽이 3.2배 부풀고, 스케일 선형성 L(2δ)/L(δ) 가 4 대신
            #      0.99 로 나왔다 — 잡음이 신호를 완전히 덮은 상태다.
            #      vs1c 직전에 RNG 상태를 되돌려 **같은 마스크**를 쓰게 한다.
            #      vs0 자체는 R13 과 동일하게 유지된다(상태를 읽기만 한다).
            rng_state = torch.cuda.get_rng_state() if ic_on else None
            vs0 = vel(policy, lang_of(policy, emb0), tl_S)
            r0 = vs0 - vt0.to(vs0.dtype)
            if self.use_struct:
                b_h = (b_j + h * u).detach()
                tl_Sh = tail_of(policy, b_h)
                with torch.no_grad():
                    tl_Th = tail_of(teach, b_h)
                    vt1 = vel(teach, lang_of(teach, emb0), tl_Th)
                vs1 = vel(policy, lang_of(policy, emb0), tl_Sh)
                r1 = vs1 - vt1.to(vs1.dtype)

            L_j = self.reduce_level(r0)
            S_j = (self.reduce_struct((r1 - r0) / h) if self.use_struct
                   else torch.zeros((), device=device))

            # ══ ★L0★ implicit CARA 항 ═════════════════════════════════════
            I_j = torch.zeros((), device=device)
            if ic_on:
                te = time.perf_counter()
                emb1 = emb0 + dl
                with torch.no_grad():
                    vt1c = vel(teach, lang_of(teach, emb1), tl_T)
                    dT = (vt1c - vt0).float()
                torch.cuda.set_rng_state(rng_state)        # ★L0★ 같은 dropout 마스크
                if ic_grad:
                    vs1c = vel(policy, lang_of(policy, emb1), tl_S)
                    dS = vs1c - vs0
                else:
                    with torch.no_grad():      # warmup — 크기만 잰다
                        vs1c = vel(policy, lang_of(policy, emb1), tl_S)
                        dS = vs1c - vs0.detach()
                I_j = self.reduce_level(dS - dT.to(dS.dtype))
                with torch.no_grad():
                    rS += dS.detach().float().flatten(1).norm(dim=1).mean()
                    rT += dT.float().flatten(1).norm(dim=1).mean()
                self.t_extra += time.perf_counter() - te
            # ═══════════════════════════════════════════════════════════════

            if chunk:
                term = (self.lam_lvl * L_j if self.lam_str is None
                        else self.lam_lvl * L_j + self.lam_str * S_j)
                if ic_grad:
                    term = term + self.lam_ic * I_j       # ★L0★
                (args.lambda_anchor * term).backward()
                lvl.append(L_j.detach()); stc.append(S_j.detach())
                ics.append(I_j.detach())
            else:
                lvl.append(L_j); stc.append(S_j); ics.append(I_j)

            if getattr(self, "_sanity", False) and j == min(self.stats):
                with torch.no_grad():
                    mj = st["mu"].to(device)[tau]
                    rel = float((b_j.mean(0) - mj.mean(0)).norm() / mj.mean(0).norm().clamp_min(1e-8))
                    st_msg = (
                        f"‖(r1−r0)/h‖={float(((r1-r0)/h).flatten(1).norm(dim=1).mean()):.4f}"
                        if self.use_struct else "structure 없음")
                    logging.info(
                        f"[L0][sanity] task{k} j={j}  ‖b̄−μ̄_j‖/‖μ̄_j‖={rel:.4f}  "
                        f"‖u‖={float(u.flatten(1).norm(dim=1).mean()):.4f}  "
                        f"teacher 유한={bool(torch.isfinite(vt0).all())}  "
                        f"‖r0‖={float(r0.flatten(1).norm(dim=1).mean()):.4f}  "
                        f"{st_msg}  null 호출={self.null_calls}")
                self._sanity = False

            # ══ ★L0★ 스테이지 첫 앵커 스텝 sanity (로그만, 중단 없음) ═══════
            if self._ic_sanity and j == min(self.stats):
                self._run_ic_sanity(policy, teach, k, j, dl, Ks, emb0, tl_S, tl_T,
                                    vs0, vt0, dS if ic_on else None,
                                    dT if ic_on else None, ic_grad, vel, lang_of,
                                    rng_state)
                self._ic_sanity = False
            # ═══════════════════════════════════════════════════════════════

        L_lvl = sum(lvl) / len(lvl)
        L_str = sum(stc) / len(stc)
        L_ic = sum(ics) / len(ics)                                    # ★L0★

        if not self.use_struct:
            self.lam_str = 0.0
        if self.lam_str is None:
            self.warm.append((float(L_lvl.detach()), float(L_str.detach())))
            if len(self.warm) >= self.a.warmup_steps:
                ml = sum(x for x, _ in self.warm) / len(self.warm)
                ms = sum(y for _, y in self.warm) / len(self.warm)
                self.lam_str = self.a.rho * self.lam_lvl * ml / max(ms, 1e-12)
            out = self.lam_lvl * L_lvl
        else:
            out = self.lam_lvl * L_lvl + self.lam_str * L_str

        # ══ ★L0★ λ_ic 자동 설정 ═══════════════════════════════════════════
        if ic_on and self.lam_ic is None:
            self.warm_ic.append((float(L_lvl.detach()), float(L_ic.detach())))
            if len(self.warm_ic) >= self.warmup_ic:
                ml = sum(x for x, _ in self.warm_ic) / len(self.warm_ic)
                mi = sum(y for _, y in self.warm_ic) / len(self.warm_ic)
                self.lam_ic = self.rho_ic * self.lam_lvl * ml / max(mi, 1e-12)
                self.lam_ic_by_stage[k] = self.lam_ic
                logging.info(
                    f"[L0] task {k} λ_ic = {self.lam_ic:.6g}  "
                    f"(mean L_level {ml:.5g} / mean L_icara {mi:.5g}, ρ_ic={self.rho_ic}, "
                    f"K_s={Ks})")
                json.dump(self.lam_ic_by_stage,
                          (self.out / "lambda_ic.json").open("w"), indent=2)
        elif ic_grad and not chunk:
            out = out + self.lam_ic * L_ic
        # ═══════════════════════════════════════════════════════════════════

        if chunk:
            out = out.detach()

        self.step += 1
        self.t_anchor += time.perf_counter() - t_loss0
        nj = max(1, len(ics))
        if ic_on:                                                     # ★L0★ 누적
            if self._acc is None or self._acc["dev"] != str(device):
                self._acc = {"dev": str(device), "n": 0,
                             "S": torch.zeros((), device=device),
                             "T": torch.zeros((), device=device),
                             "ratio": torch.zeros((), device=device),
                             "L": torch.zeros((), device=device)}
            A = self._acc
            A["n"] += 1
            A["S"] += rS / nj
            A["T"] += rT / nj
            A["ratio"] += rS / rT.clamp_min(1e-12)
            A["L"] += L_ic.detach().float()

        if self.step % self.a.log_every_anchor == 0:
            self.log.write(json.dumps({
                "task": k, "step": self.step, "L_level": float(L_lvl.detach()),
                "L_struct": float(L_str.detach()), "lambda_struct": self.lam_str,
                "h": h, "n_past": len(self.stats)}) + "\n")
            self.log.flush()
            sS, sT = float(rS) / nj, float(rT) / nj
            if sT < 1e-4 and self.warned_blind < 5:
                self.warned_blind += 1
                logging.warning(f"[L0][warn] teacher blind — resp_T={sT:.3e} "
                                f"(task {k}, step {self.step})")
            rec = {"task": k, "step": self.step, "K_s": Ks,
                   "L_level": float(L_lvl.detach()), "L_icara": float(L_ic.detach()),
                   "lambda_ic": self.lam_ic,
                   "resp_S": sS, "resp_T": sT,
                   "ratio": sS / max(sT, 1e-12),
                   "t_extra_frac": self.t_extra / max(self.t_anchor, 1e-9),
                   "ms_per_step": 1000 * self.t_anchor / max(self.step, 1)}
            self.ic.write(json.dumps(rec) + "\n"); self.ic.flush()
            logging.info(f"[L0] k={k} step={self.step:5d} L_lvl={rec['L_level']:.4f} "
                         f"L_ic={rec['L_icara']:.4f} λ_ic={self.lam_ic} "
                         f"respS={rec['resp_S']:.4f} respT={rec['resp_T']:.4f} "
                         f"ratio={rec['ratio']:.3f} extra={100*rec['t_extra_frac']:.0f}%")
        return out

    # ── sanity (스테이지당 1회, 로그만) ─────────────────────────────────────
    def _run_ic_sanity(self, policy, teach, k, j, dl, Ks, emb0, tl_S, tl_T,
                       vs0, vt0, dS, dT, ic_grad, vel, lang_of, rng_state=None):
        msgs = [f"K_s={Ks}", f"δℓ공간={self.delta_space}"]
        if dl is None:
            logging.warning(f"[L0][sanity] task{k} δℓ 없음 (K_s={Ks}<2) — icara 항 꺼짐")
            return
        with torch.no_grad():
            msgs.append(f"‖δℓ‖={float(dl.norm(dim=1).mean()):.4f}")
            msgs.append(f"resp_T={float(dT.flatten(1).norm(dim=1).mean()):.3e}"
                        if dT is not None else "resp_T=?")
            if dT is not None and float(dT.flatten(1).norm(dim=1).mean()) <= 1e-4:
                msgs.append("★teacher blind 경고★")
            # 스케일 선형성: 2δℓ 에서 L_icara 가 4배인가 (bf16 양자화 확인)
            e2 = emb0 + 2.0 * dl
            vt2 = vel(teach, lang_of(teach, e2), tl_T)
            if rng_state is not None:
                torch.cuda.set_rng_state(rng_state)   # vs0 와 같은 dropout 마스크
            vs2 = vel(policy, lang_of(policy, e2), tl_S)
            I1 = float(self.reduce_level((dS - dT.to(dS.dtype))))
            I2 = float(self.reduce_level(((vs2 - vs0) - (vt2 - vt0).to(vs0.dtype))))
            ratio = I2 / max(I1, 1e-12)
            msgs.append(f"스케일선형성 L(2δ)/L(δ)={ratio:.2f} (기대 4)")
            if not (2.0 <= ratio <= 8.0):
                msgs.append("★선형성 이탈 경고★")
        # grad 경로 확인
        msgs.append(f"학생grad={'ON' if (ic_grad and dS is not None and dS.requires_grad) else 'OFF(warmup)'}")
        msgs.append(f"teacher grad={'ON★위반★' if (dT is not None and dT.requires_grad) else 'OFF'}")
        logging.info(f"[L0][sanity-ic] task{k} j={j}  " + "  ".join(msgs))

    # ── 스테이지 종료 ───────────────────────────────────────────────────────
    def on_task_end(self, policy, k, args, instructions, device, **kw):
        A = self._acc
        if A and A["n"]:
            n = A["n"]
            row = {"stage": k, "steps": n,
                   "resp_S": float(A["S"]) / n, "resp_T": float(A["T"]) / n,
                   "ratio": float(A["ratio"]) / n, "L_icara": float(A["L"]) / n,
                   "lambda_ic": self.lam_ic_by_stage.get(k)}
            (self.out / "resp_by_stage.jsonl").open("a").write(json.dumps(row) + "\n")
            logging.info(f"[L0] stage {k} 평균  resp_T={row['resp_T']:.4f}  "
                         f"resp_S={row['resp_S']:.4f}  ratio={row['ratio']:.3f}")
        super().on_task_end(policy, k, args, instructions, device, **kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambda_level", type=float, default=3.0)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--anchor_norm", choices=["mean", "sum"], default="mean")
    ap.add_argument("--stats_batches", type=int, default=0)
    ap.add_argument("--chunk_backward", action="store_true")
    ap.add_argument("--log_every_anchor", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--teacher_bf16", action="store_true")
    ap.add_argument("--out", default=None)
    # ── L0 전용 ──
    ap.add_argument("--rho_ic", type=float, default=1.0)
    ap.add_argument("--warmup_ic", type=int, default=50)
    ap.add_argument("--delta_space", choices=["clip", "proj"], default="clip")
    ap.add_argument("--no_include_current", dest="include_current",
                    action="store_false", default=True)
    ap.add_argument("--passthru", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

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

    B1.ANCHOR = L0Anchor(args)

    argv = ["B1.py",
            "--p_drop", "0",
            "--guidance_w", "1.0",
            "--lambda_anchor", "1.0",
            "--out_dir", str(out_dir),
            "--ckpt_root", str(REPO / "outputs" / "L0")]
    if args.smoke:
        argv.append("--smoke")
    if args.teacher_bf16:
        argv.append("--teacher_bf16")
    argv += args.passthru

    json.dump({
        "arm": "L0", "base": "R13",
        "base_diff": [
            "1. R13 앵커(level, 가우시안 샘플 좌표)는 그대로 둔다.",
            "2. 같은 b·(x_t,t)·j 에서 명령어만 ℓ_j -> ℓ_j+δℓ 로 흔든 forward 를 추가한다.",
            "3. δℓ = Σ α_m(ℓ_m−ℓ̄), α~N(0,1/K_s) — 매 스텝 fresh, 배치 샘플마다 독립.",
            "4. L_icara = ‖Δ_S − Δ_T‖², λ_ic 는 스테이지 첫 50스텝에서 자동 설정.",
            "5. 새 파라미터 0 · 저장 추가 0 · forward 는 j 당 학생 2 / teacher 2.",
        ],
        "rho_ic": args.rho_ic, "warmup_ic": args.warmup_ic,
        "delta_space": args.delta_space, "include_current": args.include_current,
        "lambda_level": args.lambda_level, "n_bins": args.n_bins,
        "anchor_norm": args.anchor_norm, "chunk_backward": args.chunk_backward,
        "teacher": "rolling (1 snapshot)", "embedding": "dinov2_cls_768_frozen",
        "p_drop": 0.0, "guidance_w": 1.0, "argv": argv,
    }, (out_dir / "l0_config.json").open("w"), indent=2, ensure_ascii=False)

    old, sys.argv = sys.argv, argv
    try:
        B1.main()
    finally:
        sys.argv = old
        try:
            lam = B1.ANCHOR.lam_ic_by_stage
            cfg = json.loads((out_dir / "l0_config.json").read_text())
            cfg["lambda_ic_by_stage"] = lam
            json.dump(cfg, (out_dir / "l0_config.json").open("w"),
                      indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"[L0] l0_config.json 갱신 실패: {e}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
