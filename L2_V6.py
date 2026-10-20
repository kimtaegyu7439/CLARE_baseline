#!/usr/bin/env python
"""L2_V6 — Directional-Error-Aware (DEA) 앵커 손실.

    python L2_V6.py --anchor_form dea     --suite libero_goal ...   # 제안 방식
    python L2_V6.py --anchor_form sqerr   ...                       # 균일 제곱오차
    python L2_V6.py --anchor_form unifmd  ...                       # 균일 크기-방향

════════════════════════════════════════════════════════════════════════════
무엇이 달라지는가
════════════════════════════════════════════════════════════════════════════
속도 벡터를 크기와 방향으로 쪼개 따로 감독한다. 과거 태스크 j, 시점 t, 청크 h 에서

    v_θ = r_θ u_θ ,  v̄ = r̄ ū      (r = ‖v‖₂, ‖u‖₂ = 1)
    m_j = (r_θ,j − r̄_j)²           크기 오차
    e_j = 1 − u_θ,j·ū_j ∈ [0,2]    방향 오차

세 팔은 **e_j 의 계수만** 다르다 (노트 2절 표 그대로):

    팔        m_j 계수    e_j 계수
    sqerr     1/N        2 r_θ,j r̄_j / N      ← ‖v_θ−v̄‖² 와 항등. 즉 MSE
    unifmd    1/N        β / N                 ← 크기·방향을 분리하고 균일 배분
    dea       1/N        β w_j  (detach)       ← 방향 예산을 오차 큰 태스크로

    w_j = sg[ (e_j + ε_w) / (Σ_m e_m + N ε_w) ],   Σ_j w_j = 1

    L_anchor = (1/H) Σ_h [ (1/N) Σ_j m_j + β Σ_j w_j e_j ]
    L        = L_FM + λ · L_anchor

════════════════════════════════════════════════════════════════════════════
구현상 알아둘 것
════════════════════════════════════════════════════════════════════════════
★ w_j 는 **태스크 축에 걸쳐** 정규화된다. 그래서 어떤 j 의 손실을 역전파하기
  전에 **모든 j 의 e_j 를 먼저 알아야** 한다. 기존 chunk backward(한 태스크씩
  계산하고 즉시 backward)와 양립하지 않는다. 그래서 dea 는 2-pass 로 돈다:

      pass 1 (no_grad)  모든 j 의 e_j 를 구해 w_j 를 만든다
      pass 2 (grad)     j 마다 다시 계산해 β·w_j·e_j + m_j 를 backward

  학생 forward 가 2 배다. sqerr/unifmd 는 가중이 태스크에 의존하지 않아 1-pass 다.
  pass 1 에서 만든 x_j·cond_T·v̄ 는 재사용하므로 **teacher forward 는 늘지 않는다**.

════════════════════════════════════════════════════════════════════════════
shape 지도 — 손실이 벡터에서 스칼라로 접히는 경로
════════════════════════════════════════════════════════════════════════════
기호  B = 배치(32)   H = 청크/horizon(16)   A = 액션축(7)   P = pose 축(6)
      N = 과거 태스크 수(스테이지 k 에서 k 개)

  속도 벡터                v_s, v_t            (B,H,A) = (32,16,7)
      │  _pair  (dir_space=act 면 to_action)   (B,H,A)  ← 공간만 바뀜
      ▼
  ★ split_mag_dir — **여기서 크기/방향으로 쪼개진다**
      │  ① [..., :P] 로 gripper 를 잘라냄       (B,H,P) = (32,16,6)
      │  ② norm(dim=-1) / 내적(sum(-1))        A 축이 **접힌다**
      ▼
  크기 r_s, r_t                                (B,H)   = (32,16)
  크기 오차 m = (r_s−r_t)²                     (B,H)
  방향 오차 e = 1−cos                          (B,H)   ∈ [0,2]
  유효 마스크 valid                            (B,H)   bool
      │
      │  (dea 면) 태스크 N 개를 stack           (N,B,H) = (k,32,16)
      │  dea_weights 가 **축 0 에 걸쳐** 정규화  (N,B,H),  Σ_축0 = 1
      ▼
  계수 coef  (팔마다 다름)                     (B,H)
      │  L_m = inv_N · m.mean()
      │  L_e = where(valid, coef·e, 0).mean()   ← B x H 칸 전부 평균
      ▼
  스칼라 ()  ->  (λ_a · λ_lvl · (L_m+L_e)).backward()

핵심  A 축(액션 성분)은 split_mag_dir 에서 접히고, (B,H) 격자는 .mean() 에서
      접힌다. dea 만 그 사이에 (N,B,H) 태스크 축이 한 번 생겼다 사라진다.

★ 1/N 정규화. 기존 V4/V5 는 과거 태스크마다 λ 를 걸고 즉시 backward 해서 실효
  가중이 과거 수에 비례하는 **합**이었다(스테이지 9 에서 λ 의 9 배). 노트는 1/N
  **평균**이라 여기서는 N 으로 나눈다. --anchor_reduce sum 으로 옛 거동을 고를 수 있다.

★ 근0 teacher 마스킹. r̄ 가 0 에 가까우면 ū 가 정의되지 않는다. 그런 (t,h,j) 는
  방향 항에서 빼고 남은 태스크로 w 를 다시 정규화한다(노트의 edge case 지침).
  크기 항 m_j 는 전 태스크에 그대로 둔다.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch

import L2_V4_seq as B1


# ═════════════════════════════════════════════════════════════════════════════
#  크기 / 방향 오차
# ═════════════════════════════════════════════════════════════════════════════
def split_mag_dir(v_s: torch.Tensor, v_t: torch.Tensor, pose_dims: int | None,
                  eps: float = 1e-8):
    """★★ 여기가 앵커 손실이 **크기 항과 방향 항으로 쪼개지는 유일한 지점**이다.

    (B,H,A) 두 개 -> (m, e, r_s, r_t, valid) 각 (B,H).

    ── 무엇을 하는가 ───────────────────────────────────────────────────────
    student 와 teacher 의 속도 벡터를 각각 "크기 x 단위방향" 으로 분해한다.

        v_θ = r_θ · u_θ ,   v̄ = r̄ · ū        (r = ‖v‖₂,  ‖u‖₂ = 1)

    그리고 둘의 불일치를 **두 개의 스칼라**로 나눈다.

        m = (r_θ − r̄)²        크기 오차.  [0, ∞)
        e = 1 − u_θ·ū         방향 오차.  [0, 2]   (0 같은 방향 / 1 직각 / 2 정반대)

    ── 왜 쪼개는가: 이 항등식이 근거다 (수치 검증 완료) ────────────────────
        ‖v_θ − v̄‖²  =  (r_θ − r̄)²  +  2·r_θ·r̄·(1 − cos)
                     =      m       +  2·r_θ·r̄·  e

    즉 기존 MSE 앵커는 이미 "크기 오차 + 방향 오차" 의 합인데, 방향 항의 계수가
    2·r_θ·r̄ 로 **속도가 빠른 구간일수록 크다.** 느린 구간(r 이 작은 구간)의 방향은
    사실상 감독을 못 받는다 — 이것이 V4 의 speedinv 가중이 손대려던 문제다.
    V6 는 계수를 손으로 정해(unifmd: β, dea: β·w_j) 그 배분을 직접 통제한다.
    그래서 anchor_form=sqerr 로 두면 계수가 2·r_θ·r̄ 로 돌아가 **MSE 와 항등**이 된다
    — 세 팔을 같은 코드 경로로 비교할 수 있는 이유다 (loss() 의 표 참고).

    ── shape 이 어떻게 바뀌는가 ────────────────────────────────────────────
        입력   v_s, v_t : (B,H,A)  = (32,16,7)   B=배치, H=청크(horizon), A=액션축
        ①슬라이스       : (B,H,P)  = (32,16,6)   뒤 축만 pose P=6 으로 잘라냄
        ②norm(dim=-1)   : (B,H)    = (32,16)     ← A 축이 **사라진다**. 벡터->크기
        ③m, e           : (B,H)    = (32,16)     (t,h) 칸마다 스칼라 2개
    즉 **마지막 축(액션 성분)이 접히고, (배치 x 청크) 격자만 남는다.** 이후 loss()
    에서 이 격자에 계수를 곱하고 .mean() 으로 스칼라까지 접는다.

    ── pose_dims 로 gripper 를 빼는 이유 ──────────────────────────────────
    pose_dims 를 주면 그 앞쪽 축만 쓴다(gripper 제외). 노트의 v 는 "같은 정규화
    액션 좌표"의 벡터라 pose/gripper 를 섞어도 되지만, 이 코드베이스의 앵커는
    줄곧 pose 6 축만 가중해 왔으므로 기본을 그에 맞춘다.
    ★ 그리퍼는 정답이 사실상 ±1 이진값이라 "방향"이라는 개념이 맞지 않는다.
      6 축 pose 와 한 벡터로 묶어 각도를 재면 그리퍼 전환이 각도를 지배한다.
    """
    # P: 실제로 쓸 축 수. None 이면 전체(A), 보통 POSE_DIMS=6.
    P = v_s.shape[-1] if pose_dims is None else pose_dims
    a, b = v_s[..., :P], v_t[..., :P]                 # 각 (B,H,A)->(B,H,P)

    # ── ① 크기: 마지막 축을 L2 노름으로 접는다. (B,H,P) -> (B,H) ───────────
    r_s = a.norm(dim=-1)                              # (B,H)  student 속도 크기 r_θ
    r_t = b.norm(dim=-1)                              # (B,H)  teacher 속도 크기 r̄
    m = (r_s - r_t).pow(2)                            # (B,H)  크기 오차 m = (r_θ−r̄)²

    # ── ② 방향: 단위벡터 내적 = cos. (B,H,P) -> (B,H) ─────────────────────
    # (a*b).sum(-1) 이 내적이고, 각 크기로 나누면 단위벡터 내적이 된다.
    # 즉 u_θ·ū 를 u 를 따로 만들지 않고 한 번에 구한다(0 나눗셈만 막으면 된다).
    # 방향은 단위벡터 내적. r 이 0 이면 정의되지 않으므로 eps 로 막고 valid 로 뺀다.
    cos = (a * b).sum(-1) / (r_s.clamp_min(eps) * r_t.clamp_min(eps))
    # clamp(-1,1): 부동소수 오차로 |cos| 가 1 을 살짝 넘으면 e 가 음수가 된다.
    e = 1.0 - cos.clamp(-1.0, 1.0)                    # (B,H) ∈ [0,2]
    # valid: 어느 한쪽 크기가 ~0 이면 각도가 정의되지 않는다. 그 칸은 방향 항에서
    # 빼고(loss 의 torch.where), dea 가중의 정규화 분모에서도 뺀다.
    valid = (r_t > eps) & (r_s > eps)                 # (B,H) bool
    return m, e, r_s, r_t, valid                      # 각 (B,H) — A 축은 접혔다


def dea_weights(e_stack: torch.Tensor, valid_stack: torch.Tensor,
                eps_w: float) -> torch.Tensor:
    """(N,B,H) 방향오차 -> (N,B,H) 가중. **태스크 축(N)에 걸쳐** 합이 1.

        w_j = (e_j + ε_w) / (Σ_m e_m + N_valid·ε_w)

    노트 (2). ε_w 는 e 가 전부 0 일 때의 0/0 을 막고, 작을수록 적응이 세진다.
    무효(근0 teacher) 태스크는 분자·분모에서 모두 빼고 유효 태스크로만 정규화한다.

    ── 무엇을 위한 가중인가 ────────────────────────────────────────────────
    방향 "예산" 을 과거 태스크들 사이에 나눠 주는 것이다. Σ_j w_j = 1 이므로
    **총량은 고정**이고, 방향이 많이 틀어진 태스크가 더 큰 몫을 받는다.
    즉 균일 배분(unifmd, w_j = 1/N)의 적응형 버전이다.

    ── shape 이 어떻게 바뀌는가 ────────────────────────────────────────────
        입력  e_stack, valid_stack : (N,B,H)   N=과거 태스크 수. split_mag_dir 의
                                              (B,H) 결과를 torch.stack 으로 쌓은 것
        nv  = valid.sum(0)         : (B,H)     ← 축 0(태스크)이 접힌다
        den = e.sum(0) + nv·ε_w    : (B,H)     ← 같은 (t,h) 칸의 모든 태스크 합
        num                        : (N,B,H)
        반환 num/den (브로드캐스트) : (N,B,H)   (B,H) 가 N 축으로 자동 확장
    ★ 정규화 축이 **0(태스크)** 이라는 점이 핵심이다. (B,H) 칸마다 독립적으로
      "이 칸에서 태스크들이 예산을 어떻게 나누나" 를 계산한다. 배치나 청크 축으로
      합이 1 이 되는 게 아니다.

    ── 왜 .detach() 인가 ───────────────────────────────────────────────────
    w_j 는 **계수**이지 학습 대상이 아니다. detach 를 빼면 "e_j 를 줄이는" 대신
    "w_j 를 줄이는"(= 다른 태스크의 e 를 키우는) 경로로도 그래디언트가 흘러,
    과거 태스크를 일부러 더 망가뜨리는 쪽이 손실을 낮추는 퇴화가 생긴다.
    노트의 sg[·] (stop-gradient) 가 이것이다.

    ── 왜 2-pass 가 강제되는가 ────────────────────────────────────────────
    w_j 가 Σ_m e_m 에 의존하므로, 태스크 j 의 손실을 backward 하기 전에
    **모든 태스크의 e 를 먼저** 알아야 한다. 그래서 dea 만 pass 1(no_grad)로
    전체 e 를 모은 뒤 pass 2 에서 다시 계산한다 (loss() 참고).
    sqerr/unifmd 는 계수가 태스크에 의존하지 않아 1-pass 로 끝난다.
    """
    # 무효 칸의 e 를 0 으로 눌러 둔다 — 아래 e.sum(0) 분모에 섞이지 않게.
    e = torch.where(valid_stack, e_stack, torch.zeros_like(e_stack))
    nv = valid_stack.sum(0).clamp_min(1)                      # (B,H) 유효 태스크 수
    # 분자: 유효 태스크만 (e + ε_w), 무효는 0 -> 가중 0 을 받는다.
    num = torch.where(valid_stack, e + eps_w, torch.zeros_like(e))   # (N,B,H)
    # 분모: 같은 (t,h) 칸의 유효 태스크 e 합 + 유효 개수 x ε_w.
    # ★ N 이 아니라 nv 를 곱한다 — 무효 태스크까지 세면 Σ_j w_j < 1 이 된다.
    den = e.sum(0) + nv * eps_w                               # (B,H)
    # (N,B,H) / (B,H) -> 브로드캐스트로 (N,B,H). detach 는 위 설명 참고.
    return (num / den.clamp_min(1e-12)).detach()              # (N,B,H)


# ═════════════════════════════════════════════════════════════════════════════
#  앵커
# ═════════════════════════════════════════════════════════════════════════════
def to_action(v: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """속도 -> 그 속도가 가리키는 끝점(액션). 선형 경로 v = (a − x_t)/(1−t) 의 역.

        a = x_t + (1 − t) · v

    왜 쓰는가: v 는 공통 노이즈 ε 를 품어(‖v‖≈‖ε‖) student·teacher 가 거의 같은
    방향을 가리킨다. 각도가 t 에 따라 6 배까지 압축된다(t=0.1 에서 6.4x).
    a 로 바꾸면 그 공통 성분이 빠져 각도가 t 와 무관해진다.
    ★ m 과 e 를 **같은 공간**에서 뽑아야 한다. 섞으면
      ‖·‖² = m + 2·r·r·e 항등식이 깨져 sqerr 이 MSE 를 복원하지 못한다.

    ── shape: 바뀌지 않는다 ────────────────────────────────────────────────
        v, x_t : (B,H,A)   t : (B,)
        반환    : (B,H,A)   ← 공간만 옮기고 모양은 그대로. 그래서 split_mag_dir
                             에 v 대신 a 를 넣어도 코드가 그대로 돈다.
    t.view(-1,1,1) 이 (B,) -> (B,1,1) 로 축을 끼워, 샘플마다 자기 t 가 그 샘플의
    H x A 원소 전부에 걸리게 한다. (B,) 그대로는 뒤 축(A=7)과 맞으려 해 에러난다.

    ── 숫자 감각 (실측) ────────────────────────────────────────────────────
        ‖ε‖ ≈ √6 = 2.45   ‖a‖ ≈ 0.85   ‖v‖ ≈ 2.49
    속도의 대부분이 공통 노이즈다. 그래서 액션이 직각(90°)으로 틀어져 있어도
    속도 각도는 27° 로 보인다. 압축률이 t 에 의존하므로(t=0.1 에서 6.4x,
    t=0.9 에서 1.0x) 같은 정책 불일치가 t 에 따라 6 배 다른 e 로 들어온다.
    sqerr 은 계수 2·r_s·r̄ 가 t 와 함께 커져 자동 보정되지만, β 를 쓰는
    unifmd/dea 에는 그 보정이 없다 — dir_space=act (V7) 가 그 왜곡을 없앤다.
    """
    # a = x_t + (1−t)·v.  (B,H,A) + (B,1,1)*(B,H,A) -> (B,H,A)
    return x_t + (1.0 - t).view(-1, 1, 1) * v


class DEAAnchor(B1.L2CodebookAnchor):
    """L2CodebookAnchor 의 loss() 만 교체한다.

    코드북 생성·teacher 스냅샷·resume·수집은 전부 상속받는다. 바뀌는 것은
    "student 와 teacher 의 속도 차이를 어떤 스칼라로 바꾸는가" 하나뿐이다.
    """

    name = "L2_V6_DEA"

    def describe(self):
        return (f"L2_V6/V7 — {self.a.anchor_form} 앵커 ({getattr(self.a, 'dir_space', 'vel')} 공간)  β={self.a.beta}, "
                f"ε_w={self.a.eps_w}, reduce={self.a.anchor_reduce}, "
                f"K={self.K}, λ_lvl={self.lam_lvl}, 코드북 {len(self.books)}개")

    # ── 과거 태스크 하나의 질의 재료 ────────────────────────────────────────
    def _query(self, policy, batch, x_t, t, k, instructions, j, n, st_shape, tcol, device):
        """(x_j, cond_S 재료, v̄_j) 를 만든다. teacher 쪽은 전부 no_grad.

        x_j·v̄ 는 pass 1/2 에서 **같은 값**이어야 하므로 pass 1 에서 만들어 캐시한다.
        (dea 는 같은 태스크를 두 번 계산하는데, 그때 x_j 나 ε 가 다르면 pass 1 에서
         만든 w_j 가 pass 2 의 e_j 와 짝이 안 맞는다.)

        ── shape 흐름 ──────────────────────────────────────────────────────
            n        = B (배치 크기).  st_shape = (B, n_obs_steps, 8) = (B,2,8)
            s_t      : (B,16)   코드북에서 뽑은 가짜 state  -> view 로 (B,2,8)
            o_t      : (B,3072) 가짜 DINOv2 특징            -> reshape (B,3072)
            past     : 길이 B 의 문자열 리스트 (과거 태스크 j 의 명령어)
            cond_T   : (B, 2576)  언어 + 관측 조건 벡터
            a_hat    : (B,H,A)  teacher 가 t=0 에서 한 스텝으로 추정한 '정답 액션'
            x_j      : (B,H,A)  그 a_hat 을 끝점으로 하는 경로 위의 점
            v_t      : (B,H,A)  teacher 속도 v̄  ← 이게 앵커의 타깃

        ── 왜 a_hat 을 거쳐 x_j 를 만드는가 ───────────────────────────────
        과거 태스크 j 의 **정답 액션을 모른다**(데이터가 없으니까). 그래서 teacher
        에게 물어 a_hat 을 받고, 그 a_hat 을 정답인 셈 치고 경로를 깐다. 이것이
        teacher bootstrapping 이다. x_j 가 실제 데이터 분포 위가 아니어도, student
        와 teacher 를 **같은 x_j** 에서 비교하므로 비교 자체는 공정하다.
        """
        cb = self.books[j]
        # 코드북에서 가짜 (state, feature) 를 B 개 뽑는다. 그래프 불필요 -> detach.
        s_t, o_t = self._sample(cb, n)                # (B,16), (B,3072)
        s_t, o_t = s_t.detach(), o_t.detach()
        # batch 를 얕은 복사해 state/feature 만 가짜로 갈아끼운다. 이미지 등 나머지
        # 키는 cond_tail 이 쓰지 않으므로 그대로 둬도 된다.
        bb = dict(batch)
        bb["observation.state"] = s_t.view(st_shape).to(batch["observation.state"].dtype)
        cls_j = o_t.reshape(-1, self.cls.shape[-1]).to(x_t.dtype)   # (B,3072)
        past = [instructions[f"task{j}"]] * n         # 과거 태스크 j 의 명령어 x B

        with torch.no_grad():                        # teacher 는 전부 그래프 없음
            tl_T = B1.cond_tail(self.teacher, bb, cls_j)
            cond_T = B1.make_cond(B1.encode_lang(self.teacher, past), tl_T)   # (B,2576)
            # ① teacher 에게 "이 조건에서 뭘 하겠나" 를 한 스텝으로 묻는다.
            #    t=0 (순수 노이즈) 에서 v 를 받아 a ≈ ε + v 로 끝점을 추정.
            eps0 = torch.randn_like(x_t)                                       # (B,H,A)
            a_hat = (eps0 + self._vel(self.teacher, eps0, torch.zeros_like(t), cond_T)).detach()
            # ② 그 a_hat 을 정답 삼아 경로 위의 점 x_j 를 깐다. 현재 배치와 **같은 t**
            #    를 쓰므로(tcol), student 는 익숙한 t 분포에서 평가받는다.
            eps1 = torch.randn_like(x_t)                                       # (B,H,A)
            x_j = ((1 - tcol) * eps1 + tcol * a_hat).detach()                  # (B,H,A)
            # ③ 그 지점에서의 teacher 속도. 앵커가 보존하려는 타깃 v̄.
            v_t = self._vel(self.teacher, x_j, t, cond_T).detach()             # (B,H,A)
        # a_hat 은 진단(그리퍼 전환 위치)에서 쓴다. 손실에는 관여하지 않는다.
        return {"bb": bb, "cls_j": cls_j, "past": past, "x_j": x_j,
                "v_t": v_t, "a_hat": a_hat}

    @staticmethod
    def _vel(pol, xx, tt, cond):
        """(B,H,A) x (B,) x (B,2576) -> (B,H,A).  velocity_net 호출 한 줄 래퍼."""
        return pol.dit_flow.velocity_net(noisy_actions=xx, time=tt, global_cond=cond)

    def _pair(self, v_s, q, t):
        """(student, teacher) 벡터 쌍을 --dir_space 가 정한 공간에서 돌려준다.

        shape 은 어느 쪽이든 (B,H,A) 두 개다. act 면 to_action 으로 끝점 액션으로
        옮긴다. ★ 두 벡터를 **반드시 같은 공간**으로 옮긴다 — 한쪽만 옮기면
        m 과 e 가 다른 공간에서 나와 분해 항등식이 깨진다.
        """
        if getattr(self.a, "dir_space", "vel") == "act":
            return to_action(v_s, q["x_j"], t), to_action(q["v_t"], q["x_j"], t)
        return v_s, q["v_t"]

    def _student_v(self, policy, q, t):
        """학생 속도. cond 는 학생 인코더로 다시 만든다(그래프 필요)."""
        tl_S = B1.cond_tail(policy, q["bb"], q["cls_j"])
        cond_S = B1.make_cond(B1.encode_lang(policy, q["past"]), tl_S)
        return self._vel(policy, q["x_j"], t, cond_S)

    # ── 손실 ────────────────────────────────────────────────────────────────
    def loss(self, policy, batch, tail, x_t, t, k, instructions, rng, args, device):
        cls = getattr(self, "cls", None)
        if cls is None:
            cls = B1.rgb_cls(policy, batch)
        self.cls = cls
        self._collect(batch, cls, device)

        if k == 0 or self.teacher is None or not self.books:
            return torch.zeros((), device=device)

        t0w = time.perf_counter()
        n = batch["observation.state"].shape[0]  # B (배치 크기). 스칼라 int
        # ★ (B,16) 이 아니라 (B,S,Ds) = (B,2,8) 이다. cond_tail 이 .shape[:2] 로
        #   B, S 를 읽고 나서 flatten(1) 로 (B,16) 을 만든다 (L2_V4_seq.py:395-398).
        #   _query 의 s_t.view(st_shape) 가 (B,16) 샘플을 이 모양으로 되돌린다.
        st_shape = batch["observation.state"].shape  # (B,2,8)
        tcol = t[:, None, None] # flow matching에서 sampling되는 uniform t임. broascasting을 위해 (B,) -> (B, 1, 1)로 shape변경
        js = sorted(self.books)   # 과거 태스크 번호 오름차순. 예 [0,1,...,k-1]
        # ★ sorted 가 필수다 — pass 1 의 W[i] 와 pass 2 의 js[i] 가 같은 태스크를
        #   가리켜야 한다. dict 삽입 순서(resume 경로에 따라 달라짐)를 믿으면
        #   태스크 j 의 가중이 다른 태스크에 걸려도 에러 없이 조용히 틀린다.
        N = len(js)                                      # 과거 태스크 수
        P = getattr(self.a, "pose_dims", None) or B1.POSE_DIMS   # 보통 6 (gripper 제외)
        form = self.a.anchor_form                        # dea / sqerr / unifmd
        beta = self.a.beta                               # 방향 항 세기
        # 노트 (3) 은 크기 항에만 1/N 이 붙는다 — 방향 항은 Σ_j w_j = 1 이라
        # 가중 자체가 평균 역할을 하므로 1/N 을 또 붙이지 않는다.
        # sum 은 기존 V4/V5 거동(과거 수에 비례)을 쓰고 싶을 때의 선택지다.
        inv_N = (1.0 / N) if self.a.anchor_reduce == "mean" else 1.0

        # ── pass 1: 질의 재료 + (dea 면) 모든 e_j ──────────────────────────
        # qs[i] 는 태스크 js[i] 의 (x_j, v̄, cond 재료). teacher forward 는 여기서
        # 한 번만 돌고 pass 2 에서 **재사용**하므로, 2-pass 라도 teacher 비용은 1 배다.
        qs = [self._query(policy, batch, x_t, t, k, instructions, j, n,
                          st_shape, tcol, device) for j in js]      # 길이 N 의 list
        W = None
        if form == "dea":
            # dea 만 2-pass. w_j 가 Σ_m e_m 으로 정규화되므로 **모든 태스크의 e 를
            # 먼저** 알아야 한다. 이 pass 는 가중치 계산용이라 그래프가 필요 없다.
            es, vs = [], []
            with torch.no_grad():
                for q in qs:
                    v_s = self._student_v(policy, q, t)        # (B,H,A)
                    p_s, p_t = self._pair(v_s, q, t)           # 각 (B,H,A)
                    # ★ 여기서 크기/방향으로 쪼갠다. e 만 쓰고 m 은 버린다
                    #   (크기 항의 계수 1/N 은 태스크에 의존하지 않으므로).
                    _, e, _, _, va = split_mag_dir(p_s, p_t, P)  # e, va 각 (B,H)
                    es.append(e); vs.append(va)
            # list[(B,H)] x N  -> stack -> (N,B,H).  태스크 축이 새로 생긴다.
            W = dea_weights(torch.stack(es), torch.stack(vs), self.a.eps_w)   # (N,B,H)

        # ── pass 2: 태스크마다 손실을 만들고 즉시 backward (chunk) ─────────
        # chunk backward: 태스크 하나의 그래프를 만들고 바로 backward 해서 버린다.
        # N 개를 다 쌓아 두면 활성값이 N 배라 메모리가 터진다. 그래디언트는
        # 파라미터에 누적되므로 수학적으로는 한꺼번에 backward 한 것과 같다.
        logs = {"m": [], "e": [], "w": []}
        for i, (j, q) in enumerate(zip(js, qs)):
            # 학생 속도는 그래프가 필요하므로 pass 2 에서 다시 계산한다
            # (dea 면 학생 forward 가 총 2 배가 되는 지점).
            v_s = self._student_v(policy, q, t)            # (B,H,A)
            p_s, p_t = self._pair(v_s, q, t)               # 각 (B,H,A)

            # ★★ 분해가 일어나는 지점 ★★
            #   (B,H,A) 벡터 2개  ->  (B,H) 스칼라 격자 2개 (m: 크기, e: 방향)
            #   A 축(액션 성분)이 노름/내적으로 접히고, (배치 x 청크) 격자만 남는다.
            m, e, r_s, r_t, valid = split_mag_dir(p_s, p_t, P)   # 각 (B,H)

            # ── 노트 2절 표를 **그대로** 옮긴다 ────────────────────────────
            #   팔        m_j 계수   e_j 계수
            #   sqerr     1/N       2·r_θ·r̄ / N
            #   unifmd    1/N       β / N
            #   dea       1/N       β·w_j        <- Σ_j w_j = 1 이라 /N 이 없다
            #
            # ★ 세 팔의 차이는 **방향 항의 계수 coef 하나**뿐이다. coef 는 모두
            #   (B,H) 로, m·e 격자와 원소별로 곱해진다.
            if form == "sqerr":
                # 분해 항등식 ‖v_θ−v̄‖² = m + 2·r_θ·r̄·e 의 계수를 그대로 쓴다.
                # -> 이 팔은 **기존 MSE 앵커와 수학적으로 항등**이다 (검증 완료).
                #    세 팔을 같은 코드 경로에서 비교할 수 있는 기준선이 된다.
                # .detach(): 계수는 학습 대상이 아니다. 빼면 "r 을 줄여 계수를
                #    낮추는" 경로로 그래디언트가 흘러 속도를 0 으로 눌러 버린다.
                coef = (2.0 * r_s * r_t).detach() * inv_N        # (B,H) 칸마다 다름
            elif form == "unifmd":
                # 계수를 상수 β 로 고정 -> 느린 구간(r 작은 칸)도 빠른 구간과
                # 같은 방향 감독을 받는다. sqerr 이 못 보던 구간이 살아난다.
                coef = torch.full_like(e, beta * inv_N)          # (B,H) 전부 같은 값
            elif form == "dea":
                # 방향 예산을 **오차가 큰 태스크로** 몰아준다. W[i] 가 태스크 i 의
                # 몫이고 Σ_i W[i] = 1 이라, 여기엔 inv_N 을 곱하지 않는다.
                coef = beta * W[i]                               # (B,H), detach 됨
            else:
                raise ValueError(f"알 수 없는 anchor_form: {form}")

            # ── (B,H) 격자 -> 스칼라로 접는다 ──────────────────────────────
            # 크기 항은 전 태스크·전 (t,h) 에 그대로. 노트: "Keep magnitude
            # retention over all N tasks."
            # .mean() 이 (B,H) 의 B x H 칸 전부를 평균해 **스칼라 ()** 로 만든다.
            L_m = inv_N * m.mean()                               # () 크기 항
            # 무효 칸(teacher 속도가 ~0 이라 각도가 정의되지 않는 칸)은 0 으로
            # 바꿔 넣는다. e[valid].mean() 이 아니라 where+mean 을 쓰는 이유는
            # 분모를 B x H 로 고정해 **유효 칸 수가 배치마다 달라도 손실 스케일이
            # 흔들리지 않게** 하려는 것이다.
            L_e = torch.where(valid, coef * e, torch.zeros_like(e)).mean()   # () 방향 항

            # L_anchor = L_m + L_e.  λ 두 개를 곱해 최종 기여를 만들고 즉시 backward.
            # 반환값이 아니라 여기서 그래디언트가 흐른다 (chunk backward).
            (B1.LAMBDA_ANCHOR * self.lam_lvl * (L_m + L_e)).backward()

            # 로그는 계수를 뺀 **원 오차**를 남긴다 — 팔끼리 비교할 수 있게.
            # (손실값은 coef 가 달라 팔 간 비교가 안 된다.)
            logs["m"].append(float(m.mean().detach()))
            # 여기선 e[valid] 로 **유효 칸만** 평균한다. 손실과 달리 로그는
            # "실제 각도 오차가 얼마인가" 를 보려는 것이라 0 을 섞으면 희석된다.
            logs["e"].append(float(e[valid].mean().detach()) if valid.any() else 0.0)
            if W is not None:
                logs["w"].append(float(W[i].mean()))   # 이 태스크가 받은 평균 몫

        self.step += 1
        self.t_anchor += time.perf_counter() - t0w
        if self.step % self.a.log_every_anchor == 0:
            rec = {"task": k, "step": self.step, "form": form, "N": N,
                   "m_mean": sum(logs["m"]) / N, "e_mean": sum(logs["e"]) / N,
                   "w_mean": (sum(logs["w"]) / N) if logs["w"] else None,
                   "w_max": max(logs["w"]) if logs["w"] else None,
                   "ms_per_step": 1000 * self.t_anchor / max(self.step, 1)}
            self.log.write(json.dumps(rec) + "\n"); self.log.flush()
            logging.info(f"[V6][{form}] k={k} step={self.step:5d} "
                         f"m={rec['m_mean']:.4f} e={rec['e_mean']:.4f} "
                         + (f"w={rec['w_mean']:.3f}/max {rec['w_max']:.3f} " if logs["w"] else "")
                         + f"ms/step={rec['ms_per_step']:.1f}")
        # 이미 backward 했다. 반환은 로그용(그래프 없음).
        return torch.tensor(sum(logs["m"]) / N + sum(logs["e"]) / N, device=device)


# ═════════════════════════════════════════════════════════════════════════════
#  메인 — B1 의 CLI 를 그대로 쓰고 V6 인자만 더한다
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    import argparse
    import sys

    # B1.parse_args 를 그대로 쓰되 V6 전용 인자를 덧붙인다.
    orig = B1.parse_args

    def parse_v6():
        # B1 의 파서를 만들고 인자를 추가한 뒤 파싱한다.
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--anchor_form", choices=["dea", "sqerr", "unifmd"], default="dea",
                       help="dea = 방향오차 인지 가중(제안), sqerr = 균일 제곱오차(=MSE), "
                            "unifmd = 균일 크기-방향 분리")
        p.add_argument("--beta", type=float, default=2.0,
                       help="방향 항의 세기. 노트: β ≈ 2r₀² 가 teacher 노름 r₀ 근처의 "
                            "국소 MSE 보정이고, β=2 는 단위 노름 가정에 해당한다.")
        p.add_argument("--eps_w", type=float, default=1e-3,
                       help="w_j 의 0/0 방지 상수. 크면 균일 가중에 가까워진다.")
        p.add_argument("--dir_space", choices=["vel", "act"], default="vel",
                       help="vel = 속도 벡터로 m·e 를 분해(노트 그대로). "
                            "act = 끝점 액션 a = x_t + (1−t)v 로 분해 — 공통 노이즈가 "
                            "빠져 각도가 t 와 무관해진다. m 과 e 는 항상 같은 공간이다.")
        p.add_argument("--anchor_reduce", choices=["mean", "sum"], default="mean",
                       help="mean = 과거 태스크에 1/N (노트 정의). "
                            "sum = 기존 V4/V5 거동(과거 수에 비례).")
        known, rest = p.parse_known_args()
        sys.argv = [sys.argv[0]] + rest
        args = orig()
        for kk, vv in vars(known).items():
            setattr(args, kk, vv)
        return args

    args = parse_v6()
    B1._forbid_time_bin()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor = DEAAnchor(args)

    json.dump({
        "arm": f"L2_V6 — {args.anchor_form}",
        "anchor_form": args.anchor_form, "beta": args.beta, "eps_w": args.eps_w,
        "dir_space": args.dir_space,
        "anchor_reduce": args.anchor_reduce,
        "sampler_mode": args.sampler_mode, "codebook_k": args.codebook_k,
        "lambda_level": args.lambda_level, "seed": args.seed, "eval_seed": args.eval_seed,
        "eval_episodes": args.eval_episodes, "steps_per_task": args.steps_per_task,
        "note": "Directional-Error-Aware Anchor Loss. m=(r_s−r_t)², e=1−cos. "
                "e 계수: sqerr 2r_s·r_t/N, unifmd β/N, dea β·w_j (detach).",
    }, (out_dir / "v6_config.json").open("w"), indent=2, ensure_ascii=False)

    logging.info(f"[V6] {anchor.describe()}")
    B1._train_loop(args, anchor)


if __name__ == "__main__":
    main()
