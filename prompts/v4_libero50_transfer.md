# 작업: v3 코드에 v4 를 얹고, LOTUS 셋팅 LIBERO-50 을 돌린다

이 저장소에는 **v3 까지의 코드만** 있다. v4 를 직접 구현해서 LOTUS 셋팅
LIBERO-50 lifelong 실험을 돌리고 성능을 정리하는 것이 목표다.

---

## 0. 먼저 확인할 것 (코드를 고치기 전에)

아래를 확인하고 결과를 보고한 다음 진행하라. 하나라도 없으면 거기서 멈추고 물어라.

1. **LOTUS 드라이버 파일** — LOTUS 셋팅(base 를 joint 로 배우고 그 뒤 stage 를 하나씩
   붙이는 구조)의 lifelong 실행 스크립트가 어느 파일인지. 원본 셋팅(task 0 부터
   순차)과 헷갈리지 마라. 아래 표식으로 찾을 수 있다.
   - `sample_codebook`, `anchor_loss`, `build_codebook` 함수가 있다
   - `--suite` 에 `50` 또는 `libero_50` 같은 선택지가 있다
   - `lotus_setting.json` 또는 그에 준하는 스플릿 정의를 읽는다
2. **데이터** — LOTUS_setting 의 LIBERO-50 스플릿(base 25 task + stage 5개 x 5 task).
   각 스플릿이 LeRobot 형식(`meta/info.json`, `data/chunk-*/episode_*.parquet`)인지.
   base 는 태스크당 50 demo, lifelong 은 태스크당 10 demo 여야 한다.
3. **base 체크포인트** — 넘겨받은 LOTUS LIBERO-50 pretrain 체크포인트 경로.
   안에 `config.json` + `model.safetensors` 가 있어야 한다.
4. **action 차원 순서** — `meta/info.json` 의 `features.action.names` 를 출력하라.
   `[x, y, z, roll, pitch, yaw, gripper]` 순서여야 아래 `POSE_DIMS = 6` 이 맞다.
   **순서가 다르면 코드를 고치기 전에 보고하라.** v4 의 속도 가중이 앞 6 차원을
   pose 로 가정한다.
5. **현재 v3 가 정말 v3 인지** — 코드북에 셀별 선형 기울기 `A_k` 가 있는지
   (`build_codebook` 이 `A`, `zbar` 를 만드는지). 없으면 그건 v1 이고, v4 를 얹을
   토대가 아니다. 보고하고 멈춰라.

---

## 1. v4 가 v3 에서 바꾸는 것 — 정확히 두 가지

**FM(flow matching) 손실은 절대 건드리지 않는다.** 기존 MSE 그대로다.
속도 정규화도 EMA 도 FM 쪽에는 넣지 않는다. 바꾸는 것은 **앵커 경로뿐**이다.

### (1) 샘플러: soft_kernel → hard_linear

v3 는 코드북에서 (s̃, õ) 를 뽑을 때 커널 가중을 쓴다:

    õ = Σ_j w_j (m_j + A_j δ_j) + sqrt(Σ_j w_j σ²_{o,j}) ⊙ ε′
    w_j = softmax(−‖z(s̃) − c_j‖² / h²)

v4 는 경성 배정 + 1차 항만 쓴다. 커널 가중 w, 대역폭 h, 관측 잡음 σ_o 를 **쓰지 않는다**:

    k* ~ Cat(π)
    s̃  = μ_{s,k*} + σ_{s,k*} ⊙ ε
    z̃  = (s̃ − mean_s) / std_s
    õ  = m_{k*} + A_{k*} (z̃ − z̄_{k*})

코드북 **생성**은 건드리지 않는다. 저장 형식이 그대로라 기존 코드북 파일과 호환된다.

### (2) 앵커 손실에 속도 역제곱 가중

v3 의 앵커는 평범한 MSE 다. v4 는 teacher 가 예측한 pose 속도의 역제곱으로 가중한다:

    ṽ_T   = v_T + ε₁                        (de-noised 프레임)
    ρ[b,s] = ‖ṽ_T[b, s, 0:6]‖               pose 성분의 스텝 속도
    w      = 1 / (ρ² + c),   c = kappa · speed_med2_j
    w      = min(w, median(w) · w_clip)      위로만 자른다
    w̃      = w / mean(w)                     배치 평균 1

    L_j = (1/(H·A)) · mean_b Σ_s [ w̃[b,s] · Σ_{d<6} r²[b,s,d] + r²[b,s,6] ]

여기서 `r = v_S − v_T` 이고, **pose 6차원에만 w̃ 를 곱하고 gripper(마지막 차원)는
그대로 둔다**. `speed_med2_j = median_{데모·스텝} ‖a_s[0:6]‖²` 로, 태스크 j 의
액션에서 코드북 만들 때 같이 계산해 코드북에 넣어 둔다.

`mean(w̃) = 1` 이라 평균 스케일이 기존 MSE 와 같다. 그래서 `lambda_level` 을
다시 튜닝하지 않아도 된다.

**목적**: 로봇이 느리게 움직이는 구간(ρ 작음)에 큰 가중이 가서, "정지에 가까운
구간의 미세한 속도 차이"가 앵커 손실에 제대로 반영되게 한다. 가중이 없으면
빠른 구간의 큰 잔차가 손실을 지배해 정밀 구간이 묻힌다.

---

## 2. 추가할 코드 (그대로 붙여넣어라)

드라이버 파일에 아래를 그대로 추가한다. 이름을 바꾸지 마라.

```python
POSE_DIMS = 6      # action = [x,y,z,roll,pitch,yaw,gripper] -> 0..5 pose, 6 gripper
                   # meta/info.json 의 features.action.names 로 반드시 확인할 것


def sample_codebook_hard(cb: dict, n: int, gen=None):
    """v4 (1) 경성 배정 + 1차 항만. 커널 가중 w·대역폭 h·잡음 항 σ_o 를 쓰지 않는다."""
    pi, mu_s, sig_s = cb["pi"], cb["mu_s"], cb["sig_s"]
    k = torch.multinomial(pi, n, replacement=True, generator=gen)      # (n,)
    eps = torch.randn(n, mu_s.shape[1], device=mu_s.device, generator=gen)
    s = mu_s[k] + sig_s[k] * eps                                      # (n,16)
    z = (s - cb["mean_s"]) / cb["std_s"]                              # (n,16)
    d = (z - cb["zbar"][k]).unsqueeze(2)                              # (n,16,1)
    A = cb["A"][k].float()                                            # (n,3072,16)
    o = cb["m"][k] + torch.bmm(A, d).squeeze(2)                       # (n,3072)
    return s, o


def speed_weights(v_ref: torch.Tensor, eps_noise: torch.Tensor, c: float,
                  w_clip: float) -> torch.Tensor:
    """v4 (2) 기준 속도의 역제곱 가중. (B,H,A) 를 받아 (B,H) 를 돌려준다.

    θ 와 무관한 상수라 detach 한다.
    """
    with torch.no_grad():
        rho = (v_ref + eps_noise)[..., :POSE_DIMS].norm(dim=-1)        # (B,H)
        w = 1.0 / (rho.pow(2) + c)
        w = torch.minimum(w, w.median() * w_clip)
        return (w / w.mean().clamp_min(1e-12)).detach()                # (B,H)


def weighted_sq(resid: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """가중 제곱오차. pose 6차원에만 w 를 곱하고 gripper 는 그대로 둔다."""
    H, A = resid.shape[1], resid.shape[2]
    pose = (resid[..., :POSE_DIMS].pow(2).sum(-1) * w).sum(1)          # (B,)
    grip = resid[..., POSE_DIMS:].pow(2).sum(-1).sum(1)                # (B,)
    return ((pose + grip) / (H * A)).mean()


def action_speed_med2(actions: torch.Tensor) -> float:
    """c_j 의 재료 — median_{데모·스텝} ‖a_s[0:6]‖². actions: (N,A) 또는 (N,H,A)."""
    a = actions.reshape(-1, actions.shape[-1])
    return float(a[:, :POSE_DIMS].pow(2).sum(-1).median())
```

---

## 3. 훅 지점 — 세 군데

### (a) 코드북 만들 때 `speed_med2` 를 같이 넣는다

코드북을 만드는 루프에서, 그 태스크의 액션을 모아 `action_speed_med2` 를 계산해
코드북 dict 에 넣는다. 액션을 아직 안 모으고 있으면 (s, o) 를 모으는 루프에
`batch["action"]` 수집을 같이 붙여라 — 데이터를 한 번 더 훑지 않도록.

```python
cb["speed_med2"] = action_speed_med2(acts[local_idx])
```

로그에 `speed_med2=...` 를 찍어 태스크마다 값이 나오는지 확인하라. 액션이
MIN_MAX 로 [-1,1] 정규화돼 있으면 태스크 간 스케일 차이가 크지 않아야 한다.

### (b) 앵커 손실 안 — 샘플러 분기

```python
sampler = getattr(args, "sampler_mode", "soft_kernel") if args else "soft_kernel"
...
s_t, o_t = (sample_codebook_hard(cb, n) if sampler == "hard_linear"
            else sample_codebook(cb, n))
```

### (c) 앵커 손실 안 — 잔차를 스칼라로 줄이는 자리

기존 v3 는 이럴 것이다:

```python
resid = vs0 - vt0.to(vs0.dtype)              # (B,H,A)
L_j = resid.flatten(1).pow(2).mean(1).mean()
```

이걸 아래로 바꾼다. `eps1` 은 앵커의 x_j 를 만들 때 쓴 노이즈,
`vt0` 는 그 지점에서 잰 teacher 속도다 — **둘 다 이미 있는 변수여야 한다.**
없으면 잘못 찾은 것이니 코드를 다시 읽어라.

```python
aw = getattr(args, "anchor_weight", "mse") if args else "mse"
resid = vs0 - vt0.to(vs0.dtype)              # (B,H,A)
if aw == "speedinv":
    # ṽ_T = v_T + ε₁ 프레임에서 잰 teacher 속도로 가중.
    # 잔차는 프레임을 옮겨도 같다: ṽ_S − ṽ_T = v_S − v_T.
    w = speed_weights(vt0, eps1, cb["speed_med2"] * args.kappa, args.w_clip)
    L_j = weighted_sq(resid, w)
else:
    L_j = resid.flatten(1).pow(2).mean(1).mean()
```

---

## 4. CLI 플래그 추가

**기본값은 반드시 v3 동작이어야 한다.** 플래그를 안 주면 기존과 완전히 같은
경로를 타야 한다.

```python
ap.add_argument("--sampler_mode", choices=["soft_kernel", "hard_linear"],
                default="soft_kernel")
ap.add_argument("--anchor_weight", choices=["mse", "speedinv"], default="mse")
ap.add_argument("--kappa", type=float, default=1.0, help="c = kappa x speed_med2")
ap.add_argument("--w_clip", type=float, default=20.0)
```

---

## 5. 반드시 확인할 것 — 코드북 K 규칙 (여기서 조용히 망가진다)

LOTUS 셋팅은 base 가 태스크당 **50 demo**, lifelong 이 태스크당 **10 demo** 로
5배 차이가 난다. 둘에 같은 K(예: 96)를 쓰면 lifelong 쪽 셀당 표본이 10개 안팎으로
떨어지고 `grad_min_frames`(기본 24)에 전부 걸려 **A_k 가 모든 셀에서 0 으로 강등**된다.
그러면 이름만 v3/v4 이고 실제로는 0차 앵커(v1)가 된다. 실측으로 997 프레임 /
K=96 에서 88셀 전부 강등된 적이 있다.

K 를 demo 수에 비례시켜라:

```python
K = max(args.k_min, args.k_per_demo * ndemo[local_idx])   # k_per_demo=2, k_min=8
# 50 demo -> K=100,  10 demo -> K=20
```

코드북 로그에서 **`강등 N셀` 이 K 의 절반을 넘으면 즉시 멈추고 보고하라.**
정상 범위는 K=20 에서 0~5셀이다.

---

## 6. 스모크 — 본 run 전에 반드시

짧게 한 번 돌려서 아래를 전부 확인한다.

```
--steps 60 --batch_size 8 --num_workers 4 --eval_episodes 2 --eval_batch_size 2 \
--n_stages 1 --sampler_mode hard_linear --anchor_weight speedinv
```

확인 항목:
1. `nan` / `inf` / `Traceback` 이 로그에 없다.
2. **v3 등가성** — 플래그를 빼고(`--sampler_mode soft_kernel --anchor_weight mse`)
   같은 시드로 돌렸을 때 손실 값이 v4 코드 이전과 같다. 기본 경로를 깨지 않았다는
   확인이다. 이게 안 맞으면 훅을 잘못 넣은 것이다.
3. **속도 가중 sanity** — `ṽ_T = v_T + ε₁` 이 teacher 의 예측 액션 `â` 와 거의
   같아야 한다. 아래 비율이 **0.2 를 넘으면 잡음 규약이 어긋난 것**이니 멈추고 보고하라.
   ```python
   r = float((vt0 + eps1 - a_hat).norm(dim=-1).mean()
             / a_hat.norm(dim=-1).mean().clamp_min(1e-9))
   ```
4. `w.mean()` 이 1.0 근처, `w.max()` 가 `median(w)·w_clip` 이하.
5. 코드북 로그에 `speed_med2` 가 태스크마다 찍히고, 강등 셀이 K 의 절반 미만.

---

## 7. 본 run

```
python -u <드라이버>.py --suite 50 --steps 10000 \
    --seed <train_seed> --eval_seed 7 --eval_episodes 20 --eval_batch_size 20 \
    --num_workers 6 --sampler_mode hard_linear --anchor_weight speedinv \
    --k_per_demo 2 --k_min 8 --lambda_level 3.0 \
    --out results/l2_codebook_v4_50_seed<train_seed>
```

- **eval_seed 는 7 로 고정**한다. train seed 만 바꿔 변동을 잰다.
- 롤아웃은 칸당 20 회.
- base 체크포인트(넘겨받은 LOTUS LIBERO-50 pretrain)에서 출발한다.
- 구조: base 25 task 를 stage 0 에서 joint, 이후 stage 5개가 5 task 씩 →
  총 50 task, 평가 **225 칸**. goal/object(40칸)보다 5.6배 오래 걸린다.
- GPU 1장 기준 수십 시간이다. 백그라운드로 띄우고 PID 와 로그 경로를 남겨라.
- 120초 뒤 프로세스 생존을 확인하고, 죽었으면 로그 끝 40줄과 함께 보고하라.

---

## 8. 결과 정리

`report.txt`(SR 행렬)에서 아래를 계산해 정리하라. 정의는 LIBERO 벤치마크
(Liu et al. 2023)를 따른다. LOTUS 셋팅은 base 태스크의 "배운 스테이지"가 전부 0 이다.

    FWT = mean_t R[배운 스테이지][t]                  높을수록 좋음
    NBT = mean_t mean_{k>배운} (FWT_t − R[k][t])      **낮을수록** 좋음, + 면 망각
    AUC = mean_t mean_{k>=배운} R[k][t]               높을수록 좋음
    AvgSR = 마지막 **완성된** 행의 평균

행마다 **train seed 와 eval seed 를 같이 적어라.** 칸당 20 rollout 이면 개별 칸의
이항 표준오차가 최대 ±11%p 이므로, 시드 하나로 잰 수 %p 차이는 확정할 수 없다.

---

## 지켜야 할 것

- **원본 v3 코드의 기본 동작을 바꾸지 마라.** v4 는 전부 플래그 뒤에 있어야 하고,
  플래그 없이 돌리면 예전과 같은 결과가 나와야 한다.
- FM 손실에는 손대지 마라. 속도 정규화는 **앵커에만** 들어간다.
- 확신이 안 서는 지점(특히 4번 action 차원 순서, 5번 강등 셀 수)에서는
  추측으로 진행하지 말고 멈추고 물어라.
