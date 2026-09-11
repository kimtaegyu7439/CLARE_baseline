# l2_codebook 앵커 진단 — D1 / D3 / D2

MSE 앵커가 '느린 구간의 방향'을 과소보호하는지 본다. 학습은 하지 않았고 기존 체크포인트와 코드북으로 forward pass 와 rollout 만 수행했다.

판정은 성립 / 불성립 / 판단불가 셋 중 하나이며, 예측을 데이터에 맞게 사후 수정하지 않았다.

## 1. 탐색 결과

### 진단 대상 — 지정된 런이 아니다 (중요)

요청은 `results/l2_codebook_k96_grad` (libero_spatial, seed 42) 였으나 그 런의 **stage 체크포인트가 존재하지 않는다.** 체크포인트 경로 규칙이
`<ckpt_root>/<suite>_seed<seed>_<mode>/task_<k>` 로 **팔 이름을 담지 않아서**, 같은 suite·seed·mode 를 쓰는 다른 팔이 같은 자리에 덮어쓴다. 실제로 `outputs/L2_codebook/libero_spatial_seed42_ours/` 의 파일은 9/5 01:43~12:55 자로 `l2cb_fullcov_grad` 런(9/5 01:23~13:54)의 것이고, k96_grad 런은 9/2 03:44~15:13 이었다. 9/2 파일 중 살아남은 것은 stage 0 의 100-step 중간 체크포인트 하나뿐이다.

사용자 승인을 받아 **`l2_codebook_k96_grad_seed41`** (libero_spatial, seed 41) 로 대체했다. 같은 팔·같은 스위트이고 시드만 다르다. 이 런은 **stage 0..8 까지만** 존재한다 (stage 9 는 학습이 끝나지 않았다). 그 결과:

- D1 의 쌍은 `i = 1..8, j < i` (28 쌍). `i = 9` 는 만들 수 없다.
- D2 의 '최종' 체크포인트는 stage 9 가 아니라 **stage 8** 이고, task 는 0..8 의 9개다.
- D3 의 `j ∈ {3,5,7}` 은 전부 사용 가능하다.

### 경로와 시그니처

- 체크포인트: `/home/sa090180/clare/outputs/L2_codebook/libero_spatial_seed41_ours/task_<k>/checkpoints/005000/pretrained_model/` (k = 0..8, 각 739MB). teacher 스냅샷은 디스크에 따로 없지만, 학습 루프가 체크포인트를 저장한 **뒤** `snapshot()` 을 뜨므로 **stage i 의 teacher = stage i−1 체크포인트**로 정확히 대체된다.
- 코드북: `/home/sa090180/clare/results/l2_codebook_k96_grad_seed41/codebooks/codebook_task<j>.pt`, `torch.load` -> dict (`pi, mu_s, sig_s, c, m, sig_o, mean_s, std_s, h, A, zbar, ...`).
- 앵커 샘플링: `l2_codebook.sample_codebook(cb, n, gen) -> (s̃ (n,16), õ (n,3072))`.
- 앵커 손실: `L2CodebookAnchor.loss(policy, batch, tail, x_t, t, k, instructions, rng, args, device)` (l2_codebook.py:369).
- 정규화 버퍼가 safetensors 안에 있어(`normalize_inputs.buffer_*`), `make_policy` 를 어떤 태스크 메타로 만들어도 체크포인트 값이 덮어쓴다.

### 행동 dims — 가정대로였다

데이터셋 메타의 `action.names.motors = ['x','y','z','roll','pitch','yaw','gripper']`. 따라서 **0..5 = pose delta (0..2 translation, 3..5 orientation), 6 = gripper** 가 맞다. `observation.state` 는 8차원 (pose 6 + gripper qpos 2), 2 프레임이라 (B,2,8).

### 롤아웃·물체 pose

- `eval_policy(env, policy, n_episodes, start_seed)` (scripts/eval.py:241). 원본 프로토콜은 `n_episodes=20`, `n_envs=20`, `start_seed=args.seed=41`.
- 물체 pose 는 `env.envs[i].unwrapped._env.sim.data.body_xpos[...]` 에서 읽는다. 목표 물체는 bddl 에서 온 `obj_of_interest = [목표 그릇, 접시]` 로 직접 얻으므로 '데모의 파지 시점 위치로 대체' 하는 우회는 **필요하지 않았다**.


## 2. D1 — 앵커 주소에서의 잔차 각도·크기

**sanity** `mean‖ṽ_Tp − â‖/‖â‖` = 0.465 (최대 0.523), 임계 0.2. -> **초과**

임계를 넘었으므로 사양대로 `x_t` 정의를 코드와 다시 대조했다. `â = ε₀ + v_T(ε₀,0,cond_T)`, `x_t = (1−t)ε₁ + t·â` 로 l2_codebook.py:410-415 와 **문자 그대로 같다**. 규약 오류가 아니다.

원인은 ṽ 프레임 자체에 있다. t 를 고정해 재면:

| t | ‖ṽ−â‖/‖â‖ | ∠(ṽ,â) |
|---|---|---|
| 0.0 | 0.087 | 4.0° |
| 0.3 | 0.132 | 5.7° |
| 0.5 | 0.224 | 9.4° |
| 0.7 | 0.393 | 17.0° |
| 0.9 | 1.021 | 39.8° |
| 1.0 | 3.028 | 75.5° |

flow matching 의 속도장은 주변 기대값 `E[a−ε | x_t]` 이다. t 가 크면 `x_t ≈ a` 라 모델이 ε₁ 을 복원할 근거가 없어 `v ≈ a` 를 내고, 거기에 ε₁ 을 더하는 `ṽ = v + ε₁` 은 순수 잡음을 주입한다. 작은 t 에서는 `x_t ≈ ε₁` 이라 규약이 성립한다. 즉 **ṽ 프레임은 t 가 작을 때만 유효하다.**

이는 십분위 층화 자체를 위협한다 — 큰 t 에서 `ρ = ‖ṽ_T‖` 는 teacher 속도가 아니라 `‖ε₁‖` 을 재기 때문이다. 그래서 사양의 전 구간 분석(`all`)과 함께 **`t ≤ 0.3` 으로 제한한 판본**을 같이 계산했다. 아래 판정은 둘 다 싣는다.


### 전 t 구간 (사양 그대로)

- **P1 성립** — 느린 십분위(1–3) median θ = 11.7°, 빠른 십분위(8–10) = 4.9°, 격차 +6.8° 이고 36쌍 중 100% 가 양수. 격차와 i−j 의 상관 +0.88.
- **P2 불성립** — |m| 격차 +0.057 (느림 0.088 vs 빠름 0.031), 양수 비율 100%. |m| 은 분모가 ‖ṽ_T‖ 라 느린 십분위에서 구조적으로 부풀 수 있다 — 이 교란을 통제하지 않으면 P2 는 판정하기 어렵다.

### t ≤ 0.3 (ṽ 유효 구간)

- **P1 성립** — 느린 십분위(1–3) median θ = 10.5°, 빠른 십분위(8–10) = 3.2°, 격차 +7.2° 이고 36쌍 중 100% 가 양수. 격차와 i−j 의 상관 +0.91.
- **P2 불성립** — |m| 격차 +0.060 (느림 0.081 vs 빠름 0.021), 양수 비율 100%. |m| 은 분모가 ‖ṽ_T‖ 라 느린 십분위에서 구조적으로 부풀 수 있다 — 이 교란을 통제하지 않으면 P2 는 판정하기 어렵다.

- **P3 불성립** — raw MSE 중 t>0.8 기여 32%, (1−t)² 가중 후 2%. 가중 후 급감은 예측대로지만, raw 쪽이 과반(<50%)이라는 부분은 성립하지 않는다.

표: `d1_cells.csv` (j,i,teacher,t_sel,decile 별), `d1_taxis.csv`, `d1_gripper.csv`, `d1_sanity.csv`, `d1_sanity_by_t.csv`.  
그림: `d1_theta_by_decile.png`, `d1_absm_by_decile.png`, `d1_theta_gap_heatmap.png`, `d1_t_contribution.png`, `d1_theta_by_decile_tle03.png`.


## 3. D3 — 방향 교란 vs 크기 교란의 SR 민감도

교란은 `predict_action_chunk` 가 **실제 로봇 단위로 unnormalize 한 뒤** 돌려주는 청크에 걸었다 (modeling_dit_flow_mt.py:875-903). 평가 프로토콜은 건드리지 않았다.

`rotate` 는 **translation dims 0..2 만** 회전시킨다. orientation dims 3..5 는 축각 표현이라 회전의 의미가 모호하고 물리적으로 해석 가능한 교란이 아니어서 제외했다 (사양의 지시대로 여기 명시). gripper dim 6 은 어떤 조건에서도 건드리지 않는다. `φ = 2·arcsin(δ/2)` 라 `‖Rv − v‖ = δ‖v‖` 로 scale 의 |δ| 와 벡터 변화량이 같다.

```
kind  baseline rotate             scale                                
delta      0.0    0.1   0.2   0.3  -0.3  -0.2   -0.1    0.1   0.2   0.3
task                                                                   
3        100.0   80.0  80.0  65.0  85.0  95.0   75.0  100.0  95.0  90.0
5         80.0   90.0  45.0  45.0  95.0  80.0  100.0   80.0  60.0  70.0
7         65.0   45.0  45.0  30.0  60.0  60.0   65.0   60.0  70.0  75.0
```

- **P6 성립** — 같은 δ 비교 9건 중 rotate 가 더 나쁜 경우 100%. 평균 하락 rotate 23.3%p vs scale(더 나쁜 쪽) -3.9%p.

표 `d3_sr.csv`, `d3_p6.csv` · 그림 `d3_sr_vs_delta.png`


## 4. D2 — 옛 task 실패의 단계 귀속

단계 분할 임계값 (전부 상수): `GRIP_CLOSE > 0` (robosuite OSC_POSE 에서 gripper 액션 양수 = 닫기), `LIFT_DZ = 2cm`, `NEAR_PLATE = 5cm`, `REACH_TOL = 4cm`.

### 최종(stage 8) − 대각(stage j) 실패 수 차이
```
 task  d_reach  d_grasp  d_transport  d_place
    0        0        0            0        0
    1        0        0            2        1
    2        0        0            0       -1
    3        0        0            0        0
    4        0       -3            0        1
    5        0        0            0        1
    6        0        0            0        0
    7       -3        0           -1        1
    8       -1        0            1        0
```

- **P4 성립** — 새로 생긴 실패 7건의 단계 분포 {'reach': 0, 'grasp': 0, 'transport': 3, 'place': 4}, grasp+place 비중 57%.
- **P5 성립** — reach 실패 14건 중 비목표 그릇 4cm 이내 0%.
  - 각 task 는 서로 다른 bddl 씬이라 '다른 task 의 목표 그릇'이 그 씬에 없다. 같은 씬의 **비목표** 그릇(akita_black_bowl_2 등)에 4cm 이내인 비율로 대체했다.

표 `d2_episodes.csv`, `d2_stage_counts.csv`, `d2_delta.csv` · 그림 `d2_stage_stack.png`


## 5. 이상 징후

- **ṽ 프레임의 유효 범위**: sanity 비율이 t 와 함께 커져 t≳0.5 에서 규약이 깨진다(2절). 사양의 `all` 판정은 이 구간을 포함하므로, `t ≤ 0.3` 판본을 함께 봐야 한다. 두 판본의 결론이 갈리면 `t ≤ 0.3` 쪽을 신뢰해야 한다.
- **|m| 의 분모 편향**: `m = (‖ṽ_S‖−‖ṽ_T‖)/‖ṽ_T‖` 는 느린 십분위에서 분모가 작아 구조적으로 부풀어 오른다. P2 를 '격차 없음'으로 검정하려면 분모를 십분위 중앙값으로 고정하는 등 통제가 필요하다 — 현재 수치만으로는 P2 를 기각할 수 없다.
- **D3 표본**: 조건당 20 에피소드라 이항 표준오차가 최대 ±11%p 다. 개별 칸의 차이는 이 폭 안에서 해석해야 한다.
- **D2 표본**: 실패 에피소드가 총 42건이라 단계별로 쪼개면 칸당 표본이 한 자릿수가 될 수 있다. 임계값(4cm/2cm/5cm)에 대한 민감도는 확인하지 않았다.
- **stage 9 부재**: 대체 런에 stage 9 가 없어 D1 의 `i=9` 쌍, D2 의 task 9 및 stage 9 최종 체크포인트는 만들 수 없었다. 사양대로 재학습하지 않았다.
- **덮어쓰기 재발 방지**: `ckpt_root` 경로에 팔 이름이 없다. 이후 런은 `--ckpt_root` 를 팔마다 다르게 주지 않으면 같은 사고가 반복된다.

