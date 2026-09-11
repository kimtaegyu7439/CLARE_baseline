# B1 — replication notes

B1이 무엇을 어디서 그대로 가져왔는지 정리한다. 모든 참조는 `REPO_ROOT=/home/sa090180/clare`
기준 상대 경로 `file:line` 이다.

## 0. 결론 먼저 — 자동 발견한 것들

| 항목 | 값 | 출처 |
|---|---|---|
| REPO_ROOT | `/home/sa090180/clare` | — |
| 이전 seq-FT 실행 | E0 의 `λ=0` 팔 | `bash/E0/E0.sh:39`, `lerobot_lsy/src/lerobot/scripts/E0.py:69-71` |
| 스위트 / 태스크 순서 | `libero_spatial`, task 0→1→2→3 (benchmark id 오름차순) | `bash/E0/E0.sh:37,62-63` |
| 태스크당 스텝 | **5000 gradient step** (총합 아님) | `bash/E0/E0.sh:43`, `E0.py:450` |
| 배치 | 32 | `bash/E0/E0.sh:44` |
| 45 에피소드 | 50개 중 뒤 5개 hold-out → 학습은 ep 0..44 | `bash/E0/E0.sh:50`, `E0.py:187-191` |
| **롤아웃 / SR 칸** | **20 에피소드**, env 20개 동시, `start_seed=42` (seed 42..61) | `bash/E0/E0.sh:52-53`, `E0.py:249-273` |
| seed | 42 (학습·평가 공통) | `bash/E0/E0.sh:42` |

**이전 실험의 롤아웃 횟수 = SR 한 칸당 20회.** B1도 같은 20회를 쓴다
(`--eval_episodes` 기본값 20). 참고로 이 저장소의 다른 표들은 값이 다르다:
단일 스위트 CLARE/ER 표는 칸당 100회, libero_40 표는 칸당 20회다. B1이 비교할 상대는
E0 λ=0 이므로 **20회가 맞다.**

## 1. 초기화 — 같은 사전학습 체크포인트

- `PRETRAIN_PATH = ${CLARE_MODEL_ROOT}/dit_flow_mt_libero_90_pretrain`
  — `bash/clare/env.sh:29-31`
- task 0 은 `PRETRAIN_PATH`, task k>0 은 **직전 스테이지 체크포인트**
  `task_{k-1}/checkpoints/last/pretrained_model` 에서 출발 — `bash/E0/E0.sh:88-90`
- 정책 클래스: `DiTFlowMTPolicy` (`policy.type = ditflow_mt`)
  — `lerobot_lsy/src/lerobot/policies/dit_flow_mt/modeling_dit_flow_mt.py:794`
- 아키텍처 하이퍼파라미터는 CLI가 아니라 체크포인트 `config.json` 을 따른다
  — `lerobot_lsy/src/lerobot/configs/train.py:100-107`

## 2. Flow matching 파라미터화 — 재구현하지 않고 그대로 복제

`modeling_dit_flow_mt.py:1281-1297` 의 구성을 B1.py 가 문자 그대로 옮긴다:

```
noise      = velocity_net.sample_noise(B, device)
t          = noise_distribution.sample((B,))          # (B,)
x_t        = (1-t)[:,None,None]*noise + t[:,None,None]*action
pred       = velocity_net(noisy_actions=x_t, time=t, global_cond=cond)
target     = action - noise
L_FM       = F.mse_loss(pred, target, reduction="none").mean()
```

- `do_mask_loss_for_padding` 은 배포 체크포인트에서 False 라 마스킹 분기는 실행되지
  않는다 — `modeling_dit_flow_mt.py:1303-1305`. B1 도 마스킹하지 않는다.
- **재구현이 아님을 코드로 증명한다.** B1.py 는 시작 시 `--verify_fm` 검사를 돌려,
  RNG 상태를 고정한 뒤 `policy.forward(batch)` 가 준 손실과 B1 이 복제한 손실이
  `torch.allclose` 로 일치하는지 assert 한다. 어긋나면 즉시 죽는다.

앵커 항이 `x_t, t` 를 직접 잡아야 해서 복제가 불가피했다. `compute_loss` 안에서
샘플링이 일어나 밖에서 꺼낼 수 없기 때문이다.

## 3. 조건 벡터와 NULL 조건

명령어는 `batch["task"]`(list[str]) 로만 들어간다:
CLIP text → `language_embedding_projection` → 조건 벡터의 **앞 512차원**.
concat 순서는 `[언어 512, 상태 16, 이미지 2048] = 2576` 로 고정
— `modeling_dit_flow_mt.py:1125-1215`.

- **NULL 조건 = 빈 문자열 `""`** 을 같은 tokenizer/CLIP/projection 경로로 통과시킨다.
  저장소에 CFG용 null 임베딩이 따로 없어서 프롬프트의 기본 규칙을 따랐다.
  학습·진단·평가 훅이 모두 같은 표현을 쓴다.
- `LanguageEncoder` 는 문자열 단위 캐시가 있어 (`:161-240`) 같은 명령어 재인코딩이
  사실상 공짜다. 앵커/드롭아웃이 CLIP 비용을 늘리지 않는다.

### 조건 벡터 부분 재사용 (속도 최적화)

한 스텝에 조건만 다른 forward 가 최대 3번 필요하다(현재 명령어 / ℓ_j / teacher ℓ_j).
매번 `_prepare_global_conditioning` 을 부르면 DINOv2 가 3번 돈다. 언어 외의 부분
(상태+이미지 2064차원)은 명령어와 무관하므로 **모델당 한 번만 계산하고 앞 512차원만
갈아 끼운다.** teacher 는 projection 가중치가 달라 자기 몫을 따로 계산한다.

이 최적화도 `--verify_cond` 로 검증한다: 직접 조립한 조건 벡터가 정책의
`_prepare_global_conditioning` 출력과 `allclose` 인지 시작 시 assert 한다.

## 4. 45 에피소드 선택 규칙

- `split_episodes()` — `E0.py:187-191`: `train = range(0, total-holdout)` = **ep 0..44**,
  `holdout = 45..49`. 무작위가 아니라 **뒤에서 5개 고정** 이다.
- 적용은 `LeRobotDataset(episodes=...)` 가 아니라 **샘플러**로 한다
  — `E0.py:194-209`. `EpisodeAwareSampler(episode_indices_to_use=train_eps,
  drop_n_last_frames=7, shuffle=True)`.
  이유는 `E0.py:196-201` 주석 참조(부분집합 인덱싱 버그).
- B1 은 `episode_sampler()` 와 `split_episodes()` 를 **E0.py 에서 import 해서 쓴다.**
  복제하지 않는다.

## 5. 옵티마이저 / 스케줄러 / 정밀도

`make_optimizer_and_scheduler(cfg, policy)` 가 정책 프리셋을 읽는다
(`use_policy_training_preset=True`).

| 항목 | 값 | 출처 |
|---|---|---|
| optimizer | Adam | `configuration_dit_flow_mt.py:175-181` |
| lr | 1e-4 | `configuration_dit_flow_mt.py:158` |
| betas | (0.95, 0.999) | `:159` |
| eps | 1e-8 | `:160` |
| weight_decay | 1e-6 | `:161` |
| scheduler | cosine, warmup 500 | `:162-163, 183-187` |
| grad_clip_norm | 10.0 | `lerobot_lsy/src/lerobot/optim/optimizers.py:68` |
| use_amp | False | `lerobot_lsy/src/lerobot/configs/policies.py:63` |
| augmentation | 없음 (`image_transforms.enable=False`) | `datasets/factory.py:145-149` |
| num_workers | 8 | `bash/E0/E0.sh:45` |

스케줄러는 **태스크마다 새로 만들어진다** (E0 가 스테이지마다 프로세스를 새로 띄우므로).
B1 은 한 프로세스에서 4태스크를 도니 이 동작을 맞추려고 태스크 시작마다
`make_optimizer_and_scheduler` 를 다시 부른다. 안 그러면 cosine 스케줄이 이어져
E0 와 lr 궤적이 달라진다.

## 6. CL 세분성

**suite 안의 개별 태스크 4개** (task 0..3) 다. suite 전체를 한 태스크로 보지 않는다.
따라서 k=2 부터 앵커가 실제로 작동하고, `--mode baseline` 이 E0 λ=0 과 같은 경로가 된다.

## 7. 롤아웃 / 평가 프로토콜

`E0.py:249-273` 의 `probe_sr()` 를 그대로 따른다. B1 은 이 함수를 import 하지 않고
같은 호출을 재현한다(`cfg.probe_*` 필드가 E0Config 전용이라서).

```
env = make_env(env_cfg, n_envs=20, use_async_envs=False)   # env_cfg.task = Libero_Spatial_Task_{j}
info = eval_policy(env, policy, n_episodes=20, start_seed=42)
SR = info["aggregated"]["pc_success"]
```

- `eval_policy` 는 `lerobot/scripts/eval.py` 것을 쓴다 (E0 와 동일).
- 에피소드 시드 42..61. **매 스테이지·매 태스크가 같은 20개 초기 상태**를 쓴다.
- `episode_length=500` 스텝 상한 — `lerobot_lsy/src/lerobot/envs/configs.py:114`.
- 평가 직전 `policy.eval()`, 직후 `policy.train()` 복원 — `E0.py:271-273`.

## 8. 이전 seq-FT 기준선 (B1 이 이겨야 하는 대상)

`outputs/E0/libero_spatial/seed_42/e0_results.jsonl` 의 `run_tag="0"`:

```
stage\task     0     1     2     3
        0    100
        1     30    90
        2      0     0    95
        3      0     0    50    90

AvgSR_final 35.0    BWT -78.3
```

task 0 이 100 → 30 → 0 → 0 으로 소실된다. B1 성공 기준은 이 retention 개선이다.

## 9. B1 에서만 다른 것

프롬프트의 hard constraint 대로 `p_drop`, `lambda_anchor` **두 개만** 다르다.

| | E0 λ=0 (seq-FT) | B1 ours | B1 baseline |
|---|---|---|---|
| p_drop | 0 (개념 없음) | 0.1 | 0 |
| lambda_anchor | 0 (개념 없음) | 1.0 | 0 |
| 그 외 전부 | — | 동일 | 동일 |

`--mode baseline` 은 두 값을 0으로 두어 손실이 `L_FM` 만 남고, 조건은 항상 실제
명령어가 된다. 즉 E0 λ=0 경로와 수식이 같아진다.

## 10. 발견하지 못해 기본값을 쓴 항목

없음. 프롬프트가 "찾지 못하면 기본값을 쓰고 최상단에 표시하라"고 했으나,
`--eval_episodes`(20) 와 `--seed`(42) 를 포함해 요구된 항목을 전부 저장소에서 찾았다.
