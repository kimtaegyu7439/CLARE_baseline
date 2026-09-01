# K5b — witness 판별력 벤치

suite=libero_spatial  bins=[2, 5, 8]  n/bin=256  M_max=8  ρ=0.3
witness 규약: t=0, x_t=eps_probe, ℓ_0, state=0 (K5 와 동일)

| 후보 | E | ratio | overshoot | 평균 M | clip | d̂_before → d̂_after | diversity | 수집 s | 정련 s |
|---|---|---|---|---|---|---|---|---|---|
| a_pretrained_blocks04 | plain | 117.62 | 0.09 | 8.0 | 22% | 1.87 → 1.87 | 1.00 | 4 | 2 |
| a_pretrained_blocks04 | maha | 81.64 | 0.14 | 8.0 | 7% | 1.84 → 1.84 | 1.00 | 4 | 1 |
| b_pretrained_blocksAll | plain | 83.73 | 0.13 | 8.0 | 17% | 1.87 → 1.88 | 1.00 | 4 | 2 |
| b_pretrained_blocksAll | maha | 64.80 | 0.19 | 8.0 | 6% | 1.87 → 1.87 | 1.00 | 4 | 2 |
| c_task0snapshot_blocksAll | plain | 130.29 | 0.16 | 8.0 | 9% | 1.85 → 1.85 | 1.00 | 4 | 2 |
| c_task0snapshot_blocksAll | maha | 89.53 | 0.14 | 8.0 | 6% | 1.86 → 1.87 | 1.00 | 4 | 2 |

판정 기준: (i) ratio ≥ 3   (ii) overshoot ≤ 1.2   (iii) d̂_after ≤ 0.7·d̂_before 이고 d̂_after ≤ 1.5

**활성-통계 정련 기각 — K5 는 M=0(=R13) 으로 후퇴, negative result 로 기록. 어느 후보도 (i) ratio≥3, (ii) overshoot≤1.2, (iii) d̂ 30%↓ & ≤1.5 를 동시에 만족하지 못했다.**

(d) DINOv2 witness: **infeasible** — CLS = outputs.pooler_output (DINOv2 마지막 레이어). 주입 후 읽을 이후 레이어가 없어 구조적으로 불가.

블록별 ratio (maha)
  b_pretrained_blocksAll: b0=18.83  b1=15.73  b2=13.80  b3=9.02  b4=7.86  b5=6.11
  c_task0snapshot_blocksAll: b0=18.89  b1=17.08  b2=12.19  b3=8.86  b4=8.52  b5=7.96
