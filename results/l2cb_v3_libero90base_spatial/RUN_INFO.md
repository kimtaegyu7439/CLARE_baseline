# l2cb_v3_libero90base_spatial

- 시작: 2026-09-09 10:24:27  PID 2297702  GPU 0
- 팔: l2_codebook v3 (--grad_enable 고정된 독립 실행본 l2_codebook_id.py)
- base: /home/sa090180/Models/Lotus_clare_libero90_pretrain  (우리가 학습한 libero_90 200K step)
- 셋팅: libero_spatial 10 task 순차, 5000 step/task, 칸당 20 rollout, seed 42
- 대조: results/l2_codebook_k96_grad (같은 팔 + 기존 사전학습, AvgSR 92.5)
