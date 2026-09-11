# l2_codebook_v4_hard_only_k96

- 시작: 2026-09-08 20:17:25
- PID: 1906078
- GPU: 0
- config: results/l2_codebook_v4_hard_only_k96/config.json
- 로그: logs/l2_codebook_v4_hard_only_k96.log

```
python -u Lotus_l2cb_cl.py --suite goal --steps 10000 \
    --eval_seed 7 --eval_episodes 20 --eval_batch_size 20 \
    --num_workers 6 --out results/l2_codebook_v4_hard_only_k96 --sampler_mode hard_linear
```
- 종료: 2026-09-09 05:55:35 rc=0
