# l2_codebook_v4_hard_sf_k96

- 시작: 2026-09-08 16:18:54
- PID: 1755727
- GPU: 1
- config: results/l2_codebook_v4_hard_sf_k96/config.json
- 로그: logs/l2_codebook_v4_hard_sf_k96.log

```
python -u Lotus_l2cb_cl.py --suite goal --steps 10000 \
    --eval_seed 7 --eval_episodes 20 --eval_batch_size 20 \
    --num_workers 6 --out results/l2_codebook_v4_hard_sf_k96 --sampler_mode hard_linear --anchor_weight speedinv
```
- 종료: 2026-09-09 00:35:13 rc=0
