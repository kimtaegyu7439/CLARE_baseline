#!/usr/bin/env python
"""롤아웃이 느린 원인을 부위별로 가른다. 두 서버에서 같이 돌려 비교한다.

    python bench_rollout_parts.py

재는 것
  1. 순수 CPU (numpy 행렬곱)        — 코어 성능/BLAS
  2. MuJoCo 물리 스텝               — 시뮬레이터 자체
  3. EGL 오프스크린 렌더            — 렌더링 경로
  4. torch GPU 행렬곱               — GPU 가 제대로 잡혔는지

롤아웃 한 스텝은 (2)x50 + (3)x50xcam + 정책 forward 다. 어느 항목이 몇 배
느린지 보면 어디를 고쳐야 하는지가 정해진다.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"))

import numpy as np


def timeit(fn, n: int) -> float:
    fn()                                   # 워밍업
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3    # ms


def main() -> None:
    print(f"  threads: OMP={os.environ.get('OMP_NUM_THREADS','(미설정)')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS','(미설정)')}")
    print(f"  numpy {np.__version__}")

    # 1) 순수 CPU
    a = np.random.rand(512, 512)
    b = np.random.rand(512, 512)
    print(f"  [1] numpy 512x512 matmul      {timeit(lambda: a @ b, 20):8.2f} ms")

    # 2) MuJoCo 물리
    try:
        import mujoco
        xml = """<mujoco><worldbody>
          <body pos="0 0 1"><joint type="free"/><geom type="box" size=".1 .1 .1"/></body>
          <body pos="0 0 2"><joint type="free"/><geom type="sphere" size=".1"/></body>
          <geom type="plane" size="5 5 .1"/></worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        print(f"  [2] mujoco mj_step            "
              f"{timeit(lambda: mujoco.mj_step(m, d), 2000):8.4f} ms   (mujoco {mujoco.__version__})")

        # 3) EGL 렌더
        try:
            r = mujoco.Renderer(m, 128, 128)
            def render():
                r.update_scene(d)
                r.render()
            print(f"  [3] EGL render 128x128        {timeit(render, 50):8.2f} ms")
        except Exception as e:
            print(f"  [3] EGL render                 실패: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  [2] mujoco                     실패: {type(e).__name__}: {e}")

    # 4) GPU
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.randn(2048, 2048, device="cuda")
            torch.cuda.synchronize()
            def gpu():
                (x @ x); torch.cuda.synchronize()
            print(f"  [4] torch cuda 2048 matmul    {timeit(gpu, 20):8.2f} ms   "
                  f"({torch.cuda.get_device_name(0)})")
        else:
            print("  [4] torch cuda                 사용 불가 ★")
    except Exception as e:
        print(f"  [4] torch                      실패: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
