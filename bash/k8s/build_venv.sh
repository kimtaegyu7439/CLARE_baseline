#!/usr/bin/env bash
# 파드 이미지의 torch 2.8.0+cu128 위에 clare 의존성만 얹는 venv 를 PVC 에 만든다.
#
# ★ 서버1 의 conda env 를 그대로 옮기면 안 된다 — 거기 torch 는 2.13.0+**cu130**
#   이고 cu130 은 드라이버 580+ 를 요구한다. 이 노드는 570.124.06 이라 못 쓴다.
#   이미지의 2.8.0+cu128 이 570(=CUDA 12.8)에 맞는 조합이다.
#
# ★ --system-site-packages: torch/torchvision/numpy/mujoco 는 이미지 것을 쓴다.
#   constraints 로 못 박아 pip 이 이것들을 **교체하지 못하게** 한다. 안 막으면
#   lerobot 의 torch>=2.2.1 을 보고 최신 torch(cu13x)를 끌어와 전부 깨진다.
set -euo pipefail
VENV=/home/asdl/tether-venv
PY=/opt/anaconda3/envs/cuda12.1-pytorch2.8.0-py3.10/bin/python3

if [ ! -x "$VENV/bin/python" ]; then
    echo "[venv] 생성 (--system-site-packages)"
    "$PY" -m venv --system-site-packages "$VENV"
fi
. "$VENV/bin/activate"
python -m pip install -q -U pip setuptools wheel

cat > /tmp/constraints.txt <<'C'
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
numpy==2.2.6
mujoco==3.2.3
C

echo "[venv] 고정 패키지 (서버1 과 같은 버전)"
pip install -c /tmp/constraints.txt \
    robosuite==1.4.0 bddl==1.0.1 gymnasium==1.3.0 \
    transformers==4.47.1 diffusers==0.39.0 einops==0.8.2 \
    datasets==3.6.0 safetensors==0.8.0 av==17.1.0 \
    opencv-python PyOpenGL 2>&1 | tail -4

echo "[venv] editable 3개"
pip install -c /tmp/constraints.txt \
    -e /home/asdl/src/lerobot/lerobot_lsy \
    -e /home/asdl/src/gym-libero \
    -e /home/asdl/src/peft/peft_lsy 2>&1 | tail -6

echo "[venv] 검증"
python - <<'PY'
import importlib.util as u, sys
print("  python", sys.version.split()[0])
import torch
print("  torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
for m in ("lerobot","gym_libero","robosuite","bddl","mujoco","gymnasium","peft","numpy"):
    s = u.find_spec(m)
    print(f"  {m:12} {'OK' if s else '★없음'}")
import numpy, mujoco
print("  numpy", numpy.__version__, " mujoco", mujoco.__version__)
PY
echo "[venv] 완료 -> $VENV"
