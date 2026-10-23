#!/usr/bin/env bash
# 쿠버네티스 GPU 파드를 만든다 (A6000 클러스터, 네임스페이스 asdl).
#
#   NAME=lotus-goal GPUS=1 CMD='bash bash/k8s/lotus_seed_loop.sh' bash bash/k8s/pod.sh
#   NAME=... bash bash/k8s/pod.sh --dry          # YAML 만 출력
#
# gpu-demo_default.yaml 과 같은 구조에 **두 가지를 더한다**:
#
#  ① NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
#     기본값은 compute,utility 라 EGL 벤더 ICD(10_nvidia.json)조차 안 들어온다.
#     그러면 MuJoCo 렌더가 Mesa llvmpipe(CPU 소프트웨어)로 떨어져 롤아웃이
#     6배 이상 느려진다. ★ 이것만으로는 부족하다 — 노드가 compute-only
#     드라이버 설치라 libEGL_nvidia.so.0 자체가 없어서, ~/nvegl 번들
#     (bash/k8s/build_nvegl.sh)이 같이 있어야 GPU 렌더가 된다.
#
#  ② restartPolicy: Never
#     Lotus_l2cb_cl.py 에는 resume 이 없다. OnFailure 로 두면 죽은 런이
#     조용히 **처음부터** 다시 돌아 며칠을 날린다. 실패는 드러나는 게 낫다.
#
# ★ k8s 에서는 GPU 번호를 고르지 않는다. 파드가 GPU N 개를 요청하면 그만큼
#   할당받고 컨테이너 안에서는 항상 0..N-1 로 보인다. 그래서 "GPU 0/1/2/3 에
#   하나씩" 은 **1-GPU 파드 4개**로 구현한다.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

NAME=${NAME:?NAME 이 필요하다}
GPUS=${GPUS:-1}
CMD=${CMD:?CMD 가 필요하다}
USER_ID=${USER_ID:-asdl}
UID_N=${UID_N:-1023}
GID_N=${GID_N:-1023}
IMAGE=${IMAGE:-k8s2.local:8888/ost-hub/cuda12.1-pytorch2.8.0-py3.10:2025-09-10-robosuite}
WORKDIR=${WORKDIR:-/home/asdl/clare_1020}
KUBECONFIG=${KUBECONFIG:-/home/asdl/asdl-kubeconfig}
export KUBECONFIG

read -r -d '' YAML <<YEOF
apiVersion: v1
kind: Pod
metadata:
  name: ${NAME}
  labels:
    app: tether-lotus
spec:
  securityContext:
    runAsUser: ${UID_N}
    runAsGroup: ${GID_N}
    fsGroup: ${GID_N}
    fsGroupChangePolicy: "OnRootMismatch"
  restartPolicy: Never
  volumes:
  - name: shmdir
    emptyDir:
      medium: Memory
  - name: pvc-volume
    persistentVolumeClaim:
      claimName: ${USER_ID}-pvc
  containers:
  - name: gpu-container
    image: ${IMAGE}
    imagePullPolicy: IfNotPresent
    env:
    - name: HOME
      value: /home/${USER_ID}
    - name: NVIDIA_DRIVER_CAPABILITIES
      value: "compute,utility,graphics,display"
    volumeMounts:
    - mountPath: /dev/shm
      name: shmdir
    - mountPath: /home/${USER_ID}
      name: pvc-volume
    command: ["/bin/bash","-lc"]
    args:
      - cd ${WORKDIR} && ${CMD}
    securityContext:
      allowPrivilegeEscalation: false
    resources:
      requests:
        nvidia.com/gpu: ${GPUS}
      limits:
        nvidia.com/gpu: ${GPUS}
YEOF

if [ "${1:-}" = "--dry" ]; then
    echo "${YAML}"; exit 0
fi

kubectl get pod "${NAME}" -n "${USER_ID}" >/dev/null 2>&1 && {
    echo "✗ 파드 ${NAME} 가 이미 있다. 지우려면:"
    echo "    kubectl delete pod ${NAME} -n ${USER_ID}"
    exit 1; }

echo "${YAML}" | kubectl create -n "${USER_ID}" -f -
echo "  상태   kubectl get pod ${NAME} -n ${USER_ID}"
echo "  로그   kubectl logs -f ${NAME} -n ${USER_ID}"
