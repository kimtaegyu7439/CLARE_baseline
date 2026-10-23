#!/usr/bin/env bash
# 노드 드라이버(570.124.06)의 **64비트 그래픽 유저스페이스 라이브러리**만 추출해
# ~/nvegl 에 둔다. 파드 이미지는 compute-only 라 libEGL_nvidia.so.0 이 없고,
# 그게 없으면 EGL 이 Mesa(llvmpipe = CPU 소프트웨어 렌더)로 떨어진다.
#
# ★★ .run 안에는 32비트 사본이 ./32/ 에 같이 들어 있다. 그걸 집으면
#    dlopen 이 "wrong ELF class: ELFCLASS32" 로 실패하는데, ldd 에는 아무
#    문제가 안 보이고 eglInitialize 가 EGL_NOT_INITIALIZED(0x3001) 만 낸다.
#    그래서 반드시 ELF 클래스를 확인하고 복사한다.
#
# ★ 컴퓨트 스택(libcuda/libnvidia-ml/ptxjitcompiler)은 일부러 제외한다.
#   컨테이너에 이미 있고, LD_LIBRARY_PATH 앞에서 CUDA 를 가리면 torch 가 깨진다.
#   gpucomp/nvvm 은 eglcore 가 dlopen 하므로 포함한다(버전이 노드와 동일).
set -euo pipefail
V=570.124.06
W=/home/asdl/.nvegl_build
OUT=/home/asdl/nvegl
mkdir -p "$W"; cd "$W"
RUN="NVIDIA-Linux-x86_64-${V}.run"
[ -s "$RUN" ] || curl -fL -o "$RUN" "https://us.download.nvidia.com/tesla/${V}/${RUN}"
[ -d "NVIDIA-Linux-x86_64-${V}" ] || sh "$RUN" --extract-only --target "NVIDIA-Linux-x86_64-${V}" >/dev/null
S="$W/NVIDIA-Linux-x86_64-${V}"

rm -rf "$OUT/lib"; mkdir -p "$OUT/lib"
is64() { [ "$(od -An -t u1 -j4 -N1 "$1" 2>/dev/null | tr -d ' ')" = "2" ]; }   # ELF e_ident[EI_CLASS]==2
n64=0; n32=0
for pat in 'libEGL_nvidia.so.*' 'libGLX_nvidia.so.*' 'libGL.so.*' \
           'libnvidia-eglcore.so.*' 'libnvidia-glcore.so.*' 'libnvidia-glsi.so.*' \
           'libnvidia-glvkspirv.so.*' 'libnvidia-tls.so.*' 'libnvidia-rtcore.so.*' \
           'libnvoptix.so.*' 'libnvidia-vulkan-producer.so.*' \
           'libnvidia-egl-gbm.so.*' 'libnvidia-allocator.so.*' \
           'libnvidia-gpucomp.so.*' 'libnvidia-nvvm.so.*'; do
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        if is64 "$f"; then cp -f "$f" "$OUT/lib/"; n64=$((n64+1)); else n32=$((n32+1)); fi
    done < <(find "$S" -name "$pat" ! -path '*/32/*' 2>/dev/null)
done
echo "[nvegl] 64비트 ${n64}개 복사, 32비트 ${n32}개 제외"

cd "$OUT/lib"
# glvnd 는 .so.0 을, eglcore 는 맨이름(.so)으로 dlopen 한다 — 둘 다 걸어 둔다.
for f in *.so.${V}; do
    b="${f%.so.${V}}"
    ln -sf "$f" "${b}.so"
    ln -sf "$f" "${b}.so.0"
done
ln -sf "libnvidia-nvvm.so.${V}" libnvidia-nvvm.so.4
for f in libnvidia-egl-gbm.so.*.*.*; do [ -f "$f" ] && ln -sf "$f" libnvidia-egl-gbm.so.1; done
cat > "$OUT/10_nvidia.json" <<JSON
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "$OUT/lib/libEGL_nvidia.so.0"
    }
}
JSON
echo "[nvegl] 완료  $(du -sh "$OUT" | cut -f1)"
echo "[nvegl] ELF 클래스 검증:"
for f in libnvidia-eglcore.so.${V} libnvidia-glcore.so.${V} libEGL_nvidia.so.${V}; do
    printf '    %-40s %s\n' "$f" "$(is64 "$f" && echo 64bit || echo '★32bit')"
done
