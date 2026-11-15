#!/usr/bin/env bash
# Build 0xSero/inkling2-exl3 @ 5cf1465 (snapshot 87b3a16b) for GB10 (sm_121a) in the NGC pytorch image; nothing is installed into the image or the host.
# Expects in $WORK: repo.tgz (overlay + patches + server + bench of the author's repo with our patches 0003/0004 applied), exllamav3-v1.6.6.tgz, flashinfer-b12x.tgz
# (FlashInfer 9c509967 tree, only flashinfer/experimental/b12x). See README for how to create them. Output: $WORK/site (exllamav3 + ext), $WORK/site-deps (python deps), $WORK/flashinfer.
# Runs at low CPU priority (--cpu-shares 64) so that a server running next to it keeps its CPU.
set -euo pipefail
WORK="${WORK:?set WORK to the work directory on this host}"
IMG="${IMG:-nvcr.io/nvidia/pytorch:26.07-py3}"
mkdir -p "$WORK/src" "$WORK/site" "$WORK/site-deps" "$WORK/flashinfer"
[ -d "$WORK/src/repo" ] || { mkdir -p "$WORK/src/repo"; tar xzf "$WORK/repo.tgz" -C "$WORK/src/repo"; }
[ -d "$WORK/src/exllamav3" ] || { mkdir -p "$WORK/src/exllamav3"; tar xzf "$WORK/exllamav3-v1.6.6.tgz" -C "$WORK/src/exllamav3"; }
[ -f "$WORK/flashinfer/flashinfer/experimental/b12x/__init__.py" ] || tar xzf "$WORK/flashinfer-b12x.tgz" -C "$WORK/flashinfer"
if ! grep -q Inkling2Model "$WORK/src/exllamav3/exllamav3/architecture/architectures.py"; then
  (cd "$WORK/src/exllamav3" && patch -p1 < "$WORK/src/repo/patches/upstream-modifications.patch")
  cp -r "$WORK/src/repo/overlay/exllamav3/." "$WORK/src/exllamav3/exllamav3/"
fi
docker run --rm --name inkling-build --cpu-shares 64 --entrypoint bash -v "$WORK":/work "$IMG" -c '
  set -euo pipefail
  export TORCH_CUDA_ARCH_LIST="12.1a" MAX_JOBS=${MAX_JOBS:-6} CUDA_HOME=/usr/local/cuda
  # engine deps not in the NGC image (author lock docker/requirements-engine.lock versions where they exist for aarch64)
  pip install --no-cache-dir --target /work/site-deps rich==15.0.0 marisa-trie==1.4.1 "llguidance==1.9.1" pyyaml pillow typing_extensions \
      fastapi==0.162.7 uvicorn==0.54.0 2>&1 | tail -3
  # b12x add-on: author lock docker/requirements-b12x.lock (without nvidia-nccl-cu13: the NGC torch brings its own NCCL), installed --no-deps
  pip install --no-cache-dir --no-deps --target /work/site-deps apache-tvm-ffi==0.1.16 cuda-bindings==12.9.10 cuda-python==12.9.10 \
      nvidia-cuda-crt==13.3.73 nvidia-cuda-nvcc==13.3.73 nvidia-cuda-nvdisasm==13.5.37 nvidia-cuda-tileiras==13.3.36 \
      nvidia-cutlass-dsl==4.8.1 nvidia-cutlass-dsl-libs-base==4.8.1 nvidia-cutlass-dsl-libs-core==4.8.1 nvidia-cutlass-dsl-libs-cu12==4.8.1 \
      nvidia-cutlass-dsl-libs-cu13==4.8.1 nvidia-nvjitlink==13.5.37 nvidia-nvvm==13.3.73 protobuf==7.36.2 2>&1 | tail -5
  cd /work/src/exllamav3
  pip install --no-cache-dir --no-build-isolation --no-deps --target /work/site . 2>&1 | tail -20
'
echo "build done: $WORK/site $WORK/site-deps"
