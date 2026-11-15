#!/usr/bin/env bash
# Diagnostic probe, not a measurement: builds the b12x expert module of rank 0 for layer 5 from the real shards, compiles the kernel on GB10
# and runs it for T = 1, 8, 64, 200 tokens. Passes when the outputs are finite and not zero ("PROBE OK").
# Run on rank 0 after the build and after the shards of layer 5 exist. WORK as in start_server.sh.
set -u
WORK="${WORK:?set WORK}"; HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WORK/jitcache"
docker run --rm --gpus all --ipc host --entrypoint python \
 -e PYTHONPATH=/work/site:/work/site-deps/nvidia_cutlass_dsl/dsl_packages:/work/site-deps:/work/flashinfer:/work/flashinfer/flashinfer/experimental/b12x/_compat -e I2_B12X_PATH=/work/flashinfer \
 -e I2_EXPERT_SHARD_DIR=/work/shards -e I2_B12X=1 -e I2_EP_LAYOUT=ep2tp2 -e EXL3_LOAD_ARENA=0 -e PROBE_RANK=0 -e PYTHONUNBUFFERED=1 \
 -v "$WORK":/work -v "$WORK/model":/work/model:ro -v "$HERE":/probe:ro nvcr.io/nvidia/pytorch:26.07-py3 /probe/probe_b12x.py > "$WORK/probe-b12x.log" 2>&1
echo "exit $?" >> "$WORK/probe-b12x.log"
cat "$WORK/probe-b12x.log"
