#!/usr/bin/env bash
# scripts/start_server.sh up|down|logs|box-check   (run it on NODE0 = rank 0). Author server (0xSero/inkling2-exl3 5cf1465d + patches 0003/0004)
# for Inkling-2-Small 4.0 bpw on 4 DGX Sparks, EP2xTP2 + b12x, FP8 draft, speculative decoding always on.
# Required environment: NODE0..NODE3 (ssh names in ring order, rank i = NODEi, NODE0 = this host), MASTER_ADDR (IP of NODE0 on the LAN),
# IFACE (network interface that carries MASTER_ADDR), WORK (work directory, same path on all hosts: site, site-deps, flashinfer, src, shards, model, jitcache),
# RING_DIR (output of scripts/ring/ringenv.py, default $WORK/ring).
set -u
: "${NODE0:?}" "${NODE1:?}" "${NODE2:?}" "${NODE3:?}" "${MASTER_ADDR:?}" "${IFACE:?}" "${WORK:?}"
HOSTS=("$NODE0" "$NODE1" "$NODE2" "$NODE3"); MASTER=$MASTER_ADDR; PORT=${PORT:-29541}
IMG=${IMG:-nvcr.io/nvidia/pytorch:26.07-py3}; V=$WORK; RING_DIR=${RING_DIR:-$WORK/ring}
KVGB=${KVGB:-24}
# I2_PC_SHARD=0 I2_PEER_AR=0: token-sharded prefill and the peer all-reduce of the author need CUDA IPC between the GPUs, which only exists on one host; on four hosts they fail at cudaIpcOpenMemHandle (see README).
run() { if [ "$1" = "$(hostname)" ]; then bash -c "$2"; else ssh -n -o BatchMode=yes "$1" "$2"; fi; }
case ${1:-} in
 up)
  for r in 3 2 1 0; do h=${HOSTS[$r]}
    run $h "docker rm -f inkling-srv >/dev/null 2>&1; docker run -d --name inkling-srv --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --shm-size 16g --device /dev/infiniband \
      ${EXTRA_DOCKER:-} --entrypoint python --env-file $RING_DIR/rank$r.env -e NCCL_GRAPH_FILE=/nccl-graph.xml -e NCCL_SOCKET_IFNAME=$IFACE -e GLOO_SOCKET_IFNAME=$IFACE \
      -e RANK=$r -e WORLD_SIZE=4 -e LOCAL_RANK=0 -e MASTER_ADDR=$MASTER -e MASTER_PORT=$PORT -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
      -e PYTHONPATH=/work/site:/work/site-deps/nvidia_cutlass_dsl/dsl_packages:/work/site-deps:/work/flashinfer:/work/flashinfer/flashinfer/experimental/b12x/_compat \
      -e I2_B12X_PATH=/work/flashinfer -e I2_PC_SHARD=0 -e I2_PEER_AR=0 -e I2_B12X=1 -e I2_EP_LAYOUT=ep2tp2 -e I2_DRAFT_FP8=1 -e I2_MOE_GROUP=64 -e EXL3_LOAD_ARENA=0 -e NCCL_CUMEM_ENABLE=${NCCL_CUMEM:-1} \
      -e I2_EXPERT_SHARD_DIR=/work/shards -e B12X_COMPILE_CACHE_DIR=/work/jitcache -e MODEL_DIR=/model -e PYTHONUNBUFFERED=1 \
      -v $RING_DIR/nccl-graph-rank$r.xml:/nccl-graph.xml:ro -v $V:/work:ro -v $V/jitcache:/work/jitcache:rw -v $V/model:/model:ro \
      $IMG /work/src/repo/server/i2_server.py --model /model --port 8891 --b12x 1 --kv-gb $KVGB ${SERVER_ARGS:-}" >/dev/null || exit 1
    [ $r -ne 0 ] && sleep 1
  done; exit 0;;
 down)
  for h in "${HOSTS[@]}"; do run $h 'docker stop -t 5 inkling-srv >/dev/null 2>&1; docker rm -f inkling-srv >/dev/null 2>&1; true'; done;;
 logs) for r in 0 1 2 3; do echo "rank $r (${HOSTS[$r]})"; run ${HOSTS[$r]} 'docker logs inkling-srv 2>&1 | grep -v "NCCL INFO"'; done;;
 box-check)
  for h in "${HOSTS[@]}"; do run $h "ex=\$(ls $WORK/shards/.done-* 2>/dev/null | wc -l); ne=\$(ls $WORK/model | grep -c 'nonexpert.*safetensors\$'); echo \"\$(hostname): shard-done \$ex/192 nonexpert \$ne/24 containers \$(docker ps -q | wc -l)\"; [ \"\$ex\" = 192 ] && [ \"\$ne\" = 24 ]" || exit 1; done;;
 *) echo "usage: start_server.sh up|down|logs|box-check"; exit 2;;
esac
