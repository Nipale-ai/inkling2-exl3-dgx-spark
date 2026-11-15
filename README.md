# Inkling-2-Small (EXL3 4.0 bpw) on four NVIDIA DGX Spark

This repository makes 0xSero's Inkling-2-Small engine run on four NVIDIA DGX Spark (GB10, 128 GB unified memory each), connected in a ring without a switch.

The engine is `github.com/0xSero/inkling2-exl3`. The quant is `0xSero/Inkling-2-Small-EXL3-4bpw`. Both are the work of 0xSero. The model is Inkling-2-Small by Thinking Machines. This repository only contains the port: two patches, helper scripts and the steps that worked for us.

Pinned versions:

- Engine: `0xSero/inkling2-exl3` at commit `5cf1465d22205349b0ec3d8e54908f0467bcc2c6` (snapshot 87b3a16b).
- Quant: `0xSero/Inkling-2-Small-EXL3-4bpw` at revision `e58fa769cd946a7df2e2bbd5bec790290549b150`.
- exllamav3 v1.6.6, FlashInfer commit `9c50996794564e1e3b2a4b0683e9aa2bcc528df8` (b12x kernels, used from source).
- Base image `nvcr.io/nvidia/pytorch:26.07-py3` (torch 2.13, CUDA 13.3, NCCL 2.30.7, aarch64), `nvidia-cutlass-dsl` 4.8.1.

Speed numbers below were measured at upstream commit 5cf1465d. The quality numbers were measured earlier, at the previous state 8f12aa4e, and not repeated at 5cf1465d. This repository first targeted 8f12aa4e (the numbers of that state are kept in the tables as a comparison column).

## Quick answers

1. KV cache and context. We start the server with `--kv-gb 24`: 24 GiB of KV per rank, which is 9,125,248 tokens (2824 bytes per token per GPU). With 16 slots that is about 570,000 tokens per slot, so no slot has to share a window. The server's context limit is its default, 380,000 tokens per request. Free memory per box: right after start 6.3 to 8.6 GB available (4.7 to 7.1 GB MemFree); the lowest value during the benchmark runs, logged every 60 s, was 5.1 to 6.6 GB available per box. With the previous engine state (8f12aa4e) it was 10 to 11 GB available after start. Keep other work off the boxes.
2. Image or own build. There is no image for GB10 (the author's image is x86_64, CUDA 12.8, sm_120). You build exllamav3 v1.6.6 plus the author's overlay for sm_121a inside the NGC image, then install the author's lock-file versions of the Python dependencies (`scripts/build.sh`). Build time: 5 min 10 s for `scripts/build.sh` from zero on one idle box (container start to `build done`, nothing else running, `MAX_JOBS=6`). The same build took about 60 minutes at the earlier engine state, but at low CPU priority with a benchmark running next to it on the same box. The server start compiles the b12x kernels: 629 s from container start to ready (11 min with an empty kernel cache directory) and 641 s with a filled one, so no faster start with the cache.
3. Speed on four Sparks (5cf1465d). Decode 58.3 tok/s (code 55.6, prose 61.0). Prefill 2,355 tok/s at 32K, first token after 15.0 s. Prefill 2,167 tok/s at 128K, first token after 65.4 s. 8 requests at once: 126.2 tok/s in total. Details below.
4. Quality at reasoning effort none (measured at the previous state 8f12aa4e): MMLU-Pro 68.5 %, GPQA Diamond 58.6 %, IFEval 77.0 %, LiveCodeBench 80.0 %, RULER-lite 8K 77.5 %, 32K 70.0 %. We did not measure the unquantized original, so we cannot say how much the quant costs. The author's own figure for this quant against BF16 is KLD 0.0213 and top-1 agreement 97.35 % (his README and model card; we did not reproduce it).
5. Smallest setup. We tested only four Sparks. Two or three Sparks: not tested. Disk per box: about 91 GB (73.7 GB expert shards plus 17.4 GB for the non-expert and MTP files). Download per box: about 73.7 GB of expert byte ranges, plus 17.4 GB of non-expert and MTP files. Cabling: no switch. Each Spark is connected directly (ConnectX-7) to its two neighbours with two cables each, which makes a ring of 8 cables.

## What is in here

- `patches/0003-expert-shards-and-cache-drop.patch` and `patches/0004-multinode-localrank-server-compat.patch`: apply them in this order to a checkout of the engine at 5cf1465d.
- `scripts/reshard_experts.py`: downloads the expert slices each box needs by HTTP byte range and cuts them to the EP2xTP2 layout.
- `scripts/prepare_model_dir.py`: builds the model directory (config, tokenizer, non-expert files, MTP).
- `scripts/check_shards.py` and `scripts/test_slice.py`: check the shards.
- `scripts/ring/`: generates the NCCL settings for a switchless ring (`ringenv.py`, `pair_links.py`, `node_inventory.fn.sh`).
- `scripts/rdma_check.sh`: link bandwidth check.
- `scripts/build.sh`: build for GB10 in the NGC image.
- `scripts/start_server.sh`: starts one container per box.
- `scripts/probe_b12x.sh` and `scripts/probe_b12x.py`: diagnostic probe of the b12x kernel on GB10 (see step 4).
- `scripts/served_sweep.py` and `scripts/smoke_test.py`: the served decode sweep and a smoke test.

## Placeholders

- `NODE0` to `NODE3`: ssh names of the four boxes in ring order. Rank i runs on NODEi. Ranks 0 and 1 are one expert pair, ranks 2 and 3 the other. Neighbours in the ring are 0-1, 1-2, 2-3, 3-0. Ranks 0 and 2 are not neighbours.
- `MASTER_ADDR`: IP address of NODE0 on the LAN.
- `IFACE`: the network interface that carries `MASTER_ADDR` and the default route. On our boxes this is the 2.5 GbE LAN port, with the same name on all four. It is not a ring link: the ring links are the RoCE devices listed in `NCCL_IB_HCA` of the generated env files (ours: `rocep1s0f0`, `rocep1s0f1`, `roceP2p1s0f0`, `roceP2p1s0f1`). `ringenv.py` does not set it, `start_server.sh` does.
- `WORK`: a work directory with the same path on all boxes. It holds `site`, `site-deps`, `flashinfer`, `src`, `shards`, `model`, `jitcache` and `ring`.
- `MODEL_DIR` is `$WORK/model`, the shard directory is `$WORK/shards`.

## Requirements

- 4 x DGX Spark (GB10), each with about 91 GB free disk for the shards and model files, plus room for the build and the kernel cache.
- Switchless CX7 ring with RDMA. Check about 110 Gb/s per link before you start (`scripts/rdma_check.sh`). When we checked, all 8 cables showed between 109.25 and 109.33 Gb/s.
- Docker with the NVIDIA runtime and access to `nvcr.io`, passwordless ssh between the boxes (the two ranks of a pair send each other files over ssh), Python 3 on each host (standard library only for the helper scripts), network access to huggingface.co and github.com.
- Nothing else running on the boxes. Free memory is tight once the server is up.

System state of our boxes (the four are not identical in OS state, and it ran like this):

- All four: NVIDIA driver 580.178.04, kernel 7.0.0-1019-nvidia.
- Two boxes: DGX OS 7.4.0 (build 2026-01-26), Ubuntu 24.04.5 LTS, Docker 29.6.2.
- Two boxes: DGX OS 7.2.3 (build 2025-10-04), Docker 29.1.3 (Ubuntu package), Ubuntu 24.04.5 LTS on one and 24.04.4 LTS on the other.
- Before the loading start we had 114 to 118 GB of memory free per box.

## Steps

This is the order in which it ran for us.

### 1. Check the links and generate the ring settings

`scripts/rdma_check.sh LINKS_FILE` runs `ib_write_bw` over every cable. Each line of the links file is `server_host server_ip server_dev client_host client_dev`. All 8 cables should show at least 100 Gb/s.

`python3 scripts/ring/ringenv.py $WORK/ring NODE0 NODE1 NODE2 NODE3` reads the network inventory of the four boxes over ssh, checks that they form the ring 0-1-2-3, and writes `rank0.env` to `rank3.env` and `nccl-graph-rank0.xml` to `nccl-graph-rank3.xml`. The env files set `NCCL_ALGO=Ring`, the RoCE devices, GID index and a fixed channel count (4). We used the generated files unchanged: the copies on the boxes are byte-identical to the `ringenv.py` output. The env file and graph file of rank r go to rank r. The ring scripts come from MiaAI-Lab's GLM-5.3-Flash-DGX-Sparks-1104 recipe (Apache-2.0, `scripts/tp4.sh`): `node_inventory.fn.sh` and `pair_links.py` are the `node_inventory` and `pair_links` functions of that file, `ringenv.py` is a Python version of its graph-file and NCCL ring settings.

### 2. Download and cut the expert shards

Run on every rank R = 0..3 in parallel. The partner of rank 0 is rank 1, of 1 is 0, of 2 is 3, of 3 is 2.

`python3 scripts/reshard_experts.py --rank R --out $WORK/shards --partner PARTNER_HOST --partner-out $WORK/shards --layers 0-47 --par 8 --min-free-gb 4`

This is the command line we used on every rank, started in the background with `nohup` and its output in a log file. We did not use `--rate`. `$WORK/shards` is the same path on the partner. If one rank stalls (on one box a dead IPv6 route did, before we forced IPv4), a second process for a layer range of the same rank (for example `--layers 24-47`) can catch up; the done-markers keep the work separate.

The checkpoint stores the routed experts striped over 4 files per layer. Under EP2xTP2, rank r = 2g + h needs the experts of group g and only half h of their intermediate channels. The two ranks of a pair split the experts of their group between them, download each expert as one HTTP byte range, cut both halves, keep their own and send the other half to the partner over ssh. Each box stores 73.7 GB. Whole files would be 294 GB for the experts.

The script forces IPv4, checks `Content-Range` and the exact length of every range, is idempotent (done-markers per file) and waits while free disk is below `--min-free-gb` (default 8). In our run it ran at about 7 MB/s per rank.

Then, on every box, build the model directory:

`python3 scripts/prepare_model_dir.py --out $WORK/model`

It downloads config, tokenizer, chat template, index, the 24 non-expert files and `mtp.safetensors` (17.4 GB). With `--reuse DIR` it hardlinks identical files from an earlier copy.

### 3. Check the shards

`python3 scripts/check_shards.py --dir $WORK/shards --rank R --list shards-rR.tsv --sha`

Every (layer, file) must have both halves, each a valid safetensors file with 216 tensors. The two ranks of a pair hold different experts (they split the experts of a group by parity) under the same file names, so the lists of two ranks differ and are not meant to be compared. The command ends with `problems 0`.

Optional equality proofs, which we ran:

- `python3 scripts/reshard_experts.py --verify-file L,F --verify-base DIR` downloads one whole original file, checks its sha256 against the Hugging Face LFS hash and compares what you cut from it with the range result. For this, run step 2 once per rank with `--no-send --layers L-L --out DIR/rR`. We did layer 5, files 0 to 3: sha256 matched, 0 deviations.
- `scripts/test_slice.py` (run inside the NGC image, needs torch and safetensors) compares the shard tensors with what the author's loader reads from the original file with `get_slice`: `python3 scripts/test_slice.py --orig model-experts-05-00.safetensors --shards SHARD_DIR_OF_RANK --rank R --layer 5 --file 0`. We ran it on all four ranks at three positions (layer 5 file 0, layer 47 file 3, layer 0 file 1): 0 deviations.

### 4. Get the sources and build

Three inputs go into `$WORK`. These are the commands we used to create them.

```
git clone https://github.com/0xSero/inkling2-exl3 src-author
cd src-author
git checkout 5cf1465d22205349b0ec3d8e54908f0467bcc2c6
git apply /path/to/this-repo/patches/0003-expert-shards-and-cache-drop.patch
git apply /path/to/this-repo/patches/0004-multinode-localrank-server-compat.patch
tar czf $WORK/repo.tgz overlay patches server bench scripts docker Dockerfile
cd ..

git clone https://github.com/turboderp-org/exllamav3 exllamav3-up
(cd exllamav3-up && git archive v1.6.6 --format=tar | gzip > $WORK/exllamav3-v1.6.6.tgz)

mkdir fi && cd fi && git init -q .
git remote add origin https://github.com/flashinfer-ai/flashinfer.git
git config core.sparseCheckout true
printf 'flashinfer/experimental/b12x\nREADME.md\n' > .git/info/sparse-checkout
git fetch -q --depth 1 origin 9c50996794564e1e3b2a4b0683e9aa2bcc528df8
git checkout -q FETCH_HEAD
tar czf $WORK/flashinfer-b12x.tgz flashinfer/experimental/b12x
```

`repo.tgz` holds the author's tree with patches 0003 and 0004 applied (we applied them to a clone at 5cf1465d; `git apply --check` is clean for both, in this order). The exllamav3 archive is the full tree of v1.6.6. The FlashInfer tree is a sparse checkout (595 entries in the archive), not the full tree.

The sparse tree lacks the root file `_flashinfer_filelock.py`, which the b12x kernels need. We added it from the same commit on every box:

```
curl -sf https://raw.githubusercontent.com/flashinfer-ai/flashinfer/9c50996794564e1e3b2a4b0683e9aa2bcc528df8/_flashinfer_filelock.py -o _flashinfer_filelock.py
```

The file is 2800 bytes (sha256 `4d9bdc7c98401cedc9fc6437136e5b2b07e20ceb3c3d84c88265f8ae7c7f0e37`). Put it at `$WORK/flashinfer/_flashinfer_filelock.py` after the build has extracted the archive. The author's own Dockerfile fetches the complete FlashInfer tree at the same commit. We did not use that variant. On our boxes we also created two small files in `$WORK/flashinfer`:

```
touch $WORK/flashinfer/flashinfer/__init__.py
echo 9c50996794564e1e3b2a4b0683e9aa2bcc528df8 > $WORK/flashinfer/COMMIT
```

The `__init__.py` is empty and does not come from FlashInfer (the upstream file at that commit has 30,029 bytes and is not used). It makes `flashinfer` an importable package next to `flashinfer.experimental.b12x`. Whether the server needs it without it (as a namespace package): not tested.

Then run, on rank 0:

`WORK=$WORK bash scripts/build.sh`

It extracts the sources, applies the author's `patches/upstream-modifications.patch` to exllamav3, copies his overlay, installs the Python dependencies at the versions of his lock files into `$WORK/site-deps` (without `nvidia-nccl-cu13`, the NGC torch brings its own NCCL) and builds exllamav3 with `TORCH_CUDA_ARCH_LIST=12.1a` and `MAX_JOBS=6` into `$WORK/site`. Nothing is installed into the image or the host. `build.sh` from zero took 5 min 10 s on one idle box (rank 0) and the result loads (see "What we tested with these scripts"). For the speed numbers in this README the compiled extension of the earlier build was reused: the compiled inputs (exllamav3 v1.6.6 and its patch, the dependency locks, the FlashInfer commit) did not change between 8f12aa4e and 5cf1465d, only the Python overlay did.

The server runs from `$WORK/src/repo` on every box, so `src/repo`, `flashinfer` and the model directory must exist on all four. How we spread the build over the four boxes (we built once and copied it, one after the other over the LAN):

```
cd $WORK && for h in $NODE1 $NODE2 $NODE3; do
  tar cf - site site-deps flashinfer src/repo | ssh $h "mkdir -p $WORK && tar xf - -C $WORK"
done
```

The `exllamav3_ext` extension is then a copy of the rank 0 build on the other boxes. We did the `touch`, `COMMIT` and `_flashinfer_filelock.py` steps above on all four boxes after the copy; doing them on rank 0 before the copy gives the same result. We did not compare the four trees after copying beyond file times. The build runs inside the NGC container, so the different OS states of our boxes (see System state) only matter through Docker and the driver.

Two things we ran into with this build:

- The b12x kernels need `_flashinfer_filelock.py` from the FlashInfer root (see above). Without it the probe failed.
- The `.pth` file of `nvidia-cutlass-dsl` is not read through `PYTHONPATH`. Without the `dsl_packages` directory on `PYTHONPATH`, the image loads its own older cutlass-dsl and the kernel compile fails with `LaunchConfig preferred_smem_carveout`. `scripts/start_server.sh` sets it.

Before the first start we checked the b12x kernels on GB10 with `scripts/probe_b12x.sh` (layer 5, 192 experts, rank 0, needs the shards of layer 5): the kernel compiled in 13.3 s, and the outputs were finite and not zero for T = 1, 8, 64 and 200. It is a diagnostic, not a measurement.

### 5. Start the server

On NODE0, with `NODE0` to `NODE3`, `MASTER_ADDR`, `IFACE` and `WORK` set:

```
bash scripts/start_server.sh box-check
bash scripts/start_server.sh up
bash scripts/start_server.sh logs
```

`up` starts one container per box (`--gpus all --network host --ipc host --device /dev/infiniband --cap-add IPC_LOCK`, one GPU each). Per rank it passes the ring env file and NCCL graph file from step 1 and the following settings:

- `RANK=R WORLD_SIZE=4 LOCAL_RANK=0 MASTER_ADDR=... MASTER_PORT=29541`
- `I2_B12X=1 I2_EP_LAYOUT=ep2tp2 I2_DRAFT_FP8=1 I2_MOE_GROUP=64 EXL3_LOAD_ARENA=0` (the author's values)
- `I2_PC_SHARD=0 I2_PEER_AR=0`: the author's token-sharded prefill and peer all-reduce copy data between the GPUs with CUDA IPC, which only works inside one host. On four hosts they fail at `cudaIpcOpenMemHandle` and leave the CUDA context unusable (no fallback to NCCL), so they must stay off.
- `NCCL_CUMEM_ENABLE=1`
- `I2_EXPERT_SHARD_DIR=/work/shards`, `B12X_COMPILE_CACHE_DIR=/work/jitcache` (keep it on a persistent path), `MODEL_DIR=/model`
- `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` set to `IFACE`
- `PYTHONPATH` with `site`, the cutlass-dsl `dsl_packages` directory, `site-deps`, `flashinfer` and its b12x `_compat` directory

The server command is `server/i2_server.py --model /model --port 8891 --b12x 1 --kv-gb 24`. Rank 0 serves the OpenAI-compatible API on port 8891. Wait until `http://NODE0:8891/health` answers (`{"status":"ok","stats":{...}}`). The log of rank 0 shows `b12x fused Trellis MoE attached for verify T=[2, 3, 4, ..., 4096] in 561s`, `KV pages: 142582 x 64 = 9,125,248 tokens (per GPU 24.0 GiB, 2824 B/token)`, then `speculative graphs captured for (bucket, width) [...]` and finally `serving on http://0.0.0.0:8891 (spec decoding ON, slots 16, buckets [1, 2, 4, 8, 16])`. The first start compiles the b12x kernels for every batch size (561 s of the 629 s in our start with an empty cache directory). A start with a filled cache directory took 641 s, not less.

### 6. Check and measure

Run the smoke test on NODE0 inside the same image (`tokenizers` is part of the NGC image, nothing is installed):

```
mkdir -p $WORK/gate-out && cp scripts/smoke_test.py $WORK/
docker run --rm --network host --entrypoint python -v $WORK:/work nvcr.io/nvidia/pytorch:26.07-py3 /work/smoke_test.py --tok /work/model/tokenizer.json --out /work/gate-out/smoke.jsonl --long 120000
```

`--skip5` leaves out the long needle test. The test sends a needle test at 8K, the same request next to 7 others, a stream with a usage chunk, 16 concurrent requests and a long needle test. All passed on our boxes. It takes no speed figures.

`python3 scripts/served_sweep.py --url http://NODE0:8891 --out OUT_DIR` runs the served decode sweep of the next section.

## What differs from the author's setup

1. Hardware and processes: one GPU per host and four hosts, NCCL over the CX7 ring. The author runs 4 GPUs in one host with `torchrun`. We run one container per host with `RANK`, `WORLD_SIZE`, `MASTER_ADDR` set by hand and `LOCAL_RANK=0` for `set_device` (patch 0004).
2. `NCCL_CUMEM_ENABLE=1` instead of the author's `0`. With `0`, NCCL also connects tree channels at start, including rank 0 to rank 2. They are not neighbours in a switchless ring, there is no direct link, and the start fails with an `ibv_modify_qp` timeout. With `1`, only the ring channels are connected, lazily (`NCCL_ALGO=Ring`).
3. Weights per rank: the author's weights are unchanged. Each rank loads only the slice of the experts it needs, cut byte for byte from the author's files (step 2). Reason: 294 GB of experts per box does not fit.
4. Loader (patch 0003): reads these shards (`I2_EXPERT_SHARD_DIR`), drops the page cache of each file after use and closes the safetensors handles after each layer. Without closing the handles, the loader kept a CPU-side copy of every tensor it had read until it finished. On GB10's unified memory that ran out of memory (exit 137) while loading. With the fix, the anonymous memory of the process stayed under 2 GiB at the same GPU allocation.
5. KV budget: `--kv-gb 24` instead of the automatic size (free memory minus reserve), because the automatic size is unreliable on unified memory. The author's pool on 96 GB cards is 6,984,448 tokens in the earlier README and 6,162,368 tokens in the current one. We did not lower `--kv-gb` for the new state.
6. Server (patch 0004): `LOCAL_RANK` for `set_device` (also in `bench/i2_sweep.py`), `chat_template_kwargs.enable_thinking=false` mapped to `reasoning_effort: none`, and token-id prompts for `/v1/completions`. The last two let our benchmark harness talk to the server. (At 8f12aa4e the patch also added a `usage` chunk to the stream; the author's server sends usage itself now.)
7. Token-sharded prefill and peer all-reduce are off (`I2_PC_SHARD=0`, `I2_PEER_AR=0`). They are the author's main prefill speed-up (about 2x in his README) and need CUDA IPC between GPUs of one host. We could not use them on four hosts; the other 5cf1465d changes (4096-row prefill chunks, CUDA prefill attention, mixed prefill/decode steps, adaptive verification) are on, with the author's defaults (`--chunk 4096`, `--reserve-gb 2.75`, `--mixed 1`, `--adaptive 1`, `--state-slots 24`).
8. Build: no image of the author's for GB10. The same components are built in the NGC PyTorch image for sm_121a instead of the author's torch 2.10 / CUDA 12.8 image for sm_120.

## What the patches do

- `0003-expert-shards-and-cache-drop.patch` (`overlay/exllamav3/modules/inkling2_b12x.py`): when `I2_EXPERT_SHARD_DIR` is set, expert tensors are read from the rank-local shards. They carry the checkpoint's key names with the slice already applied, so nothing is sliced at load. Keys that are not in the shards (router and so on) still come from the model directory. After each layer the page cache of the files that were read is dropped and the handles are closed.
- `0004-multinode-localrank-server-compat.patch` (`server/i2_server.py`, `bench/i2_sweep.py`): the three changes in item 6 above.

## Results

Speed measured on 2026-11-15, at upstream commit 5cf1465d, on four DGX Spark with the 4.0 bpw quant. Setup: EP2xTP2 experts with b12x kernels, FP8 draft, speculative decoding always on with adaptive verification, 16 slots, `--kv-gb 24`, token-sharded prefill and peer all-reduce off. One setup, one date. The raw outputs are not part of this repository.

Speed, from our own benchmark harness (it is not part of this repository). 512 tokens per run, temperature 0, reasoning effort none. The "earlier state" column is the same setup at 8f12aa4e (chunk 512, no adaptive verification):

| | 5cf1465d | earlier state 8f12aa4e |
|---|---:|---:|
| Decode, code | 55.6 tok/s | 48.6 |
| Decode, prose | 61.0 tok/s (spread 14.7 %) | 61.1 |
| Decode, all | 58.3 tok/s | 54.9 |
| Prefill at 2K (2,138 to 2,264 tokens) | 1,681 tok/s, first token 1.3 s | 928 |
| Prefill at 8K (8,838 to 9,103 tokens) | 2,266 tok/s, first token 4.0 s | 1,011, 8.5 s |
| Prefill at 32K (34,371 to 35,460 tokens) | 2,355 tok/s, first token 15.0 s | 928, 38.3 s |
| Prefill at 64K (66,917 to 70,834 tokens) | 2,291 tok/s, first token 29.9 s | 834, 82.5 s |
| Prefill at 128K (141,774 to 145,797 tokens) | 2,167 tok/s, first token 65.4 s | 684, 207 s |
| 1 / 2 / 4 / 8 requests at once, in total | 52.4 / 75.0 / 100.2 / 126.2 tok/s | 50.4 / 67.6 / 84.5 / 106.8 |

The context sizes are nominal. The prompt token counts in the rows are what the server reported (`usage.prompt_tokens`, three runs per row, the server counts 8 to 12 % more tokens than the nominal size for the same text) and the tok/s values are computed on them. The first-token times are medians. The author's README also gives a first token at about 300K of 41.5 s; we did not measure that.

Served decode, in the way of the author's README: temperature 0, the default chat template (thinking on), tok/s from the server's `/health` counters with prefill excluded, 16 prompts with up to 2,048 new tokens each. The author's 16 prompts are not in his repository, so we wrote 16 chat, code and agent prompts of our own (`scripts/served_sweep.py`). Author numbers are from his current README (commit 5cf1465d, measured on 2026-11-15 on four RTX PRO 6000 cards in one host):

| Streams | Four DGX Spark, 5cf1465d | Four DGX Spark, earlier state | Author, 4 x RTX PRO 6000 |
|---|---:|---:|---:|
| 1 | 52.6 tok/s | 49.9 | 220 |
| 2 | 78.9 | 71.6 | 328 |
| 4 | 103.7 | 95.6 | 421 |
| 8 | 131.6 | 123.7 | 540 |
| 16 | 167.4 | 165.6 | 697 |

Four Sparks land at about a quarter of the author's numbers at every level (24 to 25 %). The draft gets 2.84 tokens through per step at one stream, but each step takes longer on this hardware. All 80 requests completed without error; at 16 streams one request ended at 1,871 instead of 2,048 tokens. The author's prefill numbers (8,210 / 7,722 / 7,100 tok/s at 8K / 64K / 128K, one host, token-sharded prefill) are not comparable to ours: we ran without token-sharded prefill and over RDMA instead of inside one host.

Quality at reasoning effort none, fixed question lists, temperature 0, seed 42 (measured at 8f12aa4e, not repeated at 5cf1465d):

| | Result | 95 % margin | Questions |
|---|---:|---:|---:|
| MMLU-Pro | 68.5 % | 4.5 points | 400 |
| GPQA Diamond | 58.6 % | 6.8 | 198 |
| IFEval | 77.0 % | 5.8 | 200 |
| LiveCodeBench | 80.0 % | 12.1 | 40 |
| RULER-lite 8K | 77.5 % | 12.6 | 40 |
| RULER-lite 32K | 70.0 % | 13.7 | 40 |

The model still writes its thinking first at reasoning effort none. MMLU-Pro and GPQA therefore got a limit of 4,096 tokens per answer, the same as the server's default. Answers cut at a limit count as wrong: 49 of 198 on GPQA, 13 of 400 on MMLU-Pro, and 7 of 40 on LiveCodeBench (limit 32,768 tokens). We did not measure the unquantized original.

For context, our first build on the same four boxes, with the 3.48 bpw quant, our own server and no speculative decoding, ran at 22.3 tok/s decode.

## Known limits

- The model thinks first. `reasoning_effort: "none"` (also `enable_thinking: false` with patch 0004) selects the lowest level, 0.0. It does not switch thinking off, the model still reasons visibly in `reasoning_content`, and short answer limits cut answers off. Thinking is on by default in the author's chat template (`reasoning_effort` 0.9). Because of this, "about 500 words in X seconds" cannot be computed from `content`.
- Token-sharded prefill is off. It needs CUDA IPC between the GPUs and therefore one host; on four hosts it fails and breaks the CUDA context, so our prefill is about 3.3 times lower than the author's (2,167 against 7,100 tok/s at 128K), and we have no way to say what the author's prefill would give on four Sparks.
- The first token still takes long for long prompts (15.0 s at 32K, 65.4 s at 128K). The first token at about 300K was not measured.
- Free memory per box is small once the server is up: 6.3 to 8.6 GB available right after start and 5.1 to 6.6 GB at the lowest point during the runs (right after start about 3 to 4 GB less than with 8f12aa4e). Keep other work off the boxes.
- Starting the server takes about ten minutes (kernel compile). The cache directory `B12X_COMPILE_CACHE_DIR` is set, but we measured no faster start with it (629 s empty, 641 s filled).
- The decode prose figure has a spread of 14.7 % between runs (code 3.1 %).
- Quality was measured at the earlier state 8f12aa4e. We did not repeat it at 5cf1465d.
- The author's quality check (KLD against BF16) is not in his repository and we did not run it.

## What we tested with these scripts

On 2026-11-15, with a fresh copy of this repository on four DGX Spark, the scripts exactly as they are here (placeholders only through environment variables): `scripts/rdma_check.sh` (8 cables 109.3 Gb/s), `scripts/ring/ringenv.py` (byte-identical to the files we ran the measurements with), the commands of step 4 (git clone, both patches apply, tar archives), `scripts/check_shards.py` on all four ranks (`problems 0`), `scripts/start_server.sh box-check` and `up` (ready after 657 s), `scripts/smoke_test.py` (T1 to T5 passed), `scripts/served_sweep.py --levels 1` (52.3 tok/s, 0 errors), `scripts/start_server.sh down`. Then `scripts/build.sh` from zero on one box (5 min 10 s), the probe `scripts/probe_b12x.sh` on that build (`PROBE OK`) and a second server start on all four boxes with that build (ready after 635 s, `smoke_test.py --skip5`: T1 to T4 passed), then `down`. Two scripts did not run on a host that has no ssh key to itself (`rdma_check.sh`, `ringenv.py`); both now run locally for the host they are started on.

## Not tested

- The download from zero (`scripts/reshard_experts.py`, `scripts/prepare_model_dir.py`) and `scripts/test_slice.py` with this repository state: they ran on our boxes for the first build (8f12aa4e); the shards and the model directory used for the tests above are those files, hard-linked into the test directory.
- The build commands on boxes that are not ours. Nobody compared the four boxes after copying the build beyond file times.
- `build.sh` on each of the four boxes (we ran it on one and copied the result), the server without the empty `flashinfer/__init__.py`, and the full FlashInfer tree of the author's Dockerfile (we used the sparse tree).
- Boxes with a different OS state than ours (see System state).
- Two or three Sparks.
- Other cabling or a switch.
- Other quants of this model with this engine and the patches (we ran only 4.0 bpw), and the automatic KV size.
- Image input. We only used text.
- Temperature 0.7 and thinking on for quality.

## Credits

- 0xSero: the quant (`0xSero/Inkling-2-Small-EXL3-4bpw`), the engine and server (`0xSero/inkling2-exl3`).
- Thinking Machines: the Inkling-2-Small model.
- Local Inference Lab: the b12x fused Trellis MoE kernels (now in FlashInfer, `flashinfer/experimental/b12x`).
- turboderp: exllamav3 and the EXL3 format.
- MiaAI-Lab (Mia's AI Lab): the ring discovery and the NCCL ring settings in `scripts/ring/` (from `scripts/tp4.sh` of `github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104`, Copyright 2026 Mia's AI Lab, Apache-2.0). That file in turn adapts the lab's `GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold` recipe.

## License information

- Engine `0xSero/inkling2-exl3`: MIT License, Copyright (c) 2026 0xSero (LICENSE in that repository).
- Its `NOTICE.md` lists exllamav3 (MIT License, Copyright (c) 2025 Turboderp), b12x and FlashInfer (Apache License 2.0) and NVIDIA CUTLASS DSL (installed from PyPI under NVIDIA's license). It states: "Inkling-2-Small model by Thinking Machines, Apache-2.0. The EXL3 quant weights are a derivative under the same license." The model card of the quant says `license: apache-2.0`.
- `scripts/ring/` contains code taken from `MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104`, Apache License 2.0, Copyright 2026 Mia's AI Lab (see Credits and the headers of those files).
- This repository: Apache License 2.0 (`LICENSE`), with the notices in `NOTICE.md`.
