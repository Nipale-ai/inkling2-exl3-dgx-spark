#!/usr/bin/env python3
"""T-slice: the shard tensors of one rank must equal what the author's loader (overlay inkling2_b12x.py, B12xMoEShard) reads from the
ORIGINAL checkpoint file with get_slice. Runs inside the NGC image (torch + safetensors).

usage: test_slice.py --orig model-experts-05-00.safetensors --shards DIR_OF_THIS_RANK --rank R --layer 5 --file 0
Loader semantics copied from overlay/exllamav3/modules/inkling2_b12x.py (B12xMoEShard.__init__, tp = 2):
  g, h = rank // 2, rank % 2 ; n16 = 72 ; Ih = 1152
  gate/up trellis  : get_slice(key)[:, h*n16:(h+1)*n16]      down trellis : get_slice(key)[h*n16:(h+1)*n16]
  gate/up svh, down suh : [h*Ih:(h+1)*Ih]                    gate/up suh, down svh : full
"""
import argparse, glob, os, sys
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--orig", required=True); ap.add_argument("--shards", required=True)
ap.add_argument("--rank", type=int, required=True); ap.add_argument("--layer", type=int, required=True); ap.add_argument("--file", type=int, required=True)
a = ap.parse_args()
g, h = a.rank // 2, a.rank % 2
n16, Ih = 72, 1152
E = 384; lo, hi = g * E // 2, (g + 1) * E // 2
orig = safe_open(a.orig, "pt", device="cpu")
exp = [e for e in range(lo, hi) if e % 4 == a.file]
files = sorted(glob.glob(os.path.join(a.shards, f"s-{a.layer:02d}-f{a.file}-d*.safetensors")))
got = {}
for fn in files:
    with safe_open(fn, "pt", device="cpu") as fh:
        for k in fh.keys(): got[k] = fh.get_tensor(k)
bad = n = 0
for e in exp:
    p = f"model.llm.layers.{a.layer}.mlp.experts.{e}."
    spec = {
        "gate_proj.trellis": (slice(None), slice(h * n16, (h + 1) * n16)), "up_proj.trellis": (slice(None), slice(h * n16, (h + 1) * n16)),
        "down_proj.trellis": (slice(h * n16, (h + 1) * n16),),
        "gate_proj.svh": (slice(h * Ih, (h + 1) * Ih),), "up_proj.svh": (slice(h * Ih, (h + 1) * Ih),), "down_proj.suh": (slice(h * Ih, (h + 1) * Ih),),
        "gate_proj.suh": None, "up_proj.suh": None, "down_proj.svh": None,
    }
    for name, sl in spec.items():
        key = p + name
        want = orig.get_slice(key)[sl] if sl is not None else orig.get_tensor(key)
        n += 1
        if key not in got or got[key].shape != want.shape or got[key].dtype != want.dtype or not torch.equal(got[key], want):
            bad += 1; print("DEVIATES", key, None if key not in got else (got[key].shape, want.shape))
extra = [k for k in got if not any(k.startswith(f"model.llm.layers.{a.layer}.mlp.experts.{e}.") for e in exp)]
print(f"T-slice rank {a.rank} layer {a.layer} file {a.file}: experts {len(exp)}, tensors compared {n}, deviations {bad}, extra keys in shards {len(extra)}, shard files {len(files)}")
sys.exit(0 if bad == 0 and not extra else 1)
