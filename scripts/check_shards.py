#!/usr/bin/env python3
"""Post-download check of one rank's expert shard directory (stdlib only): every (layer, file) has both halves (d<rank>, d<partner>),
each a valid safetensors file (header length + max data offset == file size), 216 tensors (24 experts x 9: the pair splits the 48 experts of the group in this file by parity), all done-markers present;
writes name, size, sha256 to --list (the two ranks of a pair hold different experts under the same file names, so their lists differ).
usage: check_shards.py --dir SHARDS --rank R [--list OUT.tsv] [--sha]"""
import argparse, hashlib, json, os, struct, sys
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--list", default=""); ap.add_argument("--sha", action="store_true"); a = ap.parse_args()
partner = a.rank ^ 1; bad = []; rows = []
def sha(p):
    h = hashlib.sha256(); n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b: break
            h.update(b); n += len(b)
            if n >= (512 << 20): os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED); n = 0
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()
def check(name):
    p = os.path.join(a.dir, name); size = os.path.getsize(p)
    with open(p, "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(hl))
    hdr.pop("__metadata__", None)
    end = max(v["data_offsets"][1] for v in hdr.values())
    msgs = []
    if 8 + hl + end != size: msgs.append(f"size {size} != header-implied {8+hl+end}")
    if len(hdr) != 216: msgs.append(f"{len(hdr)} tensors, expected 216")
    return name, size, (sha(p) if a.sha else ""), msgs
names = []
for L in range(48):
    for F in range(4):
        if not os.path.exists(os.path.join(a.dir, f".done-{L:02d}-{F}")): bad.append(f"missing done-marker {L:02d}-{F}")
        for d in (a.rank, partner):
            n = f"s-{L:02d}-f{F}-d{d}.safetensors"
            if os.path.exists(os.path.join(a.dir, n)): names.append(n)
            else: bad.append("missing " + n)
extra = sorted(set(f for f in os.listdir(a.dir) if f.endswith(".safetensors")) - set(names))
if extra: bad.append(f"unexpected files {extra[:5]}")
with ThreadPoolExecutor(4) as ex:
    for name, size, h, msgs in ex.map(check, names):
        rows.append((name, size, h)); bad += [f"{name}: {m}" for m in msgs]
if a.list: open(a.list, "w").write("".join(f"{n}\t{s}\t{h}\n" for n, s, h in sorted(rows)))
tot = sum(s for _, s, _ in rows)
print(f"rank {a.rank}: {len(rows)} files, {tot/1e9:.2f} GB, problems {len(bad)}" + ("" if not bad else "\n" + "\n".join(bad[:20])))
sys.exit(1 if bad else 0)
