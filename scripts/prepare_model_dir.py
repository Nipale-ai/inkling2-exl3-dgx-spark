#!/usr/bin/env python3
"""Build the model dir for the 4.0 bpw checkpoint on one box (stdlib only).
 - small files (config, tokenizer, chat template, index, ...): download from the pinned HF revision, size checked against the HF listing
 - non-expert shards and mtp.safetensors (17.4 GB): hardlink from an existing copy with the SAME size and sha256 (--reuse DIR), else download
 - expert tensors are NOT here: they are the rank-local shards written by reshard_experts.py (I2_EXPERT_SHARD_DIR)
usage: prepare_model_dir.py --out DIR [--reuse DIR_WITH_SAME_FILES]
"""
import argparse, hashlib, json, os, sys, urllib.request, socket
_g = socket.getaddrinfo   # IPv4 only (black-holed IPv6 route to the CDN on some hosts)
socket.getaddrinfo = lambda h, p, family=0, type=0, proto=0, flags=0: _g(h, p, socket.AF_INET, type, proto, flags)

REPO = "0xSero/Inkling-2-Small-EXL3-4bpw"; REV = "e58fa769cd946a7df2e2bbd5bec790290549b150"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--reuse", default=""); ap.add_argument("--verify-sha", action="store_true")
a = ap.parse_args()
api = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true"))
os.makedirs(a.out, exist_ok=True)

def sha256(p):
    h = hashlib.sha256(); n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b: break
            h.update(b); n += len(b)
            if n >= (256 << 20):                      # keep the page cache small (unified memory, memory guard)
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED); n = 0
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()

def fetch(name, dest):
    req = urllib.request.Request(f"https://huggingface.co/{REPO}/resolve/{REV}/{name}")
    with urllib.request.urlopen(req, timeout=300) as r, open(dest + ".part", "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b: break
            f.write(b)
    os.replace(dest + ".part", dest)

bad = 0
for s in api["siblings"]:
    n, size = s["rfilename"], s["size"]
    if "experts" in n or n == ".gitattributes": continue
    dest = os.path.join(a.out, n)
    if os.path.exists(dest) and os.path.getsize(dest) == size: print("have", n); continue
    src = os.path.join(a.reuse, n) if a.reuse else ""
    if src and os.path.exists(src) and os.path.getsize(src) == size and (not ("nonexpert" in n or n == "mtp.safetensors") or not a.verify_sha or sha256(src) == s.get("lfs", {}).get("sha256")):
        if os.path.exists(dest): os.remove(dest)
        os.link(src, dest); print("hardlink", n); continue
    fetch(n, dest); ok = os.path.getsize(dest) == size
    if ok and "lfs" in s and a.verify_sha: ok = sha256(dest) == s["lfs"]["sha256"]
    print("download", n, "OK" if ok else "BAD"); bad += 0 if ok else 1
sys.exit(1 if bad else 0)
