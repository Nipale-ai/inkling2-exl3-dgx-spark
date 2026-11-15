#!/usr/bin/env python3
"""Range-download + reshard of 0xSero/Inkling-2-Small-EXL3-4bpw for the EP2xTP2 expert layout (stdlib only, no numpy/torch).

Why: the checkpoint stores the routed experts striped over 4 files per layer (expert id % 4 = file number), but in the EP2xTP2 layout
rank r = 2*g + h needs, per layer, the experts [g*E/2, (g+1)*E/2) (spread over all 4 files) and only the h-th half of each expert's
intermediate channels (the loader of the engine slices gate/up trellis along dim 1 and down trellis along dim 0). Downloading whole
files would need 4x the bytes and disk. Instead the two ranks of a pair (g,0) and (g,1) split the group's experts between them
(parity of expert_id // 4), each downloads its experts as plain HTTP byte ranges (one contiguous ~16 MB range per expert, plus the
small tensors of the file), cuts both halves, keeps its own half and streams the partner's half to the partner over ssh.
Result on every rank: ~73.7 GB of already sliced expert tensors, same key names as the checkpoint (loader patch 0003 reads them).

usage: reshard_experts.py --rank R [--world 4 --groups 2] --out DIR --partner HOST --partner-out DIR [--layers 0-47] [--par 6] [--rate 2400k]
env:   HF_REV (default: pinned revision below)
"""
import argparse, json, os, struct, sys, threading, time, subprocess, urllib.request, urllib.error, concurrent.futures as cf, hashlib

# IPv4 only: on some hosts the IPv6 route to the CDN is a black hole and every connect waits for its timeout before the IPv4 fallback
import socket as _socket
_orig_gai = _socket.getaddrinfo
_socket.getaddrinfo = lambda host, port, family=0, type=0, proto=0, flags=0: _orig_gai(host, port, _socket.AF_INET, type, proto, flags)
REPO = "0xSero/Inkling-2-Small-EXL3-4bpw"
REV_DEFAULT = "e58fa769cd946a7df2e2bbd5bec790290549b150"
E_ALL, H, I = 384, 4608, 2304

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, default=0); ap.add_argument("--world", type=int, default=4); ap.add_argument("--groups", type=int, default=2)
ap.add_argument("--out", default=".", help="own shard dir"); ap.add_argument("--partner", help="ssh host of the pair partner (receives its half)")
ap.add_argument("--partner-out", help="shard dir on the partner"); ap.add_argument("--layers", default="0-47")
ap.add_argument("--par", type=int, default=6); ap.add_argument("--rate", default="", help="limit, e.g. 2400k (bytes/s per process, approximate)")
ap.add_argument("--verify-file", default="", help="L,f : download the whole file, check sha256 against HF, compare with the range method output in --verify-base/r<rank>/")
ap.add_argument("--verify-base", default="", help="dir holding r0..r3 (range method run with --no-send per rank)")
ap.add_argument("--check-headers", action="store_true", help="check size/offsets of all files of --layers against the HF file size")
ap.add_argument("--min-free-gb", type=float, default=8.0, help="pause while the free space of --out is below this")
ap.add_argument("--no-send", action="store_true", help="test mode: also keep the partner half locally in OUT/partner/")
a = ap.parse_args()
REV = os.environ.get("HF_REV", REV_DEFAULT)
tp = a.world // a.groups
g, h = a.rank // tp, a.rank % tp
lo, hi = g * E_ALL // a.groups, (g + 1) * E_ALL // a.groups
Ih = I // tp
n16 = Ih // 16                                   # 72 (slice width in 16-tiles)
lay_a, lay_b = (int(x) for x in a.layers.split("-")); LAYERS = range(lay_a, lay_b + 1)
os.makedirs(a.out, exist_ok=True)
LOCK = threading.Lock(); BYTES = [0]; T0 = time.time()

def log(*x): print(time.strftime("%H:%M:%S"), f"[r{a.rank}]", *x, flush=True)

def parse_rate(s):
    if not s: return 0
    return int(float(s[:-1]) * (1024 if s[-1] in "kK" else 1024 * 1024)) if s[-1] in "kKmM" else int(s)
RATE = parse_rate(a.rate)

def throttle(nbytes):
    with LOCK:
        BYTES[0] += nbytes
        if RATE:
            ahead = BYTES[0] / RATE - (time.time() - T0)
        else: ahead = 0
    if ahead > 0: time.sleep(min(ahead, 5))

def get_range(url, start, end, tries=8):
    """bytes [start, end] inclusive; exact length is checked"""
    want = end - start + 1
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "inkling-reshard/1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status not in (200, 206): raise IOError(f"status {r.status}")
                cr = r.headers.get("Content-Range", "")
                if r.status == 206 and not cr.startswith(f"bytes {start}-{end}/"): raise IOError(f"content-range {cr!r}")
                buf = bytearray()
                while True:
                    c = r.read(1 << 20)
                    if not c: break
                    buf += c; throttle(len(c))
                if len(buf) != want: raise IOError(f"short read {len(buf)} != {want}")
                return bytes(buf)
        except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as e:
            if t == tries - 1: raise
            time.sleep(min(5 * (t + 1), 30))

def fname(L, f): return f"model-experts-{L:02d}-{f:02d}.safetensors"
def url_of(L, f): return f"https://huggingface.co/{REPO}/resolve/{REV}/{fname(L, f)}"

def read_header(url):
    n = struct.unpack("<Q", get_range(url, 0, 7))[0]
    hdr = json.loads(get_range(url, 8, 8 + n - 1)); hdr.pop("__metadata__", None)
    return hdr, 8 + n

def st_bytes(tensors):
    """tensors: dict key -> (dtype, shape, bytes). returns safetensors file bytes (keys sorted, dtype grouping not needed by readers)"""
    meta, off, parts = {}, 0, []
    for k in sorted(tensors):
        dt, shape, b = tensors[k]
        meta[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(b)]}; off += len(b); parts.append(b)
    hb = json.dumps({"__metadata__": {"format": "pt"}, **meta}, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    return struct.pack("<Q", len(hb)) + hb + b"".join(parts)

def halves(L, e, small, trel, hdr, base_off, f_url, fetch_small, hh):
    """tensors of expert e for intermediate half hh, as (dtype, shape, bytes) under the checkpoint key names"""
    p = f"model.llm.layers.{L}.mlp.experts.{e}."
    out = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        meta = hdr[p + proj + ".trellis"]; shape = meta["shape"]; s, t = meta["data_offsets"]
        raw = trel[(p + proj + ".trellis")]
        if proj == "down_proj":                       # [I/16, H/16, 16K]: half = rows [hh*n16, (hh+1)*n16) -> contiguous
            rb = len(raw) // shape[0]
            out[p + proj + ".trellis"] = ("I16", [n16, shape[1], shape[2]], raw[hh * n16 * rb:(hh + 1) * n16 * rb])
        else:                                          # [H/16, I/16, 16K]: half = columns [hh*n16, (hh+1)*n16) of every row
            rb = len(raw) // shape[0]; cb = rb // shape[1]
            rows = [raw[r * rb + hh * n16 * cb: r * rb + (hh + 1) * n16 * cb] for r in range(shape[0])]
            out[p + proj + ".trellis"] = ("I16", [shape[0], n16, shape[2]], b"".join(rows))
    def sm(name):
        m = hdr[p + name]; s, t = m["data_offsets"]; return m, small(s, t)
    for name, sl in (("gate_proj.suh", None), ("up_proj.suh", None), ("down_proj.svh", None),
                     ("gate_proj.svh", 1), ("up_proj.svh", 1), ("down_proj.suh", 1)):
        m, b = sm(name)
        if sl: b = b[hh * Ih * 2:(hh + 1) * Ih * 2]; shape = [Ih]
        else: shape = m["shape"]
        out[p + name] = (m["dtype"], shape, b)
    return out

def process(L, f):
    mark = os.path.join(a.out, f".done-{L:02d}-{f}")
    if os.path.exists(mark): return 0
    url = url_of(L, f)
    hdr, base = read_header(url)
    trel_keys = sorted(k for k in hdr if k.endswith(".trellis"))
    small_end = min(hdr[k]["data_offsets"][0] for k in trel_keys)            # all I32/F16 tensors lie before the trellis block
    small_blob = get_range(url, base, base + small_end - 1)
    small = lambda s, t: small_blob[s:t]
    mine = sorted(e for e in range(lo, hi) if e % 4 == f and ((e // 4) % tp) == h)
    nb = 0
    def fetch(e):
        p = f"model.llm.layers.{L}.mlp.experts.{e}."
        ks = [p + "down_proj.trellis", p + "gate_proj.trellis", p + "up_proj.trellis"]
        a0 = hdr[ks[0]]["data_offsets"][0]; a1 = hdr[ks[-1]]["data_offsets"][1]
        assert hdr[ks[0]]["data_offsets"][1] == hdr[ks[1]]["data_offsets"][0] and hdr[ks[1]]["data_offsets"][1] == hdr[ks[2]]["data_offsets"][0], "trellis not contiguous"
        blob = get_range(url, base + a0, base + a1 - 1)
        return e, {k: blob[hdr[k]["data_offsets"][0] - a0: hdr[k]["data_offsets"][1] - a0] for k in ks}
    own, other = {}, {}
    with cf.ThreadPoolExecutor(a.par) as ex:
        for e, trel in ex.map(fetch, mine):
            nb += sum(len(v) for v in trel.values())
            own.update(halves(L, e, small, trel, hdr, base, url, None, h))
            other.update(halves(L, e, small, trel, hdr, base, url, None, 1 - h))
    name = f"s-{L:02d}-f{f}-d{a.rank}.safetensors"
    data = st_bytes(own); tmp = os.path.join(a.out, name + ".part")
    with open(tmp, "wb") as fh: fh.write(data)
    os.replace(tmp, os.path.join(a.out, name))
    pdata = st_bytes(other)
    if a.no_send:
        os.makedirs(os.path.join(a.out, "partner"), exist_ok=True)
        with open(os.path.join(a.out, "partner", name), "wb") as fh: fh.write(pdata)
    elif a.partner:
        cmd = ["ssh", "-o", "BatchMode=yes", a.partner, f"mkdir -p {a.partner_out} && cat > {a.partner_out}/{name}.part && mv {a.partner_out}/{name}.part {a.partner_out}/{name}"]
        r = subprocess.run(cmd, input=pdata, capture_output=True)
        if r.returncode: raise RuntimeError(f"send to partner failed: {r.stderr[-300:]!r}")
    open(mark, "w").write(f"{len(mine)} experts {nb} bytes {time.strftime('%F %T')}\n")
    return nb

def verify(L, f, base):
    """Equality proof for file (L,f): download the WHOLE original file, check sha256 against HF (LFS oid), derive every rank's shards
    from it (both halves, both parities, both groups) and compare byte for byte with what the range method wrote into
    BASE/r<rank>/ (own half) and BASE/r<rank>/partner/ (half meant for the partner). Prints the number of deviating files."""
    url = url_of(L, f)
    api = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true"))
    ent = [s for s in api["siblings"] if s["rfilename"] == fname(L, f)][0]
    lfs, size = ent["lfs"]["sha256"], ent["size"]
    log("full download", fname(L, f), size, "bytes, expect sha256", lfs)
    sh = hashlib.sha256(); parts = []; pos = 0; CH = 64 << 20
    while pos < size:
        b = get_range(url, pos, min(pos + CH, size) - 1); sh.update(b); parts.append(b); pos += len(b)
    whole = b"".join(parts); del parts
    ok_sha = sh.hexdigest() == lfs
    log("sha256", sh.hexdigest(), "MATCH" if ok_sha else "MISMATCH", "size", len(whole), "==", size, len(whole) == size)
    n = struct.unpack("<Q", whole[:8])[0]; hdr = json.loads(whole[8:8 + n]); hdr.pop("__metadata__", None); base_off = 8 + n
    bad = 0; checked = 0
    for g_ in range(a.groups):
        glo, ghi = g_ * E_ALL // a.groups, (g_ + 1) * E_ALL // a.groups
        for p in range(tp):                               # parity = processing rank within the pair
            exps = sorted(e for e in range(glo, ghi) if e % 4 == f and ((e // 4) % tp) == p)
            pr = g_ * tp + p
            for hh in range(tp):
                own = {}
                for e in exps:
                    pp = f"model.llm.layers.{L}.mlp.experts.{e}."
                    trel = {pp + pj + ".trellis": whole[base_off + hdr[pp + pj + ".trellis"]["data_offsets"][0]: base_off + hdr[pp + pj + ".trellis"]["data_offsets"][1]] for pj in ("gate_proj", "up_proj", "down_proj")}
                    own.update(halves(L, e, lambda s, t: whole[base_off + s: base_off + t], trel, hdr, base_off, url, None, hh))
                ref = st_bytes(own)
                path = os.path.join(base, f"r{pr}", "" if hh == p else "partner", f"s-{L:02d}-f{f}-d{pr}.safetensors")
                got = open(path, "rb").read() if os.path.exists(path) else None
                same = got == ref; checked += 1; bad += 0 if same else 1
                log(f"rank-proc {pr} half {hh} ({'own' if hh == p else 'partner'}) {len(exps)} experts: {len(ref)} bytes, {'IDENTICAL' if same else 'DEVIATES or missing'}  {path}")
    # header plausibility for the file: sizes and offsets
    ok_hdr = True
    for k, m in hdr.items():
        sz = {"I32": 4, "F16": 2, "I16": 2}[m["dtype"]]; cnt = 1
        for d in m["shape"]: cnt *= d
        if m["data_offsets"][1] - m["data_offsets"][0] != cnt * sz: ok_hdr = False
    log(f"RESULT {fname(L, f)}: sha256 {'ok' if ok_sha else 'BAD'}, header sizes {'ok' if ok_hdr else 'BAD'}, compared {checked} shard files, deviations {bad}")
    return ok_sha and ok_hdr and bad == 0

def check_headers(layers):
    """all files of the given layers: header sizes vs HF file size (dtype*shape == offsets, last offset + header = file size, contiguous trellis per expert)"""
    api = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true"))
    sz = {s["rfilename"]: s["size"] for s in api["siblings"]}; bad = 0; n = 0
    for L in layers:
        for f in range(4):
            url = url_of(L, f); hdr, base_off = read_header(url); n += 1
            end = max(m["data_offsets"][1] for m in hdr.values())
            ok = base_off + end == sz[fname(L, f)]
            for k, m in hdr.items():
                esz = {"I32": 4, "F16": 2, "I16": 2}[m["dtype"]]; cnt = 1
                for d in m["shape"]: cnt *= d
                if m["data_offsets"][1] - m["data_offsets"][0] != cnt * esz: ok = False
            bad += 0 if ok else 1
            if not ok: log("HEADER BAD", fname(L, f))
    log(f"headers checked: {n} files, bad {bad}"); return bad == 0

if a.verify_file:
    L, f = (int(x) for x in a.verify_file.split(",")); sys.exit(0 if verify(L, f, a.verify_base) else 1)
if a.check_headers:
    sys.exit(0 if check_headers(LAYERS) else 1)

tot = 0
for L in LAYERS:
    t0 = time.time(); got = 0
    for f in range(4):
        for attempt in range(1000):
            import shutil
            while shutil.disk_usage(a.out).free < a.min_free_gb * 2**30:
                log(f"free space below {a.min_free_gb} GB, waiting (layer {L} file {f})"); time.sleep(120)
            try:
                got += process(L, f); break
            except Exception as ex:                      # network hiccup, partner not reachable ...: wait and resume (done-markers make it idempotent)
                log(f"layer {L} file {f} attempt {attempt} failed: {type(ex).__name__} {str(ex)[:200]}; retry in 60 s"); time.sleep(60)
    tot += got
    log(f"layer {L} done: {got/1e9:.2f} GB in {time.time()-t0:.0f}s ({got/1e6/max(1e-3, time.time()-t0):.1f} MB/s), total {tot/1e9:.1f} GB")
log("ALL LAYERS DONE", tot)
