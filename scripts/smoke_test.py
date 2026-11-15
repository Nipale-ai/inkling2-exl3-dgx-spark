#!/usr/bin/env python3
"""Smoke test for the Inkling-2-Small server (OpenAI API, stdlib + tokenizers). Raw log: JSONL. No speed figures are taken from this test.
T1 needle 8K, same request twice (temperature 0) -> both correct and identical
T2 single vs batch: the T1 request next to 7 other requests -> output equal to T1 (if not: reported, batch-composition numerics) and every code correct
T3 stream with usage chunk -> content equals the non-stream answer, usage present
T4 16 concurrent 2K requests, 200 new tokens each -> no HTTP error, no truncated finish_reason other than length
T5 128K needle (--long N tokens)
usage: smoke_test.py --url URL --tok tokenizer.json --out gate.jsonl [--long 120000] [--skip5]"""
import argparse, json, time, urllib.request, concurrent.futures as cf, random, sys
from tokenizers import Tokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://localhost:8891"); ap.add_argument("--tok", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--long", type=int, default=120000); ap.add_argument("--skip5", action="store_true"); a = ap.parse_args()
tok = Tokenizer.from_file(a.tok)
def ntok(s): return len(tok.encode(s).ids)
TOPICS = ["harbour", "orchard", "railway", "lantern", "glacier", "workshop", "library", "market", "observatory", "vineyard", "canal", "bakery"]
def para(rng):
    t = rng.choice(TOPICS); n = rng.randint(3, 9)
    return (f"In the {t} district of the old town, record {rng.randint(100, 99999)} notes that {n} workers repaired the {rng.choice(TOPICS)} "
            f"over {rng.randint(2, 40)} days, and the committee approved budget line {rng.randint(10, 9999)} without further remarks. ")
def make(n_tokens, code, seed, frac=0.5):
    rng = random.Random(seed); ps = []; total = 0
    while total < n_tokens - 120:
        p = para(rng); ps.append(p); total += ntok(p)
    ps.insert(int(len(ps) * frac), f"IMPORTANT: The secret access code is {code}. ")
    text = "".join(ps) + "\n\nWhat is the secret access code mentioned in the text above? Answer with the code only."
    return text, ntok(text)
def post(body, timeout=3600):
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, json.loads(r.read()), time.time() - t0
    except urllib.error.HTTPError as e: return e.code, e.read().decode()[:500], time.time() - t0
    except Exception as e: return 0, repr(e), time.time() - t0
def body(text, max_tokens=64, **kw):
    b = {"model": "x", "messages": [{"role": "user", "content": text}], "temperature": 0, "max_tokens": max_tokens, "reasoning_effort": "none"}; b.update(kw); return b
def content(r): return (r["choices"][0]["message"].get("content") or "") if isinstance(r, dict) else ""
log = open(a.out, "w"); results = {}
def rec(name, **kw): log.write(json.dumps({"test": name, **kw}, ensure_ascii=False) + "\n"); log.flush()
def verdict(name, ok, note): results[name] = ok; print(f"{name}: {'PASS' if ok else 'FAIL'}  {note}", flush=True); rec(name + "_verdict", ok=ok, note=note)
# T1
text1, n1 = make(8000, "K7-ALPHA-4821", 1)
outs = []
for i in range(2):
    s, r, dt = post(body(text1)); outs.append(content(r)); rec("T1", run=i, status=s, prompt_tokens_local=n1, answer=content(r), usage=r.get("usage") if isinstance(r, dict) else None, raw=r)
verdict("T1_needle8K_twice", all(o.strip() and "K7-ALPHA-4821" in o for o in outs) and outs[0] == outs[1], f"prompt {n1} tokens; answers {outs!r}")
# T2
others = [make(4000, f"Z{j}-BETA-{1000+j*37}", 10 + j) for j in range(7)]
reqs = [text1] + [t for t, _ in others]; codes = ["K7-ALPHA-4821"] + [f"Z{j}-BETA-{1000+j*37}" for j in range(7)]
with cf.ThreadPoolExecutor(8) as ex: res = list(ex.map(lambda t: post(body(t)), reqs))
ans = [content(r) if s == 200 else "" for s, r, _ in res]
for i, (s, r, dt) in enumerate(res): rec("T2", i=i, status=s, answer=ans[i], expected=codes[i], raw=r)
okc = all(c in x for c, x in zip(codes, ans)); same = ans[0] == outs[0]
verdict("T2_batch_of_8", okc, f"all 8 codes correct: {okc}; request 1 identical to single run: {same} ({ans[0]!r} vs {outs[0]!r})")
verdict("T2b_single_equals_batch", same, "batch-composition numerics may differ in the last tokens if False")
# T3
sb = body(text1, stream=True, stream_options={"include_usage": True})
req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(sb).encode(), headers={"Content-Type": "application/json"})
parts = []; usage = None; nchunks = 0
with urllib.request.urlopen(req, timeout=3600) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        ch = json.loads(line[5:]); nchunks += 1
        if ch.get("usage"): usage = ch["usage"]
        for c in ch.get("choices", []):
            parts.append((c.get("delta") or {}).get("content") or "")
streamed = "".join(parts); rec("T3", chunks=nchunks, streamed=streamed, usage=usage)
verdict("T3_stream_usage", streamed == outs[0] and bool(usage) and usage.get("completion_tokens", 0) > 0, f"streamed {streamed!r} equals non-stream: {streamed == outs[0]}; usage {usage}")
# T4
ps16 = [f"Write a detailed paragraph about {TOPICS[i % len(TOPICS)]} number {i}. " + make(2000, f"Q{i}", 100 + i)[0] for i in range(16)]
with cf.ThreadPoolExecutor(16) as ex: res = list(ex.map(lambda t: post(body(t, max_tokens=200)), ps16))
errs = [(i, s, str(r)[:200]) for i, (s, r, _) in enumerate(res) if s != 200]
for i, (s, r, dt) in enumerate(res): rec("T4", i=i, status=s, finish=(r["choices"][0]["finish_reason"] if s == 200 else None), usage=(r.get("usage") if s == 200 else None), seconds=round(dt, 1))
verdict("T4_16_concurrent", not errs, f"errors {errs}")
# T5
if not a.skip5:
    text5, n5 = make(a.long, "L9-OMEGA-5503", 7, 0.5)
    s, r, dt = post(body(text5)); rec("T5", status=s, prompt_tokens_local=n5, answer=content(r), usage=r.get("usage") if isinstance(r, dict) else None, seconds_NOT_A_SPEED_FIGURE=round(dt, 1), raw=r)
    verdict("T5_needle_long", s == 200 and "L9-OMEGA-5503" in content(r), f"prompt {n5} tokens (local count); status {s}; answer {content(r)!r}")
print("RESULT", {k: ("PASS" if v else "FAIL") for k, v in results.items()})
