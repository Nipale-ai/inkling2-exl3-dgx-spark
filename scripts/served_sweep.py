#!/usr/bin/env python3
"""Served decode sweep in the author's way (README of 0xSero/inkling2-exl3, 'Speed: served decode'): 16 chat / code / agentic prompts, up to 2048 new tokens each,
concurrency c1/c2/c4/c8/c16, tok/s = delta of the server's /health counters (dec_tok / dec_time, prefill excluded). The author's 16 prompts are NOT in his repo;
these are our own 16 (same three kinds). Raw log JSONL + summary. Chat template default (no reasoning_effort sent) = the author's server default.
usage: served_sweep.py --url URL --out DIR [--temp 0.0] [--max-tokens 2048] [--levels 1,2,4,8,16]"""
import argparse, json, time, urllib.request, concurrent.futures as cf, os
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://localhost:8891"); ap.add_argument("--out", required=True)
ap.add_argument("--temp", type=float, default=0.0); ap.add_argument("--max-tokens", type=int, default=2048); ap.add_argument("--levels", default="1,2,4,8,16"); ap.add_argument("--effort", default="")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
P = [
 "Write a long, detailed essay on how the printing press changed European society, with concrete examples from at least five countries.",
 "Explain in depth how a modern CPU executes an instruction, from fetch to retirement, including branch prediction, out-of-order execution and caches. Use many examples.",
 "Tell a long, original short story about a lighthouse keeper who finds a message in a bottle. Include dialogue and at least eight scenes.",
 "Give a thorough guide to starting a vegetable garden on a small balcony: planning, soil, containers, watering, pests, harvesting, month by month.",
 "Compare in detail the economic systems of ancient Rome and ancient China during their peak periods: trade, taxation, currency, labour, infrastructure.",
 "Explain the history and mathematics of public-key cryptography in detail, from Diffie-Hellman to elliptic curves, with worked small-number examples.",
 "Write a complete Python implementation of an LRU cache with O(1) operations, then a thread-safe version, then unit tests, and explain every design decision.",
 "Write a Rust program that implements a simple HTTP/1.1 server using only the standard library: parsing, routing, static files, error handling. Explain each part.",
 "Implement a JSON parser in JavaScript from scratch (no JSON.parse), with error messages that include line and column. Provide tests and discuss performance.",
 "Write a C program that implements a memory allocator (malloc/free/realloc) with a free list and coalescing. Explain the data structures and edge cases in depth.",
 "Design and implement in Go a rate limiter service supporting token bucket and sliding window algorithms, with a clean API and benchmarks. Explain trade-offs.",
 "You are an autonomous agent with tools search_docs(query), run_code(language, source) and read_file(path). Plan, step by step and in detail, how you would investigate why a nightly data pipeline became 3x slower after a library upgrade. Write out each tool call you would make with its arguments and what you expect to learn.",
 "As an agent with tools get_weather(city), calendar_add(title, time) and send_email(to, subject, body), write the full transcript of handling this request: 'Plan a 3-day team offsite in Lisbon in June, check weather, schedule sessions, and email the agenda.' Include every tool call and the reasoning between calls.",
 "As a coding agent with tools run_tests(), read_file(path), edit_file(path, patch), plan and narrate in detail how you would fix a flaky integration test that fails 1 in 20 runs. Show each tool call, the output you expect, and your reasoning.",
 "Act as a research agent with tools web_search(query) and fetch(url). Describe in full detail, tool call by tool call, how you would produce a literature summary on retrieval-augmented generation in the last two years, including how you verify claims.",
 "Act as an operations agent with tools kubectl(cmd), read_logs(service) and page_oncall(msg). Write the complete incident response transcript for a production outage where checkout latency suddenly rose from 200 ms to 9 s. Include every command, expected output and decision.",
]
def health():
    return json.loads(urllib.request.urlopen(a.url + "/health", timeout=60).read())["stats"]
def post(text):
    b = {"model": "x", "messages": [{"role": "user", "content": text}], "temperature": a.temp, "max_tokens": a.max_tokens}
    if a.effort: b["reasoning_effort"] = a.effort
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=7200) as r: j = json.loads(r.read()); return 200, j, time.time() - t0
    except Exception as e: return 0, repr(e), time.time() - t0
log = open(os.path.join(a.out, f"served-sweep-temp{a.temp}.jsonl"), "w"); summ = []
for c in [int(x) for x in a.levels.split(",")]:
    h0 = health(); t0 = time.time()
    with cf.ThreadPoolExecutor(c) as ex: res = list(ex.map(post, P))
    wall = time.time() - t0; h1 = health()
    dt = h1["dec_time"] - h0["dec_time"]; dk = h1["dec_tok"] - h0["dec_tok"]; ds = h1["dec_steps"] - h0["dec_steps"]; dr = h1["dec_rows"] - h0["dec_rows"]
    toks = [r["usage"]["completion_tokens"] for s, r, _ in res if s == 200]; errs = [(i, r) for i, (s, r, _) in enumerate(res) if s != 200]
    row = {"concurrency": c, "temp": a.temp, "requests": len(P), "errors": len(errs), "min_completion_tokens": min(toks) if toks else None, "mean_completion_tokens": sum(toks) / max(1, len(toks)),
           "rows_below_2000": sum(1 for t in toks if t < 2000), "health_dec_tok": dk, "health_dec_time_s": round(dt, 2), "decode_tok_s_health": round(dk / dt, 1) if dt else None,
           "tokens_per_step": round(dk / ds, 3) if ds else None, "avg_rows_per_step": round(dr / ds, 2) if ds else None, "wall_s_NOT_A_SPEED_FIGURE": round(wall, 1)}
    summ.append(row); print(json.dumps(row), flush=True)
    for i, (s, r, d) in enumerate(res): log.write(json.dumps({"c": c, "i": i, "status": s, "usage": r.get("usage") if s == 200 else None, "finish": r["choices"][0]["finish_reason"] if s == 200 else None, "err": None if s == 200 else r}) + "\n")
    log.flush()
json.dump(summ, open(os.path.join(a.out, f"served-sweep-temp{a.temp}-summary.json"), "w"), indent=1)
