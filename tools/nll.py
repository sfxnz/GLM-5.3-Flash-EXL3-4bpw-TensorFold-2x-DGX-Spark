"""Paired NLL quality gate: score a fixed corpus through the served engine, then compare two configs token by token.

python3 tools/nll.py capture --label bf16-p4 --out DIR [--url http://127.0.0.1:8000] [--top 1]
python3 tools/nll.py compare A_DIR B_DIR [--margin 0.002] [--aa AA.json] [--json OUT]
python3 tools/nll.py corpus     # prints the resolved corpus manifest (sources, sha256, byte ranges) without a server

capture sends each corpus chunk as a text prompt to nonstreamed /v1/completions with prompt_logprobs (TensorFold
v1.2.2: one row a prompt token after the first, the prompt token's own logprob and rank first, then the best tokens;
docs/api.md:208-215, zig/src/server/logprobs.zig:94-113) and stores, per chunk, the token ids, each token's logprob,
its rank and the top-1 id. compare pairs two captures, refuses unless every token id matches, and reports the mean
NLL difference B - A in nats/token with a 95% bootstrap CI over chunks.

The gate is non-inferiority: it passes only when the CI's upper end is below --margin, and fails when any group's
CI sits wholly above it. Exit codes: 0 pass, 1 worse or inconclusive, 2 refused (captures that do not pair, an A/A
floor too close to the margin, a bad capture).

Protocol: capture the default config on two boots (A1, A2) of an idle server; `compare A1 A2 --json aa.json` must pass
before any knob is judged. Then one knob at a time: `compare A1 B1 --aa aa.json`, and when A/A was not bitwise
identical, a second boot B2 with `compare A2 B2 --aa aa.json`; both must pass.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe  # noqa: E402

FORMAT = "nll-capture/1"
CORPUS_VERSION = "nll-corpus-v1"
CORPUS_ROOT = Path.home() / "projects/data/glm53-evals/corpus-src"
# 1/10 of what 4bpw quantization itself costs: TR3's KLD against BF16 is 0.0228 nats/token
# (evidence/quality/tr3-h200-vs-root.receipt.json), so a serving knob may add at most ~10% to it (perplexity +0.2%).
MARGIN = 0.002
# A margin closer than this to the A/A floor cannot tell a knob from boot-to-boot noise.
AA_HEADROOM = 2.0
RESAMPLES = 10000
MIN_UNITS = 5
BLOCK = 512
LONG_BUCKETS = ((0, 4096), (4096, 16384), (16384, None))

G = "https://www.gutenberg.org/cache/epub/"
PY = "https://raw.githubusercontent.com/python/cpython/v3.12.3/Lib/"
RS = "https://raw.githubusercontent.com/rust-lang/rust/1.77.0/library/"
SQ = "https://raw.githubusercontent.com/sqlite/sqlite/version-3.45.0/src/"
GO = "https://raw.githubusercontent.com/golang/go/go1.22.0/src/"
GSM = "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl"

# (category, source URL, sha256 of the cached file, start byte, length in bytes). The cache file under CORPUS_ROOT is
# named by the URL, percent-encoded. A start past 0 moves to the next line start, the end back to a line end; CRLF
# becomes LF. Changing any row is a new CORPUS_VERSION.
SOURCES = {
    G + "1404/pg1404.txt": "0b92fadc9d7afd767eefd27805a20e44417ff26d1fb2f964229f39dcc7b2a65d",
    G + "2009/pg2009.txt": "27c02feed0b90e0163811a35a7565d03611dc1d3f86703d6bec4ae73d2d612b8",
    G + "3300/pg3300.txt": "e91d52dca43fd0fd69baf7776b00a5d8e74b69675b6b608ca8a0378aaf3ae58d",
    G + "408/pg408.txt": "87f8e832484792b761a510b953d67e3835fa46a9f4231a508bfeb914a98559ca",
    G + "205/pg205.txt": "2d9a76a2e3e8195c69430516ebd33c4d0757a53ad432ff6186b7b794e6fe99f9",
    G + "345/pg345.txt": "96cd16eacdbfebae8fdda5591f66e0cc8ee76be18e0cd1aca02bc00615782d28",
    G + "2000/pg2000.txt": "534f41d59f7142163fa0964076ac6351845c006ea6433778be637fac6d5b04d7",
    G + "17489/pg17489.txt": "a5de514ba7b9f2e1790e7e259c4e8b7a35ae1d29e4bf9a5f8767039c58b80503",
    G + "45334/pg45334.txt": "4cdd962f1910a49091b945eff4233798b8c13f74b011cee75a8cccc7467d3612",
    G + "55752/pg55752.txt": "0fc3dbf384544d81d87e5a731e67b7976ac3a57acb378f0f354d12a3b52bd0c7",
    G + "22367/pg22367.txt": "359d3f5983c812393f4bd7ee49a350ffffc7d476015e58d6b84b6c848dd2b0e9",
    G + "11940/pg11940.txt": "b87d2858e94d87c346b99efc3a8346c8344d2a0a99ed0ca2eb04cba82e313db7",
    G + "11024/pg11024.txt": "9cf9f2efa248146886505437f30f0f8fe906384b77bea7ac91ff72c347de1a9c",
    G + "24264/pg24264.txt": "ff1526996bf4b81807651921a85e5c1c0f1d1d123c9fa4553057ba6a3ec72011",
    G + "23962/pg23962.txt": "af3c9e408c0c58595b666ed9981b6fa1e9343f4bbc78309b1cb0818c32fc1f58",
    G + "23950/pg23950.txt": "ed422e10c16c19af7ca7b09ca0320012e60776b3d7cb054769b8f7d9a19aea7d",
    G + "1982/pg1982.txt": "7585b90b3c25951420ddf4857a53964a3b4b892c8aab5a0200c3633b53bb2a1e",
    G + "145/pg145.txt": "dc2e0107e6ae07e8e33da934b2ec4e8e600826a2e0cd48df265119cd02e5fb50",
    "https://raw.githubusercontent.com/lodash/lodash/4.17.21/lodash.js":
        "4c04561befdf653aef017a42ac5addf68ea943cdfca6bdee5ce04e04e8139f54",
    SQ + "func.c": "ee64cb9a6e86e0f8d7eabc072c008e009a278c314f607bae0df367916de9e6d5",
    SQ + "util.c": "e77b667b8a710a636303c11333cbb699a349234d0952a9b79eefef5cd05584da",
    RS + "core/src/str/pattern.rs": "98fa02e7fcb27e769d3e8b0538f7845650eda67dbe0fb19775cd0ee148c86792",
    RS + "alloc/src/collections/binary_heap/mod.rs": "6089532c516914aa8492cce9a46df0270dd5727b06795dcb532c80a163e164e8",
    RS + "core/src/iter/adapters/zip.rs": "7fc2c27972c40937855f05e327a8db6fdc743c9bb3c9945733d110abbcef8391",
    PY + "statistics.py": "b9ef01a757de37e9fbf13d32beeb9530c8271582cabec4998226dcb7e0613cd0",
    PY + "fractions.py": "cf33568f7b767f66896995a7a81cc662e8be226f2e58efb61d2028f64eb5d231",
    PY + "functools.py": "cca971c456e1bec8b751aecdf41466f34dbf72321ee0840627280ba2ccf9d033",
    PY + "textwrap.py": "62867e40cdea6669b361f72af4d7daf0359f207c92cbeddfc7c7506397c1f31c",
    GO + "bufio/scan.go": "99dea654896963b22a75a7fb1440d6e89d5037bc91876d3d4dd37bcce5bd9e5f",
    GO + "sort/sort.go": "f414ff4021ea34d3a55f9c0fc9afdf45ef6f7f88f1bd1167047642b3cea66ed7",
    GSM: "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
}
CORPUS = (
    [("prose", G + f"{b}/pg{b}.txt", s, 16000) for b, s in
     [(1404, 300000), (1404, 700000), (2009, 300000), (2009, 800000), (3300, 600000), (3300, 1500000),
      (408, 100000), (408, 250000), (205, 150000), (205, 400000), (345, 200000), (345, 600000)]]
    + [("multilingual", G + f"{b}/pg{b}.txt", s, n) for b, s, n in
       [(2000, 800000, 14000), (17489, 300000, 14000), (45334, 500000, 14000), (55752, 150000, 14000),
        (22367, 40000, 14000), (11940, 300000, 14000), (11024, 300000, 14000), (24264, 800000, 9000),
        (23962, 800000, 9000), (23950, 800000, 9000), (1982, 8000, 9000)]]
    + [("code", u, s, n) for u, s, n in
       [("https://raw.githubusercontent.com/lodash/lodash/4.17.21/lodash.js", 200000, 14000),
        (SQ + "func.c", 20000, 14000), (SQ + "util.c", 10000, 14000), (RS + "core/src/str/pattern.rs", 20000, 14000),
        (RS + "alloc/src/collections/binary_heap/mod.rs", 20000, 14000), (RS + "core/src/iter/adapters/zip.rs", 0, 14000),
        (PY + "statistics.py", 10000, 14000), (PY + "fractions.py", 10000, 14000), (PY + "functools.py", 10000, 14000),
        (PY + "textwrap.py", 0, 14000), (GO + "bufio/scan.go", 0, 14490), (GO + "sort/sort.go", 0, 10384)]]
    + [("math", GSM, s, 14000) for s in (500000, 1300000, 2100000, 2900000, 3700000)]
    + [("long", G + "145/pg145.txt", 600000, 138000)]
)


class Refused(Exception):
    """Inputs the gate will not use: a corpus source that changed, captures that do not pair, a capture left unfinished."""


def resolve(root, corpus=None, sources=None):
    """The corpus as sent: one dict a chunk with its text and provenance, or Refused naming the first source that differs."""
    corpus, sources = corpus or CORPUS, sources or SOURCES
    chunks, cache = [], {}
    for i, (category, url, start, length) in enumerate(corpus):
        if url not in cache:
            path = Path(root) / urllib.parse.quote(url, safe="")
            if not path.is_file():
                raise Refused(f"missing corpus source {path} (fetch {url})")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != sources[url]:
                raise Refused(f"{path} has sha256 {hashlib.sha256(raw).hexdigest()}, the corpus pins {sources[url]}")
            cache[url] = raw
        raw = cache[url]
        lo = start if start == 0 else raw.index(b"\n", start) + 1
        hi = min(start + length, len(raw))
        if hi < len(raw):
            hi = raw.rindex(b"\n", lo, hi) + 1
        text = raw[lo:hi].decode("utf-8").replace("\r\n", "\n").lstrip("﻿")
        chunks.append({"id": i, "category": category, "url": url, "source_sha256": sources[url], "bytes": [lo, hi],
                       "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text})
    return chunks


def corpus_sha(chunks):
    return hashlib.sha256("".join(c["text_sha256"] for c in chunks).encode()).hexdigest()


def parse_rows(rows, prompt_tokens):
    """TensorFold's prompt_logprobs as columns over prompt tokens 1..n-1: ids, logprobs, ranks, top-1 ids."""
    if not rows or rows[0] is not None or len(rows) != prompt_tokens:
        raise ValueError(f"{len(rows or [])} rows for {prompt_tokens} prompt tokens, row 0 {rows[0] if rows else None!r}")
    ids, lps, ranks, top1 = [], [], [], []
    for row in rows[1:]:
        key = next(iter(row))
        given = row[key]
        lp = float(given["logprob"])
        if not math.isfinite(lp) or lp > 1e-3:
            raise ValueError(f"token {key} has logprob {lp}")
        best = next((int(k) for k, v in row.items() if v["rank"] == 1), None)
        ids.append(int(key))
        lps.append(lp)
        ranks.append(int(given["rank"]))
        top1.append(best)
    return {"ids": ids, "logprob": lps, "rank": ranks, "top1": top1}


def capture(a):
    out = Path(a.out)
    if (out / "capture.json").exists():
        raise Refused(f"{out / 'capture.json'} exists; captures are never overwritten")
    chunks = resolve(a.corpus_root)
    _, health = recipe.get(a.url + "/health")
    _, models = recipe.get(a.url + "/v1/models")
    started = time.time()
    done = []
    for c in chunks:
        body = {"model": a.model, "prompt": c["text"], "max_tokens": 1, "temperature": 0, "prompt_logprobs": a.top}
        t0 = time.monotonic()
        status, d = recipe.post(a.url + "/v1/completions", body, timeout=a.timeout)
        wall = time.monotonic() - t0
        if status != 200:
            raise Refused(f"chunk {c['id']}: HTTP {status}: {json.dumps(d)[:400]}")
        n = d["usage"]["prompt_tokens"]
        try:
            cols = parse_rows(d["choices"][0].get("prompt_logprobs"), n)
        except ValueError as e:
            raise Refused(f"chunk {c['id']}: {e}") from None
        meta = {k: v for k, v in c.items() if k != "text"}
        done.append({**meta, "prompt_tokens": n, "wall_s": round(wall, 3), **cols})
        print(json.dumps({"event": "chunk", "id": c["id"], "category": c["category"], "prompt_tokens": n,
                          "wall_s": round(wall, 2), "tok_s": round(n / wall, 1) if wall else None}), flush=True)
    tokens = sum(c["prompt_tokens"] for c in done)
    wall = time.time() - started
    record = {"format": FORMAT, "label": a.label, "corpus": {"version": CORPUS_VERSION, "sha256": corpus_sha(chunks)},
              "server": {"url": a.url, "model": a.model, "health": health, "models": models},
              "prompt_logprobs": a.top, "started": started, "wall_s": round(wall, 1), "prompt_tokens": tokens,
              "chunks": done}
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "capture.json.part"
    tmp.write_text(json.dumps(record))
    os.replace(tmp, out / "capture.json")
    print(json.dumps({"event": "captured", "label": a.label, "out": str(out / "capture.json"), "chunks": len(done),
                      "prompt_tokens": tokens, "wall_s": round(wall, 1)}), flush=True)
    return 0


def load(path):
    p = Path(path)
    p = p / "capture.json" if p.is_dir() else p
    d = json.loads(p.read_text())
    if d.get("format") != FORMAT:
        raise Refused(f"{p} is not an {FORMAT} capture")
    return d


def pair(a, b):
    """Per chunk: its category, per-token NLL deltas B - A, both sides' logprobs and top-1 agreement; or Refused."""
    if a["corpus"] != b["corpus"]:
        raise Refused(f"corpora differ: {a['corpus']} vs {b['corpus']}")
    if len(a["chunks"]) != len(b["chunks"]):
        raise Refused(f"{len(a['chunks'])} chunks vs {len(b['chunks'])}")
    paired = []
    for ca, cb in zip(a["chunks"], b["chunks"]):
        if ca["ids"] != cb["ids"]:
            at = next((i for i, (x, y) in enumerate(zip(ca["ids"], cb["ids"])) if x != y), min(len(ca["ids"]), len(cb["ids"])))
            raise Refused(f"chunk {ca['id']} ({ca['category']}): token ids differ at prompt token {at + 1} "
                          f"({len(ca['ids'])} vs {len(cb['ids'])} scored tokens); these captures do not pair")
        deltas = [x - y for x, y in zip(ca["logprob"], cb["logprob"])]
        agree = [x == y for x, y in zip(ca["top1"], cb["top1"])]
        paired.append({"category": ca["category"], "delta": deltas, "a": ca["logprob"], "b": cb["logprob"],
                       "agree": agree})
    return paired


def bootstrap(units, resamples=RESAMPLES, seed=0):
    """Mean of sum(d)/sum(n) over (sum_d, n) units, with a 95% percentile CI from resampling units; no CI below MIN_UNITS."""
    total = sum(n for _, n in units)
    mean = sum(d for d, _ in units) / total
    if len(units) < MIN_UNITS:
        return mean, None, None
    rng, k, stats = random.Random(seed), len(units), []
    for _ in range(resamples):
        s = n = 0
        for _ in range(k):
            d, m = units[rng.randrange(k)]
            s += d
            n += m
        stats.append(s / n)
    stats.sort()
    return mean, stats[int(0.025 * resamples)], stats[int(math.ceil(0.975 * resamples)) - 1]


def summarize(paired, margin):
    groups = [("all", paired)]
    for cat in dict.fromkeys(c["category"] for c in paired):
        groups.append((cat, [c for c in paired if c["category"] == cat]))
    rows = []
    for name, cs in groups:
        units = [(sum(c["delta"]), len(c["delta"])) for c in cs]
        rows.append(row(name, "chunk", units, cs))
    for c in (c for c in paired if c["category"] == "long"):
        for lo, hi in LONG_BUCKETS:
            pos = range(lo, min(hi or len(c["delta"]), len(c["delta"])))
            if not pos:
                continue
            blocks = [range(i, min(i + BLOCK, pos.stop)) for i in range(pos.start, pos.stop, BLOCK)]
            units = [(sum(c["delta"][j] for j in r), len(r)) for r in blocks]
            part = [{"delta": c["delta"][pos.start:pos.stop], "a": c["a"][pos.start:pos.stop],
                     "b": c["b"][pos.start:pos.stop], "agree": c["agree"][pos.start:pos.stop]}]
            rows.append(row(f"long@{lo}-{hi or 'end'}", f"{BLOCK}-token block", units, part))
    overall = rows[0]
    if overall["ci95"] is None:
        status = "inconclusive"
    elif any(r["ci95"] and r["ci95"][0] > margin for r in rows):
        status = "worse"
    elif overall["ci95"][1] > margin:
        status = "inconclusive"
    else:
        status = "pass"
    return status, rows


def row(name, unit, units, cs):
    mean, lo, hi = bootstrap(units)
    n = sum(m for _, m in units)
    a = -sum(sum(c["a"]) for c in cs) / n
    b = -sum(sum(c["b"]) for c in cs) / n
    agree = sum(sum(c["agree"]) for c in cs) / n
    same = sum(sum(1 for x in c["delta"] if x == 0) for c in cs) / n
    return {"group": name, "unit": unit, "units": len(units), "tokens": n, "nll_a": a, "nll_b": b, "delta": mean,
            "ci95": None if lo is None else [lo, hi], "top1_agree": agree, "identical": same}


def compare(a):
    A, B = load(a.a), load(a.b)
    paired = pair(A, B)
    floor = None
    if a.aa:
        aa = json.loads(Path(a.aa).read_text())
        ci = aa["rows"][0]["ci95"]
        if ci is None:
            raise Refused(f"{a.aa} has no overall CI")
        floor = max(abs(ci[0]), abs(ci[1]))
        if floor * AA_HEADROOM > a.margin:
            raise Refused(f"the A/A floor in {a.aa} reaches {floor:.2e} nats/token; a {a.margin} margin needs it below "
                          f"{a.margin / AA_HEADROOM:.2e}")
    status, rows = summarize(paired, a.margin)
    print(f"{A['label']} -> {B['label']}: NLL B - A in nats/token, 95% bootstrap CI ({RESAMPLES} resamples)")
    print(f"{'group':<18}{'units':>6}{'tokens':>9}{'NLL A':>9}{'NLL B':>9}{'B - A':>11}{'CI95':>24}{'top1=':>8}{'same':>7}")
    for r in rows:
        ci = f"[{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]" if r["ci95"] else f"(n/a, <{MIN_UNITS} units)"
        print(f"{r['group']:<18}{r['units']:>6}{r['tokens']:>9}{r['nll_a']:>9.4f}{r['nll_b']:>9.4f}{r['delta']:>+11.5f}"
              f"{ci:>24}{r['top1_agree']:>8.4f}{r['identical']:>7.3f}")
    floor_note = f"A/A floor {floor:.2e}" if floor is not None else "A/A floor not checked (pass --aa)"
    print(f"margin {a.margin} nats/token, {floor_note}: {status}")
    report = {"a": A["label"], "b": B["label"], "corpus": A["corpus"], "margin": a.margin, "aa_floor": floor,
              "status": status, "rows": rows}
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1) + "\n")
    return 0 if status == "pass" else 1


def manifest(a):
    chunks = resolve(a.corpus_root)
    for c in chunks:
        print(json.dumps({k: v for k, v in c.items() if k != "text"} | {"chars": len(c["text"])}))
    print(json.dumps({"corpus": CORPUS_VERSION, "sha256": corpus_sha(chunks), "chunks": len(chunks),
                      "bytes": sum(c["bytes"][1] - c["bytes"][0] for c in chunks)}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="score the corpus through a running server")
    c.add_argument("--label", required=True)
    c.add_argument("--out", required=True, help="directory for capture.json; an existing capture is refused")
    c.add_argument("--url", default=recipe.default_url())
    c.add_argument("--model", default=recipe.setting("SERVED_NAME"))
    c.add_argument("--top", type=int, default=1, choices=range(1, 21), metavar="1..20",
                   help="best tokens a row carries (1 gives top-1 agreement)")
    c.add_argument("--timeout", type=float, default=900)
    c.add_argument("--corpus-root", default=str(CORPUS_ROOT))
    p = sub.add_parser("compare", help="paired NLL difference B - A")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--margin", type=float, default=MARGIN, help="nats/token B may lose (default %(default)s)")
    p.add_argument("--aa", help="compare --json report of two boots of one config; refused when its floor is too close")
    p.add_argument("--json", help="write the report here")
    m = sub.add_parser("corpus", help="print the resolved corpus manifest")
    m.add_argument("--corpus-root", default=str(CORPUS_ROOT))
    a = ap.parse_args(argv)
    try:
        return {"capture": capture, "compare": compare, "corpus": manifest}[a.cmd](a)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
