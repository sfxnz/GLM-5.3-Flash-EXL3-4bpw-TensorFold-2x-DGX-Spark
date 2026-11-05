"""Streamed decode speed and time to first token at each concurrency, thinking off, prose prompts, N runs, medians.

python3 tools/bench.py [--concurrency 1,4,8] [--runs 3] [--max-tokens 512] [--out evidence/run/bench.json]

Per stream: ttft_s is the first content token; decode_tok_s is (tokens - 1) over first-to-last content token. aggregate_tok_s
is all streams' tokens over the run's wall time. Above the server's --parallel, the extra streams queue, which shows in
their ttft_s. Every stream's raw numbers go to --out.
"""
import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe  # noqa: E402

PROMPTS = [
    "Write a detailed 600-word essay about the history of the printing press.",
    "Explain, step by step and at length, how a CPU pipeline handles branch misprediction.",
    "Write a long short story about a lighthouse keeper who finds a message in a bottle.",
    "Describe in detail the water cycle and its effect on regional climates.",
    "Write a long account of how the transcontinental railroads changed trade in the nineteenth century.",
    "Explain at length how vaccines train the immune system, with examples from history.",
    "Write a detailed travel guide to a fictional coastal city, district by district.",
    "Describe at length how a modern compiler turns source code into machine code.",
]


def one_stream(chat_url, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True}
    first = last = None
    deltas, usage, spec, end = 0, None, None, 0.0
    for t, chunk in recipe.stream(chat_url, body):
        for c in chunk.get("choices", []):
            if c.get("delta", {}).get("content"):
                first = t if first is None else first
                last = t
                deltas += 1
        usage = chunk.get("usage") or usage
        spec = chunk.get("speculative") or spec
        end = t
    tokens = usage["completion_tokens"] if usage else deltas
    return {"ttft_s": round(first, 3) if first is not None else None, "tokens": tokens,
            "tokens_from": "usage" if usage else "content deltas",
            "decode_tok_s": round((tokens - 1) / (last - first), 2) if first is not None and last > first else None,
            "wall_s": round(end, 3), "speculative": spec}


def run_once(chat_url, model, concurrency, max_tokens):
    out = [None] * concurrency

    def worker(i):
        try:
            out[i] = one_stream(chat_url, model, PROMPTS[i % len(PROMPTS)], max_tokens)
        except Exception as e:  # recorded and counted, so one failed stream cannot pass as a fast run
            out[i] = {"error": repr(e)}

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    wall = time.monotonic() - t0
    good = [s for s in out if "error" not in s]
    return {"concurrency": concurrency, "wall_s": round(wall, 3), "errors": concurrency - len(good),
            "aggregate_tok_s": round(sum(s["tokens"] for s in good) / wall, 2), "streams": out}


def summarize(runs):
    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 3) if xs else None

    streams = [s for r in runs for s in r["streams"] if "error" not in s]
    return {"concurrency": runs[0]["concurrency"], "runs": len(runs), "errors": sum(r["errors"] for r in runs),
            "median_ttft_s": med(s["ttft_s"] for s in streams),
            "median_decode_tok_s": med(s["decode_tok_s"] for s in streams),
            "median_aggregate_tok_s": med(r["aggregate_tok_s"] for r in runs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--concurrency", default="1,4", help="comma list, e.g. 1,4,8")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--url", default=recipe.default_url())
    ap.add_argument("--model", default=recipe.setting("SERVED_NAME"))
    ap.add_argument("--out", help="write settings, every run and the summary here as JSON")
    a = ap.parse_args(argv)
    chat_url = a.url + "/v1/chat/completions"
    levels = [int(c) for c in a.concurrency.split(",")]

    report = {"settings": {"url": a.url, "model": a.model, "concurrency": levels, "runs": a.runs,
                           "max_tokens": a.max_tokens, "thinking": False, "ignore_eos": True,
                           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              "runs": [], "summary": []}
    print(f"{'c':>3} {'runs':>4} {'err':>3} {'ttft_s':>8} {'decode tok/s':>13} {'aggregate tok/s':>16}")
    for c in levels:
        runs = [run_once(chat_url, a.model, c, a.max_tokens) for _ in range(a.runs)]
        s = summarize(runs)
        report["runs"] += runs
        report["summary"].append(s)
        print(f"{c:>3} {s['runs']:>4} {s['errors']:>3} {s['median_ttft_s']!s:>8} {s['median_decode_tok_s']!s:>13} "
              f"{s['median_aggregate_tok_s']!s:>16}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    return 1 if any(s["errors"] for s in report["summary"]) else 0


if __name__ == "__main__":
    sys.exit(main())
