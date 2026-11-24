"""Cold prompt speed: one streamed request per prompt length, each prompt new to the server, timed to its first token.

python3 tools/prefill.py 8000,32000,128000 [--runs 1] [--seed N] [--url http://127.0.0.1:8000] [--out prefill.json]

Each prompt opens with a line no earlier prompt had (the seed, the length and the run), so no kept state can cover it;
the lines after are needle.py's seeded warehouse records. prefill_tok_s is prompt_tokens over ttft_s. cached_tokens is
usage.prompt_tokens_details.cached_tokens; a run with 1% or more of its prompt cached is marked cold: false and left
out of the medians (the chat template's opening tokens are the same in every prompt).
--seed defaults to the clock, so a rerun on the same boot is cold again. Exits 1 if a request fails or none ran cold.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import needle  # noqa: E402
import recipe  # noqa: E402


def one(chat_url, model, text):
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 8,
            "chat_template_kwargs": {"enable_thinking": False}}
    first, usage = None, {}
    for t, chunk in recipe.stream(chat_url, body, timeout=7200):
        if first is None and any(c.get("delta", {}).get("content") for c in chunk.get("choices", [])):
            first = t
        usage = chunk.get("usage") or usage
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached,
            "cold": (cached or 0) * 100 < usage.get("prompt_tokens", 0),
            "ttft_s": round(first, 3) if first is not None else None,
            "prefill_tok_s": round(usage["prompt_tokens"] / first, 1) if first and usage.get("prompt_tokens") else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lengths", help="comma list of prompt tokens to aim for, e.g. 8000,32000")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=int(time.time()), help="names this invocation's prompts")
    ap.add_argument("--url", default=recipe.default_url())
    ap.add_argument("--model", default=recipe.setting("SERVED_NAME"))
    ap.add_argument("--out", help="write settings, every run and the summary here as JSON")
    a = ap.parse_args(argv)
    chat_url = a.url + "/v1/chat/completions"

    probe = needle.records(400, a.seed)
    status, d = recipe.post(chat_url, {"model": a.model, "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": False},
                                       "messages": [{"role": "user", "content": f"Probe {a.seed}.\n" + "".join(probe)}]})
    if status != 200:
        print(json.dumps({"event": "error", "stage": "calibration", "status": status, "body": d}))
        return 1
    per_line = d["usage"]["prompt_tokens"] / len(probe)

    report = {"settings": {"url": a.url, "model": a.model, "seed": a.seed, "runs": a.runs, "thinking": False,
                           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, "runs": [], "summary": []}
    print(f"{'target':>8} {'prompt_tok':>10} {'cached':>7} {'ttft_s':>8} {'prefill tok/s':>14}")
    failed = False
    for target in [int(x) for x in a.lengths.split(",")]:
        runs = []
        for r in range(a.runs):
            lines = needle.records(max(1, int(target / per_line)), a.seed + 1000 * r + target)
            try:
                run = one(chat_url, a.model, f"Batch {a.seed}-{target}-{r}.\n" + "".join(lines) + "\nReply OK.")
            except Exception as e:  # recorded, so a failed request cannot pass as a missing row
                run = {"error": repr(e), "cold": False}
                failed = True
            runs.append({"target": target, "run": r, **run})
            print(f"{target:>8} {run.get('prompt_tokens')!s:>10} {run.get('cached_tokens')!s:>7} "
                  f"{run.get('ttft_s')!s:>8} {run.get('prefill_tok_s')!s:>14}", flush=True)
        cold = [x for x in runs if x["cold"] and x.get("prefill_tok_s")]
        report["runs"] += runs
        report["summary"].append({
            "target": target, "cold_runs": len(cold),
            "median_prompt_tokens": statistics.median(x["prompt_tokens"] for x in cold) if cold else None,
            "median_ttft_s": round(statistics.median(x["ttft_s"] for x in cold), 3) if cold else None,
            "median_prefill_tok_s": round(statistics.median(x["prefill_tok_s"] for x in cold), 1) if cold else None})
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    return 1 if failed or any(s["cold_runs"] == 0 for s in report["summary"]) else 0


if __name__ == "__main__":
    sys.exit(main())
