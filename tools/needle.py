"""Needle in a haystack of about TARGET prompt tokens: one vault code hidden among warehouse records, asked back at the end.

python3 tools/needle.py 500000 [--depth 0.37] [--concurrent 1] [--url http://127.0.0.1:8000] [--out needle.json]

A target past the served window is cut to fit it; a context_length_exceeded refusal rescales the haystack and retries.
Prints one JSON line per event and exits 0 only when the code comes back. --concurrent K sends K needles at once, each at
the target with its own haystack seed, archive and code, so no two share a cached prefix and an answer that came from
another request cannot count. It prints one result line per needle, with its index, then a summary line, and exits 0
only when all K are found.
"""
import argparse
import json
import random
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe  # noqa: E402

CITIES = ["Lisbon", "Osaka", "Quito", "Tromso", "Nairobi", "Hobart", "Tbilisi", "Cusco", "Gdansk", "Leeds"]
ITEMS = ["lanterns", "barrels", "ledgers", "copper pipes", "wool bales", "clock gears", "maps", "seed sacks"]
WORDS = ["ALPHA", "BRAVO", "CEDAR", "DELTA", "EMBER", "FJORD", "GAMMA", "HAZEL"]
QUESTION = ("\nIn the records above, one line states a vault code. What is the vault code for the {} archive? "
            "Reply with the code only.")
# Room left for the chat template and the question beyond the counted lines.
MARGIN = 256
RETRIES = 3
LOCK = threading.Lock()


def needle(k):
    """Needle k's (haystack seed, archive, code). Needle 0 is the one a single run has always used."""
    return 7 + k, CITIES[(5 + k) % len(CITIES)], f"{7319 + 97 * k}-{WORDS[k % len(WORDS)]}-{42 + k}"


def records(n, seed=7):
    rng = random.Random(seed)
    return [f"Record {i}: the warehouse in {rng.choice(CITIES)} logged {rng.randint(10, 9999)} {rng.choice(ITEMS)} "
            f"on shelf {rng.randint(1, 400)} during shift {rng.randint(1, 3)}.\n" for i in range(n)]


def emit(event, **fields):
    with LOCK:
        print(json.dumps({"event": event, **fields}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", type=int, help="prompt tokens to aim for")
    ap.add_argument("--depth", type=float, default=0.37, help="where the code sits, 0 = start, 1 = end")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--concurrent", type=int, default=1, help="needles at once, each with its own haystack and code")
    ap.add_argument("--url", default=recipe.default_url())
    ap.add_argument("--model", default=recipe.setting("SERVED_NAME"))
    ap.add_argument("--out", help="append the result line(s) to this file")
    a = ap.parse_args(argv)
    chat_url = a.url + "/v1/chat/completions"

    def ask(text, max_tokens, timeout=7200):
        body = {"model": a.model, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": text}]}
        t0 = time.monotonic()
        status, d = recipe.post(chat_url, body, timeout=timeout)
        return status, d, time.monotonic() - t0

    _, models = recipe.get(a.url + "/v1/models")
    window = next((m.get("context_length") for m in models.get("data", []) if m.get("id") == a.model), None)
    target = a.target
    if window and target + a.max_tokens + MARGIN > window:
        target = window - a.max_tokens - MARGIN
        emit("rescaled", reason=f"target {a.target} does not fit the {window}-token window", target=target)

    probe = records(400)
    status, d, _ = ask("".join(probe) + "Reply OK.", 1, timeout=600)
    if status != 200:
        emit("error", stage="calibration", status=status, body=d)
        return 2
    per_line = (d["usage"]["prompt_tokens"] - 10) / len(probe)

    def hunt(k, target):
        """Needle k's result line, or None once an error stopped it (the error event is already printed)."""
        seed, archive, code = needle(k)
        tag = {"index": k} if a.concurrent > 1 else {}
        for _ in range(RETRIES + 1):
            n = max(1, int(target / per_line))
            lines = records(n, seed)
            lines.insert(int(n * a.depth), f"The vault code for the {archive} archive is {code}.\n")
            status, d, wall = ask("".join(lines) + QUESTION.format(archive), a.max_tokens)
            err = d.get("error") or {}
            if status == 200:
                break
            if err.get("code") != "context_length_exceeded":
                emit("error", **tag, stage="needle", status=status, body=d)
                return None
            msg = err.get("message", "")
            has = re.search(r"rendered prompt has (\d+) tokens", msg)
            most = re.search(r"at most (\d+) prompt tokens", msg)
            fit = int(most.group(1)) if most else (window or target) - a.max_tokens
            new = int(target * (fit - MARGIN) / int(has.group(1))) if has else int(target * 0.95)
            emit("refused", **tag, code="context_length_exceeded", message=msg, target=target, rescaled_to=new)
            target = new
        else:
            emit("error", **tag, stage="needle", reason=f"still over the window after {RETRIES} rescales")
            return None

        answer = d["choices"][0]["message"].get("content") or ""
        if tag:
            tag.update(archive=archive, vault_code=code)
        return {"event": "result", **tag, "target": a.target, "fitted_target": target,
                "tokens_per_line": round(per_line, 3), "prompt_tokens": d["usage"]["prompt_tokens"],
                "needle_depth": a.depth, "answer": answer, "found": code in answer, "wall_s": round(wall, 1),
                "prefill_tok_s": round(d["usage"]["prompt_tokens"] / wall, 1)}

    results = [None] * a.concurrent

    def worker(k):
        try:
            results[k] = hunt(k, target)
        except Exception as e:  # a dropped connection costs its own needle, not the other needles' report
            emit("error", index=k, stage="needle", reason=repr(e))

    if a.concurrent == 1:
        results[0] = hunt(0, target)
    else:
        threads = [threading.Thread(target=worker, args=(k,)) for k in range(a.concurrent)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    done = [r for r in results if r]
    lines = done if a.concurrent == 1 else done + [{"event": "summary", "found": sum(r["found"] for r in done),
                                                    "needles": a.concurrent, "errors": a.concurrent - len(done)}]
    for r in lines:
        print(json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in lines)
    if len(done) < a.concurrent:
        return 2
    return 0 if all(r["found"] for r in done) else 1


if __name__ == "__main__":
    sys.exit(main())
