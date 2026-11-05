"""Needle in a haystack of about TARGET prompt tokens: one vault code hidden among warehouse records, asked back at the end.

python3 tools/needle.py 500000 [--depth 0.37] [--url http://127.0.0.1:8000] [--out needle.json]

A target past the served window is cut to fit it; a context_length_exceeded refusal rescales the haystack and retries.
Prints one JSON line per event and exits 0 only when the code comes back.
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe  # noqa: E402

CITIES = ["Lisbon", "Osaka", "Quito", "Tromso", "Nairobi", "Hobart", "Tbilisi", "Cusco", "Gdansk", "Leeds"]
ITEMS = ["lanterns", "barrels", "ledgers", "copper pipes", "wool bales", "clock gears", "maps", "seed sacks"]
CODE = "7319-ALPHA-42"
SECRET = f"The vault code for the Hobart archive is {CODE}.\n"
QUESTION = ("\nIn the records above, one line states a vault code. What is the vault code for the Hobart archive? "
            "Reply with the code only.")
# Room left for the chat template and the question beyond the counted lines.
MARGIN = 256
RETRIES = 3


def records(n):
    rng = random.Random(7)
    return [f"Record {i}: the warehouse in {rng.choice(CITIES)} logged {rng.randint(10, 9999)} {rng.choice(ITEMS)} "
            f"on shelf {rng.randint(1, 400)} during shift {rng.randint(1, 3)}.\n" for i in range(n)]


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", type=int, help="prompt tokens to aim for")
    ap.add_argument("--depth", type=float, default=0.37, help="where the code sits, 0 = start, 1 = end")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--url", default=recipe.default_url())
    ap.add_argument("--model", default=recipe.setting("SERVED_NAME"))
    ap.add_argument("--out", help="append the result line to this file")
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

    for _ in range(RETRIES + 1):
        n = max(1, int(target / per_line))
        lines = records(n)
        lines.insert(int(n * a.depth), SECRET)
        status, d, wall = ask("".join(lines) + QUESTION, a.max_tokens)
        err = d.get("error") or {}
        if status == 200:
            break
        if err.get("code") != "context_length_exceeded":
            emit("error", stage="needle", status=status, body=d)
            return 2
        msg = err.get("message", "")
        has = re.search(r"rendered prompt has (\d+) tokens", msg)
        most = re.search(r"at most (\d+) prompt tokens", msg)
        fit = int(most.group(1)) if most else (window or target) - a.max_tokens
        new = int(target * (fit - MARGIN) / int(has.group(1))) if has else int(target * 0.95)
        emit("refused", code="context_length_exceeded", message=msg, target=target, rescaled_to=new)
        target = new
    else:
        emit("error", stage="needle", reason=f"still over the window after {RETRIES} rescales")
        return 2

    answer = d["choices"][0]["message"].get("content") or ""
    result = {"event": "result", "target": a.target, "fitted_target": target, "tokens_per_line": round(per_line, 3),
              "prompt_tokens": d["usage"]["prompt_tokens"], "needle_depth": a.depth, "answer": answer,
              "found": CODE in answer, "wall_s": round(wall, 1),
              "prefill_tok_s": round(d["usage"]["prompt_tokens"] / wall, 1)}
    print(json.dumps(result), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(result) + "\n")
    return 0 if result["found"] else 1


if __name__ == "__main__":
    sys.exit(main())
