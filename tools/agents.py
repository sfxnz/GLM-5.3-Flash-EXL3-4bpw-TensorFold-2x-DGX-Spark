"""Agent-shaped load: N clients hold multi-turn tool-using chats at once on one shared system prompt, streamed, thinking off.

python3 tools/agents.py [--clients 4] [--turns 4] [--system-tokens 8000] [--context-tokens 0] [--max-tokens 256]
                        [--temperature T] [--stagger 0] [--out evidence/run/agents.json]

Every request carries the same seeded system prompt (tool guide and project rules) and the same tools array, so all
clients can share that prefix. --context-tokens puts a seeded document of that size, different for each client, in the
client's first user message, so only that client's own prefix holds it. Turn 1 is cold; each later turn resends the
client's whole history and is resumed, so its TTFT shows whether the engine kept that client's prefix. Tool calls are
rebuilt from their streamed pieces and answered with a canned tool result each before the next user message. Sizes
assume 4 characters a token; prompt_tokens records what the server counted. No ignore_eos: replies end where the model
ends them, as an agent's do.

Per turn: ttft_s is the first content, tool-call or reasoning delta, whichever comes first; decode_tok_s is
(completion_tokens - 1) over first-to-last such delta; cached_tokens is usage.prompt_tokens_details.cached_tokens, null
when the server sends none. Per kind (cold, resumed): median and max ttft_s, median decode_tok_s, errors.
aggregate_tok_s is all completion tokens over the run's wall time; resumed_over_cold is median resumed ttft_s over
median cold ttft_s. A client stops at its first failed turn. Exits 1 if any turn errored.
"""
import argparse
import hashlib
import json
import random
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe  # noqa: E402

CHARS_PER_TOKEN = 4
KINDS = ("cold", "resumed")


def tool(name, description, **params):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
        "required": list(params)}}}


TOOLS = [
    tool("read_file", "Read a file from the repository.", path="path from the repository root"),
    tool("grep", "Search the repository for a regular expression.", pattern="regular expression", path="directory"),
    tool("run_shell", "Run a shell command at the repository root and return its output.", command="command line"),
    tool("write_file", "Replace a file's contents.", path="path from the repository root", content="new contents"),
]
TASKS = [
    "Find where the retry limit for outgoing requests is set and explain how it is used.",
    "List the functions that write to disk and say which ones skip error handling.",
    "The nightly job fails with a timeout in the scheduler. Find the likely cause.",
    "Find every place the session token is read and check that each one validates it.",
]
FOLLOWUPS = [
    "Now find the tests that cover it.",
    "Show the exact change you would make.",
    "Run the relevant tests and summarize any failures.",
    "What else depends on this code?",
]
TOOL_RESULT = "{} finished: 3 lines of output, elided by the harness."
MODULES = ["auth", "billing", "cache", "scheduler", "storage", "router", "metrics", "search"]
VERBS = ["read", "lock", "validate", "log", "flush", "check", "back up", "version"]
THINGS = ["config", "session table", "queue", "index", "ledger", "manifest", "schema", "token store"]


def fill(tokens, rng, line):
    out, size = [], 0
    while size < tokens * CHARS_PER_TOKEN:
        out.append(line(rng, len(out)))
        size += len(out[-1])
    return "".join(out)


def system_prompt(tokens):
    head = ("You are a coding agent working in a large repository. Look with the tools before you change anything, "
            "keep edits small, and say what you did.\n\nTools:\n" + json.dumps(TOOLS, indent=1)
            + "\n\nProject rules:\n")
    return head + fill(tokens - len(head) // CHARS_PER_TOKEN, random.Random(11), lambda r, i: (
        f"Rule {i + 1}: in the {r.choice(MODULES)} module, {r.choice(VERBS)} the {r.choice(THINGS)} before you "
        f"change any {r.choice(THINGS)} code.\n"))


def first_message(client, context_tokens):
    task = TASKS[client % len(TASKS)]
    if not context_tokens:
        return task
    listing = fill(context_tokens, random.Random(1000 + client), lambda r, i: (
        f"src/{r.choice(MODULES)}/part_{r.randint(1, 99)}.py:{r.randint(1, 2000)}: def {r.choice(MODULES)}_"
        f"{r.randint(100, 999)}(limit={r.randint(1, 500)}): returns the {r.choice(THINGS)}\n"))
    return f"Here is the code listing for this task:\n{listing}\n{task}"


def one_turn(chat_url, body):
    """Streams one reply; returns (the assistant message to append, the turn's numbers)."""
    first = last = usage = None
    text, calls, deltas = [], {}, 0
    for t, chunk in recipe.stream(chat_url, body):
        for c in chunk.get("choices", []):
            d = c.get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
                first = t if first is None else first
                last, deltas = t, deltas + 1
            text.append(d.get("content") or "")
            for piece in d.get("tool_calls") or []:
                call = calls.setdefault(piece["index"], {"id": None, "type": "function",
                                                         "function": {"name": "", "arguments": ""}})
                fn = piece.get("function") or {}
                call["id"] = piece.get("id") or call["id"]
                call["function"]["name"] = fn.get("name") or call["function"]["name"]
                call["function"]["arguments"] += fn.get("arguments") or ""
        usage = chunk.get("usage") or usage
    reply = {"role": "assistant", "content": "".join(text)}
    if calls:
        reply["tool_calls"] = [calls[i] for i in sorted(calls)]
    usage = usage or {}
    tokens = usage.get("completion_tokens", deltas)
    return reply, {"prompt_tokens": usage.get("prompt_tokens"),
                   "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   "ttft_s": round(first, 3) if first is not None else None,
                   "decode_tok_s": round((tokens - 1) / (last - first), 2) if first is not None and last > first
                   else None, "completion_tokens": tokens, "tool_calls": len(calls)}


def run_client(i, a, chat_url, system, turns):
    time.sleep(i * a.stagger)
    history = []
    for n in range(1, a.turns + 1):
        history.append({"role": "user", "content": first_message(i, a.context_tokens) if n == 1
                        else FOLLOWUPS[(i + n) % len(FOLLOWUPS)]})
        body = {"model": a.model, "messages": [{"role": "system", "content": system}, *history], "tools": TOOLS,
                "max_tokens": a.max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
        if a.temperature is not None:
            body["temperature"] = a.temperature
        row = {"client": i, "turn": n, "kind": "cold" if n == 1 else "resumed", "prompt_tokens": None,
               "cached_tokens": None, "ttft_s": None, "decode_tok_s": None, "completion_tokens": None, "tool_calls": 0,
               "error": None}
        turns.append(row)
        try:
            reply, numbers = one_turn(chat_url, body)
        except Exception as e:  # recorded and counted; the conversation cannot go on past a turn with no reply
            row["error"] = repr(e)
            return
        row.update(numbers)
        history.append(reply)
        history += [{"role": "tool", "tool_call_id": c["id"], "content": TOOL_RESULT.format(c["function"]["name"])}
                    for c in reply.get("tool_calls", [])]


def summarize(turns, wall):
    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 3) if xs else None

    s = {}
    for kind in KINDS:
        ts = [t for t in turns if t["kind"] == kind]
        ttft = [t["ttft_s"] for t in ts if t["ttft_s"] is not None]
        s[kind] = {"turns": len(ts), "errors": sum(t["error"] is not None for t in ts), "median_ttft_s": med(ttft),
                   "max_ttft_s": max(ttft, default=None), "median_decode_tok_s": med(t["decode_tok_s"] for t in ts),
                   "median_prompt_tokens": med(t["prompt_tokens"] for t in ts),
                   "median_cached_tokens": med(t["cached_tokens"] for t in ts)}
    cold, resumed = s["cold"]["median_ttft_s"], s["resumed"]["median_ttft_s"]
    s.update(wall_s=round(wall, 3), aggregate_tok_s=round(sum(t["completion_tokens"] or 0 for t in turns) / wall, 2),
             resumed_over_cold=round(resumed / cold, 3) if cold and resumed is not None else None)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--system-tokens", type=int, default=8000, help="shared system prompt size, tool guide included")
    ap.add_argument("--context-tokens", type=int, default=0, help="each client's own document in its first message")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, help="sent only when set; the server's default otherwise")
    ap.add_argument("--stagger", type=float, default=0.0, help="seconds between client starts")
    ap.add_argument("--url", default=recipe.default_url())
    ap.add_argument("--model", default=recipe.setting("SERVED_NAME"))
    ap.add_argument("--out", help="write settings, every turn and the summary here as JSON")
    a = ap.parse_args(argv)
    chat_url = a.url + "/v1/chat/completions"
    system = system_prompt(a.system_tokens)

    turns = []
    t0 = time.monotonic()
    threads = [threading.Thread(target=run_client, args=(i, a, chat_url, system, turns)) for i in range(a.clients)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    turns.sort(key=lambda t: (t["client"], t["turn"]))
    s = summarize(turns, time.monotonic() - t0)

    report = {"settings": {"url": a.url, "model": a.model, "clients": a.clients, "turns": a.turns,
                           "system_tokens": a.system_tokens, "context_tokens": a.context_tokens,
                           "max_tokens": a.max_tokens, "temperature": a.temperature, "stagger_s": a.stagger,
                           "thinking": False, "tools": [t["function"]["name"] for t in TOOLS],
                           "chars_per_token": CHARS_PER_TOKEN,
                           "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
                           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              "turns": turns, "summary": s}
    print(f"{'kind':<8} {'turns':>5} {'err':>3} {'ttft_s':>8} {'max_ttft':>8} {'decode tok/s':>12} {'prompt_tok':>10} "
          f"{'cached_tok':>10}")
    for kind in KINDS:
        k = s[kind]
        print(f"{kind:<8} {k['turns']:>5} {k['errors']:>3} {k['median_ttft_s']!s:>8} {k['max_ttft_s']!s:>8} "
              f"{k['median_decode_tok_s']!s:>12} {k['median_prompt_tokens']!s:>10} {k['median_cached_tokens']!s:>10}")
    print(f"aggregate {s['aggregate_tok_s']} tok/s over {s['wall_s']} s; resumed/cold ttft {s['resumed_over_cold']}",
          flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    return 1 if any(t["error"] for t in turns) else 0


if __name__ == "__main__":
    sys.exit(main())
