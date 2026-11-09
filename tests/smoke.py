"""Smoke checks against the running pair. Exits 1 if any check fails.

python3 tests/smoke.py [--url http://127.0.0.1:8000] [--out evidence/run/smoke.json]
"""
import argparse
import base64
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import recipe  # noqa: E402

QUESTION = "What is 17 * 23? Reply with the number only."
WEATHER = "Look up the weather in Paris, France in celsius. Use the tool."
SCHEMA = {"type": "object", "required": ["city", "unit"],
          "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}}
PROSE = [
    "Write a detailed paragraph about the history of the printing press.",
    "Explain how a CPU pipeline handles a branch misprediction.",
    "Describe the water cycle and its effect on regional climates.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
]


def shapes_png(w=256, h=128):
    """White canvas, a red square on the left half and a blue circle on the right half."""
    rows = []
    for y in range(h):
        row = bytearray(b"\x00")
        for x in range(w):
            if 24 <= x < 104 and 24 <= y < 104:
                row += b"\xff\x00\x00"
            elif (x - 192) ** 2 + (y - 64) ** 2 <= 40 ** 2:
                row += b"\x00\x00\xff"
            else:
                row += b"\xff\xff\xff"
        rows.append(bytes(row))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


class Smoke:
    def __init__(self, url, model, context, parallel):
        self.url, self.model, self.context, self.parallel = url, model, context, parallel
        self.chat_url = url + "/v1/chat/completions"

    def chat(self, content, thinking=True, **extra):
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], **extra}
        if not thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return recipe.post(self.chat_url, body)

    def health(self):
        status, body = recipe.get(self.url + "/health")
        return status == 200, body

    def models(self):
        _, body = recipe.get(self.url + "/v1/models")
        ours = [m for m in body.get("data", []) if m.get("id") == self.model]
        ok = bool(ours) and ours[0].get("context_length") == self.context
        return ok, {"want": {"id": self.model, "context_length": self.context}, "got": body}

    def thinking_on(self):
        status, d = self.chat(QUESTION, max_tokens=4096)
        msg = d.get("choices", [{}])[0].get("message", {}) if status == 200 else {}
        reasoning, content = msg.get("reasoning_content") or "", msg.get("content") or ""
        ok = bool(reasoning.strip()) and "391" in content and "<think>" not in content and "</think>" not in content
        return ok, {"status": status, "reasoning_chars": len(reasoning), "content": content, "raw": d}

    def thinking_off(self):
        status, d = self.chat(QUESTION, thinking=False, max_tokens=64)
        msg = d.get("choices", [{}])[0].get("message", {}) if status == 200 else {}
        reasoning, content = msg.get("reasoning_content") or "", msg.get("content") or ""
        ok = status == 200 and not reasoning and "391" in content and "<think>" not in content and "</think>" not in content
        return ok, {"status": status, "reasoning_content": reasoning, "content": content, "raw": d}

    def vision(self):
        image = "data:image/png;base64," + base64.b64encode(shapes_png()).decode()
        ask = ('The image shows two shapes. Reply with JSON only, no prose, in this form: '
               '{"left": {"color": "...", "shape": "..."}, "right": {"color": "...", "shape": "..."}}')
        status, d = self.chat([{"type": "image_url", "image_url": {"url": image}}, {"type": "text", "text": ask}],
                              thinking=False, max_tokens=200, temperature=0)
        content = d.get("choices", [{}])[0].get("message", {}).get("content") or "" if status == 200 else ""
        m = re.search(r"\{.*\}", content, re.S)
        try:
            got = json.loads(m.group(0)) if m else {}
            left, right = got["left"], got["right"]
            ok = ("red" in left["color"].lower() and "square" in left["shape"].lower()
                  and "blue" in right["color"].lower() and "circle" in right["shape"].lower())
        except (ValueError, KeyError, TypeError, AttributeError):
            ok = False
        return ok, {"status": status, "content": content}

    def speculative(self):
        status, d = self.chat(PROSE[0], thinking=False, max_tokens=256)
        spec = d.get("speculative") or {}
        return status == 200 and (spec.get("accepted") or 0) > 0, {"status": status, "speculative": spec}

    def concurrent(self):
        out = [None] * self.parallel

        def one(i):
            t0 = time.monotonic()
            try:
                status, d = self.chat(PROSE[i % len(PROSE)], thinking=False, max_tokens=128)
            except OSError as e:
                out[i] = {"status": None, "error": repr(e)}
                return
            content = d.get("choices", [{}])[0].get("message", {}).get("content") or "" if status == 200 else ""
            out[i] = {"status": status, "chars": len(content), "elapsed_s": round(time.monotonic() - t0, 2)}

        threads = [threading.Thread(target=one, args=(i,)) for i in range(self.parallel)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        return all(o and o["status"] == 200 and o["chars"] > 0 for o in out), {"requests": out}

    def tool_stream(self):
        body = {"model": self.model, "messages": [{"role": "user", "content": WEATHER}], "max_tokens": 200,
                "temperature": 0, "tool_choice": "required", "chat_template_kwargs": {"enable_thinking": False},
                "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                                            "parameters": SCHEMA}}]}
        pieces, names = [], []
        for _, chunk in recipe.stream(self.chat_url, body):
            for c in chunk.get("choices", []):
                for call in (c.get("delta") or {}).get("tool_calls") or []:
                    fn = call.get("function") or {}
                    names += [fn["name"]] if fn.get("name") else []
                    pieces += [fn["arguments"]] if fn.get("arguments") else []
        status, d = recipe.post(self.chat_url, body)
        calls = d.get("choices", [{}])[0].get("message", {}).get("tool_calls") or [] if status == 200 else []
        final = calls[0]["function"]["arguments"] if calls else None
        ok = names == ["get_weather"] and len(pieces) >= 2 and "".join(pieces) == final and set(json.loads(final)) == {"city", "unit"}
        return ok, {"names": names, "pieces": pieces, "final": final}

    def messages(self):
        status, d = recipe.post(self.url + "/v1/messages", {"model": self.model, "max_tokens": 64,
                                                           "messages": [{"role": "user", "content": QUESTION}]})
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text") if status == 200 else ""
        return status == 200 and "391" in text and d.get("stop_reason") == "end_turn", {"status": status, "raw": d}

    def messages_stream(self):
        body = {"model": self.model, "max_tokens": 200, "temperature": 0, "tool_choice": {"type": "any"},
                "messages": [{"role": "user", "content": WEATHER}],
                "tools": [{"name": "get_weather", "description": "Current weather for a city", "input_schema": SCHEMA}]}
        events = [e for _, e in recipe.stream(self.url + "/v1/messages", body)]
        kinds = [e.get("type") for e in events]
        starts = [e["content_block"] for e in events if e.get("type") == "content_block_start"]
        pieces = [e["delta"]["partial_json"] for e in events if (e.get("delta") or {}).get("type") == "input_json_delta"]
        try:
            args = json.loads("".join(pieces))
        except ValueError:
            args = None
        ok = (kinds[:1] == ["message_start"] and kinds[-1:] == ["message_stop"] and len(pieces) >= 2
              and [b.get("type") for b in starts] == ["tool_use"] and isinstance(args, dict) and set(args) == {"city", "unit"})
        return ok, {"events": kinds, "pieces": pieces}

    def auth(self):
        try:
            with urllib.request.urlopen(self.url + "/v1/models", timeout=30) as r:  # no key headers
                status = r.status
        except urllib.error.HTTPError as e:
            status = e.code
        return status == 401, {"status_without_key": status}

    CHECKS = ["health", "models", "thinking_on", "thinking_off", "vision", "speculative", "concurrent", "tool_stream",
              "messages", "messages_stream"]

    def run(self):
        results = {}
        for name in self.CHECKS + (["auth"] if os.environ.get("API_KEY") else []):
            try:
                ok, detail = getattr(self, name)()
            except Exception as e:  # a dead server or a malformed reply fails the check, not the run
                ok, detail = False, {"exception": repr(e)}
            results[name] = {"ok": ok, "detail": detail}
            shown = {k: v for k, v in detail.items() if k != "raw"} if isinstance(detail, dict) else detail
            print(f"{'PASS' if ok else 'FAIL'}  {name}  {json.dumps(shown)[:300]}", flush=True)
        return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=recipe.default_url())
    ap.add_argument("--model", default=recipe.setting("SERVED_NAME"))
    ap.add_argument("--context", type=int, default=int(recipe.setting("CONTEXT")))
    ap.add_argument("--parallel", type=int, default=int(recipe.setting("PARALLEL")))
    ap.add_argument("--out", help="write every check's full result here as JSON")
    a = ap.parse_args(argv)
    results = Smoke(a.url, a.model, a.context, a.parallel).run()
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2) + "\n")
    failed = [n for n, r in results.items() if not r["ok"]]
    print(f"smoke: {len(results) - len(failed)}/{len(results)} passed" + (f"; failed: {' '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
