"""A stand-in for TensorFold's OpenAI surface, shaped like v1.2.2's replies, so the smoke, needle and bench scripts run on a CPU.

Fault knobs (FakeServer(**knobs)): leak_think puts <think> into thinking-off content, accepted sets the speculative
block's accepted count, vision sets the image answer, window sets the context window the server admits in tokens (default: recipe.conf's CONTEXT)
(4 characters a token), and reported_window what /v1/models claims (default: window).

Chat prompts count every message and the tools array. A request with tools whose last message is the user's gets two
streamed tool calls, to the first two tools, the name in the first piece and the arguments in fragments after.
prefix_cache adds usage.prompt_tokens_details.cached_tokens, the longest prefix shared with any earlier prompt;
prefill_s_per_ktok sleeps that long per uncached thousand prompt tokens before replying. crosstalk answers each vault
question with the code from the vault request before it, as a server that mixed up concurrent requests would.
A get_weather tool gets {"city": "Paris, France", "unit": "celsius"}; whole_tool_args streams each call's arguments in
one piece, and drift_tool_args makes the non-streamed arguments differ from the streamed ones. /v1/messages answers in
the Anthropic shape (as v1.2.5's, tool_use streamed as input_json_delta pieces) from the same replies. api_key makes
every route but /health answer 401 without that key in Authorization: Bearer or x-api-key.

/v1/completions answers prompt_logprobs as v1.2.2 does (zig/src/server/logprobs.zig promptValue): null for the first
prompt token, then {id: {logprob, rank, decoded_token}} with the prompt token first. Its tokens are words; nll_shift
lowers every prompt token's logprob by that much, jitter adds seeded noise of up to that size, and id_salt tokenizes
differently.
"""
import json
import os
import random
import re
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import recipe  # noqa: E402

VOCAB = 154880
GOOD_VISION = '{"left": {"color": "red", "shape": "square"}, "right": {"color": "blue", "shape": "circle"}}'


class FakeServer:
    def __init__(self, model="glm-5.3-flash", window=int(recipe.setting("CONTEXT")), leak_think=False, accepted=5, vision=GOOD_VISION,
                 reported_window=None, nll_shift=0.0, jitter=0.0, id_salt="", prefix_cache=False,
                 prefill_s_per_ktok=0.0, crosstalk=False, whole_tool_args=False, drift_tool_args=False, api_key=None):
        self.model, self.window, self.leak_think, self.accepted, self.vision = model, window, leak_think, accepted, vision
        self.nll_shift, self.jitter, self.id_salt = nll_shift, jitter, id_salt
        self.prefix_cache, self.prefill_s_per_ktok, self.crosstalk = prefix_cache, prefill_s_per_ktok, crosstalk
        self.whole_tool_args, self.drift_tool_args, self.api_key = whole_tool_args, drift_tool_args, api_key
        self.reported_window = reported_window or window
        self.requests, self.prompts, self.last_code, self.lock = [], [], None, threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, body):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def refused(self):
                keys = {self.headers.get("x-api-key"), (self.headers.get("authorization") or "").removeprefix("Bearer ")}
                if self.path == "/health" or fake.api_key is None or fake.api_key in keys:
                    return False
                self.send(401, {"error": {"message": "invalid API key"}})
                return True

            def sse(self, events):
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for e in events:
                    self.wfile.write(b"data: " + json.dumps(e).encode() + b"\n\n")

            def do_GET(self):
                if self.refused():
                    return
                if self.path == "/health":
                    self.send(200, {"status": "ok", "model": fake.model, "max_batch_size": int(recipe.setting("PARALLEL"))})
                elif self.path == "/v1/models":
                    self.send(200, {"object": "list", "data": [{"id": fake.model, "object": "model",
                                                                "context_length": fake.reported_window,
                                                                "max_model_len": fake.reported_window}]})
                else:
                    self.send(404, {"error": {"message": "not found"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                fake.requests.append(body)
                if self.refused():
                    return
                if self.path == "/v1/messages":
                    status, reply = fake.message(body)
                    if status != 200 or not body.get("stream"):
                        return self.send(status, reply)
                    return self.sse(fake.message_events(reply))
                if self.path == "/v1/completions":
                    return self.send(*fake.complete(body))
                if "prompt_logprobs" in body:
                    return self.send(400, {"error": {"message": "prompt_logprobs are supported on /v1/completions "
                                                                "without streaming only"}})
                status, reply = fake.reply(body)
                if status != 200 or not body.get("stream"):
                    return self.send(status, reply)
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()

                def chunk(delta):
                    self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": delta}]}).encode() + b"\n\n")

                message = reply["choices"][0]["message"]
                for i, call in enumerate(message.get("tool_calls") or []):
                    fn = call["function"]
                    chunk({"tool_calls": [{"index": i, "id": call["id"], "type": "function",
                                           "function": {"name": fn["name"], "arguments": ""}}]})
                    for j in range(0, len(fn["arguments"]), len(fn["arguments"]) if fake.whole_tool_args else 5):
                        chunk({"tool_calls": [{"index": i, "function": {"arguments": fn["arguments"][j:j + 5]}}]})
                content = message["content"] or ""
                for i in range(0, len(content), 8):
                    chunk({"content": content[i:i + 8]})
                finish = "tool_calls" if message.get("tool_calls") else "length"
                final = {"choices": [{"delta": {}, "finish_reason": finish}], "usage": reply["usage"],
                         "speculative": reply["speculative"]}
                self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\ndata: [DONE]\n\n")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    @staticmethod
    def parts(message):
        c = message.get("content") or ""
        return c if isinstance(c, list) else [{"type": "text", "text": c}]

    def reply(self, body):
        last = body["messages"][-1]
        parts = self.parts(last)
        text = "".join(p.get("text", "") for p in parts)
        has_image = any(p.get("type") == "image_url" for p in parts)
        thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
        rendered = (json.dumps(body["tools"]) if body.get("tools") else "") + "".join(
            "".join(p.get("text", "") for p in self.parts(m))
            + (json.dumps(m["tool_calls"]) if m.get("tool_calls") else "") for m in body["messages"])
        prompt_tokens = len(rendered) // 4 + 10
        max_tokens = body.get("max_tokens", 256)
        if prompt_tokens + max_tokens > self.window:
            return 400, {"error": {"message": f"This model's context window is {self.window} tokens, but the rendered prompt "
                                              f"has {prompt_tokens} tokens and requests {max_tokens} reply tokens, which "
                                              f"exceeds the context window. Reduce the prompt to at most "
                                              f"{self.window - max_tokens} prompt tokens or request at most "
                                              f"{max(0, self.window - prompt_tokens)} reply tokens.",
                                   "type": "invalid_request_error", "param": None, "code": "context_length_exceeded"}}
        cached = 0
        if self.prefix_cache:
            with self.lock:
                cached = max((len(os.path.commonprefix([p, rendered])) for p in self.prompts), default=0) // 4
                self.prompts.append(rendered)
        time.sleep(self.prefill_s_per_ktok * (prompt_tokens - cached) / 1000)
        calls = None
        if has_image:
            content = self.vision
        elif "vault code" in text:
            m = re.search(r"vault code for the \w+ archive is (\S+)\.\n", text)
            code = m.group(1) if m else None
            with self.lock:
                said, self.last_code = (self.last_code if self.crosstalk else code), code
            content = said or "I could not find one."
        elif "17 * 23" in text:
            content = "391"
        elif body.get("tools") and last["role"] == "user":
            content = ""
            weather = {"city": "Paris, France", "unit": "fahrenheit" if self.drift_tool_args and not body.get("stream")
                       else "celsius"}
            calls = [{"id": f"call_{i}", "type": "function", "function": {"name": t["function"]["name"], "arguments": json.dumps(
                weather if t["function"]["name"] == "get_weather" else {"path": f"src/mod_{i}.py"})}}
                     for i, t in enumerate(body["tools"][:2])]
        else:
            content = "The printing press changed how ideas travel. " * 6
        if not thinking and self.leak_think:
            content = "<think>let me see</think>" + content
        message = {"role": "assistant", "content": content}
        if calls:
            message.update(content=None, tool_calls=calls)
        if thinking:
            message["reasoning_content"] = "17 * 23 = 17 * 20 + 17 * 3 = 391."
        tokens = max(1, len(content + "".join(c["function"]["arguments"] for c in calls or [])) // 4)
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": tokens, "total_tokens": prompt_tokens + tokens}
        if self.prefix_cache:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        return 200, {"id": "chatcmpl-fake", "object": "chat.completion", "model": self.model,
                     "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
                     "usage": usage,
                     "speculative": {"rounds": 4, "drafted": 8, "accepted": self.accepted,
                                     "acceptance_rate": self.accepted / 8, "tokens_per_round": 2.0}}

    def message(self, body):
        chat = {"messages": body["messages"], "max_tokens": body.get("max_tokens", 256), "stream": body.get("stream"),
                "chat_template_kwargs": {"enable_thinking": False},
                "tools": [{"type": "function", "function": {"name": t["name"], "parameters": t["input_schema"]}}
                          for t in body.get("tools", [])]}
        status, reply = self.reply(chat)
        if status != 200:
            return status, {"type": "error", "error": {"type": "invalid_request_error", "message": reply["error"]["message"]}}
        msg = reply["choices"][0]["message"]
        blocks = [{"type": "tool_use", "id": c["id"], "name": c["function"]["name"], "input": json.loads(c["function"]["arguments"])}
                  for c in msg.get("tool_calls") or []] or [{"type": "text", "text": msg["content"]}]
        return 200, {"id": "msg_fake", "type": "message", "role": "assistant", "model": self.model, "content": blocks,
                     "stop_reason": "tool_use" if msg.get("tool_calls") else "end_turn", "stop_sequence": None,
                     "usage": {"input_tokens": reply["usage"]["prompt_tokens"],
                               "output_tokens": reply["usage"]["completion_tokens"]}}

    def message_events(self, reply):
        yield {"type": "message_start", "message": {**reply, "content": [], "stop_reason": None}}
        for i, b in enumerate(reply["content"]):
            if b["type"] == "tool_use":
                yield {"type": "content_block_start", "index": i, "content_block": {**b, "input": {}}}
                args = json.dumps(b["input"])
                step = len(args) if self.whole_tool_args else 5
                for j in range(0, len(args), step):
                    yield {"type": "content_block_delta", "index": i,
                           "delta": {"type": "input_json_delta", "partial_json": args[j:j + step]}}
            else:
                yield {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}
                yield {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}}
            yield {"type": "content_block_stop", "index": i}
        yield {"type": "message_delta", "delta": {"stop_reason": reply["stop_reason"]}, "usage": reply["usage"]}
        yield {"type": "message_stop"}

    def complete(self, body):
        k = body.get("prompt_logprobs")
        if k is not None and (type(k) is not int or not 0 <= k <= 20):
            return 400, {"error": {"message": "prompt_logprobs must be an integer between 0 and 20, or null"}}
        if k is not None and body.get("stream"):
            return 400, {"error": {"message": "prompt_logprobs are supported on /v1/completions without streaming only"}}
        pieces = re.findall(r"\S+\s*", body["prompt"])
        ids = [zlib.crc32((self.id_salt + p).encode()) % VOCAB for p in pieces]
        if len(ids) + body.get("max_tokens", 16) > self.window:
            return 400, {"error": {"message": f"This model's context window is {self.window} tokens",
                                   "type": "invalid_request_error", "param": None, "code": "context_length_exceeded"}}
        reply = {"id": "cmpl-fake", "object": "text_completion", "model": self.model,
                 "choices": [{"index": 0, "text": " the", "finish_reason": "length"}],
                 "usage": {"prompt_tokens": len(ids), "completion_tokens": 1, "total_tokens": len(ids) + 1}}
        if k is not None:
            reply["choices"][0]["prompt_logprobs"] = [None] + [self.row(ids, t, k) for t in range(1, len(ids))]
        return 200, reply

    def row(self, ids, t, k):
        h = zlib.crc32(f"{ids[t - 1]}:{ids[t]}".encode())
        lp = -(h % 8000) / 1000 - self.nll_shift
        if self.jitter:
            lp += random.Random(f"{t}:{ids[t]}:{self.jitter}").uniform(-self.jitter, self.jitter)
        lp = min(lp, 0.0)
        rank = 1 if h % 3 == 0 else 2 + h % 50
        out = {str(ids[t]): {"logprob": lp, "rank": rank, "decoded_token": f"<{ids[t]}>"}}
        best = [ids[t]] if rank == 1 else []
        while len(best) < k:
            other = (h // 7 + 13 * len(best)) % VOCAB
            if other not in best and other != ids[t]:
                best.append(other)
                out[str(other)] = {"logprob": -0.05 * len(best), "rank": len(best), "decoded_token": f"<{other}>"}
            else:
                h += 1
        return out

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
