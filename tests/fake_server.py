"""A stand-in for TensorFold's OpenAI surface, shaped like v1.2.2's replies, so the smoke, needle and bench scripts run on a CPU.

Fault knobs (FakeServer(**knobs)): leak_think puts <think> into thinking-off content, accepted sets the speculative
block's accepted count, vision sets the image answer, window sets the context window the server admits in tokens (default: recipe.conf's CONTEXT)
(4 characters a token), and reported_window what /v1/models claims (default: window).

/v1/completions answers prompt_logprobs as v1.2.2 does (zig/src/server/logprobs.zig promptValue): null for the first
prompt token, then {id: {logprob, rank, decoded_token}} with the prompt token first. Its tokens are words; nll_shift
lowers every prompt token's logprob by that much, jitter adds seeded noise of up to that size, and id_salt tokenizes
differently.
"""
import json
import random
import re
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import recipe  # noqa: E402

VOCAB = 154880
GOOD_VISION = '{"left": {"color": "red", "shape": "square"}, "right": {"color": "blue", "shape": "circle"}}'


class FakeServer:
    def __init__(self, model="glm-5.3-flash", window=int(recipe.setting("CONTEXT")), leak_think=False, accepted=5, vision=GOOD_VISION,
                 reported_window=None, nll_shift=0.0, jitter=0.0, id_salt=""):
        self.model, self.window, self.leak_think, self.accepted, self.vision = model, window, leak_think, accepted, vision
        self.nll_shift, self.jitter, self.id_salt = nll_shift, jitter, id_salt
        self.reported_window = reported_window or window
        self.requests = []
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

            def do_GET(self):
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
                content = reply["choices"][0]["message"]["content"]
                for i in range(0, len(content), 8):
                    self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": content[i:i + 8]}}]}).encode() + b"\n\n")
                final = {"choices": [{"delta": {}, "finish_reason": "length"}], "usage": reply["usage"],
                         "speculative": reply["speculative"]}
                self.wfile.write(b"data: " + json.dumps(final).encode() + b"\n\ndata: [DONE]\n\n")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def reply(self, body):
        msg = body["messages"][-1]["content"]
        parts = msg if isinstance(msg, list) else [{"type": "text", "text": msg}]
        text = "".join(p.get("text", "") for p in parts)
        has_image = any(p.get("type") == "image_url" for p in parts)
        thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
        prompt_tokens = len(text) // 4 + 10
        max_tokens = body.get("max_tokens", 256)
        if prompt_tokens + max_tokens > self.window:
            return 400, {"error": {"message": f"This model's context window is {self.window} tokens, but the rendered prompt "
                                              f"has {prompt_tokens} tokens and requests {max_tokens} reply tokens, which "
                                              f"exceeds the context window. Reduce the prompt to at most "
                                              f"{self.window - max_tokens} prompt tokens or request at most "
                                              f"{max(0, self.window - prompt_tokens)} reply tokens.",
                                   "type": "invalid_request_error", "param": None, "code": "context_length_exceeded"}}
        if has_image:
            content = self.vision
        elif "vault code" in text:
            content = "7319-ALPHA-42" if "7319-ALPHA-42" in text else "I could not find one."
        elif "17 * 23" in text:
            content = "391"
        else:
            content = "The printing press changed how ideas travel. " * 6
        if not thinking and self.leak_think:
            content = "<think>let me see</think>" + content
        message = {"role": "assistant", "content": content}
        if thinking:
            message["reasoning_content"] = "17 * 23 = 17 * 20 + 17 * 3 = 391."
        tokens = max(1, len(content) // 4)
        return 200, {"id": "chatcmpl-fake", "object": "chat.completion", "model": self.model,
                     "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": tokens,
                               "total_tokens": prompt_tokens + tokens},
                     "speculative": {"rounds": 4, "drafted": 8, "accepted": self.accepted,
                                     "acceptance_rate": self.accepted / 8, "tokens_per_round": 2.0}}

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
