import json, sys, threading, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
PROMPTS = [
    "Write a detailed 600-word essay about the history of the printing press.",
    "Explain, step by step and at length, how a CPU pipeline handles branch misprediction.",
    "Write a long short story about a lighthouse keeper who finds a message in a bottle.",
    "Describe in detail the water cycle and its effect on regional climates.",
]


def one(i, prompt, out):
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": 400,
            "stream": True, "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time(); first = None; n = 0; spec = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("speculative"):
                spec = d["speculative"]
            for c in d.get("choices", []):
                if c.get("delta", {}).get("content"):
                    if first is None:
                        first = time.time()
                    n += 1
            if d.get("usage"):
                n = d["usage"].get("completion_tokens", n)
    t1 = time.time()
    out[i] = {"ttft_s": round(first - t0, 2), "tokens": n, "decode_tok_s": round((n - 1) / (t1 - first), 2),
              "wall_s": round(t1 - t0, 2), "acceptance": spec and spec.get("acceptance_rate")}


for c in [int(x) for x in sys.argv[1:]] or [1, 4]:
    out = [None] * c
    t0 = time.time()
    ts = [threading.Thread(target=one, args=(i, PROMPTS[i % 4], out)) for i in range(c)]
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.time() - t0
    print(json.dumps({"concurrency": c, "aggregate_tok_s": round(sum(o["tokens"] for o in out) / wall, 2),
                      "streams": out}))
