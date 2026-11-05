import json, random, sys, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
target = int(sys.argv[1])
rng = random.Random(7)
CITIES = ["Lisbon", "Osaka", "Quito", "Tromso", "Nairobi", "Hobart", "Tbilisi", "Cusco", "Gdansk", "Leeds"]
ITEMS = ["lanterns", "barrels", "ledgers", "copper pipes", "wool bales", "clock gears", "maps", "seed sacks"]


def line(i):
    return (f"Record {i}: the warehouse in {rng.choice(CITIES)} logged {rng.randint(10, 9999)} "
            f"{rng.choice(ITEMS)} on shelf {rng.randint(1, 400)} during shift {rng.randint(1, 3)}.\n")


def ask(text, max_tokens):
    body = {"model": "glm-5.3-flash", "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": text}]}
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"}),
                               timeout=7200)
    d = json.load(r)
    return d, time.time() - t0


probe_lines = [line(i) for i in range(400)]
d, _ = ask("".join(probe_lines) + "Reply OK.", 1)
per_line = (d["usage"]["prompt_tokens"] - 10) / 400
n = int(target / per_line)
lines = [line(i) for i in range(n)]
secret = "The vault code for the Hobart archive is 7319-ALPHA-42."
lines.insert(int(n * 0.37), secret + "\n")
question = "\nIn the records above, one line states a vault code. What is the vault code for the Hobart archive? Reply with the code only."
d, wall = ask("".join(lines) + question, 40)
ans = d["choices"][0]["message"]["content"]
print(json.dumps({"target": target, "tokens_per_line": round(per_line, 3), "prompt_tokens": d["usage"]["prompt_tokens"],
                  "needle_depth": 0.37, "answer": ans, "found": "7319-ALPHA-42" in ans, "wall_s": round(wall, 1),
                  "prefill_tok_s": round(d["usage"]["prompt_tokens"] / wall, 1)}))
