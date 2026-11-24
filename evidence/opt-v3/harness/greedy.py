"""greedy.py NAME: four fixed prompts at temperature 0 (thinking off, 300 tokens), texts kept in runs/NAME/greedy.json.
Two boots that claim the same replies must give the same texts."""
import json, sys, urllib.request
from pathlib import Path
W = Path(__file__).resolve().parent
PROMPTS = ["Explain how a hash table handles collisions, in three short paragraphs.",
           "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
           "List the planets of the solar system with one fact each.",
           "Write a bash script that prints the ten largest files under a directory given as its first argument."]
out = []
for p in PROMPTS:
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 300,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    out.append({"prompt": p, "text": r["choices"][0]["message"]["content"], "tokens": r["usage"]["completion_tokens"]})
(W / "runs" / sys.argv[1] / "greedy.json").write_text(json.dumps(out, indent=1))
print("greedy", [o["tokens"] for o in out])
