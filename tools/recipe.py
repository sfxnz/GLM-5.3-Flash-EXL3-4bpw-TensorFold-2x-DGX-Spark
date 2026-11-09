"""Settings from recipe.conf (the environment wins, as in run.sh) and the HTTP calls the smoke, needle and bench scripts share."""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

CONF = Path(__file__).resolve().parent.parent / "recipe.conf"


def setting(name):
    if name in os.environ:
        return os.environ[name]
    m = re.search(rf"^{name}=\$\{{{name}:-(.*)\}}$", CONF.read_text(), re.M)
    if not m:
        raise KeyError(f"{name} has no default in {CONF}")
    return m.group(1)


def default_url():
    return f"http://127.0.0.1:{setting('PORT')}"


def headers():
    """The JSON content type, and the key from API_KEY in both header forms the server reads, when it is set."""
    h = {"content-type": "application/json"}
    if os.environ.get("API_KEY"):
        h.update({"authorization": f"Bearer {os.environ['API_KEY']}", "x-api-key": os.environ["API_KEY"]})
    return h


def get(url, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers()), timeout=timeout) as r:
        return r.status, json.load(r)


def post(url, body, timeout=900):
    """Returns (status, parsed JSON body); an HTTP error answers with its status and error body instead of raising."""
    req = urllib.request.Request(url, json.dumps(body).encode(), headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"error": {"message": raw.decode(errors="replace")}}


def stream(url, body, timeout=900):
    """Yields (seconds since the request, chunk) for each SSE data chunk of a streamed reply."""
    req = urllib.request.Request(url, json.dumps({**body, "stream": True}).encode(), headers())
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:") and line != "data: [DONE]":
                yield time.monotonic() - t0, json.loads(line[5:])
