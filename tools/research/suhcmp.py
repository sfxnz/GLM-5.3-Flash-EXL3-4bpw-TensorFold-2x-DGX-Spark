import sys, json, struct, subprocess, re, hashlib
url = sys.argv[1]
def rng(a, b):
    return subprocess.run(["curl", "-sL", "-r", f"{a}-{b}", url], capture_output=True).stdout
n = struct.unpack("<Q", rng(0, 7))[0]
h = json.loads(rng(8, 8 + n - 1)); h.pop("__metadata__", None)
base = 8 + n
ks = sorted(k for k in h if k.endswith(("gate_proj.suh", "up_proj.suh", "down_proj.svh")))[:8]
import array
for k in ks:
    a, b = h[k]["data_offsets"]
    raw = rng(base + a, base + b - 1)
    v = struct.unpack("<6e", raw[:12])
    print(k, h[k]["dtype"], h[k]["shape"], hashlib.sha256(raw).hexdigest()[:12], [round(x, 5) for x in v[:6]])
