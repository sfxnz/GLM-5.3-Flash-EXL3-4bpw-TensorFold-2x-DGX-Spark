import sys, json, struct, subprocess, re, collections
url = sys.argv[1]
def rng(a, b):
    return subprocess.run(["curl", "-sL", "-r", f"{a}-{b}", url], capture_output=True).stdout
n = struct.unpack("<Q", rng(0, 7))[0]
h = json.loads(rng(8, 8 + n - 1))
h.pop("__metadata__", None)
g = {}
for k, v in h.items():
    kk = re.sub(r"\.\d+\.", ".N.", k)
    g.setdefault(kk, (v["dtype"], v["shape"]))
for k, v in sorted(g.items()):
    print(v[0], v[1], k)
