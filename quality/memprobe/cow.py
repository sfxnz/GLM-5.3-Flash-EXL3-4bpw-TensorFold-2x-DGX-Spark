import os, sys, json
import torch
from safetensors import safe_open
pack = sys.argv[1]
idx = json.load(open(os.path.join(pack, "model.safetensors.index.json")))["weight_map"]
keys = sorted(k for k in idx if k.startswith("model.language_model.layers.10.mlp.experts.") and k.endswith(".trellis"))[:300]
shard = idx[keys[0]]; keys = [k for k in keys if idx[k] == shard]
path = os.path.realpath(os.path.join(pack, shard))
def mapping():
    out, cur = {}, False
    for line in open("/proc/self/smaps"):
        h = line.split()
        if "-" in h[0] and not h[0].endswith(":"):
            cur = len(h) > 5 and h[5] == path
            if cur: out["perms"] = h[1]
        elif cur and h[0] in ("Rss:", "Anonymous:"):
            out[h[0][:-1]] = out.get(h[0][:-1], 0) + int(h[1]) // 1024
    return out
f = safe_open(os.path.join(pack, shard), framework="pt", device="cpu")
groups = [keys[i::4] for i in range(4)]
def nbytes(ks): return sum(f.get_slice(k).get_shape()[0] for k in ks)
print("open", mapping())
t = [f.get_slice(k)[...] for k in groups[0]]
print("eager views", t[0].untyped_storage().nbytes() > 10**9, mapping())
s = sum(float(x.float().sum()) for x in t); print("cpu read group0", mapping())
d = [x.to("cuda") for x in t]; torch.cuda.synchronize(); print("to(cuda) group0 after cpu read", mapping())
t1 = [f.get_slice(k)[...] for k in groups[1]]
d1 = [x.to("cuda") for x in t1]; torch.cuda.synchronize(); print("to(cuda) group1 cold", mapping())
t2 = [f.get_slice(k)[...].clone() for k in groups[2]]
print("clone group2", mapping())
d2 = [x.to("cuda") for x in t2]; torch.cuda.synchronize(); print("clone+to(cuda) group2", mapping())
del t, t2, d, d2
print("dropped tensors", mapping())
import ctypes, time
libc = ctypes.CDLL(None, use_errno=True); libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
for line in open("/proc/self/maps"):
    h = line.split(None, 5)
    if len(h) == 6 and h[5].strip() == path:
        lo, hi = (int(v, 16) for v in h[0].split("-")); t0 = time.monotonic()
        print("madvise", libc.madvise(lo, hi - lo, 4), ctypes.get_errno(), "%.4fs" % (time.monotonic() - t0))
print("after DONTNEED", mapping())
again = [f.get_slice(k)[...].to("cuda") for k in groups[1]]; torch.cuda.synchronize()
print("refault equal", all(torch.equal(a, b) for a, b in zip(again, d1)), mapping())
