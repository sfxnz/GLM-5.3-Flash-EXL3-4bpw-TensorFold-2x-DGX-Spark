import sys, json, struct, subprocess
import numpy as np
def rng(url, a, b):
    return subprocess.run(["curl", "-sL", "-r", f"{a}-{b}", url], capture_output=True).stdout
def hdr(url):
    n = struct.unpack("<Q", rng(url, 0, 7))[0]
    return json.loads(rng(url, 8, 8 + n - 1)), 8 + n
def fetch(url, h, base, k, nbytes=None):
    a, b = h[k]["data_offsets"]
    if nbytes: b = a + nbytes
    return rng(url, base + a, base + b - 1)
def e4m3(u8):
    u = u8.astype(np.int32); s = np.where(u & 0x80, -1.0, 1.0); e = (u >> 3) & 0xF; m = u & 7
    v = np.where(e == 0, m / 8.0 * 2.0**-6, (1 + m / 8.0) * 2.0**(e - 7.0))
    return s * v
bf_url, fp_url, key = sys.argv[1], sys.argv[2], sys.argv[3]
hb, bb = hdr(bf_url); hf, bf = hdr(fp_url)
shape = hb[key]["shape"]; cols = shape[1]; rows = 128
print(key, "bf16", hb[key]["dtype"], shape, "fp8", hf[key]["dtype"], hf[key]["shape"], hf[key + "_scale_inv"]["dtype"], hf[key + "_scale_inv"]["shape"])
wb = np.frombuffer(fetch(bf_url, hb, bb, key, rows * cols * 2), dtype=np.uint16).reshape(rows, cols)
wb = (wb.astype(np.uint32) << 16).view(np.float32)
wf = e4m3(np.frombuffer(fetch(fp_url, hf, bf, key, rows * cols), dtype=np.uint8).reshape(rows, cols))
sc = np.frombuffer(fetch(fp_url, hf, bf, key + "_scale_inv"), dtype=np.float32 if hf[key + "_scale_inv"]["dtype"] == "F32" else np.uint16)
if hf[key + "_scale_inv"]["dtype"] == "BF16": sc = (sc.astype(np.uint32) << 16).view(np.float32)
sc = sc.reshape(hf[key + "_scale_inv"]["shape"])[0]
deq = wf * np.repeat(sc, 128)[None, :cols]
deq_bf = ((deq.astype(np.float32).view(np.uint32) + 0x7FFF + ((deq.astype(np.float32).view(np.uint32) >> 16) & 1)) >> 16 << 16).view(np.float32)
print("exact bf16 match frac:", float(np.mean(deq_bf == wb)), "max abs diff:", float(np.max(np.abs(deq - wb))), "rel rms:", float(np.sqrt(np.mean((deq - wb) ** 2)) / np.sqrt(np.mean(wb ** 2))))
# distinct values per 128x128 block in bf16 (fp8 upcast gives <= 256 distinct)
blk = wb[:, :128]; print("distinct bf16 values in block 0:", len(np.unique(blk)))
