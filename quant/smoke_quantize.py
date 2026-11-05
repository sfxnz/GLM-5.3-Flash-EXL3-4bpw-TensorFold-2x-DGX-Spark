import importlib.metadata as md
import json

import torch
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

K, N, ROWS = 512, 512, 4096
torch.manual_seed(0)
dev = torch.device("cuda:0")
w = torch.randn(K, N, device=dev) / K**0.5
x = torch.randn(ROWS, K, device=dev)
h_data = {
    "H": x.T @ x, "packed": False, "n": K, "first_key": "smoke", "count": ROWS, "finalized": False,
    "num_total": x.numel(), "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev,
}
quant_args = {"seed": 0, "K": 4, "mcg": True, "devices": [0], "apply_out_scales": None}
_, proxy_err, t = quantize_exl3(w.clone(), h_data, quant_args, False)

lin = LinearEXL3(None, K, N, suh=t["suh"], svh=t["svh"], trellis=t["trellis"], mcg=t["mcg"])
w_q = lin.get_weight_tensor().float()
y, y_q = x[:256] @ w, (x[:256].half() @ w_q.half()).float()
report = {
    "exllamav3": md.version("exllamav3"),
    "torch": torch.__version__,
    "device": torch.cuda.get_device_name(dev),
    "capability": list(torch.cuda.get_device_capability(dev)),
    "tensors": {k: [str(v.dtype).removeprefix("torch."), list(v.shape)] for k, v in t.items()},
    "mcg_marker_hex": hex(t["mcg"].view(torch.uint32).item()),
    "proxy_err": proxy_err,
    "weight_nmse": ((w_q - w).pow(2).sum() / w.pow(2).sum()).item(),
    "output_nmse": ((y_q - y).pow(2).sum() / y.pow(2).sum()).item(),
}
print(json.dumps(report, indent=1))
assert report["tensors"]["trellis"] == ["int16", [K // 16, N // 16, 64]], "trellis is i16 [K/16, N/16, 64] at 4 bpw"
assert report["tensors"]["suh"] == ["float16", [K]] and report["tensors"]["svh"] == ["float16", [N]], "suh/svh f16"
assert report["mcg_marker_hex"] == "0xcbac1fed", "mcg marker"
assert report["weight_nmse"] < 0.02, "4 bpw reconstruction error in the expected range"
print("SMOKE PASS")
