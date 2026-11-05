"""CPU check of the suite's EXL3 sidecar gate on codebook-marker shape, inside the fidelity image.

Usage (PYTHONPATH=/opt/fidelity/suite/bin:/opt/fidelity/suite/engines/tools):
    python3 verify_marker.py <pack dir with quantization_config.json> [--module NAME]

Runs trellis_checkpoint_plan, the call that refused TR3 at cf36287c, on the real pack, then decodes one
real module through materialize_trellis_subset. It then builds a synthetic one-module pack per marker
variant: shape [] and [1] must be admitted and decode to identical weights, while [2] and I64 must be refused.
Exits 1 on any unexpected outcome.
"""
import argparse
import json
import os
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import exl3hf_surface as xs
import layer_outer as lo

MODULE = "model.language_model.layers.0.mlp.experts.0.down_proj"
failures = []


def report(name, ok, detail=""):
    print("[%s] %s%s" % ("ok" if ok else "FAIL", name, (" - " + detail) if detail else ""))
    if not ok:
        failures.append(name)


def plan_or_refusal(config, keys, model_dir):
    try:
        return lo.trellis_checkpoint_plan(config, keys, model_dir=model_dir), None
    except lo.LayerOuterError as exc:
        return None, str(exc)


def real_pack(pack, module):
    config = json.load(open(os.path.join(pack, "config.json")))
    weight_map = json.load(open(os.path.join(pack, "model.safetensors.index.json")))["weight_map"]
    plan, refusal = plan_or_refusal(config, list(weight_map), pack)
    report("real pack: trellis_checkpoint_plan admits the sidecar declaration", plan is not None,
           refusal or "%d groups, sidecar sha256 %s" % (
               len(plan["_observed"]["tensor_storage"]) // 4,
               plan["declared_tensor_storage_source"]["sha256"]))
    if plan is None:
        return
    marker_key = next(k for k in weight_map if k.startswith(module + ".") and k.rsplit(".", 1)[1] in ("mcg", "mul1"))
    names = [module + ".trellis", module + ".suh", module + ".svh", marker_key]
    shard = os.path.join(pack, weight_map[marker_key])
    with safe_open(shard, framework="pt", device="cpu") as handle:
        lazy = {name: handle.get_slice(name) for name in names}
        shape = list(lazy[marker_key].get_shape())
        stats = {"decoded_modules": 0, "trellis_bits": 0}
        out = lo.materialize_trellis_subset(lazy, plan, torch.bfloat16, stats)
    weight = out[module + ".weight"]
    report("real pack: %s (marker shape %s) decodes through the marker-value check" % (module, shape),
           stats["decoded_modules"] == 1 and bool(torch.isfinite(weight.float()).all()),
           "weight %s %s" % (tuple(weight.shape), weight.dtype))


def synthetic(marker, bits=4):
    generator = torch.Generator().manual_seed(7)
    tensors = {
        MODULE + ".trellis": torch.randint(-(2 ** 15), 2 ** 15, (8, 8, 16 * bits), generator=generator,
                                           dtype=torch.int16),
        MODULE + ".suh": torch.randn(128, generator=generator).to(torch.float16),
        MODULE + ".svh": torch.randn(128, generator=generator).to(torch.float16),
        MODULE + ".mcg": marker,
    }
    inline = {"quant_method": "exl3", "bits": bits, "head_bits": 16, "codebook": "mcg"}
    storage = {MODULE: {"quant_format": "exl3", "bits_per_weight": bits, "stored_tensors": {
        key: {"shape": list(t.shape), "dtype": str(t.dtype), "n_bytes": t.numel() * t.element_size()}
        for key, t in tensors.items()}}}
    with tempfile.TemporaryDirectory(prefix="mcg-marker-") as td:
        save_file(tensors, os.path.join(td, "model.safetensors"))
        with open(os.path.join(td, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {key: "model.safetensors" for key in tensors}}, f)
        with open(os.path.join(td, "quantization_config.json"), "w") as f:
            json.dump(dict(inline, tensor_storage=storage), f)
        plan, refusal = plan_or_refusal({"quantization_config": inline}, list(tensors), td)
        if plan is None:
            return None, refusal
        with safe_open(os.path.join(td, "model.safetensors"), framework="pt", device="cpu") as handle:
            lazy = {key: handle.get_slice(key) for key in tensors}
            out = lo.materialize_trellis_subset(lazy, plan, torch.bfloat16, {"decoded_modules": 0, "trellis_bits": 0})
        return out[MODULE + ".weight"], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--module", default="model.language_model.layers.10.mlp.experts.0.down_proj")
    args = ap.parse_args()
    print("suite", open("/opt/fidelity/suite/COMMIT").read().strip(), "patches",
          open("/opt/fidelity/suite/PATCHES").read().split()[1::2] if os.path.exists("/opt/fidelity/suite/PATCHES")
          else "none")
    real_pack(args.pack, args.module)
    mcg = xs.CODEBOOK_OBJECTS["mcg"]
    scalar, refusal = synthetic(torch.tensor(mcg, dtype=torch.int32))
    report("synthetic: I32 marker shape [] admitted and decoded", scalar is not None, refusal or "")
    rank1, refusal = synthetic(torch.tensor([mcg], dtype=torch.int32))
    report("synthetic: I32 marker shape [1] admitted and decoded", rank1 is not None, refusal or "")
    if scalar is not None and rank1 is not None:
        report("synthetic: [] and [1] markers decode to bitwise-equal weights", torch.equal(scalar, rank1))
    for name, marker in (("shape [2]", torch.tensor([mcg, mcg], dtype=torch.int32)),
                         ("dtype I64", torch.tensor(mcg, dtype=torch.int64))):
        weight, refusal = synthetic(marker)
        report("synthetic: marker %s refused" % name, weight is None and "codebook marker" in (refusal or ""),
               refusal or "admitted")
    print("verify_marker: %s" % ("FAIL: " + "; ".join(failures) if failures else "all checks passed"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
