#!/usr/bin/env python3
# Builds a tiny synthetic BF16 source and EXL3 candidate, then runs check_layout.py on the good pack and on
# one broken copy per contract rule. Usage: test_check_layout.py WORKDIR
import json
import os
import struct
import subprocess
import sys

CHECK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_layout.py")
SIZE = {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4}
TEXT = {"num_hidden_layers": 2, "first_k_dense_replace": 1, "num_nextn_predict_layers": 1,
        "n_routed_experts": 2, "hidden_size": 128, "moe_intermediate_size": 32}
MCG = struct.pack("<I", 0xCBAC1FED)


def write_pack(path, config, shards):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    weight_map = {}
    for shard, tensors in shards.items():
        header, blobs, off = {}, [], 0
        for name, (dtype, shape, data) in tensors.items():
            n = SIZE[dtype]
            for d in shape:
                n *= d
            data = data if data is not None else bytes(n)
            assert len(data) == n, name
            header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
            blobs.append(data)
            off += n
            weight_map[name] = shard
        h = json.dumps(header).encode()
        with open(os.path.join(path, shard), "wb") as f:
            f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)


def expert(l, e, p):
    return f"model.language_model.layers.{l}.mlp.experts.{e}.{p}"


def source():
    h, m = TEXT["hidden_size"], TEXT["moe_intermediate_size"]
    t = {"lm_head.weight": ("BF16", (8, h), None), "model.language_model.embed_tokens.weight": ("BF16", (8, h), None),
         "model.language_model.layers.0.self_attn.A_log": ("F32", (4,), None)}
    for l in (1, 2):
        t[f"model.language_model.layers.{l}.mlp.gate.weight"] = ("BF16", (2, h), None)
        for e in range(2):
            for p, shape in (("gate_proj", (m, h)), ("up_proj", (m, h)), ("down_proj", (h, m))):
                t[expert(l, e, p) + ".weight"] = ("BF16", shape, None)
    return {"text_config": dict(TEXT)}, {"model-1.safetensors": t}


def candidate(shared_suh=True):
    config, shards = source()
    config["quantization_config"] = {"quant_method": "exl3", "bits": 4, "codebook": "mcg", "head_bits": 16}
    t = {}
    for name, v in shards["model-1.safetensors"].items():
        if not name.endswith("_proj.weight") or ".experts." not in name:
            t[name] = v
            continue
        n_out, k_in = v[1]
        base = name.removesuffix(".weight")
        e = int(base.split(".experts.")[1].split(".")[0])
        suh = struct.pack("<e", 1.0) * k_in if shared_suh or e == 0 else struct.pack("<e", -1.0) * k_in
        t[base + ".trellis"] = ("I16", (k_in // 16, n_out // 16, 64), None)
        t[base + ".suh"] = ("F16", (k_in,), suh)
        t[base + ".svh"] = ("F16", (n_out,), None)
        t[base + ".mcg"] = ("I32", (), MCG)
    return config, {"model-1.safetensors": t}


def run(cand, src):
    r = subprocess.run([sys.executable, "-I", CHECK, cand, src], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    root = sys.argv[1]
    write_pack(os.path.join(root, "src"), *source())
    src = os.path.join(root, "src")

    def case(name, mutate=None, shared_suh=True):
        config, shards = candidate(shared_suh)
        if mutate:
            mutate(config, shards["model-1.safetensors"])
        path = os.path.join(root, name)
        write_pack(path, config, shards)
        return run(path, src)

    def set_(key, value):
        def m(_, t):
            t[key] = value
        return m

    def drop(key):
        return lambda _, t: t.pop(key)

    good = case("good")
    assert good[0] == 0 and "RESULT PASS" in good[1] and "layer 1: 1 distinct" in good[1], good[1]
    print("good pack: PASS", flush=True)
    unshared = case("unshared", shared_suh=False)
    assert unshared[0] == 0 and "shx=false" in unshared[1], unshared[1]
    print("per-expert suh: PASS, reported shx=false")

    tr = expert(2, 1, "down_proj")
    broken = {
        "bits_float": (lambda c, _: c["quantization_config"].update(bits=4.0), "want the JSON integer 4"),
        "bits_text_config_only": (lambda c, _: c["text_config"].update(quantization_config=c.pop("quantization_config")), "no top-level quantization_config"),
        "codebook_mul1": (lambda c, _: c["quantization_config"].update(codebook="mul1"), 'want "mcg"'),
        "trellis_shape": (set_(tr + ".trellis", ("I16", (2, 8, 32), None)), "want I16 [2, 8, 64]"),
        "suh_f32": (set_(tr + ".suh", ("F32", (32,), None)), "want F16 [32]"),
        "marker_value": (set_(tr + ".mcg", ("I32", (), struct.pack("<I", 0x83DCD12D))), "want 0xcbac1fed"),
        "mtp_expert_missing": (drop(tr + ".svh"), f"{tr}.svh missing"),
        "weight_left": (set_(tr + ".weight", ("BF16", (128, 32), None)), "left in the pack"),
        "nonexpert_f16": (set_("lm_head.weight", ("F16", (8, 128), None)), "source BF16"),
        "nonexpert_f32_narrowed": (set_("model.language_model.layers.0.self_attn.A_log", ("BF16", (4,), None)), "source F32"),
        "nonexpert_missing": (drop("model.language_model.embed_tokens.weight"), "embed_tokens.weight missing"),
        "nonexpert_extra": (set_("model.language_model.layers.0.self_attn.qkv_proj.weight", ("BF16", (4, 128), None)), "not in the source"),
    }
    for name, (mutate, needle) in broken.items():
        code, out = case(name, mutate)
        assert code == 1 and "RESULT FAIL" in out and needle in out, f"{name}: {out}"
        line = next(l for l in out.splitlines() if needle in l)
        print(f"{name}: FAIL as expected ->{line}")
    print("ALL CHECKER TESTS PASS")


if __name__ == "__main__":
    main()
