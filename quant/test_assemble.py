#!/usr/bin/env python3
# Builds a tiny random BF16 source and exllamav3-style qtensors/, then runs assemble.py extract, assemble and verify,
# check_layout.py on the result, and one refusal per bad qtensors input. Usage: test_assemble.py EMPTY_WORKDIR
import importlib.util
import json
import os
import random
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("tcl", os.path.join(HERE, "test_check_layout.py"))
tcl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tcl)
SIZE, MCG = tcl.SIZE, tcl.MCG
TEXT = {"num_hidden_layers": 2, "first_k_dense_replace": 1, "num_nextn_predict_layers": 1,
        "n_routed_experts": 2, "hidden_size": 128, "moe_intermediate_size": 32}
FILES = ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "processor_config.json",
         "generation_config.json", "LICENSE", "README.md")
rng = random.Random(0)
L = "model.language_model.layers"


def t(dtype, *shape):
    n = SIZE[dtype]
    for d in shape:
        n *= d
    return dtype, shape, rng.randbytes(n)


def projs(l, e):
    h, m = TEXT["hidden_size"], TEXT["moe_intermediate_size"]
    return {f"{L}.{l}.mlp.experts.{e}.{p}.weight": t("BF16", *s) for p, s in
            (("gate_proj", (m, h)), ("up_proj", (m, h)), ("down_proj", (h, m)))}


def write_source(path):
    shards = {
        "model-00001-of-00003.safetensors": {"lm_head.weight": t("BF16", 8, 128), "model.language_model.embed_tokens.weight": t("BF16", 8, 128),
                                             f"{L}.0.self_attn.A_log": t("F32", 3), f"{L}.0.mlp.up_proj.weight": t("BF16", 64, 128), **projs(1, 0)},
        "model-00002-of-00003.safetensors": {**projs(1, 1), f"{L}.1.mlp.gate.weight": t("BF16", 2, 128),
                                             f"{L}.1.mlp.gate.e_score_correction_bias": t("F32", 2)},
        "model-00003-of-00003.safetensors": {**projs(2, 0), **projs(2, 1), f"{L}.2.mlp.gate.weight": t("BF16", 2, 128), f"{L}.2.eh_proj.weight": t("BF16", 128, 256),
                                             "model.language_model.norm.weight": t("BF16", 128), "model.visual.blocks.0.attn.qkv.weight": t("BF16", 6, 4)},
    }
    tcl.write_pack(path, {"architectures": ["Glm5NextForConditionalGeneration"], "text_config": dict(TEXT)}, shards)
    for n in FILES:
        with open(os.path.join(path, n), "w") as f:
            f.write(f"{n} contents\n")
    return shards


def qlayer(l):
    """What exllamav3 writes for one MoE layer: the expert groups plus fused/F16 copies the assembler must ignore."""
    h, m = TEXT["hidden_size"], TEXT["moe_intermediate_size"]
    out = {f"{L}.{l}.self_attn.qkv_proj.weight": t("F16", 24, 128), f"{L}.{l}.mlp.gate.weight": t("F16", 2, 128),
           f"{L}.{l}.mlp.shared_experts.up_proj.trellis": t("I16", 8, 2, 64)}
    for e in range(TEXT["n_routed_experts"]):
        for p, (n_out, k_in) in (("gate_proj", (m, h)), ("up_proj", (m, h)), ("down_proj", (h, m))):
            k = f"{L}.{l}.mlp.experts.{e}.{p}"
            out.update({f"{k}.trellis": t("I16", k_in // 16, n_out // 16, 64), f"{k}.suh": t("F16", k_in),
                        f"{k}.svh": t("F16", n_out), f"{k}.mcg": ("I32", (), MCG)})
    return out


def write_q(path, layers):
    tcl.write_pack(path, {}, {f"{L}.{l}.safetensors": v for l, v in layers.items()})
    os.remove(os.path.join(path, "config.json"))
    os.remove(os.path.join(path, "model.safetensors.index.json"))


def tensors(path):
    """Independent reader: {name: (dtype, shape, bytes)} in data order."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        data = f.read()
    h.pop("__metadata__", None)
    return {k: (v["dtype"], tuple(v["shape"]), data[v["data_offsets"][0]:v["data_offsets"][1]])
            for k, v in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"])}


def run(*args):
    r = subprocess.run([sys.executable, "-I", *args], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def tool(cmd, *args):
    return run(os.path.join(HERE, "assemble.py"), cmd, *args)


def check(*args):
    return run(os.path.join(HERE, "check_layout.py"), *args)


def snapshot(d):
    return {n: open(os.path.join(d, n), "rb").read() for n in sorted(os.listdir(d))}


def ok(result, needle):
    code, out = result
    assert code == 0 and needle in out, out
    return out


def main():
    root = sys.argv[1]
    src, ext, q, out = (os.path.join(root, n) for n in ("src", "extract", "qtensors", "out"))
    shards = write_source(src)
    qt = {l: qlayer(l) for l in (1, 2)}
    write_q(q, qt)

    ok(tool("extract", src, ext), "extract: 3 shards")
    for shard, ts in shards.items():
        got = tensors(os.path.join(ext, shard))
        want = {k: v for k, v in tensors(os.path.join(src, shard)).items() if ".experts." not in k}
        assert list(got.items()) == list(want.items()), shard
    for n in FILES + ("config.json", "model.safetensors.index.json"):
        assert open(os.path.join(ext, n), "rb").read() == open(os.path.join(src, n), "rb").read(), n
    print("extract: non-expert tensors byte-equal in source order, non-weight files and index copied")

    ok(tool("assemble", ext, q, out), "12 expert projections written, 0 left out")
    ok(check(out, src), "RESULT PASS")
    ok(tool("verify", out, ext, q), "VERIFY PASS")
    config = json.load(open(os.path.join(out, "config.json")))
    assert config["quantization_config"] == {"quant_method": "exl3", "bits": 4, "codebook": "mcg", "head_bits": 16}
    assert type(config["quantization_config"]["bits"]) is int and config["text_config"] == TEXT
    assert set(os.listdir(out)) == {*shards, *FILES[:-1], "config.json", "model.safetensors.index.json"}, os.listdir(out)
    packed = {}
    for shard in shards:
        packed.update(tensors(os.path.join(out, shard)))
    for l, qs in qt.items():
        for k, v in qs.items():
            assert (packed.get(k) == v) == (".experts." in k), k
    for ts in shards.values():
        for k, v in ts.items():
            assert (packed.get(k) == v) == (".experts." not in k), k
    print("assemble: check_layout PASS, verify PASS, expert parts byte-equal to qtensors, exllamav3 extras left out, bits int 4")

    first = snapshot(out)
    ok(tool("assemble", ext, q, out), "assemble:")
    assert snapshot(out) == first
    out2 = os.path.join(root, "out2")
    ok(tool("assemble", ext, q, out2), "assemble:")
    with open(os.path.join(out2, "model-00002-of-00003.safetensors"), "r+b") as f:
        f.truncate(100)
    open(os.path.join(out2, "model-00003-of-00003.safetensors.tmp"), "wb").write(b"stale")
    ok(tool("assemble", ext, q, out2), "assemble:")
    assert snapshot(out2) == first
    print("deterministic: rerun in place and a fresh dir give identical bytes; a truncated shard and a stale .tmp converge")

    def refuse(name, mutate, needle):
        qq, oo = os.path.join(root, f"q-{name}"), os.path.join(root, f"out-{name}")
        layers = {l: dict(v) for l, v in qt.items()}
        mutate(layers)
        write_q(qq, layers)
        code, msg = tool("assemble", ext, qq, oo)
        assert code != 0 and needle in msg and not os.path.exists(oo), f"{name}: {msg}"
        print(f"refuses {name}: {msg.strip().splitlines()[-1]}")

    e = f"{L}.2.mlp.experts.1.down_proj"
    refuse("missing_mtp_svh", lambda ls: ls[2].pop(f"{e}.svh"), "1 expert parts missing")
    refuse("missing_mtp_layer", lambda ls: ls.pop(2), "layers.2.safetensors missing")
    refuse("packed_su_sv", lambda ls: ls[1].update({f"{L}.1.mlp.experts.0.up_proj.su": t("I16", 8)}), "1 unexpected")
    refuse("mul1_marker", lambda ls: ls[2].update({f"{e}.mcg": ("I32", (), struct.pack("<I", 0x83DCD12D))}), "not the I32 mcg marker")
    refuse("trellis_shape", lambda ls: ls[2].update({f"{e}.trellis": t("I16", 2, 8, 32)}), "want I16 [2, 8, 64]")
    refuse("suh_f32", lambda ls: ls[2].update({f"{e}.suh": t("F32", 32)}), "want F16 [32]")

    bad = os.path.join(root, "out-tampered")
    shutil.copytree(out, bad)
    path = os.path.join(bad, "model-00003-of-00003.safetensors")
    blob = bytearray(open(path, "rb").read())
    blob[-1] ^= 1
    open(path, "wb").write(bytes(blob))
    code, msg = tool("verify", bad, ext, q)
    assert code == 1 and "differs from" in msg and "VERIFY FAIL" in msg, msg
    print("verify catches a flipped byte:", next(l for l in msg.splitlines() if "differs" in l).strip())

    pext, pout = os.path.join(root, "extract-partial"), os.path.join(root, "out-partial")
    ok(tool("extract", src, pext, "--shards", "1,3"), "extract: 2 shards")
    ok(tool("assemble", pext, q, pout, "--layers", "0,2"), "6 expert projections written, 3 left out")
    ok(tool("verify", pout, pext, q), "VERIFY PASS")
    code, msg = check(pout, src)
    assert code == 1 and "RESULT FAIL" in msg, msg
    ok(check(pout, src, "--layers", "0,2"), "RESULT PASS")
    code, msg = check(pout, src, "--layers", "0,1,2")
    assert code == 1 and f"{L}.1.mlp.gate.weight missing" in msg, msg
    code, msg = tool("assemble", pext, q, out, "--layers", "0,2")
    assert code != 0 and "1 shards this run does not write" in msg, msg
    print("partial: shards 1,3 with --layers 0,2 pass check_layout --layers 0,2 and fail the full check and --layers 0,1,2")
    print("ALL ASSEMBLER TESTS PASS")


if __name__ == "__main__":
    main()
