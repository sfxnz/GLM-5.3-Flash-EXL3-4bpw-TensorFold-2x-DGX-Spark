#!/usr/bin/env python3
# Builds the TensorFold GLM EXL3 pack (quant/README-contract.md) from the BF16 source and exllamav3's qtensors/.
#   assemble.py extract SOURCE EXTRACT [--shards N,N,...]     on spark2: non-expert tensors + every non-weight file
#   assemble.py assemble EXTRACT QTENSORS OUT [--layers L,...] on spark1: one output shard per source shard
#   assemble.py verify OUT EXTRACT QTENSORS                    byte-compares OUT against the source hashes and qtensors
# --shards and --layers build a partial pack for testing: only those source shards (1-based numbers), and only those
# layers' routed experts; other layers' experts are left out. Standard library only (spark2's python has no numpy).
import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time

DTYPE_SIZE = {"F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2}
# Data order inside a shard, as the safetensors library writes it and the BF16 source and TR3 shards are laid out:
# wider dtypes first, then name. It keeps every tensor aligned to its element size.
DTYPE_RANK = {"F32": 0, "I32": 1, "BF16": 2, "F16": 3, "I16": 4}
EXPERT = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight")
EXPERT_PART = re.compile(r"(model\.language_model\.layers\.(\d+)\.mlp\.experts\.\d+\.(?:gate|up|down)_proj)\.(?:mcg|suh|svh|trellis)")
PROJS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("mcg", "suh", "svh", "trellis")
MCG = struct.pack("<I", 0xCBAC1FED)
QUANT = {"quant_method": "exl3", "bits": 4, "codebook": "mcg", "head_bits": 16}
PACK_FILES = ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "processor_config.json",
              "generation_config.json", "LICENSE")
INDEX = "model.safetensors.index.json"
MANIFEST = "extract-manifest.json"
CHUNK = 64 << 20


def read_header(path):
    """{name: (dtype, shape, abs_start, abs_end)} in data order; refuses a truncated or non-contiguous file."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        if hlen > size - 8:
            raise SystemExit(f"{path}: header length {hlen} exceeds the file ({size} bytes)")
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    base, at, out = 8 + hlen, 0, {}
    for name, v in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"]):
        a, b = v["data_offsets"]
        if a != at:
            raise SystemExit(f"{path}: {name} starts at {a}, previous tensor ended at {at}")
        out[name] = (v["dtype"], tuple(v["shape"]), base + a, base + b)
        at = b
    if base + at != size:
        raise SystemExit(f"{path}: tensors end at {base + at}, file is {size} bytes (truncated or padded)")
    return out


def nbytes(dtype, shape):
    n = DTYPE_SIZE[dtype]
    for d in shape:
        n *= d
    return n


def write_shard(path, tensors, hash_names=()):
    """tensors: [(name, dtype, shape, src_path, abs_start)] in data order. Writes tmp, fsyncs, renames.
    Returns the sha256 of each tensor named in hash_names."""
    header, off = {}, 0
    for name, dtype, shape, _, _ in tensors:
        n = nbytes(dtype, shape)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    text = json.dumps(dict(sorted(header.items())), separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    hashes, files = {}, {}
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as out:
            out.write(struct.pack("<Q", len(text)) + text)
            for name, dtype, shape, src, start in tensors:
                f = files.get(src) or files.setdefault(src, open(src, "rb"))
                f.seek(start)
                left = nbytes(dtype, shape)
                h = hashlib.sha256() if name in hash_names else None
                while left:
                    buf = f.read(min(left, CHUNK))
                    if not buf:
                        raise SystemExit(f"{src}: short read for {name}")
                    out.write(buf)
                    if h:
                        h.update(buf)
                    left -= len(buf)
                if h:
                    hashes[name] = h.hexdigest()
            out.flush()
            os.fsync(out.fileno())
    finally:
        for f in files.values():
            f.close()
    os.replace(tmp, path)
    return hashes, 8 + len(text) + off


def write_bytes(path, data):
    with open(path + ".tmp", "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)


def slurp(path):
    with open(path, "rb") as f:
        return f.read()


def clear_tmp(out):
    for n in os.listdir(out):
        if n.endswith(".tmp"):
            os.remove(os.path.join(out, n))


def text_config(config):
    return config.get("text_config", config)


def int_list(s):
    return sorted({int(x) for x in s.split(",")}) if s else None


def extract(src, out, shards):
    t0 = time.time()
    weight_map = json.load(open(os.path.join(src, INDEX)))["weight_map"]
    names = sorted(set(weight_map.values()))
    if shards:
        names = [names[i - 1] for i in shards]
    os.makedirs(out, exist_ok=True)
    clear_tmp(out)
    for n in sorted(os.listdir(src)):
        p = os.path.join(src, n)
        if os.path.isfile(p) and not n.endswith(".safetensors"):
            write_bytes(os.path.join(out, n), slurp(p))
    manifest, total = {}, 0
    for shard in names:
        entries = read_header(os.path.join(src, shard))
        keep = [(n, d, s, os.path.join(src, shard), a) for n, (d, s, a, _) in entries.items() if not EXPERT.fullmatch(n)]
        hashes, size = write_shard(os.path.join(out, shard), keep, {k[0] for k in keep})
        manifest[shard] = [[n, d, list(s), hashes.get(n)] for n, (d, s, _, _) in entries.items()]
        total += size
        print(f"{shard}: {len(keep)} non-expert of {len(entries)} tensors, {size} bytes", flush=True)
    write_bytes(os.path.join(out, MANIFEST), json.dumps({"source": os.path.realpath(src), "shards": manifest}).encode())
    print(f"extract: {len(names)} shards, {total} bytes, {time.time() - t0:.0f} s")


def load_qlayer(qdir, layer, n_exp, want_shape):
    """The four EXL3 parts of every routed expert in one layer's qtensors file: {name: (dtype, shape, path, start)}."""
    path = os.path.join(qdir, f"model.language_model.layers.{layer}.safetensors")
    if not os.path.isfile(path):
        raise SystemExit(f"layer {layer}: {path} missing")
    entries = read_header(path)
    prefix = f"model.language_model.layers.{layer}.mlp.experts."
    got = {n: v for n, v in entries.items() if n.startswith(prefix)}
    want = {f"{prefix}{e}.{p}.{part}" for e in range(n_exp) for p in PROJS for part in PARTS}
    if set(got) != want:
        missing, extra = sorted(want - set(got)), sorted(set(got) - want)
        raise SystemExit(f"layer {layer}: {len(missing)} expert parts missing, e.g. {missing[:3]}; "
                         f"{len(extra)} unexpected (packed su/sv, bias or mul1 need a converter), e.g. {extra[:3]}")
    with open(path, "rb") as f:
        for name, (dtype, shape, a, b) in got.items():
            proj, part = name.rsplit(".", 2)[1:]
            n_out, k_in = want_shape[proj]
            spec = {"trellis": ("I16", (k_in // 16, n_out // 16, 64)), "suh": ("F16", (k_in,)), "svh": ("F16", (n_out,))}
            if part == "mcg":
                f.seek(a)
                if dtype != "I32" or shape not in ((), (1,)) or f.read(b - a) != MCG:
                    raise SystemExit(f"{path}: {name} is not the I32 mcg marker {MCG.hex()}")
            elif (dtype, shape) != spec[part]:
                raise SystemExit(f"{path}: {name} is {dtype} {list(shape)}, want {spec[part][0]} {list(spec[part][1])}")
    return {n: (d, s, path, a) for n, (d, s, a, _) in got.items()}


def assemble(ext, qdir, out, only_layers):
    t0 = time.time()
    config = json.load(open(os.path.join(ext, "config.json")))
    if "quantization_config" in config:
        raise SystemExit("source config.json already has a quantization_config; want the BF16 source")
    tc = text_config(config)
    first, n_layers, n_exp = tc["first_k_dense_replace"], tc["num_hidden_layers"], tc["n_routed_experts"]
    moe = range(first, n_layers + tc.get("num_nextn_predict_layers", 0))
    hidden, inter = tc["hidden_size"], tc["moe_intermediate_size"]
    want_shape = {"gate_proj": (inter, hidden), "up_proj": (inter, hidden), "down_proj": (hidden, inter)}
    manifest = json.load(open(os.path.join(ext, MANIFEST)))["shards"]
    layers = [l for l in moe if only_layers is None or l in only_layers]

    src_experts = {}
    for shard, entries in manifest.items():
        for name, dtype, shape, _ in entries:
            m = EXPERT.fullmatch(name)
            if m:
                if (dtype, tuple(shape)) != ("BF16", want_shape[m[3]]):
                    raise SystemExit(f"source {name} is {dtype} {shape}, want BF16 {list(want_shape[m[3]])}")
                src_experts[name] = int(m[1])
    if only_layers is None:
        all_shards = set(json.load(open(os.path.join(ext, INDEX)))["weight_map"].values())
        if set(manifest) != all_shards:
            raise SystemExit(f"extract holds {len(manifest)} of {len(all_shards)} source shards; full assembly needs all")
        want = {f"model.language_model.layers.{l}.mlp.experts.{e}.{p}.weight" for l in moe for e in range(n_exp) for p in PROJS}
        if set(src_experts) != want:
            raise SystemExit(f"source routed experts differ from layers {moe.start}..{moe.stop - 1} x {n_exp}: "
                             f"{len(want - set(src_experts))} missing, {len(set(src_experts) - want)} unexpected")

    q = {}
    for l in layers:
        q.update(load_qlayer(qdir, l, n_exp, want_shape))
        print(f"qtensors layer {l}: {n_exp} experts x 3 x {len(PARTS)} parts ok", flush=True)

    os.makedirs(out, exist_ok=True)
    clear_tmp(out)
    stray = sorted(n for n in os.listdir(out) if n.endswith(".safetensors") and n not in manifest)
    if stray:
        raise SystemExit(f"{out} holds {len(stray)} shards this run does not write, e.g. {stray[:3]}; use an empty dir")
    weight_map, total, dropped, written = {}, 0, 0, 0
    for shard in sorted(manifest):
        path = os.path.join(ext, shard)
        entries = read_header(path)
        rest = [(n, d, tuple(s)) for n, d, s, _ in manifest[shard] if not EXPERT.fullmatch(n)]
        if [(n, d, s) for n, (d, s, _, _) in entries.items()] != rest:
            raise SystemExit(f"{path} does not match {MANIFEST}; re-run extract")
        tensors = [(n, d, s, path, a) for n, (d, s, a, _) in entries.items()]
        for name, *_ in manifest[shard]:
            if name not in src_experts:
                continue
            if src_experts[name] not in layers:
                dropped += 1
                continue
            base = name.removesuffix(".weight")
            tensors += [(f"{base}.{part}", *q[f"{base}.{part}"]) for part in PARTS]
            written += 1
        tensors.sort(key=lambda t: (DTYPE_RANK[t[1]], t[0]))
        _, size = write_shard(os.path.join(out, shard), tensors)
        total += sum(nbytes(d, s) for _, d, s, _, _ in tensors)
        weight_map.update((t[0], shard) for t in tensors)
        print(f"{shard}: {len(tensors)} tensors, {size} bytes", flush=True)

    for n in PACK_FILES:
        write_bytes(os.path.join(out, n), slurp(os.path.join(ext, n)))
    config["quantization_config"] = dict(QUANT)
    write_bytes(os.path.join(out, "config.json"), (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode())
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    write_bytes(os.path.join(out, INDEX), (json.dumps(index, indent=2) + "\n").encode())
    print(f"assemble: {len(manifest)} shards, {written} expert projections written, {dropped} left out "
          f"(partial), {total} tensor bytes, {time.time() - t0:.0f} s")


def verify(out, ext, qdir):
    """Every pack tensor equals its origin: non-experts by sha256 against the source hashes taken at extract time,
    expert parts byte-for-byte against qtensors."""
    t0 = time.time()
    fails = []
    bits = json.load(open(os.path.join(out, "config.json"))).get("quantization_config", {}).get("bits")
    if type(bits) is not int or bits != 4:
        fails.append(f"config bits {json.dumps(bits)}, want the JSON integer 4")
    for n in PACK_FILES:
        if slurp(os.path.join(out, n)) != slurp(os.path.join(ext, n)):
            fails.append(f"{n} differs from the source copy")
    manifest = json.load(open(os.path.join(ext, MANIFEST)))["shards"]
    shards = sorted(set(json.load(open(os.path.join(out, INDEX)))["weight_map"].values()))
    qheaders, n_rest, n_parts = {}, 0, 0
    for shard in shards:
        src = {n: (d, tuple(s), h) for n, d, s, h in manifest[shard]}
        got = read_header(os.path.join(out, shard))
        rest = {n for n in src if not EXPERT.fullmatch(n)}
        if rest - set(got):
            fails.append(f"{shard}: {len(rest - set(got))} source non-expert tensors missing")
        with open(os.path.join(out, shard), "rb") as f:
            for name, (dtype, shape, a, b) in got.items():
                f.seek(a)
                data = f.read(b - a)
                if name in rest:
                    if (dtype, shape) != src[name][:2] or hashlib.sha256(data).hexdigest() != src[name][2]:
                        fails.append(f"{shard}: {name} differs from the source")
                    n_rest += 1
                    continue
                m = EXPERT_PART.fullmatch(name)
                if not m or m[1] + ".weight" not in src:
                    fails.append(f"{shard}: {name} is neither a source tensor nor an expert part of this shard")
                    continue
                qpath = os.path.join(qdir, f"model.language_model.layers.{m[2]}.safetensors")
                qh = qheaders.get(qpath) or qheaders.setdefault(qpath, read_header(qpath))
                if qh.get(name, ())[:2] != (dtype, shape):
                    fails.append(f"{shard}: {name} is {dtype} {list(shape)}, qtensors has {qh.get(name, ())[:2]}")
                    continue
                with open(qpath, "rb") as g:
                    g.seek(qh[name][2])
                    if g.read(qh[name][3] - qh[name][2]) != data:
                        fails.append(f"{shard}: {name} differs from {qpath}")
                n_parts += 1
    print(f"verify: {len(shards)} shards, {n_rest} non-expert tensors sha256-equal to the source, "
          f"{n_parts} expert parts byte-equal to qtensors, config bits {json.dumps(bits)}, {time.time() - t0:.0f} s")
    for m in fails[:20]:
        print(f"  {m}")
    print("VERIFY", "FAIL" if fails else "PASS")
    return not fails


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("source")
    e.add_argument("extract")
    e.add_argument("--shards", type=int_list, help="1-based source shard numbers (partial test packs)")
    a = sub.add_parser("assemble")
    a.add_argument("extract")
    a.add_argument("qtensors")
    a.add_argument("out")
    a.add_argument("--layers", type=int_list, help="write only these layers' routed experts (partial test packs)")
    v = sub.add_parser("verify")
    v.add_argument("out")
    v.add_argument("extract")
    v.add_argument("qtensors")
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.source, args.extract, args.shards)
    elif args.cmd == "assemble":
        assemble(args.extract, args.qtensors, args.out, args.layers)
    else:
        sys.exit(0 if verify(args.out, args.extract, args.qtensors) else 1)


if __name__ == "__main__":
    main()
