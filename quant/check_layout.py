#!/usr/bin/env python3
# Offline check that a candidate pack meets TensorFold 1.2.2's GLM EXL3 contract (quant/README-contract.md).
# Usage: check_layout.py CANDIDATE SOURCE [--cache DIR] [--suh-sample N] [--layers L,L,...]
#   --layers: check a partial pack. Only tensors of these layers, and those outside any layer, are compared.
#   CANDIDATE, SOURCE: a local pack directory, or hf:owner/repo@rev (read by HTTP range requests).
# Reads config.json, the index and every shard header; of tensor data it reads only the 4-byte mcg markers and
# the gate/up suh vectors (all experts, or N evenly spaced per layer with --suh-sample). Exit 0 on PASS, 1 on FAIL.
import argparse
import collections
import concurrent.futures
import http.client
import json
import os
import re
import struct
import sys
import urllib.request
from dataclasses import dataclass

MCG = 0xCBAC1FED
PROJS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("trellis", "suh", "svh", "mcg")
SHOW = 5
LAYER = re.compile(r"model\.language_model\.layers\.(\d+)\.")


@dataclass(frozen=True)
class Tensor:
    dtype: str
    shape: tuple
    file: str
    start: int
    end: int


class Pack:
    def __init__(self, loc, cache):
        self.loc = loc
        self.remote = loc.startswith("hf:")
        self.cache = None
        if self.remote:
            repo, rev = loc[3:].split("@")
            self.base = f"https://huggingface.co/{repo}/resolve/{rev}/"
            self.cache = os.path.join(cache, f"{repo.replace('/', '--')}@{rev}") if cache else None
            token = os.environ.get("HF_TOKEN") or _read(os.path.expanduser("~/.cache/huggingface/token"))
            self.headers = {"Authorization": f"Bearer {token.strip()}"} if token else {}
        self.config = json.loads(self._cached("config.json", lambda: self._get("config.json")))
        index = json.loads(self._cached("model.safetensors.index.json", lambda: self._get("model.safetensors.index.json")))
        shards = sorted(set(index["weight_map"].values()))
        self.tensors = {}
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            for shard, (hlen, header) in zip(shards, pool.map(self._header, shards)):
                header.pop("__metadata__", None)
                for name, v in header.items():
                    a, b = v["data_offsets"]
                    self.tensors[name] = Tensor(v["dtype"], tuple(v["shape"]), shard, 8 + hlen + a, 8 + hlen + b)
        unindexed = set(index["weight_map"]) ^ set(self.tensors)
        if unindexed:
            raise SystemExit(f"{loc}: index and shard headers disagree on {len(unindexed)} tensors, e.g. {sorted(unindexed)[:SHOW]}")

    def _get(self, file, start=None, end=None):
        if not self.remote:
            with open(os.path.join(self.loc, file), "rb") as f:
                if start is None:
                    return f.read()
                f.seek(start)
                return f.read(end - start)
        req = urllib.request.Request(self.base + file, headers=dict(self.headers))
        if start is not None:
            req.add_header("Range", f"bytes={start}-{end - 1}")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if start is not None and len(data) != end - start:
                    raise OSError(f"short range read {len(data)} of {end - start}")
                return data
            except (OSError, http.client.HTTPException):
                if attempt == 4:
                    raise

    def _cached(self, key, fetch):
        path = self.cache and os.path.join(self.cache, key)
        if path and os.path.isfile(path):
            return _read(path)
        data = fetch()
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
        return data

    def _header(self, shard):
        def fetch():
            n = struct.unpack("<Q", self._get(shard, 0, 8))[0]
            return json.dumps({"hlen": n, "header": json.loads(self._get(shard, 8, 8 + n))}).encode()
        h = json.loads(self._cached(f"headers/{shard}.json", fetch))
        return h["hlen"], h["header"]

    def read(self, names):
        """Bytes of the named tensors, nearby spans in one file merged into one read."""
        spans = collections.defaultdict(list)
        for n in names:
            t = self.tensors[n]
            spans[t.file].append((t.start, t.end, n))
        jobs = []
        for file, s in spans.items():
            s.sort()
            group = [s[0]]
            for item in s[1:]:
                if item[0] - group[-1][1] > 65536:
                    jobs.append((file, group))
                    group = []
                group.append(item)
            jobs.append((file, group))

        def run(job):
            file, group = job
            lo, hi = group[0][0], max(e for _, e, _ in group)
            blob = self._get(file, lo, hi)
            return {n: blob[a - lo:b - lo] for a, b, n in group}

        out = {}
        with concurrent.futures.ThreadPoolExecutor(8 if self.remote else 1) as pool:
            for part in pool.map(run, jobs):
                out.update(part)
        return out


def _read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


def text_config(config):
    return config.get("text_config", config)


def expert_key(layer, e, proj, part=None):
    k = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}"
    return f"{k}.{part}" if part else k


def check(cand, src, suh_sample, only_layers=None):
    def in_scope(name):
        m = only_layers is not None and LAYER.match(name)
        return not m or int(m[1]) in only_layers

    fails = collections.defaultdict(list)
    report = []

    q = cand.config.get("quantization_config")
    if not isinstance(q, dict):
        fails["config"].append("no top-level quantization_config (TF config.zig reads exl3 only from the top level)")
        q = {}
    if q.get("quant_method") != "exl3":
        fails["config"].append(f"quant_method {q.get('quant_method')!r}, want \"exl3\"")
    bits = q.get("bits")
    if type(bits) is not int or bits != 4:
        fails["config"].append(f"bits {json.dumps(bits)} ({type(bits).__name__}), want the JSON integer 4")
    if q.get("codebook") != "mcg":
        fails["config"].append(f"codebook {json.dumps(q.get('codebook'))}, want \"mcg\"")

    tc = text_config(src.config)
    first, n_layers = tc["first_k_dense_replace"], tc["num_hidden_layers"]
    layers = [l for l in range(first, n_layers + tc.get("num_nextn_predict_layers", 0)) if only_layers is None or l in only_layers]
    n_exp, hidden = tc["n_routed_experts"], tc["hidden_size"]
    for k in ("first_k_dense_replace", "num_hidden_layers", "num_nextn_predict_layers", "n_routed_experts", "hidden_size", "moe_intermediate_size"):
        if text_config(cand.config).get(k) != tc.get(k):
            fails["config"].append(f"text_config.{k} {text_config(cand.config).get(k)} differs from the source's {tc.get(k)}")
    if hidden % 128:
        fails["config"].append(f"hidden_size {hidden} is not a multiple of 128 (cuda_exl3.zig load)")
    report.append(f"routed experts: layers {layers[0]}..{layers[-1]} x {n_exp} experts x 3 projections" if layers else "routed experts: none in --layers")
    if only_layers is not None:
        report.append(f"partial check, --layers {sorted(only_layers)}: other layers' tensors are ignored in both packs")

    expert_weights = {expert_key(l, e, p, "weight") for l in layers for e in range(n_exp) for p in PROJS}
    missing_src = sorted(expert_weights - set(src.tensors))
    if missing_src:
        raise SystemExit(f"source lacks {len(missing_src)} routed expert weights, e.g. {missing_src[:SHOW]}")

    markers = []
    for l in layers:
        for e in range(n_exp):
            for p in PROJS:
                n_out, k_in = src.tensors[expert_key(l, e, p, "weight")].shape
                want = {"trellis": ("I16", (k_in // 16, n_out // 16, 64)), "suh": ("F16", (k_in,)), "svh": ("F16", (n_out,))}
                if k_in % 16 or n_out % 16:
                    fails["expert"].append(f"{expert_key(l, e, p)}: [{n_out}, {k_in}] does not tile by 16")
                for part, (dt, shape) in want.items():
                    t = cand.tensors.get(expert_key(l, e, p, part))
                    if t is None:
                        fails["expert"].append(f"{expert_key(l, e, p, part)} missing")
                    elif (t.dtype, t.shape) != (dt, shape):
                        fails["expert"].append(f"{expert_key(l, e, p, part)} is {t.dtype} {list(t.shape)}, want {dt} {list(shape)}")
                m = cand.tensors.get(expert_key(l, e, p, "mcg"))
                if m is None:
                    fails["expert"].append(f"{expert_key(l, e, p, 'mcg')} missing")
                elif m.dtype != "I32" or m.shape not in ((), (1,)):
                    fails["expert"].append(f"{expert_key(l, e, p, 'mcg')} is {m.dtype} {list(m.shape)}, want I32 numel 1")
                else:
                    markers.append(expert_key(l, e, p, "mcg"))
                if expert_key(l, e, p, "weight") in cand.tensors:
                    fails["expert"].append(f"{expert_key(l, e, p, 'weight')} left in the pack")
    for name, raw in cand.read(markers).items():
        if len(raw) != 4 or struct.unpack("<I", raw)[0] != MCG:
            fails["marker"].append(f"{name} = {raw.hex()}, want {MCG:#x} little-endian")
    report.append(f"mcg markers read: {len(markers)}")

    expert_parts = {expert_key(l, e, p, part) for l in layers for e in range(n_exp) for p in PROJS for part in PARTS}
    src_rest = {n: t for n, t in src.tensors.items() if n not in expert_weights and in_scope(n)}
    cand_rest = {n: t for n, t in cand.tensors.items() if n not in expert_parts and n not in expert_weights and in_scope(n)}
    for n in sorted(set(src_rest) - set(cand_rest)):
        fails["non-expert"].append(f"{n} missing")
    for n in sorted(set(cand_rest) - set(src_rest)):
        fails["non-expert"].append(f"{n} not in the source")
    for n in sorted(set(src_rest) & set(cand_rest)):
        s, c = src_rest[n], cand_rest[n]
        if (s.dtype, s.shape) != (c.dtype, c.shape):
            fails["non-expert"].append(f"{n} is {c.dtype} {list(c.shape)}, source {s.dtype} {list(s.shape)}")
    for n, t in src_rest.items():
        if t.dtype not in ("BF16", "F32"):
            fails["non-expert"].append(f"source {n} is {t.dtype}; TF converts only bf16 and f32 (cuda_weights.zig add)")
    dtypes = collections.Counter(t.dtype for t in cand_rest.values())
    report.append(f"non-expert tensors: {len(cand_rest)} in candidate, {len(src_rest)} in source, dtypes {dict(dtypes)}")
    report.append(f"candidate tensors total: {len(cand.tensors)}")

    picks = range(n_exp) if not suh_sample else sorted({round(i * (n_exp - 1) / max(suh_sample - 1, 1)) for i in range(suh_sample)})
    want_suh = [expert_key(l, e, p, "suh") for l in layers for e in picks for p in ("gate_proj", "up_proj")]
    want_suh = [n for n in want_suh if n in cand.tensors]
    got = cand.read(want_suh)
    scope = f"all {n_exp} experts" if not suh_sample else f"{len(picks)} sampled experts {list(picks)}"
    report.append(f"gate/up suh sharing per layer over {scope} (TF shx: one input rotation per layer):")
    for l in layers:
        vecs = {got[n] for n in got if f".layers.{l}." in n}
        report.append(f"  layer {l}: {len(vecs)} distinct gate/up suh -> shx={'true' if len(vecs) == 1 else 'false'}")
    return report, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("source")
    ap.add_argument("--cache", help="directory caching remote config, index and shard headers")
    ap.add_argument("--suh-sample", type=int, default=0, help="experts per layer read for suh sharing (0 = all)")
    ap.add_argument("--layers", type=lambda s: {int(x) for x in s.split(",")}, help="comma-separated layers (partial pack)")
    a = ap.parse_args()
    cand, src = Pack(a.candidate, a.cache), Pack(a.source, a.cache)
    print(f"candidate {a.candidate}\nsource    {a.source}")
    report, fails = check(cand, src, a.suh_sample, a.layers)
    print("\n".join(report))
    for kind, msgs in fails.items():
        print(f"FAIL {kind}: {len(msgs)} problem(s)")
        for m in msgs[:SHOW]:
            print(f"  {m}")
        if len(msgs) > SHOW:
            print(f"  ... {len(msgs) - SHOW} more")
    print("RESULT", "FAIL" if fails else "PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
