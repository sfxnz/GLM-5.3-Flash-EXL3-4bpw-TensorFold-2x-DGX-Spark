#!/usr/bin/env python3
# Writes the two pack-specific inputs hf_capture needs for a --role quant capture, read from the pack's own bytes.
# Usage (inside glm53-fidelity, suite on sys.path): capture_inputs.py PACK OUT_DIR --repo OWNER/NAME --revision REV
#   OUT_DIR/allowlist.json  every index key on a layer >= text_config.num_hidden_layers (the MTP block, which the
#                           transformers model has no module for), as a sorted compact JSON array.
#   OUT_DIR/scope.json      engines/tools/exl3_scope.py's scope, with native dtypes read from the local shard headers
#                           instead of the Hub (exl3_scope only reads headers over HTTP).
# Prints the sha256 pair hf_capture binds the allowlist by (--unexpected-tensors-allowlist-sha256 / -name-sha256).
import argparse
import hashlib
import json
import os
import re
import struct
import sys

import exl3_scope

LAYER = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.")


def local_dtypes(pack):
    def read(repo, revision, weight_map):
        dtypes = {}
        for shard in sorted(set(weight_map.values())):
            with open(os.path.join(pack, shard), "rb") as f:
                header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
            dtypes.update({k: v["dtype"] for k, v in header.items() if k != "__metadata__"})
        return dtypes
    return read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("out_dir")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    args = ap.parse_args()
    config = json.load(open(os.path.join(args.pack, "config.json")))
    index = os.path.join(args.pack, "model.safetensors.index.json")
    layers = (config.get("text_config") or config)["num_hidden_layers"]
    names = sorted(k for k in json.load(open(index))["weight_map"]
                   if (m := LAYER.match(k)) and int(m.group(1)) >= layers)
    if not names:
        sys.exit("no key on a layer >= %d: nothing for the allowlist" % layers)
    raw = json.dumps(names, separators=(",", ":")).encode()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "allowlist.json"), "wb") as f:
        f.write(raw)
    exl3_scope.shard_dtypes = local_dtypes(args.pack)
    rc = exl3_scope.main(["--index", index, "--config", os.path.join(args.pack, "config.json"),
                          "--repo", args.repo, "--revision", args.revision,
                          "--out", os.path.join(args.out_dir, "scope.json"), "--dtypes-from-hub"])
    # The file bytes are the suite's canonical form, so the artifact and name digests are one value.
    print("allowlist %d names sha256 %s" % (len(names), hashlib.sha256(raw).hexdigest()))
    return rc


if __name__ == "__main__":
    sys.exit(main())
