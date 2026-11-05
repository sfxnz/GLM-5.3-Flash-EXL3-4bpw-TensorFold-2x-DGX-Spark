import json, os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
tr3 = "brandonmusic--GLM-5.3-Flash-tr3-4bpw@a5fee929"
bf16 = "zai-org--GLM-5.3-Flash-BF16@f9fed3bbc82291073373d67452e9f25317ef7577"
shutil.rmtree(dst, ignore_errors=True)
os.makedirs(dst)
shutil.copytree(os.path.join(src, tr3), os.path.join(dst, tr3))
os.symlink(os.path.join(src, bf16), os.path.join(dst, bf16))
p = os.path.join(dst, tr3, "config.json")
c = json.load(open(p)); c["quantization_config"]["bits"] = 4.0
json.dump(c, open(p, "w"), indent=2)
p = os.path.join(dst, tr3, "headers", "model-00010-of-00120.safetensors.json")
h = json.load(open(p)); k = "model.language_model.layers.13.mlp.experts.100.gate_proj.trellis"
assert h["header"][k]["dtype"] == "I16"; h["header"][k]["dtype"] = "F16"
json.dump(h, open(p, "w"))
print("broken: config bits 4 -> 4.0;", k, "dtype I16 -> F16 (same byte size, offsets unchanged)")
