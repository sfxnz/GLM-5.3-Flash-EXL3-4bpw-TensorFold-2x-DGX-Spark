#!/usr/bin/env bash
# Usage: quant/convert.sh <variant> [extra convert.py args...]
#   quant/convert.sh probe-a --max_module 6     # embed, hc_expand, dense layers 0-2, MoE layers 3-4
#   quant/convert.sh a                          # baseline
#   quant/convert.sh b --moe_affinity_eps 0.01  # affinity-weighted expert Hessians
#   quant/convert.sh a --resume                 # resume an interrupted job (container name must be free)
# Env: IMAGE, MEM (cgroup cap, no swap), SRC_REPO (HF cache repo dir of the BF16 source).
# The NGC base sets TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which forces TF32 matmuls; convert.py runs with it unset.
# Only long option names reach convert.py: the base's nvidia-cutlass-dsl parses sys.argv with its own argparse
# (-h, -diagnostic) when a MoE layer first imports it, so -hb read as -h printed its help and exited 0 mid-run.
# The output dir is a 1 GiB tmpfs: convert.py always ends by compiling a stock ~176 GB pack that TensorFold cannot
# read and spark2 has no room for. It fails there with ENOSPC after every qtensors/ file is written;
# quant/assemble.py builds the served pack from qtensors/ and the BF16 source.
set -euo pipefail
variant=${1:?usage: convert.sh <variant> [extra convert.py args...]}
shift
for a in "$@"; do
    [[ $a == -[!-]* ]] && { echo "convert.sh: pass long option names only (got $a)" >&2; exit 2; }
done
here=$(cd "$(dirname "$0")" && pwd)
image=${IMAGE:-glm53-quant:exl3-1.6.6}
mem=${MEM:-100g}
repo=${SRC_REPO:-$HOME/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash-BF16}
src=$repo/snapshots/f9fed3bbc82291073373d67452e9f25317ef7577
root=$HOME/projects/data/glm53-exl3/$variant
name=glm53-quant-$variant

python3 -I - "$src" <<'PY'
import json, os, sys
src = sys.argv[1]
if not os.path.isfile(os.path.join(src, "model.safetensors.index.json")):
    sys.exit(f"source incomplete: no model.safetensors.index.json in {src}")
index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
shards = sorted(set(index["weight_map"].values()))
missing = [s for s in shards if not os.path.isfile(os.path.join(src, s))]
if missing or not os.path.isfile(os.path.join(src, "config.json")):
    sys.exit(f"source incomplete: {len(missing)} of {len(shards)} shards missing, e.g. {missing[:3]}")
print(f"source complete: {len(shards)} shards")
PY
if docker container inspect "$name" >/dev/null 2>&1; then
    echo "container $name exists; inspect it (docker logs $name) and remove it yourself before relaunching" >&2
    exit 1
fi
mkdir -p "$root/work"
cmd=(docker run -d --name "$name" --restart no --gpus all --network none
    --memory "$mem" --memory-swap "$mem" --oom-score-adj 500 --user 1000:1000
    -e HOME=/tmp -e PYTHONUNBUFFERED=1
    -v "$repo:$repo:ro" -v "$here:/recipe:ro" -v "$root:$root"
    --mount "type=tmpfs,destination=$root/out,tmpfs-size=1g"
    --entrypoint env "$image" -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3 convert.py
    --in_dir "$src" --work_dir "$root/work" --out_dir "$root/out" --recipe /recipe/experts4.yaml
    --bits 4.0 --codebook mcg --head_bits 16 --mtp_bits 4 --vision_bits 16 --embed_bits 16
    --cal_rows 250 --cal_cols 2048 --devices 0 "$@")
printf '%q ' "${cmd[@]}" > "$root/launch-$(date -u +%Y%m%dT%H%M%SZ).cmd"
"${cmd[@]}"
echo "started $name; follow with: docker logs -f $name"
