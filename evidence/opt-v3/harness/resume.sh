#!/usr/bin/env bash
# resume.sh NAME SEED: one client, three turns on a 32k document: what a resumed turn costs with nothing else running.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=~/projects/ai-lab/recipes/GLM-5.3-Flash-EXL3-4bpw-TensorFold-2x-DGX-SPARK
OUT=$W/runs/${1:?name}; mkdir -p "$OUT"
exec 9>"$W/.gpu.lock"; flock 9
python3 "$R/tools/agents.py" --clients 1 --turns 3 --context-tokens 32000 --seed "${2:?seed}" --out "$OUT/resume.json" > "$OUT/resume.txt" 2>&1
echo "resume rc $?"; cat "$OUT/resume.txt"
