#!/usr/bin/env bash
# agents.sh NAME SEED: the 8-agent, 4-turn run on the pair serving now, into runs/NAME/, with MemAvailable logged.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=~/projects/ai-lab/recipes/GLM-5.3-Flash-EXL3-4bpw-TensorFold-2x-DGX-SPARK
NAME=${1:?name}; OUT=$W/runs/$NAME; mkdir -p "$OUT"
exec 9>"$W/.gpu.lock"; flock 9
"$R/tools/memwatch.sh" "$OUT/memavail.txt" 30 < /dev/null & MW=$!
python3 "$R/tools/agents.py" --clients 8 --turns 4 --context-tokens 32000 --seed "${2:?seed}" --out "$OUT/agents.json" > "$OUT/agents.txt" 2>&1
echo "$(date +%T) agents rc $?"; kill "$MW"; echo AGENTS_DONE
