#!/usr/bin/env bash
# qualify.sh NAME: the ship gates on the pair that is serving now, into runs/NAME/ (smoke, NLL against the v2 default,
# agents, a 50% needle, a 1M needle), with MemAvailable logged on both nodes throughout.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=~/projects/ai-lab/recipes/GLM-5.3-Flash-EXL3-4bpw-TensorFold-2x-DGX-SPARK
N=~/projects/data/glm53-exl3/nll
NAME=${1:?name}; OUT=$W/runs/$NAME; mkdir -p "$OUT"
exec 9>"$W/.gpu.lock"; flock 9
"$R/tools/memwatch.sh" "$OUT/memavail.txt" 30 < /dev/null & MW=$!
step() { echo "$(date +%T) $*"; }
step smoke; python3 "$R/tests/smoke.py" --out "$OUT/smoke.json" > "$OUT/smoke.txt" 2>&1; echo "smoke rc $?"
step nll; python3 "$R/tools/nll.py" capture --label "$NAME" --out "$N/$NAME" > "$OUT/nll-capture.txt" 2>&1; echo "capture rc $?"
python3 "$R/tools/nll.py" compare "$N/${REF:-v2-default}" "$N/$NAME" --json "$OUT/nll.json" > "$OUT/nll.txt" 2>&1; echo "nll rc $?"
step agents; python3 "$R/tools/agents.py" --clients 8 --turns 4 --context-tokens 32000 --seed "${SEED:-31}" --out "$OUT/agents.json" > "$OUT/agents.txt" 2>&1; echo "agents rc $?"
step needle-half; python3 "$R/tools/needle.py" 524288 --out "$OUT/needle-half.jsonl" > "$OUT/needle-half.txt" 2>&1; echo "needle-half rc $?"
step needle-1m; python3 "$R/tools/needle.py" 1048576 --out "$OUT/needle.jsonl" > "$OUT/needle.txt" 2>&1; echo "needle-1m rc $?"
kill "$MW"
step QUALIFY_DONE
