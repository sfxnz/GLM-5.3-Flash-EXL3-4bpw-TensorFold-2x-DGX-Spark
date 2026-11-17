#!/usr/bin/env bash
# panel.sh NAME [decode|full|prefill]: measure the pair that is serving now into runs/NAME/ and append a ledger line.
set -euo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=~/projects/ai-lab/recipes/GLM-5.3-Flash-EXL3-4bpw-TensorFold-2x-DGX-SPARK
NAME=${1:?name}; MODE=${2:-full}
OUT=$W/runs/$NAME; mkdir -p "$OUT"
exec 9>"$W/.gpu.lock"; flock -n 9 || { echo "another panel holds the GPU lock"; exit 2; }
state() {
  echo "$(date -u +%FT%TZ) $1 head load $(cut -d' ' -f1-3 /proc/loadavg) hot $(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1) gpu $(nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader | tr -d ' ') avail $(awk '/^MemAvailable:/ {printf "%.1f", $2/1048576}' /proc/meminfo) | worker $(ssh -o BatchMode=yes spark2 "cut -d' ' -f1-3 /proc/loadavg; cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1; awk '/^MemAvailable:/ {printf \"%.1f\", \$2/1048576}' /proc/meminfo" | xargs)"
}
# An idle host: wait up to 4 min for the 1-min load to fall under 1.5 on both nodes, then record whatever it is.
for _ in $(seq 12); do
  h=$(cut -d' ' -f1 /proc/loadavg); w=$(ssh -o BatchMode=yes spark2 "cut -d' ' -f1 /proc/loadavg")
  awk -v h="$h" -v w="$w" 'BEGIN {exit !(h < 1.5 && w < 1.5)}' && break; sleep 5
done
other=$(docker ps --format '{{.Names}}' | grep -v '^tf-glm53-rank0$' || true)
state start > "$OUT/state.txt"; echo "other containers on head: ${other:-none}" >> "$OUT/state.txt"
if [[ $MODE == gated* ]]; then  # gated or gated=PLAN: one run at a time, only while the head is quiet
  plan=${MODE#gated}; plan=${plan#=}
  python3 "$W/gated.py" "$NAME" ${plan:+--plan "$plan"} > "$OUT/gated.txt" 2>&1 || echo "gated rc $?" >> "$OUT/state.txt"
  python3 "$W/greedy.py" "$NAME" >> "$OUT/state.txt" 2>&1 || echo "greedy rc $?" >> "$OUT/state.txt"
  state after-gated >> "$OUT/state.txt"; tail -1 "$OUT/gated.txt"; exit 0
fi
if [[ $MODE == quick ]]; then
  python3 "$R/tools/bench.py" --concurrency 1 --runs 6 --warmup 1 --out "$OUT/bench-c1.json" > "$OUT/bench.txt"
  python3 "$R/tools/bench.py" --concurrency 8 --runs 3 --warmup 1 --out "$OUT/bench-c48.json" >> "$OUT/bench.txt"
  python3 "$R/tools/bench.py" --concurrency 1 --runs 4 --warmup 1 --temperature 0 --out "$OUT/bench-greedy.json" > "$OUT/bench-greedy.txt"
  state after-bench >> "$OUT/state.txt"
  python3 "$R/tools/prefill.py" 32000 --out "$OUT/prefill.json" > "$OUT/prefill.txt" || echo "prefill rc $?" >> "$OUT/state.txt"
  state after-prefill >> "$OUT/state.txt"
  python3 "$W/ledger.py" "$NAME"; exit 0
fi
if [[ $MODE != prefill ]]; then
  python3 "$R/tools/bench.py" --concurrency 1 --runs 6 --warmup 1 --out "$OUT/bench-c1.json" > "$OUT/bench.txt"
  python3 "$R/tools/bench.py" --concurrency 4,8 --runs 4 --warmup 1 --out "$OUT/bench-c48.json" >> "$OUT/bench.txt"
  state after-bench >> "$OUT/state.txt"
  python3 "$R/tools/bench.py" --concurrency "${GREEDY_C:-1}" --runs 6 --warmup 1 --temperature 0 --out "$OUT/bench-greedy.json" > "$OUT/bench-greedy.txt"
  state after-greedy >> "$OUT/state.txt"
fi
if [[ $MODE != decode ]]; then
  python3 "$R/tools/prefill.py" 8000,32000,128000 --out "$OUT/prefill.json" > "$OUT/prefill.txt" || echo "prefill rc $?" >> "$OUT/state.txt"
  state after-prefill >> "$OUT/state.txt"
fi
python3 "$W/ledger.py" "$NAME"
