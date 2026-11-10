#!/usr/bin/env bash
# Log MemAvailable (GiB) on the head and the worker until killed: tools/memwatch.sh OUT [SECONDS]
# One line a sample, "HH:MM:SS head GiB worker GiB", the format of evidence/knobs/*/memavail.txt.
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=recipe.conf
source "$DIR/recipe.conf"
OUT=${1:?usage: tools/memwatch.sh OUT [SECONDS]}
EVERY=${2:-30}
avail='awk "/^MemAvailable:/ {printf \"%.1f\", \$2 / 1048576}" /proc/meminfo'
while :; do
  echo "$(date -u +%T) $(hostname -s) $(bash -c "$avail") $WORKER $(ssh -o BatchMode=yes "$WORKER" "$avail")" >> "$OUT"
  sleep "$EVERY"
done
