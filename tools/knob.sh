#!/usr/bin/env bash
# Boot one config fresh and record it: tools/knob.sh NAME [VAR=VALUE ...]
# Runs ./stop.sh, then ./run.sh with the overrides, and writes evidence/knobs/NAME/: run.txt (run.sh's output),
# boot.log and boot-rank1.log (each rank's log without request lines), config.txt (the overrides and rank 0's argv).
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=recipe.conf
source "$DIR/recipe.conf"
NAME=${1:?usage: tools/knob.sh NAME [VAR=VALUE ...]}
shift
OUT=$DIR/evidence/knobs/$NAME
mkdir -p "$OUT"
"$DIR/stop.sh"
env "$@" "$DIR/run.sh" 2>&1 | tee "$OUT/run.txt"
quiet='"(GET|POST) /'
docker logs "${NAME_PREFIX}0" 2>&1 | grep -Ev "$quiet" > "$OUT/boot.log" || true
ssh -o BatchMode=yes "$WORKER" docker logs "${NAME_PREFIX}1" 2>&1 | grep -Ev "$quiet" > "$OUT/boot-rank1.log" || true
{
  echo "knob $NAME: ${*:-(recipe.conf defaults)}"
  docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}{{join .Config.Cmd " "}}' "${NAME_PREFIX}0" |
    grep -E '^(TF_|TENSORFOLD_|NCCL_IB_HCA)|serve'
} > "$OUT/config.txt"
grep -E 'CUDA sm_|L2PF|serving' "$OUT/boot.log" || true
