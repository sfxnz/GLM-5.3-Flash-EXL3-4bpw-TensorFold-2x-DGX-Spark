#!/usr/bin/env bash
# Usage: quant/move_qtensors.sh <variant> [dest]
# Runs on spark1 while quant/convert.sh encodes on spark2. Moves each finished qtensors/*.safetensors file to dest
# (default ~/projects/data/glm53-exl3/full/qtensors) so spark2 holds only the checkpoints and the newest layers:
# spark2 cannot hold ~173 GB of qtensors beside the BF16 source and a 96 GB checkpoint peak. exllamav3 only writes
# qtensors/ (convert_model.py:1726, 1905) and never reads it before compile, so moving finished files is safe.
# save_file writes in place, so a file moves only when it is not the newest one and its size equals its own
# safetensors header's end. Exits when the encoder container has stopped and nothing is left to move.
set -euo pipefail
variant=${1:?usage: move_qtensors.sh <variant> [dest]}
dest=${2:-$HOME/projects/data/glm53-exl3/full/qtensors}
src=projects/data/glm53-exl3/$variant/work/qtensors
name=glm53-quant-$variant
mkdir -p "$dest"
complete_files() {
    ssh spark2 "python3 -I - $src $name" <<'PY'
import json, os, struct, sys
d = sys.argv[1]
files = sorted((os.path.getmtime(os.path.join(d, f)), f) for f in os.listdir(d) if f.endswith(".safetensors"))
running = os.popen("docker inspect -f '{{.State.Running}}' " + sys.argv[2] + " 2>/dev/null").read().strip()
for _, f in (files if running != "true" else files[:-1]):
    p = os.path.join(d, f)
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    end = max((v["data_offsets"][1] for k, v in h.items() if k != "__metadata__"), default=0)
    if os.path.getsize(p) == 8 + n + end:
        print(f)
PY
}
while :; do
    running=$(ssh spark2 "docker inspect -f '{{.State.Running}}' $name 2>/dev/null" || echo gone)
    mapfile -t ready < <(complete_files)
    for f in "${ready[@]}"; do
        rsync -a --remove-source-files "10.100.8.2:$src/$f" "$dest/$f"
        echo "$(date -u +%FT%TZ) moved $f $(stat -c %s "$dest/$f")"
    done
    if [[ $running != true && ${#ready[@]} -eq 0 ]]; then
        echo "$(date -u +%FT%TZ) encoder $name stopped ($running); nothing left to move"
        exit 0
    fi
    sleep 120
done
