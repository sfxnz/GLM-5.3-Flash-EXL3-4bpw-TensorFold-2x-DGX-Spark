#!/usr/bin/env bash
# Usage: quality/compare.sh <label>
#   Scores ~/projects/data/glm53-exl3/fidelity/<label>/dataset against the BF16 root on CPU, with no GPU and no network.
#   Writes <label>/compare/{comparison-receipt.json,tokenwise-kld.npy} and <label>/compare.log.
#   --vocab-chunk 8192 is the replay profile the root's own qualification compare ran.
#   Exit codes are the suite's: 0 strict, 2 advisory (e.g. cross_stack: a GB10 capture against the H200 root), 3 refused.
# Env: IMAGE, ROOT (reference dataset dir), CPUS (cores for the replay and estimator, default 8 so a GPU job on the
#   same box keeps the rest).
set -euo pipefail
label=${1:?usage: compare.sh <label>}
image=${IMAGE:-glm53-fidelity:cf36287c-mcg1-stream}
base=$HOME/projects/data/glm53-exl3/fidelity
root=${ROOT:-$base/root-v1@bdd25fe0}
cpus=${CPUS:-8}
cand=$base/$label/dataset
out=$base/$label/compare
[ -f "$cand/fidelity-dataset.json" ] || { echo "compare.sh: no dataset at $cand" >&2; exit 1; }
[ ! -e "$out" ] || { echo "compare.sh: $out exists; move it aside first" >&2; exit 1; }
docker run --rm --network none --user 1000:1000 -e HOME=/tmp \
    --cpus "$cpus" -e OMP_NUM_THREADS="$cpus" -e OPENBLAS_NUM_THREADS="$cpus" \
    -v "$root:/data/root:ro" -v "$cand:/data/cand:ro" -v "$base/$label:/data/out" \
    --entrypoint fidelity-dataset "$image" compare --reference /data/root --candidate /data/cand \
    --out /data/out/compare --vocab-chunk 8192 \
    --reference-label "$(basename "$root")" --candidate-label "$label" 2>&1 | tee "$base/$label/compare.log"
