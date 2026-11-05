#!/usr/bin/env bash
# Usage: REPO=owner/name REVISION=<40-hex> AUTHOR=<hf-handle> quality/capture.sh <label> <pack_dir> [extra hf_capture args]
#   Captures the pack's final hidden states on the 25 x 2048 Flash panel with the suite's hf-transformers engine
#   (layer-outer, one decoder layer resident, EXL3 experts decoded to BF16 in torch) on this box's GPU, and writes
#   ~/projects/data/glm53-exl3/fidelity/<label>/{inputs/,dataset/,memory.json,capture.log}. Score it with compare.sh.
#   REPO/REVISION name the weights in the dataset's provenance and scope notes.
#   TOKENIZER_ROOT (default: the pack) is where the panel binding reads tokenizer.json, tokenizer_config.json and
#   chat_template.jinja. A pack whose chat_template.jinja differs from the root's needs a root holding the root's copy
#   under .reference/ (quant-fidelity-suite bin/fidelity/panel.py:558); the panel is token ids, so the template never
#   reaches the score.
#   INPUTS_ONLY=1 stops after the CPU-only inputs (panel binding, unexpected-tensor allowlist, scope); nothing touches
#   a GPU until then.
# Every argument the container sees is a long option: the NGC base's nvidia-cutlass-dsl parses sys.argv with its own
# argparse (-h, -diagnostic) when first imported, so a short option or a --help prefix could end the run early.
# The image sets TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0, so fp32 matmuls start at "highest" (no TF32).
set -euo pipefail
label=${1:?usage: capture.sh <label> <pack_dir> [extra hf_capture args]}
pack=$(realpath "${2:?usage: capture.sh <label> <pack_dir>}")
shift 2
: "${REPO:?set REPO=owner/name}" "${REVISION:?set REVISION=<40-hex>}" "${AUTHOR:?set AUTHOR=<hf-handle>}"
for a in "$@"; do
    [[ $a == -[!-]* || ( ${#a} -ge 3 && --help == "$a"* ) ]] && { echo "capture.sh: long options only (got $a)" >&2; exit 2; }
done
image=${IMAGE:-glm53-fidelity:cf36287c-mcg1-stream}
out=$HOME/projects/data/glm53-exl3/fidelity/$label
[ ! -e "$out/dataset" ] || { echo "capture.sh: $out/dataset exists; move it aside first" >&2; exit 1; }
[ -f "$pack/model.safetensors.index.json" ] || { echo "capture.sh: no model.safetensors.index.json in $pack" >&2; exit 1; }
# An HF cache snapshot holds symlinks into ../../blobs, so mount the whole repo cache dir.
case $pack in */snapshots/*) mnt=${pack%/snapshots/*} ;; *) mnt=$pack ;; esac
tok=$(realpath "${TOKENIZER_ROOT:-$pack}")
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
run=(docker run --rm --network none --user 1000:1000 -e HOME=/tmp -v "$mnt:$mnt:ro" -v "$out:$out")
[[ $tok == "$pack" || $tok == "$out"/* ]] || run+=(-v "$tok:$tok:ro")
panel=engines/panels/panel--glm53.brandonmusic.final25

# Inputs are built fresh every time; existing inputs must equal them byte for byte, so a rerun converges.
fresh=$(mktemp -d -p "$out" inputs.XXXXXX)
trap 'rm -rf "$fresh"' EXIT
"${run[@]}" --entrypoint fidelity-dataset "$image" panel-binding \
    --panel "$panel" --tokenizer-root "$tok" --out "$fresh/panel.binding.json"
"${run[@]}" -v "$here:/recipe:ro" -e PYTHONPATH=/opt/fidelity/suite/bin:/opt/fidelity/suite/engines/tools \
    --entrypoint python3 "$image" /recipe/capture_inputs.py "$pack" "$fresh" --repo "$REPO" --revision "$REVISION"
if [ -d "$out/inputs" ]; then
    diff -rq "$fresh" "$out/inputs" >/dev/null || { echo "capture.sh: $out/inputs differs from a fresh build; move it aside" >&2; exit 1; }
else
    mv "$fresh" "$out/inputs"
fi
[ "${INPUTS_ONLY:-0}" = 1 ] && exit 0

binding_sha=$(sha256sum "$out/inputs/panel.binding.json" | cut -d' ' -f1)
allow_sha=$(sha256sum "$out/inputs/allowlist.json" | cut -d' ' -f1)
image_id=$(docker image inspect --format '{{.Id}}' "$image")
set +e
"${run[@]}" --gpus all -e STACKPRINT_IMAGE_PIN="$image@$image_id" --entrypoint fidelity-dataset "$image" capture \
    --engine hf-transformers --out "$out/dataset" --form hidden --role quant --lane streaming -- \
    --model "$pack" --model-revision "$REVISION" --weights-repository "$REPO" \
    --panel "$panel" --panel-tokenizer-root "$tok" --panel-binding "$out/inputs/panel.binding.json" --panel-binding-sha256 "$binding_sha" \
    --schedule layer-outer --layer-residency stream --device cuda --dtype bfloat16 --sanity-expect Paris \
    --scope-file "$out/inputs/scope.json" --codec exl3-mcg --declared-bits 4.0 \
    --unexpected-tensors-allowlist "$out/inputs/allowlist.json" \
    --unexpected-tensors-allowlist-sha256 "$allow_sha" --unexpected-tensors-name-sha256 "$allow_sha" \
    --dataset-id "fidelity--glm53flash.$AUTHOR.quant.$label" --dataset-name "fidelity--glm53flash.$AUTHOR.quant.$label" \
    --run-name "$label-1" --cold-run "$label-1" --author "$AUTHOR" --memory-report "$out/memory.json" \
    "$@" 2>&1 | tee "$out/capture.log"
status=${PIPESTATUS[0]}
set -e
# 2 is "sealed with caveat disclosures" (run_count 1, the intentionally unused MTP tensors), not a failure.
[ "$status" = 0 ] || [ "$status" = 2 ] || { echo "capture.sh: capture exited $status" >&2; exit "$status"; }
echo "capture.sh: sealed $out/dataset (exit $status); next: quality/compare.sh $label"
