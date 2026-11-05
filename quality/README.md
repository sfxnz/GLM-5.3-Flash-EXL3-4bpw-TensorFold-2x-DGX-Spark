# Panel KLD: ours vs TR3

Paired full-vocab KLD(BF16 || candidate) on the 25 x 2048 Flash panel (`panel--glm53.brandonmusic.final25`, 51,175 positions).

## Inputs (pinned)

- Reference: `malaiwah/glm53-flash-fidelity-root-v1@bdd25fe0` (MIT). It is a capture of `zai-org/GLM-5.3-Flash-BF16@a6c167b6`. All 120 weight shards are sha-equal to `@f9fed3bb`.
- TR3: `malaiwah/glm53-flash-fidelity-exl3-tr3-4bpw-miaailab-v1@3156c323` (MIT). It is a capture of `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw@024db9f7`. All 120 shards and `config.json` are sha-equal to `brandonmusic/GLM-5.3-Flash-tr3-4bpw@a5fee929`.
- Code: `malaiwah/quant-fidelity-suite@cf36287c` (MIT, Michel Belleau; the repo was formerly named `glm53-flash-fidelity-suite`) plus two patches, `patches/0001-mcg-marker-rank1.patch` (see Suite patch) and `patches/0002-drop-private-shard-pages.patch` (see GB10 capture memory). We run it and do not copy it.

## Image

`glm53-fidelity:cf36287c-mcg1-stream` is the NGC base `nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699…` (torch 2.13.0a0, CUDA 13.3, sm_120 cubins that GB10 runs) plus transformers 5.16.1, accelerate 1.14.0 and typer, installed `--no-deps` with the suite lock's hashes (`requirements-capture.txt`). Torch, numpy, tokenizers and safetensors stay the base's. The suite lands at `/opt/fidelity/suite`: upstream's `bin/BUNDLE.txt` set, the two CLI wrappers, `LICENSE`, `fidelity-stats` with `jointstd`, and `engines/tools/exl3_scope.py`. The two patches are applied on top, in order, and `/opt/fidelity/suite/PATCHES` holds their sha256s.

```sh
S=~/projects/data/glm53-exl3/src/quant-fidelity-suite
git init $S && git -C $S remote add origin https://github.com/malaiwah/quant-fidelity-suite.git
git -C $S fetch --depth 1 --filter=blob:none origin cf36287c51a18e9361e009ca00a21eb8c0021ab1
git -C $S show cf36287c51a18e9361e009ca00a21eb8c0021ab1:bin/BUNDLE.txt | grep -v -e '^#' -e '^$' \
  | git -C $S sparse-checkout set --no-cone --stdin
git -C $S sparse-checkout add /LICENSE /bin/fidelity-dataset /bin/fidelity-stats /bin/fidelity_stats.py \
  '/bin/jointstd/*.py' /engines/tools/exl3_scope.py
git -C $S checkout cf36287c51a18e9361e009ca00a21eb8c0021ab1         # about 11 MB of objects
docker build -f quality/Dockerfile --build-context suite=$S -t glm53-fidelity:cf36287c-mcg1-stream quality/
quality/verify-image.sh <pack dir>    # CPU only: pins, TF32 default, CLIs, 2-layer forward, no cutlass import
```

## Suite patch

At cf36287c, `exl3_storage_inventory` (`engines/tools/layer_outer.py:1266-1268`, mirrored at `bin/fidelity/hfmeta.py:1062-1064`) refuses an EXL3 codebook marker unless its `quantization_config.json` declaration is I32 shape `[]`. TR3 `@a5fee929` ships that sidecar and stores all 37,152 `.mcg` markers as I32 shape `[1]` (value -877912083, the mcg multiplier). The sidecar declares the same shape, so the cf36287c capture refuses TR3 before it loads a layer.

The rule is newer than the published TR3-h200 dataset. That capture ran `layer_outer.py` sha256 `870724873c…` (b832c8d0) and `hf_capture.py` `200ba12c…` (254fa350), per its `runtime/capture-runtime.json`, and was created 2026-09-06T10:24Z. Neither file reads the sidecar. Upstream 57ed8bb6 (2026-09-09, "jobs: bind hydrated mixed storage…") added the sidecar reader and, with it, the `[]`-only rule. No upstream test pins `[1]` as invalid. The suite's own decoder reads a marker of any rank (8171360, "read a lazy safetensors slice at any rank"; `layer_outer.py:1619` checks `int(marker.reshape(-1)[0])`), and the `[]` fixture records what the checkpoints upstream had seen write, not a format rule.

The patch admits shape `[]` or `[1]` (one I32 value) in both copies and changes nothing else. Shape `[2]` and dtype I64 are still refused. Ours and TR3 are captured with the same patched image, so their capture code files are equal. `verify_marker.py` checks it on CPU inside the image.

```sh
P=<pack dir>; M=${P%/snapshots/*}
docker run --rm --network none --user 1000:1000 -e HOME=/tmp -v $M:$M:ro -v $PWD/quality:/recipe:ro \
  -e PYTHONPATH=/opt/fidelity/suite/bin:/opt/fidelity/suite/engines/tools --entrypoint python3 \
  glm53-fidelity:cf36287c-mcg1-stream /recipe/verify_marker.py $P
```

On TR3 the unpatched image fails the real-pack and `[1]` checks (`evidence/quality/verify-marker-cf36287c-unpatched.txt`), and the patched image passes all seven (`evidence/quality/verify-marker-cf36287c-mcg1.txt`, and `verify-marker-cf36287c-mcg1-stream.txt` with 0002 added). Upstream's `selftest_trellis_decode_offline.py` passes 119 checks, with 0 failures and 1 skip, on both images (`evidence/quality/selftest-trellis-decode-offline-mcg1.txt`).

Two NGC traps are handled in the image and the launcher.

- The base sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`. In torch 2.13 that only starts `float32_matmul_precision` at `high` (`aten/src/ATen/Context.h:495`). The suite pins TF32 off and `highest` itself before an EXL3 capture loads any layer (`layer_outer.py:2951`), so our captures would be safe either way. It does not pin a non-EXL3 capture, and it checks only `NVIDIA_TF32_OVERRIDE`. The image sets the variable to `0`, which leaves `allow_tf32=False` and `highest` (`evidence/quality/image-verify.txt`).
- `nvidia-cutlass-dsl` parses `sys.argv` with `parse_known_args` (`-h`, `-diagnostic`) on first import. A CPU forward of the real architecture through the suite's import path loads no cutlass module. The CUDA path is not checked yet, so `capture.sh` passes only long options and refuses `-x` and `--help` prefixes in extra arguments.

## Run order

```sh
# 1. capture on one GB10 (GPU). Inputs first, CPU only, then the capture.
#    Our pack ships zai's f9fed3bb chat_template.jinja; the root bound the a6c167b6 one, which the panel binding
#    needs under .reference/ in a tokenizer root (the panel is token ids, so the template never reaches the score).
T=~/projects/data/glm53-exl3/fidelity/ours/tokroot; mkdir -p $T/.reference
cp <our pack dir>/{tokenizer.json,tokenizer_config.json,chat_template.jinja} $T/
hf download zai-org/GLM-5.3-Flash-BF16 chat_template.jinja --revision a6c167b62691b2bac901344b65cb651a70f53e43 --local-dir $T/.reference
TOKENIZER_ROOT=$T REPO=sfxnz/GLM-5.3-Flash-EXL3-4bpw REVISION=<40-hex> AUTHOR=sfxnz quality/capture.sh ours <our pack dir>
REPO=brandonmusic/GLM-5.3-Flash-tr3-4bpw REVISION=a5fee929cf4888b1824323e33e8a19b60129e025 AUTHOR=sfxnz \
  quality/capture.sh tr3-gb10 ~/.cache/huggingface/hub/models--brandonmusic--GLM-5.3-Flash-tr3-4bpw/snapshots/a5fee929cf4888b1824323e33e8a19b60129e025
# 2. score each against the root (CPU, about 4 minutes on 8 cores)
quality/compare.sh ours; quality/compare.sh tr3-gb10
# 3. paired per-window delta, ours minus TR3 on the same stack; then the GB10-vs-H200 stack offset for TR3
F=~/projects/data/glm53-exl3/fidelity
python3 -I quality/paired_delta.py $F/ours/compare $F/tr3-gb10/compare --out evidence/quality/ours-vs-tr3-gb10.json
python3 -I quality/paired_delta.py $F/tr3-gb10/compare $F/tr3-h200/compare --out evidence/quality/tr3-gb10-vs-h200.json
```

Data layout under `~/projects/data/glm53-exl3/fidelity/`: `root-v1@bdd25fe0/` is the reference, and each `<label>/` holds `dataset/`, `inputs/`, `compare/` and the logs. The published `receipts/root-qualification.json` of both datasets is not in their `checksums.txt`, and the seal gate refuses an unlisted file, so it sits in `published-receipts/` instead.

`compare.sh` exits with the suite's code: 0 strict, 2 advisory, 3 refused. Any pair whose capture code files differ is `cross_stack` and so advisory. That includes TR3-h200 against root and every GB10 capture. It still computes the full number.

## Capture inputs (`capture_inputs.py`, run by `capture.sh`)

- **Unexpected-tensor allowlist.** The allowlist is every index key on a layer at or above `text_config.num_hidden_layers` (45). That is the MTP block, which `Glm5NextForConditionalGeneration` has no module for. It is written as a sorted compact JSON array, which is the suite's canonical form, so the file sha256 and the name sha256 are one value. On TR3 `@a5fee929` this gives 3,481 names with sha256 `37b68b12…217c`, equal to the `canonical_sorted_names_sha256` in TR3-h200's runtime record. Our pack has the same layout, so it should give the same set. The capture refuses any mismatch.
- **Scope file.** A `--role quant` capture requires one. The suite's `exl3_scope.py` writes it, with native dtypes read from the local shard headers (upstream reads them over HTTP). On TR3 its `scope_digest` equals the published TR3-h200 one.
- **Panel binding.** `fidelity-dataset panel-binding` with the pack as tokenizer root. On TR3 the binding block equals the one sealed in TR3-h200's runtime record.

## TR3 reference baseline (measured)

The published TR3-h200 dataset against root gives KLD(root || TR3-h200) = **0.022802 nats**. Top-1 agreement is 0.9546 over 51,175 positions in 25 windows. Per-window means run from 0.0051 to 0.0609 (stdev 0.0135). Windows 0 and 4 match `kld_slice.py` to within 5e-8. The receipt is `evidence/quality/tr3-h200-vs-root.receipt.json`, with the tokenwise array beside it. This is an H200 capture and cannot be paired with a GB10 capture of ours. The claim rests on `ours` against `tr3-gb10`, and TR3-h200 only sizes the stack offset.

## GB10 capture memory (measured, fixed by patch 0002)

The first TR3 GB10 capture on `cf36287c-mcg1` passed the audit and the sidecar gate. The kernel OOM killer stopped it after it had loaded layers 0 to 26 of 45, 1,222 s into the run. At the kill, python3 held anon-rss 85.5 GB, and host used memory peaked at 120.9 of 121.7 GiB (`evidence/quality/capture-tr3-gb10-mcg1-oom.{log,mem.tsv,kernel.txt}`). The resident layer stayed at 18.49 GB throughout.

The cause is copy-on-write in the shard mappings. safetensors maps each shard MAP_PRIVATE (`rw-p`). On GB10, a CUDA copy straight out of a cold page of that mapping faults the page in for write, which leaves a private anonymous copy of the file bytes. The suite opens every shard once and keeps it mapped until the run ends, so every byte it has read stays resident and cannot be evicted. The evidence:

- A load-and-free loop with no forwards (`memprobe/loadloop.py`) grows RssAnon by 3.7 GB per MoE layer. Over the same layers, glibc's in-use heap stays at 1.3 GB or less, `gc` reaches no CPU tensor, the CUDA host allocator holds 0 bytes, and `memory_allocated` returns to baseline after each free. `/proc/self/smaps` puts the growth inside the shard mappings, as Anonymous pages (`evidence/quality/stream-fix-ab.txt`).
- On one shard, a CPU read leaves its pages file-backed. A `.to("cuda")` of cold pages makes 150 MiB anonymous. `.clone()` first makes none, and `madvise(MADV_DONTNEED)` returns them all. The refaulted bytes are equal to the first copy (`memprobe/cow.py`, `evidence/quality/stream-cow-microtest.txt`).
- The run is not limited by the window count. With `--windows 2`, a capture still grew 3.86 GB per layer.

On an H200, a pageable copy is staged through the CPU, so the pages stay file-backed and evictable. That is probably why upstream's `peak_rss_bytes` matches the checkpoint size and the run still finishes (inferred, not measured on an H200).

`patches/0002-drop-private-shard-pages.patch` adds `_drop_private_pages` to `layer_outer.py` and calls it in `do_free`, the per-layer free. It calls `madvise(MADV_DONTNEED)` on this process's private mappings of the opened shards. Nothing writes to a checkpoint mapping, so no data is lost, and the next read refaults the file's own bytes. TR3's files are untouched.

Numerics are unchanged. On layers 0 to 6, all 181 layer parameters and all 14 per-layer hidden states (panel windows 0 and 1) are sha256-equal between `cf36287c-mcg1` and `cf36287c-mcg1-stream`. RssAnon after each free stays at 1.85 GB, where the old image grew to 21.6 GB (`evidence/quality/stream-fix-ab.txt`, digests in `stream-ab-digest-*.json`). Upstream's `selftest_layer_outer.py` passes the same 61 checks on both images (`evidence/quality/selftest-layer-outer-mcg1-stream.txt`). `verify-image.sh` and `verify_marker.py` give the same results as before (`image-verify-mcg1-stream.txt`, `verify-marker-cf36287c-mcg1-stream.txt`).

The full TR3 capture on `cf36287c-mcg1-stream` sealed with exit 0 on spark1 in 2,104 s. All 45 layers loaded, and the sanity probe passed with top-1 " Paris" at p 0.53. The suite's `peak_rss_bytes` is 6.67 GB and `peak_cuda_allocated_bytes` is 33.71 GB. In 5 s samples, RssAnon peaked at 5.74 GiB, container memory at 18.43 GiB and host used memory at 48.1 GiB, and MemAvailable never fell below 73.5 GiB. MoE layers load in 37.1 s on average, of which 34.8 s is trellis decode (`evidence/quality/capture-tr3-gb10-mcg1-stream.{log,mem.tsv,memory.json}`). `evidence/quality/upstream-issue-retention.md` is the draft upstream report.

```sh
# rerun the memory probe and the byte-equality check (GPU, about 13 min per image)
P=<TR3 snapshot>; M=${P%/snapshots/*}
for tag in cf36287c-mcg1 cf36287c-mcg1-stream; do mkdir -p /tmp/ab-$tag
  docker run --rm --network none --user 1000:1000 -e HOME=/tmp --gpus all -v $M:$M:ro -v $PWD/quality/memprobe:/probe:ro \
    -v /tmp/ab-$tag:/out --entrypoint python3 glm53-fidelity:$tag /probe/loadloop.py $P 0 6 --digest /out/digest.json --windows 2
done
python3 quality/memprobe/abcmp.py /tmp/ab-cf36287c-mcg1/digest.json /tmp/ab-cf36287c-mcg1-stream/digest.json
```

## TR3 on GB10 against root (measured)

KLD(root || TR3-gb10) = **0.025558 nats**, with top-1 agreement 0.9529 over 51,175 positions in 25 windows. The median is 0.00151, p99 is 0.373 and the max is 8.27. `compare.sh` exits 2, which is advisory, from `cross_stack_capture` and `weights_reconstructed` (`evidence/quality/tr3-gb10-vs-root.{log,receipt.json,tokenwise-kld.npy}`).

Against TR3-h200 (0.022802), the GB10 stack reads 0.002756 nats higher, or 12% (`evidence/quality/tr3-gb10-vs-h200.json`). The per-window delta has a mean of 0.00276 and a stdev of 0.00924, so its standard error over 25 windows is 0.00185, about 1.5 standard errors. GB10 is lower on 12 of 25 windows, and the largest window delta is 0.0423. The two captures differ in device, kernels and capture code: H200 ran `layer_outer.py` b832c8d0, and GB10 ran cf36287c plus 0001 and 0002. That offset is why the claim pairs ours against tr3-gb10 on the same stack and never against TR3-h200.

## Paired delta

`fidelity-stats paired-delta` refuses compare receipts. It reads stream_score `kld-report.json` (`per_window[].summary.mean` plus `teacher_receipt_sha256`), and a compare receipt carries `per_context[]` (`evidence/quality/fidelity-stats-paired-delta-on-compare-receipt.txt`). `paired_delta.py` pairs two compare receipts by window. It refuses unless the reference `dataset_sha256`, the panel token hash, the replay backend, the vocab chunk and every window's id and row count match. It reports `same_capture_stack` (equal candidate lane identity and stack fingerprint), which must be true for ours against tr3-gb10 and false for tr3-gb10 against tr3-h200. `test_paired_delta.py WORKDIR` covers one pairing and four refusals.

## Spot checks without the suite

`kld_slice.py` is a numpy-only, CPU-only, independent implementation of the same estimator on a few windows. It also scores the top-K view that TensorFold `prompt_logprobs` would return.

```sh
python3 -I quality/kld_slice.py ROOT_DIR CAND_DIR --windows 0 4 --out evidence/quality/x.json
```

The results for TR3 vs root on windows 0 and 4 are in `evidence/quality/tr3-vs-root.window-{0,4}.json`.
