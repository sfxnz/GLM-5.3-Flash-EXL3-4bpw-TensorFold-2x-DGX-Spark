# Paired panel KLD for our EXL3 pack vs TR3

**Goal.** Find a reproducible way to score mean full-vocab KLD(BF16 || candidate) for our pack and for TR3 on the same 25 x 2047 panel. That number lets us claim "no worse than TR3" when we publish.

Sources are pinned as follows. Suite: `malaiwah/quant-fidelity-suite@cf36287c`. The repo was renamed from `glm53-flash-fidelity-suite`, and its LICENSE is MIT, "NO third-party source code". Datasets: root `@bdd25fe0`, TR3 `@3156c323`. TensorFold: tag `v1.2.2` = `00e7dddb`.

## Premise correction (measured)

- The registry numbers (floor 0.01151, TR3 0.02550, turbo 0.02553) are not scored against the root dataset. Their `reference_ref` is `reference--brandonmusic.glm53-bf16-fp32-logits.final25`, the deleted teacher logits, and their pipeline is `glm53-stream-packed-kld` (`stream_score.py --source tr3`), per `data/measurements.jsonl`. We cannot reproduce those numbers for a new quant.
- The root-dataset pipeline is a separate group, `pipeline--malaiwah.fidelity-dataset-hf.h200-layer-outer`. Its floor is a self-compare at 0.0, and its only quant row is k3 at 0.0506. The registry note says "do not read it beside the 13 older Flash rows". No TR3-vs-root row exists yet.
- The TR3 hidden dataset captures `Mia-AiLab/...@024db9f7`. All 120 of its shards and its `config.json` are sha-equal to `brandonmusic/GLM-5.3-Flash-tr3-4bpw@a5fee929`. The root captures BF16 `@a6c167b6`, and all 120 of those shards are sha-equal to `@f9fed3bb`.

## 1. How the stream lane makes candidate hidden states

- **Program.** `engines/tools/hf_capture.py` (MIT), with `--schedule layer-outer --layer-residency stream` from `layer_outer.py`. It runs a plain transformers 5.16.1 eager forward on torch 2.11.0+cu130. Per layer, it builds the model on meta, materializes one decoder layer, pushes all 25 windows through it sequentially, then frees it. The lm_head input is tapped with a forward pre-hook (`hf_capture.py:1-50`, `layer_outer.py:1-60`, CLI at `hf_capture.py:2830-2905`).
- **Not exllamav3.** EXL3 experts are decoded to BF16 in pure torch by `exl3hf_surface.py::decode_payload_hf`, which transcribes the mcg LUT from exllamav3 (`exl3hf_surface.py:1-55,194`). The forward then runs dense BF16.
- **Hardware.** TR3's own capture ran on an H200 (`tr3/runtime/capture-runtime.json`). It took 186 s, read 171.6 GB, and peaked at 33.7 GB CUDA allocated, 36.6 GB reserved and 18.5 GB resident weights. The 175 GB peak RSS is evictable safetensors mmap page cache (`layer_outer.py:4071-4076`).
- **One GB10.** It should fit (inferred, not run): about 37 GB of device need against 128 GB unified, with only one layer resident. Unknowns: no aarch64 torch is pinned, because `bin/requirements-cu130-py312.lock` is x86_64 only. Nobody has run this on sm_121. My guess is decode and forward take tens of minutes on GB10 rather than 3.
- **Scorer.** `bin/fidelity-dataset compare` (`bin/fidelity/dscompare.py`) computes `hidden_fp32 @ head_fp32.T` in numpy (`_replay`, 1022-1047), then fp64 log_softmax and KL over the full 154,880 vocab, as a token mean (`token_kld`, 957-1019). Its gates refuse a pair whose form, panel/token hash, head digest, geometry or coverage differ (234-500). A differing lane identity, stack fingerprint or source files mark the pair `cross_stack`/advisory but still compute it (422-476).

## 2. TensorFold 1.2.2 at TP=2

- Nonstreamed `/v1/completions` takes vLLM-style `prompt_logprobs` from 0 to 20 on the GLM-5.3-Flash CUDA engine. Each prompt row returns the given token's logprob and rank plus the top-k. Such a request never resumes from the prompt cache (`docs/api.md:208-212`, `zig/src/server/fields.zig:314-318`, `lanes/logprob.zig:4`). The max k is 20.
- It works at TP: each rank runs its head columns and the rows are gathered to rank 0 (`families/glm/cuda_prompt_logprobs.zig:1,67-84`). The gathered logits are bf16 (`:33-35`, `vocab * 2` bytes).
- There is no full-vocab output and no GLM hidden-state dump. The only logit dump, `TF_FLASHNEXT_LOGITS`, belongs to the flashnext family and writes the last row only (`families/flashnext/cuda_rows.zig:208-209`). No `TF_GLM_*` variable dumps anything.
- The suite has no top-k mode. Its receipt says "No truncation, top-k or clamp" (`dscompare.py:1783`).
- **Measured top-k quality.** I scored TR3 vs root with `quality/kld_slice.py`, simulating what TF would return (bf16 logits, top-20 plus the given token, merged into a {listed, rest} bucket). That estimate is a strict lower bound on the KL. Results on windows 0 and 4:

| window | full fp64 KLD | cand bf16 logits | top-20 bound | bound / full | per-pos corr |
|---|---|---|---|---|---|
| 0 | 0.03363 | 0.03391 | 0.02629 | 0.78 | 0.957 |
| 4 | 0.06093 | 0.06113 | 0.05172 | 0.85 | 0.977 |

  Top-20 loses 15-22% of the KL, and that fraction varies by window. That variation is larger than the ours-vs-TR3 margin we need to resolve, so I reject TF as the scoring path. TF is still useful as a separate smoke check of the served stack.
- Two checks on the estimator. It matches the suite's `token_kld` exactly (max abs diff 0.0 on 64 positions), and a self-compare gives 0.0.

## 3. Recommendation: suite capture on one Spark, TR3 recaptured as the control

No new scoring code is needed. Run on one Spark when it is free, in the suite's MIT checkout. The full argv template is in suite `bin/README.md:421-432`.

```sh
# panel binding (once)
bin/fidelity-dataset panel-binding --panel engines/panels/panel--glm53.brandonmusic.final25 \
  --tokenizer-root $BF16 --out panel.binding.json
# capture, twice: X = ours ($OURS), tr3 ($TR3 @a5fee929)
bin/fidelity-dataset capture --engine hf-transformers --out ds-$X --form hidden --role quant --lane streaming -- \
  --model $X_DIR --model-revision <sha> --weights-repository <repo> \
  --panel engines/panels/panel--glm53.brandonmusic.final25 --panel-binding panel.binding.json --panel-binding-sha256 <sha> \
  --schedule layer-outer --layer-residency stream --device cuda --dtype bfloat16 --sanity-expect Paris \
  --unexpected-tensors-allowlist <derive_unexpected_allowlist.py output> \
  --dataset-id fidelity--glm53flash.<us>.quant.$X --dataset-name ... --run-name $X-gb10-1 --cold-run $X-gb10-1 --role quant --memory-report mem-$X.json
# score, CPU
for C in ds-ours ds-tr3 hf://malaiwah/glm53-flash-fidelity-exl3-tr3-4bpw-miaailab-v1@3156c323; do
  bin/fidelity-dataset compare --reference hf://malaiwah/glm53-flash-fidelity-root-v1@bdd25fe0 --candidate $C --out cmp-$(basename $C)
done
```

Report KLD(root || ours-gb10) beside KLD(root || tr3-gb10) as a paired per-window delta from the tokenwise arrays, with the same-lane floor at 0.0. Then report TR3-gb10 vs TR3-h200 as the size of the GB10 stack offset. Run the TR3-h200 compare now on CPU, because it already gives the TR3 baseline. It needs no GPU and about 0.8 GB more pulled (the other 23 hidden windows of each dataset). Measured cost here is 13 min per window on one core, so 25 windows run as parallel processes should finish in well under an hour (projection).

**Open items.**
- Pin an aarch64 torch 2.11/cu130 build.
- Derive our unexpected-tensor allowlist (TR3's covered 3,481 MTP and other keys).
- Confirm that `fidelity-stats paired-delta` reads compare outputs. If it does not, the paired per-window delta is about 20 lines over the tokenwise files.

**WAN pulled:** 1.34 GB in total. That is the head (1.27 GB) plus hidden windows 0 and 4 from each dataset (67 MB), all sha-checked against `checksums.txt`, plus about 4 MB of metadata, suite sources and registry rows.
