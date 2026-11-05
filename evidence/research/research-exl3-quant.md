# EXL3 4bpw quant of GLM-5.3-Flash for TensorFold: research

Goal: publish our own EXL3 quant (sfxnz/) of GLM-5.3-Flash with routed experts at 4 bpw on the mcg codebook and every other tensor BF16. It must load in TensorFold v1.2.2's CUDA GLM reader on 2x GB10, use no ShapleyMCG code, and match or beat TR3 (KLD 0.02455).

Pins used for every citation:
- exllamav3 v1.6.6 = `bd1f5191`, cloned at `scratchpad/src/exllamav3`.
- TensorFold v1.2.2 = `e45cca70`, read with `git show v1.2.2:` from `scratchpad/src/tf-upstream`.
- TR3 = HF `brandonmusic/GLM-5.3-Flash-tr3-4bpw@a5fee929`.
- BF16 source = `zai-org/GLM-5.3-Flash-BF16@f9fed3bb`. FP8 source = `zai-org/GLM-5.3-Flash@46a050be`.

## Bottom line

A stock exllamav3 *output directory* does not load in TF. Stock exllamav3 can still *encode the experts* in exactly the layout TF reads. So the recipe has two steps:

1. Run convert.py with a recipe that puts the experts at 4 bpw and everything else at 16.
2. Run our own small assembler. It copies every non-expert tensor byte-for-byte from the BF16 source and swaps in the encoded experts.

TR3 did the same thing: its config says `non_routed_dtype_policy: official_source_native`. turboderp's stock exllamav3 4.05bpw (every tensor quantized, from the FP8 source) already scores the same KLD as TR3 on the same lane: 0.025526 vs 0.025503 (registry `measurements.jsonl`). An experts-only quant with BF16 elsewhere should therefore match or beat TR3 (inferred).

## 1. exllamav3 v1.6.6 and glm5_next

- **Architecture.** `exllamav3/architecture/glm5_next.py` registers text, mtp and vision components (29-42).
  - KDA is a GatedDeltaNet with **fused** `qkv_proj` (180-181).
  - DSA uses MLAttention (201-224).
  - Dense layers are GatedMLP. MoE is BlockSparseMLP with per-expert keys `experts.{i}.{gate,up,down}_proj` (278-280) plus a GatedMLP shared expert (292-306).
  - `lm_head` takes `head_bits` (334-343). `calibration_all_experts = True` (348).
- **MTP.** `glm5_next_mtp.py` sets the MTP layer as `model.language_model.layers.{num_hidden_layers}` (37-47). Every MTP linear takes the single `mtp_bits` value (52, 89, 123, 137). The recipe does not cover MTP (`allocation.py:231`).
- **Vision.** With `-vb 16` no vision model is built, so the tower is not quantized (`convert_model.py:426-441`). Its tensors are copied as extras with `allow_bf16=True` (`compile.py:126`).
- **Experts-only at 4.0 + mcg.** This works with a recipe YAML (`-rcp`, `convert_model.py:82, 243-269`). Patterns resolve top to bottom and the last match wins (`allocation.py:228-255`). 16 means unquantized (`convert_model.py:703-708`). `-cb mcg` (100, 572-573) writes the `.mcg` marker `0xCBAC1FED` (`quantize.py:24, 36-46`).
- **Output layout.** The EXL3 tensors are `trellis`, `suh` and `svh` as fp16, plus the marker (`modules/quant/exl3.py:103-116`). `config.json` gets `{"quant_method":"exl3","version":"1.6.6","bits":round(final_bpw,2),"head_bits",...,"codebook","calibration":{rows,cols},"out_scales"}` (`compile.py:196-242`, `convert_model.py:1445`). An FP8 source config also adds `original_quantization_config` (`compile.py:230-233`).

## 2. TR3 vs stock output vs TF's reader

**TR3 = BF16 source with only the expert `.weight` tensors replaced.**
- 150,226 tensors with source names kept, including separate `q/k/v_proj` and `q/k/v_conv1d`. Non-experts are BF16 or F32 as in the source, including the 347 vision tensors and the MTP layer 45. Read from the shard 1, 103 and 120 headers.
- Experts are `trellis` I16 `[256,128,64]` / `[128,256,64]`, `suh`/`svh` F16, and `mcg` I32 `[1]` (`quantization_config.json`, `exl3-mcg-storage-abi.json`).
- The storage ABI was written with exllamav3 0.0.43 (`c5d9c657`). The source was BF16 `a6c167b6`. Its 120 weight-shard LFS oids are identical to `f9fed3bb` (measured), so the weights are the same.
- **How it was made.** Uniform K4 over routed experts only (`quantization/recipe.json`). A custom "R10/TR3" encoder is derived from exllamav3 0.0.43 (`reproducibility/r10/README.md:53-62`). It uses 640 rolling calibration windows of the BF16-teacher corpus, with the 25 final windows held out (`docs/HISTORICAL_MODEL_CARD_2026-08-27.md:23-40`).
- **One TR3-specific trait.** All 288 experts in a layer share one gate/up `suh` and one down `svh` (measured by range-reading 8 vectors from shard 10).

**TF v1.2.2 admission and load rules (`zig/src/families/glm/cuda_exl3.zig`):**
- `bits` must be the JSON **integer** 4 and `codebook` must be `"mcg"` (27-44). exllamav3 writes `4.0`, so TF refuses stock output (inferred from the JSON typing).
- Per expert it requires `trellis` i16 rank-3 `[K/16, N/16, 64]`, an i32 `.mcg` with the magic value (numel 1, so exllamav3's rank-0 marker passes), and f16 `suh` [K] / `svh` [N] (64-122). It covers the MTP layer's experts too (`cuda_weights.zig:437-441`).
- Non-expert projections are read as **bf16** with source names: separate q/k/v (`cuda_weights.zig:205-268`), bf16 dense MLP (336-340), bf16 embed and head (545-551). Conversion accepts only bf16 or f32 (149-170).

**What stock exllamav3 output gets wrong for TF.** Seen in turboderp's 4.05bpw shard headers (`2a30229e`) and consistent with the v1.6.6 code:
- fused `self_attn.qkv_proj` and `conv1d [24576,1,4]`;
- unquantized linears stored as **F16** (`b_proj`, `f_a_proj`, `kv_b_proj`, `mlp.gate.weight`, indexer weights);
- `hc_*_fn` stored as F32;
- `mul1` codebook by default.

Hence the assembler. A second gap: exllamav3 folds per-tensor input-channel scales and a refit into `suh` (`quantize.py:1113-1116, 1517-1521, 1880-1890`). Our `suh` therefore differs per expert, and TF sets `shx=false`. That path exists and is tested both ways (`zig/tests/cuda/glm_exl3_tests.zig:69-80`), but prefill rotates once per pick instead of once per row. The speed cost is unmeasured.

**License.**
- exllamav3 is MIT. Both zai-org repos are MIT.
- The TR3 LICENSE file is actually titled "Local Inference Lab Attribution License 1.0" (LICENSE:1-8), although the card calls it ShapleyMCG 1.0. It reaches every "Derivative Work… quantizations, re-encodings, format conversions" of *the Work* (LICENSE:43-45).
- It binds us only if we copy TR3 bytes or code: its weights, its `eval/kld` code, or its `r7_encoder`. Take every non-expert tensor from zai-org, not from TR3. Reading TR3's public index to compare names is a check, not a copy.

## 3. Quality levers at a fixed 4.0 bpw on experts

TF accepts exactly 4.0 on mcg (`k2=8`, `cuda_exl3.zig:116-120`). So mul1, 3inst, mixed rates and `-hq` are out. Every lever below changes only values and keeps the layout.

- **Calibration coverage.** `calibration_all_experts` runs every calibration token through every expert during capture (`convert_model.py:1045-1048`). Even rarely routed experts see all 512k tokens, but against one shared gate/up Hessian per layer. With `-mae 0.01`, each expert's Hessian is weighted by its routing (`doc/convert.md:32-34`, `-mar` default 262144 rows).
- **Calibration data.** `-cr/-cc` (default 250 x 2048). `-cd` takes packed rows from `sc_trace.py` (self-sampled in-domain chat and reasoning traces; `--api_model` can sample from a local server). This is how turboderp made his `sc_*` branches. The default mix has no chat-templated data (`calibration_data.py:73-81`).
- **Encoder.** It is already tail-biting Viterbi (`quantize_tiles_kernel.cuh:3-6`). The scale refit polish is on by default (`quantize.py:1880-1890`). `-sh` (scaled Hessian, experimental, flag at `convert_model.py:105`). `--out_scales auto|always`.
- **YAQA `-hess`.** Not usable here. Its tool skips fused MoE expert tensors (`util/yaqa_hessians.py`, header) and needs a BF16 Transformers backward pass.

## 4. Feasibility on one GB10

- **Memory (inferred from code).**
  - The calibration state is mHC 4-stream fp32 (`hyperconnections.py`, `ExpandStreams`): 250 x 2048 x 4 x 4096 x 4 B = 33.5 GB.
  - One MoE layer in fp16 is about 14.5 GB. Down Hessians take 288 x 16 MB = 4.8 GB. `-mae` adds about 19 GB of gate/up Hessians.
  - Estimated peak is 60-90 GB of 128 GB. `--no_swap` is automatic on integrated GPUs (`convert_model.py:1367-1368`). Raising `-cr` grows the state linearly.
- **Wall time (inferred).**
  - The Viterbi work per step is `32768>>K` units x `2^K` candidates, so it does not depend on K (`quantize_tiles_kernel.cuh:40-41`).
  - Measured locally: 1.8-2.2 s per 5120x2304 DSV4.1 expert on GB10 (`dsv41-requant-viterbi/units/00004.json`).
  - Scaling gives about 1.4 s per 4096x2048 GLM expert. 37,152 expert tensors then take about 15 h of encoding.
  - With all-experts forwards, LDLQ and checkpoints, expect about 20-30 h. No glm5_next timing is published anywhere I found.
- **Disk.**
  - BF16 source: 642.6 GB (index `total_size`).
  - Work dir: about 175 GB of qtensors, plus up to 2 x 33.5 GB of `ckpt/state` (`convert_model.py:1832-1839`).
  - Stock out_dir: about 176 GB, deletable after the run.
  - Assembled output: about 176 GB.
  - Total is about 1.06-1.24 TB. spark1 has 760 GB free and spark2 has 844 GB (measured `df`). spark1 also holds 574 GB of NVFP4 caches.
  - spark2 has a partial BF16 download at `f9fed3bb` (51 GB, 8 `.incomplete` blobs). Someone may be downloading now, so it was left untouched.
- **FP8 as source.**
  - The BF16 repo is **not** an upcast of FP8. 128x128 blocks hold 405-570 distinct BF16 values, more than the 254 an FP8 block allows. Only 89-96% of elements equal FP8 x scale, with a relative RMS difference of 0.16-0.18% (measured on 4 tensors by range reads).
  - The cards are identical apart from the title (diff), so no provenance statement exists.
  - The official FP8 model scores 0.0206 against a 0.0127 floor on the crosscheck lane (registry), so FP8 is not lossless.
  - exllamav3 reads FP8 block-scaled tensors packed and dequantizes them exactly (`convert_model.py:413-415`). turboderp's FP8-sourced quant still matched TR3.
  - Recommendation: BF16 for the experts. FP8 (328 GB) plus about 20 GB of BF16 non-expert tensors is the disk fallback.
- **Two machines.** Stock convert runs layer by layer through one state and has only single-host `-d` multi-GPU (`doc/convert.md:68-72`). The local precedent split work across nodes only because it was Hessian-free. The best use of two Sparks is two variants in parallel, for example baseline vs `-mae`.

## 5. Measuring quality without running BF16

- `brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits` now returns "Repository not found", even authenticated.
- `malaiwah/glm53-flash-fidelity-root-v1@bdd25fe0` (MIT, 1.57 GiB) holds everything needed for the same 25 x 2048 panel (51,175 positions): `panel/tokens/*.json`, masks, BF16 final hidden states (post-norm, pre-head) and the BF16 `head/weight.safetensors`. KLD is head(hidden) on both sides.
- TR3's own capture in the same form is `malaiwah/glm53-flash-fidelity-exl3-tr3-4bpw-miaailab-v1`.
- On the streaming lane: floor 0.01151, TR3 0.02550, turboderp 4.05 0.02553, K6 0.01371.
- For a paired comparison, run our quant and TR3 through the same stack (TF on 2 Sparks) and compute KLD against the root. Report the result relative to the floor.

## Recommended recipe

```yaml
# experts4.yaml
target_bpw: 4.0
tensors:
  "*": 16
  "model.language_model.layers.*.mlp.experts.*": 4
```

```sh
python convert.py -i $BF16 -w $WORK -o $WORK/stock_out -rcp experts4.yaml -b 4.0 \
  -cb mcg -hb 16 -mb 4 -vb 16 -eb 16 -cr 250 -cc 2048 -d 0     # A
# B (other Spark): same plus  -mae 0.01   (later: -cd trace.safetensors from sc_trace.py)
```

`-mb 4` quantizes all MTP linears, but the assembler keeps only the MTP experts.

The assembler is our own MIT code. For each source shard:
- Copy every tensor verbatim except the routed-expert `.weight` tensors in layers 3-45.
- Insert `{key}.trellis/.suh/.svh/.mcg` from `$WORK/qtensors/*.safetensors` (`convert_model.py:1726`).
- Write the source `config.json` with `quantization_config={"quant_method":"exl3","bits":4,"codebook":"mcg","head_bits":16,"version":"1.6.6"}` and a new index.

Gates, in order:
1. Every TF `exl3.load` check passes offline (dtype, shape, marker).
2. The expected count holds: 43 layers x 288 x 3 x 4 expert tensors.
3. Non-expert bytes are sha-equal to the source.
4. TF loads on 2 Sparks.
5. Panel KLD comes in at or below TR3's on the same stack.

## Unknowns

- Real conversion peak memory and s/tensor for glm5_next on GB10. Time one or two MoE layers with `--max_module` first.
- Whether exllamav3 v1.6.6 builds and runs on sm_121 aarch64 (the 1.5.1 precedent did).
- TF's prefill and decode cost with `shx=false`, and whether any other TF path assumes the shared rotation.
- Whether TF can emit final hidden states or logits on the panel (needed for §5).
- How much `-mae`, self-calibration data, `-sh` or `out_scales` change KLD on this model.
- Whether the BF16 teacher-logits dataset is gone for good or only private.
- Whether the TR3 card's "ShapleyMCG 1.0" and its LICENSE file's "LIL Attribution 1.0" are the same instrument.
