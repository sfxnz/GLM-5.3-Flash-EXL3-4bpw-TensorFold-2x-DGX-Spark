---
license: mit
base_model: zai-org/GLM-5.3-Flash-BF16
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- glm
- moe
- exl3
- 4-bit
- quantized
- vision
- mtp
- tensorfold
- dgx-spark
---

# GLM-5.3-Flash EXL3 4bpw

[zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) at revision `f9fed3bbc82291073373d67452e9f25317ef7577`, with the routed experts encoded in EXL3 at 4 bits on the `mcg` codebook and every other tensor left as Z.ai's own BF16 and F32 bytes. It is built for the native CUDA GLM family of [TensorFold](https://github.com/ashhart/TensorFold) 1.2.2, which reads GLM EXL3 in this layout only. It was made with stock exllamav3 v1.6.6 and a small assembler of our own, from MIT sources only, so anyone can rebuild it.

Every number on this card comes from a file under `evidence/` in [the recipe repo](https://github.com/sfxnz/GLM-5.3-Flash-EXL3-4bpw-TensorFold-2x-DGX-Spark). Paths below are relative to it.

## What is quantized

| Part | Format | Source |
|---|---|---|
| Routed experts in layers 3 to 44, 288 a layer, each with `gate_proj`, `up_proj` and `down_proj` | EXL3 4.00 bpw on the `mcg` codebook, stored as `trellis` I16, `suh` and `svh` F16, and one I32 `mcg` marker | exllamav3 v1.6.6 |
| Routed experts in the MTP layer, 45 | The same | exllamav3 v1.6.6 |
| Everything else, which is the embedding, the LM head, attention, dense layers 0 to 2, shared experts, routers, norms, the MTP layer's other tensors and the vision tower | BF16 and F32, unchanged | byte copy of `zai-org/GLM-5.3-Flash-BF16@f9fed3bb` |

That is 43 layers × 288 experts × 3 projections, 37,152 EXL3 projections (`evidence/quant/full/assemble.txt`). The 1,618 non-expert tensors are byte-equal to the source (`evidence/quant/full/verify.txt`). The expert `.weight` tensors are left out of the pack, because TensorFold never reads them on the EXL3 path.

`config.json` is the source's, with one top-level block added:

```json
"quantization_config": {"quant_method": "exl3", "bits": 4, "codebook": "mcg", "head_bits": 16}
```

`bits` is the JSON integer 4. TensorFold 1.2.2 admits the pack only when `bits` is a JSON integer equal to 4 (`if (bits != .integer or bits.integer != 4)` in `zig/src/families/glm/cuda_exl3.zig`, in `evidence/research/tf-source-lines.txt`), so it refuses the `4.0` that stock exllamav3 writes. That is read from the source, not run. `chat_template.jinja`, the tokenizer files, `processor_config.json`, `generation_config.json` and `LICENSE` are byte copies of the source's.

TensorFold can also pack the BF16 dense linears and the head to 8 bits when it loads them (`TF_GLM_DENSE_BITS=8`, `TF_GLM_HEAD_BITS=8`). That happens in the engine, not in this pack. The recipe turns it on by default, because it measured within noise of BF16 serving and faster. See [Serving](#how-to-serve).

## Sizes

| Part | Size |
|---|---|
| BF16 source | 598.5 GiB (`evidence/decisions.tsv`, row `disk` 12:47:53) |
| Non-expert tensors, copied unchanged | 19,339,724,144 bytes (`evidence/decisions.tsv`, row `quant` 15:44:44) |
| This pack, 120 shards | 175,642,102,880 bytes of shards (`evidence/run/stage-env.txt`) |
| Weights on each rank at TP=2, BF16 dense | 86.41 GiB (`evidence/run/first-boot.txt`) |
| Weights on each rank at TP=2, dense and head packed to 8 bits at load | 82.87 GiB (`evidence/run/boot-final.txt`) |

The pack keeps the source's shard layout, one output shard for each of the 120 source shards.

## How to serve

Use [the recipe](https://github.com/sfxnz/GLM-5.3-Flash-EXL3-4bpw-TensorFold-2x-DGX-Spark). On two DGX Sparks joined by their direct cable, `./run.sh` builds the TensorFold 1.2.2 image, downloads this pack at a pinned revision, copies both to the second Spark, and serves an OpenAI-style API on the first. `./stop.sh` stops it.

Its defaults: 8 requests at once, one request up to 1,048,576 tokens, one shared BF16 KV pool of 2,056 pages (about 1.05M tokens) on each rank, vision, thinking and MTP drafts on, dense linears and head packed to 8 bits at load. Measured with those defaults, 5 runs a cell, 512 tokens, thinking off (`evidence/knobs/d8h8-p8-1m-r105/bench.json`):

| Streams | Aggregate tok/s, median (range) |
|---|---|
| 1 | 33.8 (32.4 to 36.2) |
| 4 | 45.5 (44.6 to 48.8) |
| 8 | 58.8 (51.8 to 61.0) |

A needle at 37% depth was found in a 1,045,624-token prompt (`evidence/knobs/d8h8-p8-1m-r105/needle.jsonl`).

Only TensorFold 1.2.2's native CUDA GLM family has loaded this layout here. Other loaders are untested. Stock exllamav3 output differs from this pack, so do not assume that a loader for stock EXL3 reads it.

## Quality

Two measures, and they disagree about TR3, the one other pack in this layout. Read both.

**Panel KLD.** Full-vocabulary KLD(BF16 || candidate) on a panel of 25 windows of 2,048 tokens, 51,175 positions. The BF16 side is `malaiwah/glm53-flash-fidelity-root-v1@bdd25fe0`. Both 4-bit packs were captured with `malaiwah/quant-fidelity-suite@cf36287c` plus two patches, on the same GB10, and scored on CPU in float64 (`quality/README.md` in the recipe).

| Pack | KLD, nats | Top-1 agreement | Evidence |
|---|---|---|---|
| This pack, GB10 capture | 0.029440 | 0.9488 | `evidence/quality/ours-gb10-vs-root.receipt.json` |
| TR3 `brandonmusic/GLM-5.3-Flash-tr3-4bpw@a5fee929`, GB10 capture | 0.025558 | 0.9529 | `evidence/quality/tr3-gb10-vs-root.receipt.json` |

Paired window by window, this pack is 0.00388 nats higher than TR3 (stdev 0.0081 over 25 windows, about 2.4 standard errors), and lower in 4 of the 25 windows (`evidence/quality/ours-vs-tr3-gb10.json`). On this panel TR3 is better.

The panel is the 25 held-out final windows of the corpus TR3 was calibrated on. TR3's encoder used 640 rolling windows of that corpus (`evidence/research/research-exl3-quant.md`, section 2). This pack used exllamav3's default calibration mix. So the panel is TR3's own domain.

**Engine NLL on an independent corpus.** `tools/nll.py` scores 41 chunks of `nll-corpus-v1` (178,793 tokens of prose, multilingual text, code, math and one long document) through TensorFold with `prompt_logprobs`, and pairs two packs token by token. Both packs served with the same image and flags, BF16 dense (`evidence/knobs/tr3-bf16-p4/nll.txt`):

| Group | TR3 minus this pack, nats/token | 95% CI |
|---|---|---|
| All | -0.00030 | -0.00343 to +0.00363 |
| Prose | +0.00057 | -0.00626 to +0.00615 |
| Multilingual | -0.00312 | -0.01121 to +0.00508 |
| Code | +0.00551 | -0.00016 to +0.01198 |
| Math | -0.00067 | -0.00385 to +0.00434 |

On independent text we detect no difference between the two packs, though the CI is wider than the recipe's 0.002 gate.

Do not compare these KLDs with numbers from other stacks. The same TR3 bytes read 0.022802 from an H200 capture and 0.025558 from our GB10 capture (`evidence/quality/tr3-gb10-vs-h200.json`). TR3's own card reports 0.0246 against a BF16 teacher-logits repo that no longer exists (`evidence/decisions.tsv`, row `quality` 14:38:01).

## How it was made

The encode chose `--moe_affinity_eps 0.01`, which weights each expert's Hessian by how often the router picks it. On MoE layer 3, with the same calibration, it cut the layer's output error from rfn 0.02227 to 0.01330 and raised its SQNR from 50.78 dB to 53.31 dB, for 7.6% more time (without the flag `evidence/quant/probe-a-layers-3-4.txt`, with it `evidence/quant/probe/probe-b-log.txt`). Calibration is exllamav3's default set, 250 rows of 2,048 tokens. On one GB10, layers 0 to 3 took about 27 minutes as a probe, and layers 4 to 45 then took 14.1 hours of wall time, from 15:55:54Z to 06:04:21Z (`evidence/quant/full-encode-b.times.txt`; per-layer times in `full-encode-b.log`).

To reproduce, from the recipe repo:

```sh
# 1. The quantizer image: exllamav3 v1.6.6 (bd1f5191) for sm_121a on nvcr.io/nvidia/pytorch:26.07-py3
git clone https://github.com/turboderp-org/exllamav3 /path/to/exllamav3
docker build -f quant/Dockerfile --build-context exllamav3=/path/to/exllamav3 -t glm53-quant:exl3-1.6.6 quant/

# 2. The BF16 source
hf download zai-org/GLM-5.3-Flash-BF16 --revision f9fed3bbc82291073373d67452e9f25317ef7577

# 3. Encode the experts (routed and MTP) at 4 bpw, mcg; everything else stays 16
quant/convert.sh b --moe_affinity_eps 0.01
# Optional, when the encoding node is short of disk: run on the other node to move finished layers off it
quant/move_qtensors.sh b

# 4. Build the pack from the source and qtensors/, then check it
python3 -I quant/assemble.py extract SOURCE_SNAPSHOT EXTRACT
python3 -I quant/assemble.py assemble EXTRACT QTENSORS OUT
python3 -I quant/assemble.py verify OUT EXTRACT QTENSORS
python3 -I quant/check_layout.py OUT hf:zai-org/GLM-5.3-Flash-BF16@f9fed3bbc82291073373d67452e9f25317ef7577
```

- `convert.sh` passes `--recipe quant/experts4.yaml --bits 4.0 --codebook mcg --head_bits 16 --mtp_bits 4 --vision_bits 16 --embed_bits 16 --cal_rows 250 --cal_cols 2048`. It unsets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`, which the NGC base sets. It passes long option names only, because the base's `nvidia-cutlass-dsl` reads `sys.argv` and took `-hb` for `-h` (`evidence/quant/probe/run1-cutlass-argparse-exit.log`).
- exllamav3 always ends by compiling its own stock pack. TensorFold cannot read that pack, so `convert.sh` points it at a 1 GiB tmpfs, and the compile stops there after every file in `qtensors/` is written. `assemble.py` builds the real pack from `qtensors/`.
- The MTP layer reports rfn, cos and SQNR of 0. exllamav3 prints zeros for side models it quantizes without calibration (`convert_model.py:1913`). They are not measurements.
- Our encode ran layers 0 to 3 as a probe and continued with `--resume`. A run from scratch passes the same arguments. We did not test whether it gives byte-identical `qtensors/`.
- `assemble.py verify` found all 1,618 non-expert tensors sha256-equal to the source and all 148,608 expert parts byte-equal to `qtensors/` (`evidence/quant/full/verify.txt`).
- `check_layout.py` asserts TensorFold 1.2.2's layout rules offline, each tied to the TensorFold source line that enforces it (`quant/README-contract.md`). On this pack it reports `RESULT PASS` (`evidence/quant/check-layout-ours.txt`).

## Limits

- exllamav3 folds input-channel scales into each expert's `suh`, so this pack's gate and up `suh` vectors differ per expert (576 distinct a layer), and TensorFold sets `shx=false`. TR3 shares one per layer. On the same engine and flags we measured no cost from it. This pack prefilled the NLL corpus in 188 to 202 s over 8 boots, and TR3 in 219 s on one boot (`evidence/nll/capture-walls.txt`).
- The `mcg` markers are I32 of shape `[]`, as exllamav3 writes them. TR3's are shape `[1]`. TensorFold accepts both (`quant/README-contract.md`).
- The KV cache is BF16. Stock TensorFold 1.2.2 refuses FP8 KV on its paged pool.

## License and credits

MIT, the same as the base model. The pack holds Z.ai's tensors and `LICENSE` unchanged, plus expert tensors that exllamav3 encoded from them.

- GLM-5.3-Flash is by [Z.ai](https://huggingface.co/zai-org), MIT.
- The encoder, [exllamav3](https://github.com/turboderp-org/exllamav3), is by turboderp, MIT.
- The engine this layout targets, [TensorFold](https://github.com/ashhart/TensorFold), is by ash hart and the TensorFold contributors, Apache-2.0.
- The KLD method uses [quant-fidelity-suite](https://github.com/malaiwah/quant-fidelity-suite) and the BF16 fidelity root dataset, both by Michel Belleau (malaiwah), MIT.
- TR3, `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, served only as the measured control. This pack holds none of its bytes and none of its code. The assembler copies non-expert tensors from the zai-org source, not from TR3.
