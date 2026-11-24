# Ceilings: GLM-5.3-Flash EXL3 4bpw, TP=2 on two GB10

## Bytes (from the pack's safetensors headers, whole model = both ranks)

| Component | GiB as stored | GiB as served | Read per forward row |
|---|---:|---:|---|
| Routed experts, 42 MoE layers x 288, EXL3 4 bit | 142.165 | 142.165 | 8/288 a row: 3.949 GiB |
| Dense: attention/KDA 11.282 + shared experts, dense MLPs, router 2.905 (BF16) | 14.187 | 7.315 at 8 bit (+fp16 scale per 64) | all, once a round |
| LM head (BF16) | 1.182 | 0.610 at 8 bit | all, once a row group |
| MTP layer: routed 3.385 (4 bit), dense 0.344 (BF16) | 3.729 | 3.56 | 0.881 GiB a draft step (its 8/288, its dense, the head) |
| Embedding 1.182, vision tower 1.05 | 2.23 | 2.23 | rows only / images only |

One plain forward of one row: 3.949 + 7.315 + 0.610 = 11.874 GiB over both ranks, 5.94 GiB (6.375 GB) a rank.
The dense 8-bit layers are 62% of it. That is why DENSE_BITS moved decode so much (26 -> 33 tok/s at 16 -> 8 bits,
39 at 4 bits, which failed quality) and why drafts pay less here than on a dense model: every verified row reads
its own 8 experts a layer, only the dense part is shared.

## Decode ceiling (bandwidth bound)

Device floor: 273 GB/s (LPDDR5X spec; upstream's docs/optimizing.md says GB10 decode projections reach 73-87% of it,
so 200-237 GB/s practical). Measured copy bandwidth on this pair: see "Measured hardware" below.

c1 tok/s = accepted tokens a round / ((verify bytes + draft bytes) / bandwidth), a rank's bytes:

| Bandwidth | plain | k=1, a=1.6 | k=2, a=2.0 | k=3, a=2.4 | k=3, a=3.0 |
|---|---:|---:|---:|---:|---:|
| 200 GB/s | 31.4 | 35.9 | 35.1 | 34.8 | 43.5 |
| 237 GB/s | 37.2 | 42.6 | 41.6 | 41.2 | 51.5 |
| 273 GB/s | 42.8 | 49.0 | 48.0 | 47.5 | 59.3 |

(k drafts verified, a tokens kept a round; verify reads the dense part once and the union of k+1 rows' experts.)
So at one stream, sampled prose with MTP acceptance ~0.6, the practical ceiling is about 42 tok/s and the device
floor about 49. A drafter only moves it if it keeps 3 of 3 (code, copies, greedy).

c8 aggregate, no drafts: union of 8 rows' experts 28.7 GiB + dense 7.3 + head 0.6 = 36.6 GiB a round over both ranks
-> 19.65 GB a rank -> 83 ms at 237 GB/s -> 8 tokens a round -> ~96 tok/s practical, 111 at the floor.

## Prefill ceiling (compute bound)

Active params a token: dense 7.62 B + 8 experts x 25.17 M x 42 layers = 8.46 B -> 16.1 B -> 32.2 GFLOP a token
(sparse attention over 2,048 picked tokens adds little). Each rank does half.
Practical ceiling = 0.5 x measured GEMM peak x 2 ranks / 32.2 GFLOP: see below.

## KV and state

13,056 bytes a token a rank (11 MLA layers' 512-wide BF16 latents + index keys). 1,048,576 tokens = 12.75 GiB a rank.
A kept state's slot is 175 MiB (34 KDA layers x 32 heads a rank x fp32 128x128, double-buffered); TF_GLM_KV_LIVE_HALF
keeps one half (71 MB less).

## Startup floor

82.95 GiB a rank from NVMe. Disk read speed and the unpack are measured below.

## Measured hardware

`runs/hw-spark1.json` (pair stopped, torch 2.13 in the quant image, best of 6-8):

- Device copy of a 4 GiB buffer: 112.4 GB/s copied = 224.7 GB/s of traffic (one read stream + one write stream).
  A read-mostly decode kernel can therefore approach ~225 GB/s; that is 82% of the 273 GB/s spec.
- BF16 GEMM: 88.3 TFLOPS at 4096^3, 95.4 at 8192^3 (FP16 the same). No thermal drop at the larger size.

Ceilings at the measured 225 GB/s and 95 TFLOPS:

| Metric | Baseline (1.2.5) | Practical ceiling | Baseline / ceiling |
|---|---:|---:|---:|
| c1 decode, sampled prose (a ~1.6 a round) | 35.3 tok/s | ~40 (plain forward 35.3) | 88% |
| c8 aggregate, no drafts counted | 59.4 tok/s | ~92 | 65% |
| Cold prefill | ~1,090 tok/s | 0.5 x 190 TFLOPS / 32.2 GFLOP = ~2,950 | 37% |

Prefill is not purely compute bound here: a 2,048-row pass touches every routed expert (about 57 rows each), so it
reads all 142 GiB of experts once a pass: 76 GB a rank / 225 GB/s = 0.34 s of a 1.88 s pass at 1,090 tok/s. Wider
passes (TF_GLM_CHUNK_SOLO=4096) halve that share.

One stream meets its ceiling early: it cannot gain more than ~13% without fewer bytes a token (DENSE_BITS below 8;
4 failed the quality gate) or drafts that keep 3 of 3.

## Corrections after the first runs (19:10)

- The GPU clock is capped at 2,200 MHz of 3,003 by `nvidia-lock-clocks.service` on both nodes (the owner's
  power-off workaround). The 225 GB/s and 95 TFLOPS above were measured under it, so they are the ceilings of the
  machine as it is run, not of the silicon.
- A 1.2.5 c1 round is 52-54 ms with 1.7 drafts verified (2.7 rows): dense + head 3.96 GiB + 2.7 x 1.975 GiB of experts
  = 9.3 GiB = 9.98 GB a rank, 44 ms at 225 GB/s. The engine is at 83% of that bound, which matches "c1 at 85-88%".
- c8: a round verifies about 32 rows (8 streams x 4). The union of 32 rows' experts is 1 - (1 - 8/288)^32 = 59% of a
  layer's experts, 42.2 GiB a rank, plus the dense 3.96: 49.6 GB, 220 ms at 225 GB/s against 272-286 ms measured
  (77-81%). At 17.8 tokens a round that bound is 81 tok/s; the engine gives 59-60. (The table above counted no drafts.)
- Prefill stays the widest gap (37% of the compute bound under the cap). It is kernel work inside the engine.
