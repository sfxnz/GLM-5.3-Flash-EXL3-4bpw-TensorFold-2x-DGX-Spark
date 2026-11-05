# TensorFold native GLM-5.3-Flash on 2x GB10: research notes

Goal (user's): a new clone + ./run.sh + ./stop.sh recipe serving our own EXL3 4bpw GLM-5.3-Flash on two DGX Sparks over RoCE with TensorFold 1.2.2 native, with vision, >=4 streams, ~500k context, correct thinking, and speculative decoding on by default.

Tree: HEAD `85e53093b8b6b6de47c7d5e2f2d432268211a8af` (v1.2.2-228). Tag v1.2.2 is `00e7dddbdb2dd0c481f3d7e8d850fa3e1f2e80b7`. `cuda_exl3.zig`, `cuda_start.zig` paging and `exl3_load.zig` are unchanged between the two. The `cuda_weights.zig` diff is csf0 NVFP4 only (`git diff v1.2.2 HEAD`). Paths below are relative to the repo root.

## 1. EXL3 checkpoint contract (native CUDA GLM, TP=2)

**Detection and admission**
- EXL3 is detected by `quant_method == "exl3"` in `quantization_config`, either top level or under `text_config` (zig/src/families/glm/cuda_exl3.zig:10-24).
- `admit` needs `bits` to be the integer 4 and `codebook` to be `"mcg"`. Anything else raises `UnsupportedQuant`, and the log says why: "EXL3 at {bits} bits: the CUDA family reads 4-bit EXL3 experts" or "EXL3 codebook {x}: the CUDA family reads \"mcg\"" (cuda_exl3.zig:27-44). The engine runs this before it loads anything (cuda_engine.zig:113-117).
- Trap. `config.parse` sets `c.exl3` only from the top-level `quantization_config` (config.zig:168-171). A block that sits only under `text_config` passes `admit` but then takes the NVFP4 expert path. Put the block at the top level.
- Unit-test fixture: `{"quant_method":"exl3","bits":4,"codebook":"mcg","head_bits":16}` (cuda_exl3.zig:180-194). `head_bits` is never checked.

**Only the routed experts are EXL3, the MTP layer's included**
- Routed-expert tensor names are `model.language_model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.{trellis,mcg,suh,svh}` (prefix at cuda_weights.zig:21 and 88-91; names at cuda_exl3.zig:80-113).
- `trellis` must be i16 with shape [K/16, N/16, 64]. That last dim of 64 means uniform 4 bit, the kernel's `uniform_k2 = 8` (cuda_exl3.zig:81, 117-119, 132).
- The `.mcg` marker must be an i32 equal to 0xCBAC1FED (cuda_exl3.zig:82-85; exl3_load.zig:10-12).
- `suh` and `svh` must be **f16 per row**. The packed `.su`/`.sv` sign words that the common scanner accepts (exl3_load.zig:153-176) are refused here with `NotExl3Mcg4` (cuda_exl3.zig:98-99, 110-111).
- Shape rules: `hidden % 128 == 0`, and each rank's `moe_inter` cut must fall on multiples of 128 (cuda_exl3.zig:65; cuda_shard.zig:38). `run` also needs `dims % 256 == 0` (cuda_exl3.zig:158).
- No mixed widths, no other codebooks, no per-layer bit choice. The recipe says the same: "4-bit mcg-codebook routed experts with BF16 weights elsewhere, not arbitrary EXL3 layouts" (docs/recipes/glm-5.3-flash.md:61-62).
- The MTP layer (index `num_hidden_layers`) also takes the EXL3 expert path. The comment reads "the MTP layer's experts too" (cuda_weights.zig:441-445; cuda_mtp_load.zig:19). A bf16 MTP expert therefore fails whenever drafts are on. MTP loads only when drafts are on (cuda_weights.zig:557).

**Everything else must be a plain `.weight` tensor**
- Format checks read `{module}.weight`, so a trellis-only tensor there fails with `UnreadableFormat` (cuda_weights.zig:114-138).
- Allowed formats per tensor:
  - Dense layers: bf16 (cuda_weights.zig:337-344).
  - DSA attention: bf16 or fp8_pb. The MTP layer may also use MXFP8 (cuda_weights.zig:242-262).
  - KDA projections: bf16 or MXFP8 (cuda_weights.zig:209-232).
  - Shared experts: bf16 or fp8_pb (cuda_weights.zig:447-450).
  - Router and bias: bf16 and f32 (cuda_weights.zig:435-437).
  - `embed_tokens`: bf16.
  - `lm_head`: bf16 or MXFP8 (cuda_weights.zig:552-556).
- MTP norms and `eh_proj` are bf16. The draft head is quantized from `lm_head` at load (cuda_mtp_load.zig:22-26, 33-60).
- A float32 tensor that bf16 cannot hold exactly is refused with `InexactBf16` (cuda_weights.zig:167-169; docs/quantization.md:138-145).

**Vision**
- Needs `vision_config` in config.json with `hidden_act: "silu"`. Without it the server says "this checkpoint has no vision tower: serve it without --vision" (cuda_images.zig:55-60; vision_config.zig:36-47).
- Tower tensors live under `model.visual.` or `vision_tower.` (vision_names.zig:6) and must be bf16. MXFP8 or NVFP4 are read only when a recipe declares them (cuda_vision_weights.zig:46-68). Otherwise `MissingVisionTensor`.

**Other refusals**
- Config: `NotGlm53Flash` for any width or count mismatch, `vocab % 128 != 0`, or `n_group`/`topk_group` other than 1. Also `BadConfig` and `UnsupportedActivationDtype` (config.zig:153-232).
- `TooManyRanks` (world >16), `NoRank`, `NoCommId` (cuda_native.zig:118-131, 199-200).
- `RankMismatch` when the ranks' fingerprints differ: settings, capacity, MTP, config, binary, checkpoint digest, media limits, drafter (cuda_engine.zig:122-139).
- `GlmShareNeedsPages`, `GlmSoloWithDrafter`, `GlmKvHostNeedsPages`, `Fp8OnPages`, `DrafterNeedsDrafts`, `NoVisionTower`, `BadRoceCap` (cuda_native.zig:232-235, 270-276, 299-300, 467-485).
- Weights larger than the budget (native/cuda.zig:310-313).
- `UnsupportedQuant` has a misleading explanation ("reads ModelOpt NVFP4 and MIXED_PRECISION") (cuda_native.zig:482). The log carries the real reason. `NotExl3Mcg4` has no explanation and surfaces as the generic "cannot load ... (NotExl3Mcg4)" (native/cuda.zig:343-349).

**Checkpoint IDs named**
- `brandonmusic/GLM-5.3-Flash-tr3-4bpw` and its byte-identical re-host `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` (docs/recipes/glm-5.3-flash.md:24-25, 59-63; docs/recipes/cuda.md:69, 74-76; README.md:181).
- That checkpoint is 164 GB in total (CHANGELOG.md:1179).
- The NVFP4 comparators are `nvidia/GLM-5.3-Flash-NVFP4@423acf37…` and `Bizuayeu/GLM-5.3-Flash-XSFP4`.

## 2. Serve knobs and memory

**Flags**
- Vision: `--vision`, `--vision-urls`, `--vision-max-images`, `--vision-max-videos`, `--vision-image-tokens`, `--vision-video-tokens`, `--max-frames`.
- Context and streams: `--context`, `--parallel`.
- KV and cache: `--kv-cache bf16|fp8`, `--prompt-cache-gib`.
- Drafting: `--drafter`, `--drafter-bits 0|4`, `--no-drafts`, `--mtp-drafts`, `--mtp-confidence`.
- Thinking: `--reasoning-effort low|medium|high|xhigh`, `--thinking`, `--no-thinking`, `--thinking-budget`, `--thinking-reserve`.
- Sources: zig/src/server/cli.zig:57-120. `--context` defaults to 8192 for GLM (cuda_native.zig:23).

**KV pages**
- `TF_KV_PAGES` defaults to `auto`, which means pages when world >= 2 and the cache is not fp8. `on` together with fp8 is refused (cuda_start.zig:47-53).
- fp8 therefore keeps one full window per stream. Its quality qualification is still "pending" (docs/recipes/glm-5.3-flash-cuda.md:46-47).

**Env** (read at families/glm/cuda_ranks.zig:9-27 and 32-131, plus the files cited)
- `TF_GLM_KV_HOST`, `_SHARED`, `_GIB`: device cost drops from 13,056 to 768 bytes a token (glm-5.3-flash-cuda.md:52-56).
- `TF_GLM_HOST_CACHE=1` (glm-5.3-flash-cuda.md:48-51).
- `TF_GLM_KV_SHARE=N` (cuda_native.zig:80-93).
- `TF_GLM_VISION_HOST=1`: the tower takes 240 MiB on device instead of 1.05 GiB (glm-5.3-flash-cuda.md:41-43).
- `TF_GLM_CHUNK_SOLO=4096`, `TF_GLM_DENSE_BITS=4` (lossy), `TF_GLM_DRAFT_CONFIDENCE` (default 0.7, dflash.zig:37-39), `TF_GLM_PREFILL_REDUCE` (default split).
- `TF_HEAT_HIGH`/`TF_HEAT_LOW`.
- `TF_GLM_MTP` does not exist in native Zig (no reader). It belongs to the Python 0.6.6 engine (glm-5.3-flash.md:50-55).

**Drafter**
- Native default is the MTP head whenever drafts are on (cuda_native.zig:263).
- DFlash2 runs only with an explicit `--drafter <incoai/GLM-5.3-Flash-DFlash2>`, which the hub resolves (server/main.zig:113). Its weights are CC BY-NC-ND (cuda_native.zig:46; THIRD_PARTY_NOTICES.md:240). It adds a 40 MiB ring per stream (dflash.zig:35; cuda_native.zig:357).
- A DFlash2 bug at TP>1 is fixed only after 1.2.2: "drafter reads complete prompt taps on every rank (it read stale residual rows...)". Acceptance is lower on 1.2.2 (CHANGELOG.md:9).

**Admission**
- Room = MemAvailable minus the reserve (native/cuda_family_memory.zig:18-42; native/cuda_memory.zig:117-121).
- The reserve on unified memory is max(4 GiB, total/10), which is 12.2 GiB on a GB10. `TENSORFOLD_MEMORY_RESERVE_GIB` can lower it to as little as 2 (cuda_memory.zig:127-139).
- Rank 0 admits by the smallest room across the ranks (cuda_native.zig:316-322).
- Startup line format: native/cuda_startup.zig:2.

**Measured GB10 runs**
- No EXL3 run is recorded at `--parallel 4` with context >= 500k.
- The nearest run is TP=2 `--parallel 4 --context 524288 TF_KV_PAGES=on` with `Bizuayeu/GLM-5.3-Flash-XSFP4` plus MTP, published as a speed table only with no memory figures (glm-5.3-flash-cuda.md:87-100).
- NVFP4 TP=2 with `--vision` at 131072 x4 (docs/receipts/glm-kv-pages-8e705835/tp2/serve/r2vis/server-rank0.log:3): "model 94.33 GiB; 4 streams ... 561 pages (3.49 GiB) ... 340 MiB fixed each, of 8.5 GiB left after a 12.2 GiB reserve". That implies about 115 GiB available before load (inferred).
- KV per rank at capacity 245,824 is 3,397,789,184 B in bf16 and 1,899,246,080 B in fp8 (receipts/.../ci/mutations/paged-table-kept.log:6). That works out to about 13.8 KB and 7.7 KB a token.

**Is 524288 x 4 plausible at about 82 GiB per rank plus vision? (inferred)**
- One full 524k window is 6.75 GiB in bf16 or 3.77 GiB in fp8.
- Rank 0 room is about 115 − 12.2 − 82 − 1.05 (tower) − 4 × 0.34 ≈ 18.4 GiB, before the prompt cache and any drafter.
- bf16 on pages: 4 streams are admitted, since the pool's floor is one full window. Only about 2.7 full windows can be resident at once, so a further long request waits (receipts README:51). Four full windows need about 27 GiB.
- fp8 windows: 4 × 3.77 = 15.1 GiB fits, leaving about 3 GiB of prompt cache. Quality is unqualified.
- So four concurrent full 524k streams do not fit in bf16 without cutting the reserve to about 4 GiB, or without `TF_GLM_KV_HOST`. KV_HOST's benefit on unified memory is unproven.

## 3. Thinking

- `--thinking` is on by default (server/cli.zig:211). The startup log says so (serve.zig:268-269).
- A request can set `enable_thinking` and `reasoning_effort`. `chat_template_kwargs.thinking` is heard as `enable_thinking` (chat.zig:139-142; CHANGELOG.md:1684). They are passed to the template (prompt.zig:31-41).
- With thinking off, a dangling `<think>` is closed with `</think>` (prompt.zig:50-58).
- The reply is split on `</think>`, with the open marker taken from the prompt (reply_text.zig:9-16, 111-130). The result is `reasoning_content` plus `content`.
- If a reply hits `max_tokens` while still thinking, the server logs a warning and `content` is empty (chat.zig:582).
- `clear_thinking` is always passed to GLM templates (`model_type glm5_next`). `TF_GLM_CLEAR_THINKING` unset or 0 means false, so earlier turns keep their reasoning. 1 means true, and any other value is refused (hf_text.zig:12-19, 62-68, 371).
- A request's `chat_template_kwargs.clear_thinking` wins over the env (hf_text.zig:467-525). This was added because "TR3's template clears unset" (hf_text.zig:14; CHANGELOG.md:804).
- Tool calls use GLM's `<tool_call>NAME<arg_key>..</arg_key><arg_value>..</arg_value>` format. Values are decoded by the tool's schema (tool_glm_stream.zig:1; docs/api.md:319-325, 379; glm-5.3-flash.md:293-297).
- Streaming GLM tool arguments arrived after 1.2.2 (CHANGELOG.md:32).
- No GLM thinking bug is listed in docs/known-issues.md.

## 4. Build and install (aarch64, sm_121)

**Release v1.2.2** (published 2026-11-03 from `main`)
- Assets: `tensorfold-1.2.2-linux-aarch64.tar.gz` (45.8 MB) plus `.sha256` and `.sha256.sig` (`gh api .../releases/tags/v1.2.2`).
- Both Linux archives are CUDA builds with sm_80, 86, 120 and 121 fatbins (packaging/README.md:13, 36-37).
- Install steps: RUNBOOK.md:194-204. Needs driver R580+ (CUDA 13.0) and glibc 2.28+ (RUNBOOK.md:186-187).

**Not in the archive**
- `libnccl.so.2` (CUDA 13), `libibverbs.so.1`, and `libcublasLt.so.13` for `--vision` (packaging/RUNTIME.md:33-42).
- `libturbojpeg` for JPEG images (RUNTIME.md:13-15).
- `libtfvideo` for video.

**Source build and packaging**
- `zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc` produces zig-out/native/bin/tensorfold-native (RUNBOOK.md:238-249). It needs Zig 0.17.0 (.zig-version; build.zig.zon:5).
- nvcc 13.3 (receipt README:14) and 13.4 (RUNBOOK.md:256) were used on GB10.
- There is no Dockerfile anywhere upstream (`find` returned nothing). Receipts ran in Ubuntu 24.04 containers with NCCL 2.30.7 and `--cpuset-cpus=0-4,10-14` (receipts/glm-kv-pages-8e705835/README.md:11-14).

**RUNBOOK for GB10**
- Heat bands: `TF_HEAT_HIGH=92 TF_HEAT_LOW=88`. The receipts also used `TF_HEAT_CEILING=93` (RUNBOOK.md:228-233; receipt README:60).
- Pin to efficiency cores with `taskset -c 0-4,10-14` (RUNBOOK.md:259-271).
- Cap the GPU clock with `nvidia-smi -lgc 300,2200`, set by a systemd unit (RUNBOOK.md:273-308).
- NCCL. The recipe says to set `NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA` (glm-5.3-flash-cuda.md:27). The receipt used `enp1s0f1np1` and `rocep1s0f1` (r2vis/server-rank0.log:12-13).

## 5. Two-host CLI

**Launch**
- Same args on every rank, with `TF_GLM_RANK`, `TF_GLM_WORLD=2` and `NCCL_COMM_ID=<rank0 IPv4>:port`. Start rank 1, then rank 0 (glm-5.3-flash-cuda.md:12-25).
- The liveness line uses port+1. Ranks must connect within 5 minutes and be ready within 15 (cuda/tp_watch.zig:37-51, 134).
- The engine sets `NCCL_RUNTIME_CONNECT=0` and `NCCL_CUMEM_*=0` itself, and drops `NCCL_COMM_ID` on followers when `NCCL_SOCKET_IFNAME` is set (cuda_native.zig:244-249).
- RoCE gathers are on by default at two ranks. `TF_ROCE=0`, `TF_ROCE_HCA` and `TF_ROCE_MAX_KB` (default 4096) tune them (docs/multi-rank-receipts.md:36-38; cuda_native.zig:181-187).

**Endpoints** (served by rank 0 only; rank 1 never serves, cuda_native.zig:336)
- `/health` lists status, model, `max_batch_size`, memory and live rates (docs/api.md:800-821).
- `/v1/models` also reports reasoning support (CHANGELOG.md:1245).
- `/metrics` and `/v1/metrics` (docs/api.md:103) need `--metrics-open` when API keys are set (api.md:21).

**Speculative acceptance**
- Each response carries a top-level `speculative` object with rounds, drafted, accepted, acceptance_rate and tokens_per_round (chat.zig:551-560; openai.zig:328).
- The log line prints `rounds= accepted=a/d` (chat.zig:591).
- Metrics: `tensorfold:mtp_drafted_total` and `mtp_accepted_total`, plus vLLM-style `spec_decode_num_*` (metrics.zig:122, 199-200).

**Shutdown**
- SIGTERM or SIGINT on rank 0 stops it cleanly (serve.zig:38-45). `close` sends the end step, and followers release and exit 0 (cuda_native.zig:336-349, 445-454).
- If either rank fails, the other stops (`tp_watch.fail`; follower `_exit(1)`, cuda_native.zig:337-341).
- A stale process holding port+1 causes `BindFailed` (tp_watch.zig:50).

## Unknowns

1. The true per-rank bytes of our quant: does the ~82 GiB include the MTP layer (about 2 GiB) and the 1.05 GiB tower? Nothing upstream measures EXL3 per-rank memory.
2. Does our exllamav3 output write f16 `suh`/`svh` (not packed `su`/`sv`), the `.mcg` marker, the `model.language_model.` prefix, a quantized MTP layer, and a top-level `quantization_config`? Each one is a hard refusal if wrong. Check before building run.sh.
3. Do `TF_GLM_KV_HOST`, `TF_GLM_VISION_HOST` and `TF_GLM_HOST_CACHE` save anything on GB10 unified memory? Pinned host memory is the same DRAM, so I'm guessing no physical saving. Admission accounting for it is unverified.
4. No EXL3 or GB10 run is recorded at 524288 x 4. The XSFP4 run gives no memory figures.
5. DFlash2 acceptance at TP=2 on 1.2.2 is degraded until the post-1.2.2 fix, and its NC-ND license may not fit the use.
6. `--reasoning-effort` has no `max` choice in the native CLI. GLM's template default is Max (glm-5.3-flash.md:157-159). A per-request `reasoning_effort: "max"` was not traced.
7. Whether the v1.2.2 aarch64 archive's GLM path was qualified on the EXL3 checkpoint at that tag was not run here.
