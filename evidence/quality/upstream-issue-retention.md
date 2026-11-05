# Draft upstream issue (not filed): malaiwah/quant-fidelity-suite

**Title.** layer-outer stream residency keeps every byte it has read as anonymous memory on GB10 (MAP_PRIVATE copy-on-write)

**Version.** cf36287c51a18e9361e009ca00a21eb8c0021ab1, `engines/tools/layer_outer.py`. NGC `pytorch:26.07-py3` (torch 2.13.0a0, safetensors 0.8.0, driver 580.178.04), one DGX Spark GB10 (sm_121, 128 GB unified memory).

## What happens

`hf_capture.py --schedule layer-outer --layer-residency stream --device cuda` on `brandonmusic/GLM-5.3-Flash-tr3-4bpw@a5fee929` is killed by the kernel OOM killer after layer 26 of 45. The resident layer stays at 18.49 GB. The process's `RssAnon` grows by about 3.7 GB per MoE layer and is never returned, which is about the checkpoint bytes per layer (175.64 GB / 45). At the kill, python3 held anon-rss 85.5 GB.

## Where the memory is

Per layer, after `free_layer`, on a load-and-free loop with no forwards (layers 0 to 6):

- glibc `mallinfo2` stays at 0.56 GB in use, and no CPU tensor is reachable from `gc`.
- `torch.cuda.host_memory_stats()` reports 0 bytes, and `memory_allocated` returns to 3.44 GB after each free.
- `/proc/self/smaps` shows the growth inside the shard mappings themselves. Each shard is mapped `rw-p` (safetensors' `UntypedStorage.from_file(shared=False)`, so MAP_PRIVATE), and its resident pages are `Anonymous`. For example, a 1.28 GiB shard holds 1,280 MiB Rss, all of it Anonymous.

A micro test on one shard (`get_slice(k)[...]` views, no suite code) isolates the trigger:

| step | mapping Rss | Anonymous |
|---|---|---|
| CPU read of group 0 | 101 MiB | 0 |
| `.to("cuda")` of group 0, already faulted by the CPU | 101 MiB | 0 |
| `.to("cuda")` of group 1, cold pages | 220 MiB | 150 MiB |
| `.clone()` then `.to("cuda")` of group 2 | 301 MiB | 150 MiB (no change) |
| `madvise(MADV_DONTNEED)` on the mapping | 0 | 0 |
| re-read group 1 and copy to cuda | equal to the first copy, bit for bit | |

On GB10, a device copy straight out of a cold MAP_PRIVATE page faults it in for write. That breaks copy-on-write and leaves a private anonymous copy of the file bytes. The copy lives as long as the mapping. `build_streamed_model` opens every shard once and holds the handles until `StreamedModel.close()`, because `layer_subset` keeps lazy slices for all layers. So every layer's bytes stay resident as unevictable anonymous memory, and "stream" residency holds the whole checkpoint by the end of the run. Both the trellis decode path (`payload[...].to(device)`) and the transformers converter path do this.

On a discrete GPU (H200), a pageable host-to-device copy goes through the driver's staging buffer, so the CPU reads the pages and they stay file-backed and evictable. That is probably why the published TR3-h200 run reports `peak_rss_bytes` 175.47 GB, about the checkpoint size, and still finishes. This is inferred from the GB10 micro test, not measured on an H200.

## Proposed fix

In `do_free`, after `gc.collect()` and `torch.cuda.empty_cache()`, call `madvise(MADV_DONTNEED)` on the process's private mappings of the opened shards (`_drop_private_pages` in the attached patch: one 30-line helper and one call). Nothing in the capture writes to a checkpoint mapping, so this loses no data. The next read refaults the file's own bytes. The loaded parameters (sha256 of every parameter of layers 0 to 6) and the hidden states of panel windows 0 and 1 after each of those layers are byte-equal with and without the patch. With the patch, `selftest_layer_outer.py` passes the same 61 checks, and the full TR3 capture on one GB10 seals in 2,104 s. Its `peak_rss_bytes` is 6.67 GB, against 175.47 GB for the H200 run, and RssAnon peaks at 5.74 GiB. The patch is `quality/patches/0002-drop-private-shard-pages.patch` in our recipe.

A smaller, always-on signal would have shown this at a glance. `memory.json` could record `RssAnon` and `RssFile` from `/proc/self/status` at each layer boundary, next to `peak_rss_bytes`. `ru_maxrss` cannot tell evictable page cache from anonymous memory, and that difference decides whether a run fits.
