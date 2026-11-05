# TensorFold 1.2.2 GLM EXL3 layout contract

`check_layout.py` asserts the rules below on a candidate pack, offline and without a GPU. Each rule cites the TensorFold source that enforces it. Rules marked **ours** are not enforced by TensorFold. They are the assembler's invariant: every non-expert tensor comes from the BF16 source unchanged.

Pins:
- TensorFold `v1.2.2` is tag object `e45cca70`, commit `00e7dddbdb2dd0c481f3d7e8d850fa3e1f2e80b7`. Read with `git show v1.2.2:<path>`.
- The BF16 source is `zai-org/GLM-5.3-Flash-BF16@f9fed3bbc82291073373d67452e9f25317ef7577`.

## config.json

| Rule | TensorFold source |
|------|-------------------|
| A top-level `quantization_config` with `quant_method: "exl3"`. A copy under `text_config` alone is not enough, because the family config sets `exl3` from the top level only. | `zig/src/families/glm/config.zig:167-170` sets the loader's `exl3` flag. Admission (`cuda_engine.zig:113` through `cuda_exl3.zig:9-24`) also accepts `text_config`, so a `text_config`-only pack is admitted and then loaded down the NVFP4 path. |
| `bits` is the JSON integer `4`. `4.0`, which stock exllamav3 writes, is refused. | `zig/src/families/glm/cuda_exl3.zig:34-38` |
| `codebook` is `"mcg"`. | `cuda_exl3.zig:39-43` |
| `hidden_size` is a multiple of 128. | `cuda_exl3.zig:65` |

## Routed experts

The routed experts are in every layer `i` with `first_k_dense_replace <= i < num_hidden_layers`, plus the MTP layer `num_hidden_layers`. For GLM-5.3-Flash that is layers 3 to 45, with 288 experts each. TensorFold sources:
- `config.zig:63-65` (`isMoe`), `config.zig:172` and `config.zig:174` for the range and the count.
- `cuda_weights.zig:438-439` and `zig/src/families/glm/cuda_mtp_load.zig:15,19` cover the MTP layer.

Names sit under `model.language_model.` (`cuda_weights.zig:21`). For each expert `e` and `proj` in `gate_proj`, `up_proj` and `down_proj`, the input width is K and the output width is N. Both come from the source `.weight`, which is `[N, K]`. Gate and up are `[2048, 4096]`. Down is `[4096, 2048]`.

| Tensor | Requirement | TensorFold source |
|--------|-------------|-------------------|
| `layers.{i}.mlp.experts.{e}.{proj}.trellis` | `I16`, shape `[K/16, N/16, 64]` | `cuda_exl3.zig:80-81` |
| `....{proj}.mcg` | `I32`, numel 1, value `0xCBAC1FED` little-endian | `cuda_exl3.zig:82-85`, `zig/src/core/exl3_load.zig:10-11` |
| `....{proj}.suh` | `F16`, numel K | `cuda_exl3.zig:98-99` |
| `....{proj}.svh` | `F16`, numel N | `cuda_exl3.zig:110-111` |
| `....{proj}.weight` | absent (**ours**). TensorFold never reads it on the EXL3 path. A leftover adds about 14.5 GB per layer of dead bf16 bytes. | not enforced |

The checker reads the 4-byte value of every marker. Shape and dtype come from the headers.

## suh sharing (reported, not enforced)

TensorFold compares every gate and up `suh` in a layer against the first one. If all of them are equal it sets `shx = true` and rotates a prompt row once. Otherwise it rotates once per pick (`cuda_exl3.zig:68, 75, 101-105`; `cuda_weights.zig:440`). Both paths load. The checker prints the count of distinct gate/up `suh` vectors per layer and the resulting `shx`. TR3 shares one rotation per layer. Stock exllamav3 folds input-channel scales into `suh`, so our pack is expected to report `shx=false`.

## Every other tensor (**ours**)

- The candidate's non-expert names equal the source's names with the routed-expert `.weight` tensors removed. Each tensor's dtype and shape equal the source's.
- TensorFold reads these tensors by their source names. The fused `qkv_proj` from stock exllamav3 would not be found, since TensorFold reads separate `q/k/v_proj` and `q/k/v_conv1d` (`cuda_weights.zig:205-220`).
- TensorFold converts only bf16 and f32 (`cuda_weights.zig:149-170`). It reads the dense MLP, embedding and head as bf16 (`cuda_weights.zig:336-337, 546-550`). The f16 linears from stock exllamav3 would be refused.
- The equality check keeps the source's own F32 tensors (`A_log`, `dt_bias`, `conv1d` and others) as F32.

## Running it

```sh
python3 -I quant/check_layout.py <candidate dir | hf:owner/repo@rev> <source dir | hf:owner/repo@rev> \
    [--cache DIR] [--suh-sample N]
python3 -I quant/test_check_layout.py <empty scratch dir>   # synthetic PASS plus one FAIL per rule
```

A remote location is read with HTTP range requests. `--cache` keeps the config, index and shard headers. `--suh-sample N` reads gate/up `suh` for N evenly spaced experts per layer instead of all 288. Use it for a remote pack, where reading every vector would mean 24,768 range reads. Markers are always read for every expert.
