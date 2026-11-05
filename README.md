# GLM-5.3-Flash EXL3 4bpw · TensorFold 1.2.2 · 2× DGX Spark

Serve [sfxnz/GLM-5.3-Flash-EXL3-4bpw](https://huggingface.co/sfxnz/GLM-5.3-Flash-EXL3-4bpw) across two NVIDIA DGX Spark (GB10) nodes at tensor parallel 2, with the stock [TensorFold](https://github.com/ashhart/TensorFold) 1.2.2 release and its native CUDA GLM family. No fork and no patches. Vision, thinking and MTP speculative decoding are on by default. Up to 8 requests run at once, and one request can use a 1,048,576-token window.

```sh
git clone https://github.com/sfxnz/GLM-5.3-Flash-EXL3-4bpw-TensorFold-2x-DGX-Spark
cd GLM-5.3-Flash-EXL3-4bpw-TensorFold-2x-DGX-Spark
./run.sh     # http://<head>:8000/v1, model glm-5.3-flash
./stop.sh
```

Every number below comes from a file under `evidence/`.

## At a glance

| | Default | Evidence |
|---|---|---|
| 1 stream | 33.8 tok/s (range 32.4 to 36.2) | `evidence/knobs/d8h8-p8-1m-r105/bench.json` |
| Aggregate, 4 streams | 45.5 tok/s (44.6 to 48.8) | same |
| Aggregate, 8 streams | 58.8 tok/s (51.8 to 61.0), TTFT 0.86 s | same |
| Longest prompt answered | a needle found at 1,045,624 tokens, 857 tok/s prefill | `evidence/knobs/d8h8-p8-1m-r105/needle.jsonl` |
| MTP acceptance | 0.62 at 1 stream, 0.41 to 0.43 at 4 and 8 | `evidence/knobs/d8h8-p8-1m-r105/bench.json` |
| `tests/smoke.py` through `run.sh` | 7 of 7 | `evidence/run/smoke-final.txt` |
| Load time after the first boot | 2.4 min to `/health` (`ready after 145s`), 2 min 31 s for all of `run.sh` | `evidence/run/boot-final.txt` |

Rates are aggregate tok/s, medians of 5 runs of 512 tokens, thinking off. Against the same recipe with BF16 dense layers, 4 slots and a 524,288 window, measured on two boots and taken as their mean, that is about 28% faster at 1 stream, 7% at 4 (within the runs' spread) and 35% at 8 (see [Speed knobs](#speed-knobs)).

## What it serves and why

GLM-5.3-Flash is Z.ai's MoE model with a vision tower and an MTP head. Its BF16 weights are 598.5 GiB (`evidence/decisions.tsv`, row `disk` 12:47:53), so it does not fit one 128 GB Spark. TensorFold 1.2.2 reads a GLM EXL3 pack in one layout only. The routed experts are EXL3 at 4 bits on the `mcg` codebook, and every other tensor is BF16 (`evidence/research/research-tensorfold-glm.md`, section 1). The one other published pack in that layout, TR3, was made with a custom encoder and carries its own attribution license (`evidence/research/research-exl3-quant.md`, section 2). We built ours with stock tools from the MIT sources, so anyone can rebuild it. The routed experts, the MTP layer's included, are encoded with stock exllamav3 v1.6.6. Every other tensor is copied byte for byte from `zai-org/GLM-5.3-Flash-BF16@f9fed3bb`. Each Spark holds one rank's share of the weights, and NCCL runs over the direct RoCE cable between them.

At load, TensorFold packs the BF16 dense linears and the LM head to symmetric 8-bit integers with an fp16 scale for each 64 values (`TF_GLM_DENSE_BITS=8`, `TF_GLM_HEAD_BITS=8`; `zig/src/families/glm/cuda_qpack.zig:1,18` at v1.2.2, in `evidence/research/tf-source-lines.txt`). That frees 3.8 GiB on each rank for KV pages and costs 0.0003 nats a token, within our quality gate (see [Speed knobs](#speed-knobs)). The KV cache stays BF16. `DENSE_BITS=16 HEAD_BITS=16 ./run.sh` serves the dense layers at BF16 from the same files.

## Quick start

On the head node (rank 0, which serves HTTP):

```sh
VALIDATE_ONLY=1 ./run.sh   # read-only: every check, both docker argvs, exit 1 if run.sh would refuse
./run.sh
```

`run.sh` acts only on what is missing, so a rerun converges. In order, it:

1. Checks `recipe.conf`, ssh to the worker, the tools on both nodes, the link, the HCAs, and that no other container holds a GPU.
2. Builds the image `tf-glm53:1.2.2` from `docker/Dockerfile` on the head. The build downloads the TensorFold 1.2.2 aarch64 release archive and checks its sha256 (`17632c30…`), plus libjpeg-turbo 3.1.4.1 and libwebp 1.6.0. The image is 1,549,502,371 bytes (`evidence/image/labels.txt`). Its apt step took 883 s in our build (`evidence/image/build.log`).
3. Copies the image to the worker with `docker save | ssh docker load` over the direct link when the two image IDs differ, then checks that they match.
4. Downloads the pack at the pinned revision with `hf download` into `HF_CACHE` on the head, and checks the shard count and total bytes. The pack is 175,642,102,880 bytes in 120 shards.
5. Copies the snapshot to the worker with `rsync -aL` over the direct link when the worker's copy is incomplete, and checks it again there.
6. Checks the ports and the memory gate, starts rank 1 on the worker, then rank 0 on the head, and waits for `/health`.

When `/health` answers, `run.sh` prints rank 0's startup lines and the endpoint. With the pack already on the head, the first boot took 9 min 54 s, most of it the copy to the worker (`evidence/run/first-boot.txt`). Later boots take about 2.5 min.

To stop:

```sh
./stop.sh
```

`stop.sh` stops rank 0 first, whose SIGTERM ends the session, then rank 1 over ssh. Each container gets 60 s to exit and is then removed. It touches only containers that carry the label `ai-lab.recipe=glm53-exl3-tensorfold` and the names `tf-glm53-rank0` and `tf-glm53-rank1`. It is safe to run when nothing runs (`evidence/run/weights-plan-and-stop.txt`).

## Requirements

- Two DGX Spark (GB10, sm_121) nodes. Our pair runs driver 580.178.04 with CUDA driver API 13.0 (`evidence/engine-probe/tr3-run2-rank0.log`).
- The direct QSFP cable between them, with an IPv4 address on `enp1s0f1np1` on each node. Our pair uses 10.100.8.1 for the head and 10.100.8.2 for the worker. `run.sh` reads the addresses from the interface.
- Both RoCE HCAs for that port, `rocep1s0f1` and `roceP2p1s0f1`, ACTIVE on each node. NCCL uses both (`NCCL_IB_HCA`). TensorFold's RDMA all-gather uses `rocep1s0f1` (`TF_ROCE_HCA`).
- Key-based ssh from the head to the worker, by the name in `WORKER` (default `spark2`). `run.sh` uses `ssh -o BatchMode=yes`, so a password prompt counts as a failure.
- The tools `docker ssh curl rsync hf ip rdma ss free stat` on the head, and `docker rsync ip rdma ss free stat` on the worker.
- Docker with the NVIDIA runtime (`--gpus all`) and `/dev/infiniband` on both nodes.
- Internet on the head only, for the image build and the `hf download`. The worker gets both from the head.
- Disk for one copy of the pack, 175.6 GB, on each node at the same absolute `HF_CACHE` path.
- MemAvailable of at least 109 GiB on each node before load (`MEM_GATE_GIB`). That is 82.9 GiB of weights per rank, 12.8 GiB of KV pages, 2.4 GiB for the other 7 streams and the 10.5 GiB reserve. Our boots had 114 to 118 GiB (`evidence/run/first-boot.txt`, `boot-final.txt`).

### What run.sh checks and refuses

`run.sh` refuses to start when any of these holds. It checks the first five before it builds, downloads or stops anything. It checks the ports and the memory after the image and the weights are in place.

- `REVISION`, `SHARDS` or `SHARD_BYTES` is not a pinned value.
- ssh to the worker fails, or a node lacks a tool listed above.
- `IFACE` is down or has no IPv4 address on either node, or an explicit `HEAD_IP` or `WORKER_IP` differs from the interface's address.
- An HCA in `HCA` is not ACTIVE.
- Another container holds a GPU on either node. `run.sh` never stops it. Stop it yourself with its own stop script.
- `PORT`, `NCCL_PORT` or `NCCL_PORT+1` already listens on either node.
- MemAvailable is below `MEM_GATE_GIB` on either node, unless `FORCE=1`.

It also exits with an error when the built image carries another archive sha256, when a snapshot is incomplete after the download or the copy, when a rank does not start, or when `/health` does not answer within `READY_TIMEOUT`. In the last two cases it prints the last 40 log lines of both ranks and runs `stop.sh`.

If both ranks already run and `/health` answers, `run.sh` prints the endpoint and exits 0 without a restart. If both run and `/health` does not answer yet, it waits for them. Leftover containers of ours that are not a running pair are removed before a fresh start.

## Configuration

`recipe.conf` holds every setting. A variable already set in the environment wins, for example `PARALLEL=4 ./run.sh`. Both `run.sh` and `stop.sh` source the file, and the Python tools read the same defaults (`tools/recipe.py`).

| Variable | Default | What it sets |
|---|---|---|
| `MODEL` | `sfxnz/GLM-5.3-Flash-EXL3-4bpw` | The HF repo. |
| `REVISION`, `SHARDS`, `SHARD_BYTES` | `d75e26a4…`, `120`, `175642102880` | The pinned commit, the shard count and the total shard bytes. |
| `HF_INCLUDE` | `*.safetensors *.json *.jinja LICENSE` | What `hf download` fetches, one `--include` a pattern. |
| `HF_CACHE` | `$HOME/.cache/huggingface` | The cache root. It must be the same absolute path on both nodes. |
| `IMAGE`, `TF_VERSION`, `TF_SHA256` | `tf-glm53:1.2.2`, `1.2.2`, `17632c30…` | The image tag and the release archive it is built from. They match the `ARG` defaults in `docker/Dockerfile`. |
| `SERVED_NAME` | `glm-5.3-flash` | The model name clients send. |
| `PORT` | `8000` | The HTTP port on rank 0. |
| `CONTEXT` | `1048576` | One request's window (`--context`). |
| `PARALLEL` | `8` | Requests that run at once (`--parallel`). |
| `DENSE_BITS`, `HEAD_BITS` | `8`, `8` | `TF_GLM_DENSE_BITS` and `TF_GLM_HEAD_BITS`: 4, 6, 8, or 16 for BF16. |
| `MEMORY_RESERVE_GIB` | `10.5` | `TENSORFOLD_MEMORY_RESERVE_GIB`, the memory each node keeps for the host. The engine's own default is 12.2. |
| `WORKER` | `spark2` | The worker's ssh name. |
| `IFACE` | `enp1s0f1np1` | The direct-link interface. NCCL binds to it. |
| `HEAD_IP`, `WORKER_IP` | empty | Empty means read from `IFACE` on each node. |
| `HCA`, `ROCE_HCA` | `rocep1s0f1,roceP2p1s0f1`, `rocep1s0f1` | `NCCL_IB_HCA` and `TF_ROCE_HCA`. |
| `NCCL_PORT` | `29583` | Rank 0's NCCL port. The liveness line takes the next port. |
| `MEM_GATE_GIB` | `109` | The MemAvailable floor on each node. |
| `FORCE` | `0` | `1` starts below the memory gate after printing it. |
| `READY_TIMEOUT` | `1200` | Seconds to wait for `/health`. |

`VALIDATE_ONLY=1 ./run.sh` runs every check without writing anything. It prints the config, each check as `ok` or `REFUSE`, each action it would take as `WOULD`, and both ranks' full `docker run` argvs. It exits 1 if `run.sh` would refuse.

Both ranks get the same serve argv from one function, `rank_argv` in `run.sh`. Only `TF_GLM_RANK` and the container name differ, because the ranks compare their settings at startup and refuse to serve on a mismatch. The serve flags are `--context`, `--parallel`, `--vision`, `--name`, `--host`, `--port` and `--no-update-check`. The engine variables are the NCCL and RoCE ones plus `TF_GLM_DENSE_BITS`, `TF_GLM_HEAD_BITS` and `TENSORFOLD_MEMORY_RESERVE_GIB`. The engine's own defaults set the rest. They are BF16 KV pages (`cache bf16` in `evidence/knobs/d8h8-p8-1m/boot.log`), thinking on (`evidence/run/smoke-final.txt`, check `models`), MTP drafts on, and sampling at temperature 1.0 and top_p 0.95 (`evidence/run/boot-final.txt`).

## API

Rank 0 serves an OpenAI-style API at `http://<head>:8000/v1`. Rank 1 never serves HTTP.

| Endpoint | Use |
|---|---|
| `POST /v1/chat/completions` | Chat, streamed or not, with images and thinking. |
| `POST /v1/completions` | Plain completions. A nonstreamed request can ask for `prompt_logprobs` (`tools/nll.py` uses it). |
| `GET /v1/models` | Lists `glm-5.3-flash` with its `context_length`. |
| `GET /health` | Status, model, memory and live rates. |
| `GET /metrics` | Prometheus counters, the MTP ones included. |

The server has no API key, and it listens on `0.0.0.0`. Rank 0 logs `non-loopback server has no API key; requests are open`. Keep the port on a trusted network.

```sh
curl -s http://spark1:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "glm-5.3-flash",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "max_tokens": 4096
}'
```

With the `openai` Python client, pass the extra fields in `extra_body`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://spark1:8000/v1", api_key="none")
r = client.chat.completions.create(
	model="glm-5.3-flash",
	messages=[{"role": "user", "content": "What is 17 * 23?"}],
	max_tokens=4096,
	extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(r.choices[0].message.content)
```

### Thinking

Thinking is on by default. The server splits the reply on `</think>`, so the reasoning arrives in `message.reasoning_content` and the answer in `message.content`. Neither field carries the think tags (`tests/smoke.py`, checks `thinking_on` and `thinking_off`).

- To turn thinking off for one request, send `"chat_template_kwargs": {"enable_thinking": false}`. The server also reads `chat_template_kwargs.thinking` as `enable_thinking`.
- A request can set `reasoning_effort`. The server passes it to the chat template.
- Earlier turns keep their reasoning by default. To clear it, send `"chat_template_kwargs": {"clear_thinking": true}`.
- If a reply reaches `max_tokens` while it is still thinking, `content` is empty and rank 0 logs a warning. Give thinking requests a large `max_tokens`.

Sources for these rules are in `evidence/research/research-tensorfold-glm.md`, section 3.

### Vision

Send images as `image_url` parts. A base64 data URL works:

```sh
IMG=$(base64 -w0 shapes.png)
curl -s http://spark1:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "glm-5.3-flash",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG"'"}},
    {"type": "text", "text": "Which shape is on the left, and what color is it?"}
  ]}],
  "chat_template_kwargs": {"enable_thinking": false},
  "max_tokens": 200
}'
```

The image carries libjpeg-turbo 3.1.4.1 and libwebp 1.6.0, the exact decoder versions TensorFold accepts (`docker/Dockerfile`). It has no video library, so video parts do not work (`evidence/image/capabilities-summary.txt`). The pack ships zai-org's `chat_template.jinja` from `f9fed3bb`, which writes the image placeholder. TR3's template does not, and with it TensorFold warns that image requests will fail (`evidence/knobs/tr3-bf16-p4/boot.log`). `tests/smoke.py` draws a red square left of a blue circle and checks that the model names both.

### Speculative decoding stats

MTP drafts are on by default. Rank 0's `serving` line says `drafts: MTP + copies`. Each response carries a top-level `speculative` object with `rounds`, `drafted`, `accepted`, `acceptance_rate` and `tokens_per_round`. Rank 0 logs `rounds=N accepted=a/d` for every request, and `/metrics` counts `tensorfold:mtp_drafted_total` and `tensorfold:mtp_accepted_total` (`evidence/research/research-tensorfold-glm.md`, section 5).

```sh
docker logs tf-glm53-rank0 2>&1 | grep '^\[tensorfold\] done'
```

## Capacity: one shared pool of about 1.05M tokens

`--parallel 8 --context 1048576` does not give 8 windows of a million tokens. It gives one pool of KV pages per rank that all 8 requests share, and one request may fill it. Rank 0 reports (`evidence/run/boot-final.txt`):

```text
model 82.87 GiB; 8 streams at once, the engine's own and 7 more taking 2056 pages of 512 rows (12.80 GiB) as they
need them (enough for one at a 1048576-token window), 340 MiB fixed each, of 17.7 GiB left after a 10.5 GiB reserve
```

- The pool is 2,056 pages of 512 rows, which is 1,052,672 tokens, in 12.80 GiB on each rank.
- That is 13,056 bytes a token on each rank. TP=2 does not halve it, because each rank holds the whole MLA latent (`evidence/decisions.tsv`, row `scope` 15:56:30).
- When requests together need more than the pool, they wait for pages. Three concurrent prompts of 416,767 tokens, 1.25M together, all came back correct, finishing at 459 s, 854 s and 1,208 s. Nothing failed, and the pool was whole again afterwards (`evidence/run/pool-overcommit/`, `rank0-log-excerpt.txt`).
- Host MemAvailable stayed at 10.4 GiB or more on the head and 13.8 GiB on the worker across the bench and the 1M needle, and at 8.8 GiB or more on the head during the overcommit (`evidence/knobs/d8h8-p8-1m-r105/memavail.txt`, `evidence/run/pool-overcommit/memavail.txt`).

At the engine's default 12.2 GiB reserve, the 1M window does not fit beside 8 streams. The engine refuses at load. A 1,048,576-token window is 2,048 pages, 12.75 GiB at 13,056 bytes a token, and the other 7 streams need 2.36 GiB more, against 13.77 GiB of budget (`evidence/knobs/d8h8-p8-1m/boot.log`). That is why the recipe sets the reserve to 10.5.

FP8 KV would roughly double the pool, but stock 1.2.2 refuses `--kv-cache fp8` on its paged pool (`Fp8OnPages`, `zig/src/families/glm/cuda_start.zig:51`), and upstream `main` at `85e53093` still does (`evidence/research/tf-source-lines.txt`). MiaAI-Lab's GLM recipe reaches about 2.1M tokens of FP8 pages with a TensorFold fork of 105 patches (`evidence/decisions.tsv`, row `scope` 16:00:27). This recipe uses the stock release.

## Speed knobs

Each knob had to pass two gates before it became a default: an engine NLL within 0.002 nats a token of the BF16 default, and a measured speed or capacity gain. The NLL gate scores 41 chunks of `nll-corpus-v1` (178,793 tokens of prose, multilingual text, code, math and one long document) through the served engine with `prompt_logprobs`, and pairs two configs token by token (`tools/nll.py`). Two fresh boots of the BF16 default gave bitwise-identical logprobs, so the engine is deterministic and any change is the knob's (`evidence/nll/aa.txt`).

| Config | NLL change, nats/token (95% CI) | 1 stream | 4 streams | 8 streams | Verdict |
|---|---|---|---|---|---|
| BF16 dense, `--parallel 4`, 524,288 window | 0 | 25.8, 27.0 | 41.9, 43.1 | 42.8, 44.0 (TTFT 20 s, queued) | the baseline, two boots |
| `DENSE_BITS=8` | +0.00031 (-0.00072, +0.00129) | 32.6 | 49.0 | 48.5 | pass |
| `DENSE_BITS=4` | +0.03973 (+0.03224, +0.04590) | 39.0 | 51.6 | 51.3 | **fails** quality |
| `DENSE_BITS=8 HEAD_BITS=8` | +0.00034 (-0.00069, +0.00131) | 34.6 | 47.4 | 48.7 | pass |
| the same, `--parallel 8` | +0.00034, the same logprobs | 33.4 | 49.1 | 56.1 (TTFT 0.85 s) | pass |
| the same, 1,048,576 window, 10.5 GiB reserve | +0.00034, the same logprobs | 33.8 | 45.5 | 58.8 | pass, **the default** |
| `--kv-cache fp8` | | | | | refused by 1.2.2 |

Speeds are aggregate tok/s, medians of 5 runs of 512 tokens with thinking off (`tools/bench.py`). Each row is one boot except the baseline, which ran on two boots. Per-run numbers, ranges and boot logs are in `evidence/knobs/<config>/` and `evidence/bench/`. At 4 streams the default's range (44.6 to 48.8) overlaps the baseline's (39.7 to 45.8), so call that gain small.

## Quality

Two measures, and they disagree about TR3, the one other pack in this layout.

**Panel KLD.** Full-vocabulary KLD(BF16 || candidate) on a panel of 25 windows of 2,048 tokens, 51,175 positions. The BF16 side is `malaiwah/glm53-flash-fidelity-root-v1@bdd25fe0`. Both packs were captured with `malaiwah/quant-fidelity-suite@cf36287c` plus our two patches on the same GB10, a transformers forward one decoder layer at a time with the EXL3 experts decoded to BF16, and scored on CPU in float64. `quality/README.md` has every command and pin.

| Pack | KLD, nats | Top-1 agreement | Evidence |
|---|---|---|---|
| Ours | 0.029440 | 0.9488 | `evidence/quality/ours-gb10-vs-root.receipt.json` |
| TR3 `brandonmusic/GLM-5.3-Flash-tr3-4bpw@a5fee929` | 0.025558 | 0.9529 | `evidence/quality/tr3-gb10-vs-root.receipt.json` |

Paired window by window, ours is 0.00388 nats higher (stdev 0.0081 over 25 windows, about 2.4 standard errors) and lower in 4 of 25 windows (`evidence/quality/ours-vs-tr3-gb10.json`). On this panel TR3 is better. The panel is the 25 held-out final windows of the corpus that TR3 was calibrated on, 640 rolling windows of it (`evidence/research/research-exl3-quant.md`, section 2). Ours used exllamav3's default calibration mix.

**Engine NLL on an independent corpus.** Both packs through the same engine, image and flags, BF16 dense, scored on `nll-corpus-v1` (`evidence/knobs/tr3-bf16-p4/nll.txt`):

| Group | TR3 minus ours, nats/token | 95% CI |
|---|---|---|
| All | -0.00030 | -0.00343 to +0.00363 |
| Prose | +0.00057 | -0.00626 to +0.00615 |
| Multilingual | -0.00312 | -0.01121 to +0.00508 |
| Code | +0.00551 | -0.00016 to +0.01198 |
| Math | -0.00067 | -0.00385 to +0.00434 |

On independent text we detect no difference between the two packs, though the CI is wider than our 0.002 gate, so the tool calls it inconclusive. On the same engine, ours ran faster than TR3 at 4 streams, 42 to 43 against 38.3 tok/s with disjoint ranges. At 1 stream the ranges overlap, 25.8 to 27.0 against 24.3 (`evidence/knobs/tr3-bf16-p4/bench.json`, one boot of TR3).

Notes on the method:

- The capture needs the root's `chat_template.jinja` under `.reference/` in its tokenizer root, because our pack ships zai's newer template. The panel is token ids and never applies the template, and our panel binding is byte-identical to TR3's (`evidence/decisions.tsv`, row `quality`, the `panel_unresolvable` fix).
- Do not compare these KLDs with numbers from other stacks. The same TR3 bytes read 0.022802 from an H200 capture and 0.025558 from our GB10 capture (`evidence/quality/tr3-gb10-vs-h200.json`).
- TR3's model-card KLD of 0.0246 was scored against a BF16 teacher-logits repo that no longer exists (`evidence/decisions.tsv`, row `quality` 14:38:01).
- TensorFold's `prompt_logprobs` returns at most the top 20 tokens, and that bound loses 15% to 22% of the KL, by an amount that varies per window (`evidence/quality/tr3-vs-root.window-0.json`, `window-4.json`). So the engine gate scores NLL, not KL.

## How the quant was made

`quant/` holds every step. The model card (`quant/MODEL_CARD.md`) has the commands.

1. **Source.** `zai-org/GLM-5.3-Flash-BF16@f9fed3bb`. The BF16 repo is not an upcast of the FP8 one. Its 128×128 blocks hold 405 to 570 distinct values, more than the 254 an FP8 block allows (`evidence/research/research-exl3-quant.md`, section 4).
2. **Encoder.** Stock exllamav3 v1.6.6 (`bd1f5191`), built for sm_121a in `quant/Dockerfile`. Its smoke test passes on both Sparks (`evidence/quant/smoke-quantize-spark1.txt`, `smoke-quantize-spark2.txt`).
3. **Recipe.** `quant/experts4.yaml` puts the routed experts at 4 bits and everything else at 16, which means unquantized. `quant/convert.sh` adds `--codebook mcg --head_bits 16 --mtp_bits 4 --vision_bits 16 --embed_bits 16 --cal_rows 250 --cal_cols 2048`. It runs with `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` unset, because the NGC base forces TF32 otherwise. It passes long option names only, because the base's `nvidia-cutlass-dsl` parses `sys.argv` and read `-hb` as `-h`, which exited 0 at the first MoE layer (`evidence/quant/probe/run1-cutlass-argparse-exit.log`).
4. **`--moe_affinity_eps 0.01`.** We encoded MoE layer 3 both ways with the same calibration. Its output error fell from rfn 0.02227 to 0.01330 and its SQNR rose from 50.78 dB to 53.31 dB, for 7.6% more time (without the flag `evidence/quant/probe-a-layers-3-4.txt`, with it `evidence/quant/probe/probe-b-log.txt`). The full encode kept it. Layers 4 to 45 then took 14.1 hours of wall time on one GB10 after the probe's layers 0 to 3 (`evidence/quant/full-encode-b.times.txt`; the per-layer times in `full-encode-b.log` add up to 14.0 hours).
5. **Assembly.** Stock exllamav3 output does not load in TensorFold. It writes `bits` as `4.0`, fuses `qkv_proj`, and stores unquantized linears as F16 (`evidence/research/research-exl3-quant.md`, section 2). So `convert.sh` points the stock output at a 1 GiB tmpfs and keeps only `qtensors/`. `quant/assemble.py` then builds one output shard per source shard, copies every non-expert tensor byte for byte from the zai-org source, inserts the encoded expert parts, and writes `quantization_config` as `{"quant_method": "exl3", "bits": 4, "codebook": "mcg", "head_bits": 16}`. `assemble.py verify` found all 1,618 non-expert tensors sha256-equal to the source and all 148,608 expert parts byte-equal to `qtensors/` (`evidence/quant/full/verify.txt`).
6. **Layout check.** `quant/check_layout.py` asserts TensorFold 1.2.2's contract offline, each rule tied to the TensorFold source line that enforces it (`quant/README-contract.md`). It reports `RESULT PASS` on our pack (`evidence/quant/check-layout-ours.txt`), passes on TR3, and fails on a broken copy of TR3 (`evidence/quant/check-layout-tr3-pass.txt`, `check-layout-tr3-broken-fail.txt`).

## Known limits

- **TR3 scores better on its own calibration domain.** See [Quality](#quality). On independent text we detect no difference, within a CI of about ±0.0035 nats a token.
- **One shared KV pool.** About 1.05M tokens across all 8 requests on each rank. See [Capacity](#capacity-one-shared-pool-of-about-105m-tokens).
- **Long prompts are slow to fill.** The 1,045,624-token needle took 1,220 s, about 857 tok/s of prefill (`evidence/knobs/d8h8-p8-1m-r105/needle.jsonl`). The engine keeps no prompt state past its prompt-cache budget, so a second turn on a conversation that long fills from the start again. Rank 0 logged `prompt cache: kept nothing at 415744 tokens: 5358 MiB passes the 2665 MiB budget` for a 416k prompt, transcribed from its log (`evidence/run/pool-overcommit/rank0-log-excerpt.txt`).
- **KV is BF16.** FP8 KV on pages is refused by stock 1.2.2.
- **No API key.** Requests to port 8000 are open.
- **No video.** The image has no `libtfvideo`.
- **DFlash2 is not used.** In 1.2.2 it reads stale prompt taps at TP>1, fixed after the release, and its weights are CC BY-NC-ND (`evidence/decisions.tsv`, row `frame` 12:48:26).
- **Tool-call arguments are not streamed.** Streaming GLM tool arguments arrived after 1.2.2 (`evidence/research/research-tensorfold-glm.md`, section 3).
- **No `max` reasoning effort on the command line.** The native CLI takes `low`, `medium`, `high` and `xhigh`. A per-request `"max"` was not traced (`evidence/research/research-tensorfold-glm.md`, Unknowns).
- **The archive signature is not checked.** The image build trusts the release archive's sha256. Upstream documents no key for the `.sig` (`evidence/decisions.tsv`, row `image` 13:24:53).
- **If one rank fails, the other stops.** Restart both with `./stop.sh && ./run.sh`.

## Troubleshooting

Run `VALIDATE_ONLY=1 ./run.sh` first. Each `REFUSE` line names the cause.

- **`the pack is not pinned`.** `REVISION`, `SHARDS` or `SHARD_BYTES` in `recipe.conf` or the environment is not a pinned value.
- **`cannot ssh spark2 (BatchMode)`.** Set up key-based ssh from the head to `WORKER`. Test it with `ssh -o BatchMode=yes spark2 true`.
- **`runs GPU container NAME IMAGE`.** Another job holds a GPU. Stop it with its own stop script. `run.sh` never stops it.
- **`MemAvailable below the 109 GiB gate`.** `run.sh` prints `free -h` for both nodes. Free memory without `sudo` or cache drops, or set `FORCE=1` and accept a possible load failure.
- **`--parallel 8 needs ... GiB for its other streams' fixed parts`** at load. The node had less free memory than at our boots. Lower `PARALLEL` or `CONTEXT`.
- **`port(s) ... already listen`.** A stale rank or another server holds the port. A stale rank on `NCCL_PORT+1` makes the start fail with `BindFailed`. Find it with `ss -ltnp`.
- **`RoCE HCA not ACTIVE` or `IFACE ... is not up`.** Check the cable and `rdma link show` on both nodes.
- **`rank 1 exited` or `rank 0 exited` while waiting.** `run.sh` prints both logs. A `RankMismatch` line means the ranks differ in settings, model, binary or checkpoint digest. Check that both image IDs and both snapshots match.
- **`no /health after 1200s`.** The ranks must connect within 5 minutes and be ready within 15 (`evidence/research/research-tensorfold-glm.md`, section 5). Read both logs. Raise `READY_TIMEOUT` only if the weights are still loading.
- **HTTP 400 `maximum context length is 1048576 tokens`.** The rendered prompt, chat template included, leaves no room for a reply. Shorten it.
- **Empty `content` with thinking on.** The reply reached `max_tokens` while it was still thinking. Raise `max_tokens`, or turn thinking off for that request.
- **Startup warning `the chat template writes no image placeholder`.** The snapshot has the wrong `chat_template.jinja`. Our pack ships zai-org's. Check the snapshot against the published revision.

## Licenses and credits

This recipe's own files are MIT (`LICENSE`). `NOTICE` lists what comes from elsewhere.

- **GLM-5.3-Flash.** The weights are Z.ai's, MIT (`zai-org/GLM-5.3-Flash-BF16`). Our pack copies every non-expert tensor and the `LICENSE` file from that repo, and is MIT too.
- **exllamav3.** turboderp, MIT. It encodes the experts. The quantizer image builds it from git at `bd1f5191`. No exllamav3 code is in this repo.
- **TensorFold.** ash hart and the TensorFold contributors, Apache-2.0. `docker/Dockerfile` installs the 1.2.2 release archive. Its `NOTICE` and `THIRD_PARTY_NOTICES.md` ship inside the archive.
- **quant-fidelity-suite.** Michel Belleau (malaiwah), MIT. `quality/Dockerfile` builds it from git at `cf36287c`. `quality/patches/` holds our two patches against it, with a few of its lines as context. The reference dataset `malaiwah/glm53-flash-fidelity-root-v1` is MIT.
- **DFlash2.** `incoai/GLM-5.3-Flash-DFlash2` is CC BY-NC-ND. This recipe does not download or use it.
- **TR3.** `brandonmusic/GLM-5.3-Flash-tr3-4bpw` is under its own attribution license. We used it only as a measured control, served and captured from the author's repo. This repo and our pack hold none of its bytes and none of its code (`evidence/research/research-exl3-quant.md`, section 2).

## For developers

**Tests.** All CPU, no GPU, no network:

```sh
python3 -m unittest discover -s tests -v                # run.sh and stop.sh on a stubbed pair, smoke, tools, nll
python3 -I quant/test_check_layout.py <empty dir>       # the layout checker: one pass, one FAIL per rule
python3 -I quant/test_assemble.py <empty dir>           # the assembler on synthetic packs
python3 -I quality/test_paired_delta.py <empty dir>     # one pairing and four refusals
shellcheck -x -S warning run.sh stop.sh
```

The unittest suite runs 58 tests (`evidence/run/cpu-tests.txt`).

**Measuring.** After `./run.sh`:

```sh
python3 tests/smoke.py --out evidence/run/smoke.json
python3 tools/bench.py --concurrency 1,4,8 --runs 5 --out evidence/run/bench.json
python3 tools/needle.py 1000000 --out evidence/run/needle.json
python3 tools/nll.py capture --label mine --out DIR        # then: tools/nll.py compare A_DIR B_DIR
```

`tools/bench.py` streams prose prompts with thinking off and `ignore_eos`. It reports the median stream's decode rate after its first token, the aggregate tokens over each run's wall time, and the TTFT, and it counts errors.

**`evidence/`.** Every number in this README points into it. `evidence/decisions.tsv` is the decision trail in order: one row a decision, with why, the evidence and the result. A later row that supersedes an earlier one says so. `evidence/research/` holds the source reading behind the engine contract, the quant and the KLD method. `evidence/engine-probe/` holds the first boots on TR3. `evidence/knobs/` holds one directory per serving config with its boot log, NLL comparison and bench. `evidence/image/`, `evidence/quant/`, `evidence/quality/`, `evidence/nll/`, `evidence/bench/` and `evidence/run/` hold the receipts of each part.

**`quant/`.** The quantizer image, the recipe, the launcher, the qtensors mover for a short disk, the assembler, the layout checker and their tests. `quant/README-contract.md` is TensorFold's layout contract with its source lines.

**`quality/`.** The fidelity image, the two suite patches, `capture.sh`, `compare.sh` and `paired_delta.py`, plus the memory probes behind patch 0002. `quality/README.md` is the full procedure.

**`tools/nll.py` needs a local corpus.** It reads each corpus source from `~/projects/data/glm53-evals/corpus-src/`, one file a source URL, named by the percent-encoded URL. It checks every file against a pinned sha256. It does not download anything. A missing file stops it with `missing corpus source PATH (fetch URL)`, and a changed file stops it with both hashes. `python3 tools/nll.py corpus --corpus-root DIR` prints the resolved manifest without a server. `evidence/quality/nll-tool/corpus-v1.manifest.jsonl` is that manifest for `nll-corpus-v1`.

**Changing the engine.** Bump `TF_VERSION` and `TF_SHA256` in `recipe.conf` and the matching `ARG` lines in `docker/Dockerfile`. A test checks that the two agree. A new engine needs new evidence, so measure again, starting with the NLL A/A floor.
