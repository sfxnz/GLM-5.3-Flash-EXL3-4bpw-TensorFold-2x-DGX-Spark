"""Load and free TR3 layers through the suite's streamer, no forwards; report host memory per layer.

usage: loadloop.py <pack_dir> <first_layer> <last_layer> [--trim] [--digest OUT.json] [--windows N]
With --windows N (needs first_layer 0) each layer's output for panel windows 0..N-1 is hashed too,
pushed through the layers as the suite's run_panel does.
"""
import ctypes
import gc
import hashlib
import json
import sys
import time

sys.path[:0] = ["/opt/fidelity/suite/bin", "/opt/fidelity/suite/engines/tools"]
import torch  # noqa: E402
import transformers  # noqa: E402

import hf_capture  # noqa: E402
import layer_outer  # noqa: E402

libc = ctypes.CDLL("libc.so.6")


class Mallinfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in
                ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks",
                 "uordblks", "fordblks", "keepcost")]


libc.mallinfo2.restype = Mallinfo2


def status():
    out = {}
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith(("RssAnon", "RssFile", "RssShmem")):
                key, value = line.split(":")
                out[key] = int(value.split()[0]) * 1024
    return out


def cpu_tensors():
    count = total = 0
    seen = set()
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.device.type == "cpu" and not obj.is_meta:
                ptr = obj.untyped_storage().data_ptr()
                if ptr in seen:
                    continue
                seen.add(ptr)
                count += 1
                total += obj.untyped_storage().nbytes()
        except Exception:
            pass
    return count, total


def smaps(index):
    rows, cur = [], None
    with open("/proc/self/smaps") as handle:
        for line in handle:
            head = line.split()
            if "-" in head[0] and not head[0].endswith(":"):
                lo, hi = (int(x, 16) for x in head[0].split("-"))
                cur = {"range": head[0], "perms": head[1], "name": head[5] if len(head) > 5 else "", "size": hi - lo}
                rows.append(cur)
            elif head[0] in ("Anonymous:", "Rss:", "Locked:"):
                cur[head[0][:-1]] = int(head[1]) * 1024
    by = {}
    for r in rows:
        key = r["name"] or "[anon:%s]" % r["perms"]
        agg = by.setdefault(key, [0, 0, 0])
        agg[0] += 1; agg[1] += r.get("Anonymous", 0); agg[2] += r["size"]
    top = sorted(by.items(), key=lambda kv: -kv[1][1])[:8]
    big = sorted((r for r in rows if r.get("Anonymous", 0) > 64 << 20), key=lambda r: -r["Anonymous"])[:12]
    print("SMAPS " + json.dumps({"layer": index, "top": top, "big": big}), flush=True)


def snapshot(tag, index):
    gc.collect()
    m = libc.mallinfo2()
    count, total = cpu_tensors()
    host = {}
    try:
        stats = torch.cuda.host_memory_stats()
        host = {k: stats[k] for k in stats if k.endswith(("allocated_bytes.current", "reserved_bytes.current"))
                or k in ("allocations.current", "segment.current")}
    except Exception as exc:
        host = {"error": str(exc)}
    row = dict(tag=tag, layer=index, **status(),
               malloc_in_use=m.uordblks, malloc_free=m.fordblks, malloc_mmapped=m.hblkhd,
               malloc_arena=m.arena, cpu_tensors=count, cpu_tensor_bytes=total,
               cuda_allocated=torch.cuda.memory_allocated(), cuda_reserved=torch.cuda.memory_reserved(),
               host_alloc=host)
    print("MEM " + json.dumps(row), flush=True)


def main():
    pack, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    trim = "--trim" in sys.argv
    digest_out = sys.argv[sys.argv.index("--digest") + 1] if "--digest" in sys.argv else None
    config, cls, _ = hf_capture._model_config(pack)
    if cls is None:
        cls = getattr(transformers, config.architectures[0])
    log = lambda **kw: print(json.dumps(kw, default=str)[:300], flush=True)  # noqa: E731
    streamer = layer_outer.build_streamed_model(pack, cls, config, "bfloat16", "cuda", log)
    snapshot("built", -1)
    digests = {}
    nwin = int(sys.argv[sys.argv.index("--windows") + 1]) if "--windows" in sys.argv else 0
    if nwin:
        import numpy as np
        panel = hf_capture.load_panel("/opt/fidelity/suite/engines/panels/panel--glm53.brandonmusic.final25",
                                      None, nwin, None, int(config.get_text_config().vocab_size))
        schedule = layer_outer.LayerOuterSchedule(streamer.layers).install()
        memo = [None] * nwin
    for index in range(first, last + 1):
        started = time.monotonic()
        streamer.load_layer(index)
        torch.cuda.synchronize()
        snapshot("loaded", index)
        for w in range(nwin):
            window = panel.windows[w]
            schedule.active, schedule.replay, schedule.captured, schedule.calls = index, memo[w], None, 0
            ids = torch.tensor([window["token_ids"]], dtype=torch.long, device="cuda")
            mask = torch.from_numpy(np.asarray(window["mask"], dtype=np.int64).reshape(1, -1)).to("cuda")
            try:
                with torch.inference_mode():
                    streamer.model(input_ids=ids, attention_mask=mask, use_cache=False)
            except layer_outer._Suspend:
                pass
            assert schedule.calls == 1, schedule.calls
            memo[w] = schedule.captured
            parts = memo[w] if isinstance(memo[w], (tuple, list)) else (memo[w],)
            h = hashlib.sha256()
            for part in parts:
                if isinstance(part, torch.Tensor):
                    h.update(part.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
            digests.setdefault("hidden", {})["%d/%d" % (index, w)] = h.hexdigest()
        if digest_out:
            head = "%s.%d." % (streamer.layers_prefix, index)
            layer = {}
            for name, param in streamer.model.named_parameters():
                if name.startswith(head):
                    flat, h = param.detach().contiguous().view(torch.uint8).reshape(-1), hashlib.sha256()
                    for at in range(0, flat.numel(), 1 << 28):
                        h.update(flat[at:at + (1 << 28)].cpu().numpy().tobytes())
                    layer[name] = h.hexdigest()
            digests[str(index)] = layer
        streamer.free_layer(index)
        if trim:
            libc.malloc_trim(0)
        snapshot("freed", index)
        smaps(index)
        print("LAYER %d %.1fs" % (index, time.monotonic() - started), flush=True)
    if digest_out:
        with open(digest_out, "w") as handle:
            json.dump(digests, handle, sort_keys=True, indent=0)


main()
