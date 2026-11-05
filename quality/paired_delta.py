#!/usr/bin/env python3
# Per-window paired delta between two `fidelity-dataset compare` outputs scored against the same reference.
# Usage: paired_delta.py A_DIR B_DIR [--out FILE]   (each DIR holds comparison-receipt.json)
# delta = KL(ref || A) - KL(ref || B) per window, in nats; negative means A is closer to the reference.
# fidelity-stats paired-delta cannot do this: it reads stream_score kld-report.json (per_window[].summary.mean
# plus teacher_receipt_sha256), and a compare receipt carries per_context[] and no teacher receipt.
import argparse
import json
import os
import statistics

MUST_MATCH = (("reference", "dataset_sha256"), ("panel", "suite_token_hash_sha256"),
              ("comparator", "replay_backend"), ("comparator", "vocab_chunk"))


def paired(a, b):
    for block, key in MUST_MATCH:
        if a[block][key] != b[block][key]:
            raise SystemExit("not paired: %s.%s is %r vs %r" % (block, key, a[block][key], b[block][key]))
    wa = {c["index"]: c for c in a["per_context"]}
    wb = {c["index"]: c for c in b["per_context"]}
    same = lambda i: (wa[i]["window_id"], wa[i]["scored_rows"]) == (wb[i]["window_id"], wb[i]["scored_rows"])
    if wa.keys() != wb.keys() or not all(same(i) for i in wa):
        raise SystemExit("not paired: the two receipts score different windows")
    delta = {i: wa[i]["mean"] - wb[i]["mean"] for i in sorted(wa)}
    d = list(delta.values())
    lanes = [r["candidate"].get(k) for r in (a, b) for k in ("lane_identity_sha256", "stack_fingerprint_sha256")]
    return {"a": a["candidate"]["label"], "b": b["candidate"]["label"], "windows": len(d),
            "same_capture_stack": lanes[:2] == lanes[2:] and None not in lanes,
            "kld_a": a["metric"]["value"], "kld_b": b["metric"]["value"],
            "token_mean_delta": a["metric"]["value"] - b["metric"]["value"],
            "window_mean_delta": statistics.fmean(d), "window_delta_stdev": statistics.stdev(d),
            "windows_a_lower": sum(x < 0 for x in d), "max_window_delta": max(d),
            "per_window": [{"index": i, "window_id": wa[i]["window_id"], "kld_a": wa[i]["mean"],
                            "kld_b": wb[i]["mean"], "delta": delta[i]} for i in delta]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_dir")
    ap.add_argument("b_dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    load = lambda d: json.load(open(os.path.join(d, "comparison-receipt.json")))
    result = paired(load(args.a_dir), load(args.b_dir))
    text = json.dumps(result, indent=1)
    if args.out:
        open(args.out, "w").write(text + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_window"}, indent=1))


if __name__ == "__main__":
    main()
