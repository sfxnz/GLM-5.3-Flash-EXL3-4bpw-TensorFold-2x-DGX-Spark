#!/usr/bin/env python3
# Runs paired_delta.py on synthetic compare receipts: known per-window deltas, then each pairing refusal.
# Usage: test_paired_delta.py WORKDIR
import json
import os
import subprocess
import sys

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paired_delta.py")


def receipt(label, means, ref="r" * 64, backend="numpy:cpu:float32", rows=2047):
    per = [{"index": i, "window_id": "w%d" % i, "scored_rows": rows, "mean": m} for i, m in enumerate(means)]
    return {"reference": {"dataset_sha256": ref}, "candidate": {"label": label},
            "panel": {"suite_token_hash_sha256": "p" * 64},
            "comparator": {"replay_backend": backend, "vocab_chunk": 8192},
            "metric": {"value": sum(means) / len(means)}, "per_context": per}


def run(work, name_a, a, name_b, b):
    for name, doc in ((name_a, a), (name_b, b)):
        os.makedirs(os.path.join(work, name), exist_ok=True)
        json.dump(doc, open(os.path.join(work, name, "comparison-receipt.json"), "w"))
    out = os.path.join(work, name_a + "-vs-" + name_b + ".json")
    p = subprocess.run([sys.executable, TOOL, os.path.join(work, name_a), os.path.join(work, name_b), "--out", out],
                       capture_output=True, text=True)
    return p, (json.load(open(out)) if p.returncode == 0 else None)


def main():
    work = sys.argv[1]
    p, r = run(work, "ours", receipt("ours", [0.01, 0.03, 0.02]), "tr3", receipt("tr3", [0.02, 0.02, 0.04]))
    assert p.returncode == 0, p.stderr
    assert [round(w["delta"], 12) for w in r["per_window"]] == [-0.01, 0.01, -0.02], r["per_window"]
    assert abs(r["window_mean_delta"] - (-0.02 / 3)) < 1e-12, r
    assert abs(r["token_mean_delta"] - (0.02 - 0.08 / 3)) < 1e-12, r
    assert r["windows_a_lower"] == 2 and r["windows"] == 3 and (r["a"], r["b"]) == ("ours", "tr3"), r
    checks = (("other reference", receipt("x", [0.1] * 3, ref="s" * 64), "reference.dataset_sha256"),
              ("other replay backend", receipt("x", [0.1] * 3, backend="torch:cuda:float32"), "comparator.replay_backend"),
              ("missing window", receipt("x", [0.1] * 2), "different windows"),
              ("other scored rows", receipt("x", [0.1] * 3, rows=2046), "different windows"))
    for what, doc, message in checks:
        p, _ = run(work, "ours", receipt("ours", [0.1] * 3), "bad", doc)
        assert p.returncode != 0 and message in p.stderr, (what, p.returncode, p.stderr)
    print("test_paired_delta: 1 pairing and %d refusals OK" % len(checks))


if __name__ == "__main__":
    main()
