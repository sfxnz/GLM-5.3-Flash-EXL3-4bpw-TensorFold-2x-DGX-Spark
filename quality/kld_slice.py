#!/usr/bin/env python3
"""Paired full-vocab KLD(reference || candidate) on a slice of two hidden-form fidelity datasets.

CPU, numpy only. Reads the layout of malaiwah/glm53-flash-fidelity-* (fidelity-dataset spec v1):
capture/hidden_NNNN.safetensors [2047, 4096] bf16 after the final norm, head/weight.safetensors
[154880, 4096] bf16, panel/tokens/context-NNNN.json. Logits are hidden @ head.T in fp32, then
log-softmax and KL in fp64 over the full vocabulary: the estimator of quant-fidelity-suite
(MIT, Michel Belleau) bin/fidelity/dscompare.py token_kld, written independently here.

It also scores what a TensorFold /v1/completions prompt_logprobs=K reply could give for the
candidate (bf16 logits, top K plus the prompt token), to show how far that is from the full number.

    python3 -I kld_slice.py REF_DATASET CAND_DATASET --windows 0 4 --out result.json
"""

import argparse
import json
import os
import time

import numpy as np


def read_bf16(path, key):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    meta = header[key]
    if meta["dtype"] != "BF16":
        raise SystemExit("%s: %s is %s, expected BF16" % (path, key, meta["dtype"]))
    start, end = meta["data_offsets"]
    raw = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + n + start, shape=((end - start) // 2,))
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])


def to_bf16(x):
    bits = x.view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.view(np.float32)


def log_softmax64(logits32):
    x = logits32.astype(np.float64)
    x -= x.max(axis=1, keepdims=True)
    x -= np.log(np.exp(x).sum(axis=1, keepdims=True))
    return x


def coarse_kl(lp, lq, ids, valid):
    """KL of P and Q merged to {valid listed ids} + {everything else}: a lower bound on KL(P || Q)."""
    lp_list = np.where(valid, np.take_along_axis(lp, ids, axis=1), -np.inf)
    lq_list = np.where(valid, np.take_along_axis(lq, ids, axis=1), -np.inf)
    p_list = np.exp(lp_list)
    p_rest = np.clip(1.0 - p_list.sum(axis=1), 0.0, None)
    q_rest = np.clip(1.0 - np.exp(lq_list).sum(axis=1), 1e-300, None)
    listed = (p_list * np.where(valid, lp_list - np.where(valid, lq_list, 0.0), 0.0)).sum(axis=1)
    rest = np.where(p_rest > 0, p_rest * (np.log(np.maximum(p_rest, 1e-300)) - np.log(q_rest)), 0.0)
    return listed + rest


def score_window(ref_root, cand_root, head_t, w, ks, block):
    name = "capture/hidden_%04d.safetensors" % w
    h_ref = read_bf16(os.path.join(ref_root, name), "hidden_states")
    h_cand = read_bf16(os.path.join(cand_root, name), "hidden_states")
    with open(os.path.join(ref_root, "panel/tokens/context-%04d.json" % w)) as f:
        tokens = np.asarray(json.load(f), dtype=np.int64)
    with open(os.path.join(cand_root, "panel/tokens/context-%04d.json" % w)) as f:
        if not np.array_equal(tokens, np.asarray(json.load(f), dtype=np.int64)):
            raise SystemExit("window %d: reference and candidate panels differ" % w)
    n = h_ref.shape[0]
    if h_cand.shape != h_ref.shape or tokens.shape[0] != n + 1:
        raise SystemExit("window %d: shapes %s %s tokens %d" % (w, h_ref.shape, h_cand.shape, tokens.shape[0]))
    given = tokens[1:]

    kl = np.empty(n)
    kl_bf16 = np.empty(n)
    nll_ref = np.empty(n)
    nll_cand = np.empty(n)
    top1 = 0
    coarse = {k: np.empty(n) for k in ks}
    for s in range(0, n, block):
        e = min(n, s + block)
        lp = log_softmax64(h_ref[s:e] @ head_t)
        cand_logits = h_cand[s:e] @ head_t
        lq = log_softmax64(cand_logits)
        kl[s:e] = (np.exp(lp) * (lp - lq)).sum(axis=1)
        top1 += int((lp.argmax(axis=1) == lq.argmax(axis=1)).sum())
        g = given[s:e, None]
        nll_ref[s:e] = -np.take_along_axis(lp, g, axis=1)[:, 0]
        nll_cand[s:e] = -np.take_along_axis(lq, g, axis=1)[:, 0]

        lq16 = log_softmax64(to_bf16(cand_logits))
        kl_bf16[s:e] = (np.exp(lp) * (lp - lq16)).sum(axis=1)
        part = np.argpartition(-lq16, max(ks) - 1, axis=1)[:, :max(ks)]
        order = np.take_along_axis(part, np.argsort(-np.take_along_axis(lq16, part, axis=1), axis=1), axis=1)
        for k in ks:
            ids = np.concatenate([order[:, :k], g], axis=1)
            valid = np.ones(ids.shape, dtype=bool)
            valid[:, k] = ~(order[:, :k] == g).any(axis=1)
            coarse[k][s:e] = coarse_kl(lp, lq16, ids, valid)
            if (coarse[k][s:e] > kl_bf16[s:e] + 1e-9).any():
                raise SystemExit("window %d: top-%d coarse KL exceeds the full KL it bounds" % (w, k))
    if not (np.isfinite(kl).all() and np.isfinite(kl_bf16).all()):
        raise SystemExit("window %d: non-finite KLD" % w)
    return {
        "window": w,
        "positions": n,
        "kld_full_fp64": float(kl.mean()),
        "kld_full_cand_logits_bf16": float(kl_bf16.mean()),
        "top1_agreement": top1 / n,
        "delta_nll_given_token": float((nll_cand - nll_ref).mean()),
        "kld_coarse_topk_lower_bound": {str(k): float(coarse[k].mean()) for k in ks},
        "coarse_over_full": {str(k): float(coarse[k].mean() / kl.mean()) for k in ks},
        "per_position_corr_coarse_vs_full": {str(k): float(np.corrcoef(coarse[k], kl)[0, 1]) for k in ks},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--windows", type=int, nargs="+", required=True)
    ap.add_argument("--topk", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    head_t = np.ascontiguousarray(read_bf16(os.path.join(a.reference, "head/weight.safetensors"), "lm_head.weight").T)
    cand_head = os.path.join(a.candidate, "head/weight.safetensors")
    if os.path.exists(cand_head) and not np.array_equal(read_bf16(cand_head, "lm_head.weight")[:64], head_t.T[:64]):
        raise SystemExit("candidate head differs from the reference head")
    rows = []
    for w in a.windows:
        r = score_window(a.reference, a.candidate, head_t, w, sorted(a.topk), a.block)
        r["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(r), flush=True)
        rows.append(r)
    total = sum(r["positions"] for r in rows)
    summary = {
        "reference": os.path.abspath(a.reference),
        "candidate": os.path.abspath(a.candidate),
        "windows": rows,
        "token_mean_kld_full_fp64": sum(r["kld_full_fp64"] * r["positions"] for r in rows) / total,
        "estimator": "logits = hidden_fp32 @ head_fp32.T (numpy), fp64 log_softmax, full vocab, mean over positions",
    }
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=1)
    print("token-mean KLD %.6f over %d positions, %d windows, %.0f s" %
          (summary["token_mean_kld_full_fp64"], total, len(rows), time.time() - t0))


if __name__ == "__main__":
    main()
