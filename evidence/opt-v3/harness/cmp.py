"""cmp.py NAME...: the tagged runs of those panels side by side. A run counts when the head's other work stayed under
6 cores and under 1,000 swapped-out pages while it ran; the cell is the median tok/s of the runs that count, with
(counted/all, best). c/g: prose sampled/greedy, k/j: code sampled/greedy, p: cold prefill; the digits are streams
(or thousands of prompt tokens)."""
import json, statistics, sys
from pathlib import Path
W = Path(__file__).resolve().parent
ok = lambda r: r["other"] < 6 and r["swapout"] < 1000 and not r["errors"]
names = sys.argv[1:]; data = {}
for n in names:
    runs = {}
    for f in sorted((W / "runs" / n).glob("gated*.json")):
        for k, v in json.loads(f.read_text())["runs"].items(): runs.setdefault(k, []).extend(v)
    data[n] = runs
keys = sorted({k for d in data.values() for k in d}, key=lambda k: ("cgkjp".index(k[0]), int(k[1:])))
print(f"{'':12}" + "".join(f"{k:>22}" for k in keys))
for n in names:
    cells = []
    for k in keys:
        rs = data[n].get(k, []); good = [r for r in rs if ok(r)]
        if not rs: cells.append(""); continue
        use = good or rs
        cells.append(f"{statistics.median(r['tok_s'] for r in use):7.1f} ({len(good)}/{len(rs)}, {max(r['tok_s'] for r in rs):.1f})")
    print(f"{n:12}" + "".join(f"{c:>22}" for c in cells))
    if "-v" in sys.argv: pass
print()
for n in names:
    for k in keys:
        rs = [r for r in data[n].get(k, []) if ok(r) and "ms_round" in r]
        if rs: print(f"  {n:10} {k:4} ms/round {statistics.median(r['ms_round'] for r in rs):6.1f}  tok/round {statistics.median(r['tok_round'] for r in rs):.2f}  accept {statistics.median(r['accept'] for r in rs):.3f}")
