import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
params = [(l, n) for l in sorted(set(a) - {"hidden"}, key=int) for n in a[l]]
pe = sum(a[l][n] == b.get(l, {}).get(n) for l, n in params)
he = sum(a["hidden"][k] == b["hidden"].get(k) for k in a["hidden"])
print("layers %s; parameters byte-equal %d/%d; hidden states byte-equal %d/%d (layer/window: %s)" % (
    ",".join(sorted(set(a) - {"hidden"}, key=int)), pe, len(params), he, len(a["hidden"]), " ".join(sorted(a["hidden"]))))
print("same key sets:", set(a) == set(b) and all(set(a[k]) == set(b[k]) for k in a))
