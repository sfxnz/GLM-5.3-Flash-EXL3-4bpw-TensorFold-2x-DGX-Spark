"""GLM-5.3-Flash token counts of the nll-corpus-v1 chunks, for the runtime estimate (the engine reports the real counts).

~/projects/code/.venv-tf/bin/python evidence/quality/nll-tool/count_tokens.py <tokenizer.json>
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import nll  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

tk = Tokenizer.from_file(sys.argv[1])
per, n = collections.Counter(), collections.Counter()
for c in nll.resolve(nll.CORPUS_ROOT):
    t = len(tk.encode(c["text"], add_special_tokens=False).ids)
    per[c["category"]] += t
    n[c["category"]] += 1
    print(f"{c['id']:>3} {c['category']:<13}{t:>7} tokens  {c['url']}")
for k in per:
    print(f"{k:<13}{n[k]:>3} chunks {per[k]:>7} tokens")
print(f"total        {sum(n.values()):>3} chunks {sum(per.values()):>7} tokens")
