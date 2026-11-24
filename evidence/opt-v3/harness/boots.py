"""boots.py NAME...: one line a boot from runs/NAME/: engine, seconds to ready, the checkpoint digest line, the memory
budget, the pool's pages and the kept states the engine reported (tab-separated)."""
import re, sys
from pathlib import Path
W = Path(__file__).resolve().parent
def find(p, text, d=""):
    m = re.search(p, text); return m.group(1) if m else d
print("run\tengine\tready_s\tdigest\tbudget_gib\tpages\tkept_of_16\tnote")
for n in sys.argv[1:]:
    d = W / "runs" / n
    run = (d / "run.txt").read_text(); log = (d / "boot.log").read_text() if (d / "boot.log").exists() else run
    kept = find(r"keeps (\d+) states", log) or ("16" if "pages of 512" in log else "")
    print("\t".join([n, find(r"TensorFold ([\d.]+),", run), find(r"ready after (\d+)s", run, "refused"),
                     find(r"checkpoint digest: (.*?cached files)", log), find(r"of ([\d.]+) GiB left", log),
                     find(r"taking (\d+) pages", log), kept, find(r"\ntensorfold: (.*)", run) or find(r"\nerror: (.*)", run)]))
