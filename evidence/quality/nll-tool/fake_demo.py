"""A/A and A/B through tests/fake_server.py on the real nll-corpus-v1: the report format and the CPU cost, not model numbers.

python3 evidence/quality/nll-tool/fake_demo.py SCRATCH_DIR
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))
from fake_server import FakeServer  # noqa: E402

d, nll = sys.argv[1], str(ROOT / "tools/nll.py")
print("Fake TensorFold on the real nll-corpus-v1; the fake's tokens are words, so token counts are not GLM's.")
for label, knobs in [("a", {}), ("a2", {"jitter": 1e-4}), ("b", {"nll_shift": 0.001, "jitter": 0.05})]:
    with FakeServer(**knobs) as s:
        r = subprocess.run([sys.executable, nll, "capture", "--label", label, "--out", f"{d}/{label}", "--url", s.url],
                           capture_output=True, text=True)
    print(f"$ nll.py capture --label {label}   fake knobs {knobs}, rc={r.returncode}\n{r.stdout.strip().splitlines()[-1]}{r.stderr}")
for args in (["a", "a2", "--json", f"{d}/aa.json"], ["a", "b", "--aa", f"{d}/aa.json"], ["a", "b", "--margin", "0.0008"]):
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, nll, "compare", f"{d}/{args[0]}", f"{d}/{args[1]}", *args[2:]], capture_output=True, text=True)
    print(f"\n$ nll.py compare {' '.join(a.replace(d + '/', '') for a in args)}   rc={r.returncode}, {time.monotonic() - t0:.1f} s")
    print(r.stdout + r.stderr, end="")
