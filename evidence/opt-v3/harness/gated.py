"""gated.py NAME [--plan c1:6,c8:3,g1:3,p32:2]: measure the serving pair one run at a time, each run only while the
head is quiet, and rerun any run the head disturbed. Writes runs/NAME/gated.json and one ledger line.

Quiet: everything but the engine used under QUIET cores over the last 5 s. A run is clean when the mean of that over
the run stays under CLEAN cores, no 5 s sample passes BURST, and nothing was swapped out. Up to TRIES tries a run;
the dirty ones are kept and marked. Medians are over clean runs only (all runs if none was clean).
"""
import glob, json, os, statistics, subprocess, sys, tempfile, threading, time
from pathlib import Path
W = Path(__file__).resolve().parent
R = Path.home() / "projects/ai-lab/recipes/GLM-5.3-Flash-EXL3-4bpw-TensorFold-2x-DGX-SPARK"
HZ = os.sysconf("SC_CLK_TCK"); QUIET, CLEAN, BURST = 1.0, 1.2, 4.0
TRIES, WAIT = int(os.environ.get("GATED_TRIES", 1)), int(os.environ.get("GATED_WAIT", 10))  # 1 try, no long wait: tag, don't gate

def cpu():
    f = open("/proc/stat").readline().split()[1:]
    tot = (int(f[0]) + int(f[1]) + int(f[2]) + int(f[5]) + int(f[6])) / HZ
    eng = 0.0
    for p in glob.glob("/proc/[0-9]*/stat"):
        try: s = open(p).read()
        except OSError: continue
        if s[s.index("(") + 1:s.rindex(")")].startswith("tensorfold"):
            x = s[s.rindex(")") + 2:].split(); eng += (int(x[11]) + int(x[12])) / HZ
    return tot - eng
def swapout():
    return next(int(l.split()[1]) for l in open("/proc/vmstat") if l.startswith("pswpout "))
def gpu():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=clocks.current.graphics,power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=4).stdout.split(",")
        return {"mhz": int(o[0]), "w": float(o[1]), "gpu_c": int(o[2])}
    except Exception: return {}

def capped_us():
    """Microseconds the driver has held the GPU clock down for power, ever (nvidia-smi's SW Power Capping counter)."""
    try:
        out = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"], capture_output=True, text=True, timeout=4).stdout
        return next(int(l.split(":")[1].split()[0]) for l in out.splitlines() if "SW Power Capping" in l)
    except Exception:
        return 0


class Watch(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.stop = threading.Event(); self.samples = []; self.sw0 = swapout(); self.g = []; self.k0 = capped_us()
    def run(self):
        c0, t0 = cpu(), time.monotonic()
        while not self.stop.wait(5):
            c1, t1 = cpu(), time.monotonic(); self.samples.append((c1 - c0) / (t1 - t0)); c0, t0 = c1, t1; self.g.append(gpu())
    def end(self):
        self.stop.set(); self.join()
        s = self.samples or [0.0]; mhz = [x["mhz"] for x in self.g if x]
        return {"other": round(sum(s) / len(s), 2), "burst": round(max(s), 2), "swapout": swapout() - self.sw0, "capped_ms": round((capped_us() - self.k0) / 1000),
                "mhz": round(statistics.median(mhz)) if mhz else None, "gpu_c": max((x["gpu_c"] for x in self.g if x), default=None)}

def wait_quiet():
    t0 = time.monotonic()
    while True:
        c0 = cpu(); time.sleep(5); o = (cpu() - c0) / 5
        if o < QUIET or time.monotonic() - t0 > WAIT: return round(time.monotonic() - t0), round(o, 2)

def bench(c, greedy, code=False):
    with tempfile.TemporaryDirectory() as d:
        cmd = ["python3", str(R / "tools/bench.py"), "--concurrency", str(c), "--runs", "1", "--warmup", "0", "--out", f"{d}/b.json"]
        if greedy: cmd += ["--temperature", "0"]
        if code: cmd += ["--prompts", "code"]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        r = json.load(open(f"{d}/b.json"))["runs"][0]
    sp = [s["speculative"] for s in r["streams"] if s.get("speculative")]
    rounds = sum(s["rounds"] for s in sp) / max(len(sp), 1)
    dec = statistics.median(s["tokens"] / s["decode_tok_s"] for s in r["streams"])
    return {"tok_s": r["aggregate_tok_s"], "errors": r["errors"], "ttft": statistics.median(s["ttft_s"] for s in r["streams"]),
            "ms_round": round(1000 * dec / rounds, 1) if rounds else None,
            "tok_round": round(sum(s["tokens"] for s in r["streams"]) / max(sum(s["rounds"] for s in sp), 1), 2),
            "accept": round(sum(s["accepted"] for s in sp) / max(sum(s["drafted"] for s in sp), 1), 3)}

def prefill(n):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["python3", str(R / "tools/prefill.py"), str(n), "--out", f"{d}/p.json"], capture_output=True, text=True)
        r = json.load(open(f"{d}/p.json"))["runs"][0]
    return {"tok_s": r.get("prefill_tok_s"), "ttft": r.get("ttft_s"), "prompt_tokens": r.get("prompt_tokens"), "errors": 0 if r.get("cold") else 1}

def main():
    name = sys.argv[1]; plan = "c1:6,c8:3,g1:3,p32:2"
    if "--plan" in sys.argv: plan = sys.argv[sys.argv.index("--plan") + 1]
    out = W / "runs" / name; out.mkdir(parents=True, exist_ok=True)
    bench(8 if "c8" in plan else 1, False)  # warmup
    res = {}
    for item in plan.split(","):
        key, n = item.split(":"); runs = []
        for _ in range(int(n)):
            for attempt in range(TRIES):
                waited, o = wait_quiet(); w = Watch(); w.start()
                # c sampled prose, g greedy prose, k sampled code, j greedy code, p cold prefill (thousands of tokens)
                r = prefill(int(key[1:]) * 1000) if key[0] == "p" else bench(int(key[1:]), key[0] in "gj", key[0] in "kj")
                r.update(w.end()); r["waited"] = waited
                r["clean"] = r["other"] < CLEAN and r["burst"] < BURST and r["swapout"] == 0 and not r["errors"]
                runs.append(r); print(key, json.dumps(r), flush=True)
                if r["clean"]: break
        res[key] = runs
    summ = {}
    for key, runs in res.items():
        clean = [r for r in runs if r["clean"]]; use = clean or runs
        summ[key] = {"n_clean": len(clean), "n": len(runs), "tok_s": round(statistics.median(r["tok_s"] for r in use if r["tok_s"]), 2),
                     "best": max(r["tok_s"] for r in runs if r["tok_s"]),
                     **({"ms_round": statistics.median(r["ms_round"] for r in use), "tok_round": statistics.median(r["tok_round"] for r in use),
                         "accept": statistics.median(r["accept"] for r in use)} if key[0] != "p" else {})}
    env = W / "configs" / f"{name}.env"
    row = {"name": name, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gated": True,
           "overrides": env.read_text().split("\n")[:-1] if env.exists() else None, "summary": summ}
    (out / "gated.json").write_text(json.dumps({"row": row, "runs": res}, indent=1) + "\n")
    with open(W / "ledger.jsonl", "a") as f: f.write(json.dumps(row) + "\n")
    print("SUMMARY", name, json.dumps(summ))

if __name__ == "__main__":
    main()
