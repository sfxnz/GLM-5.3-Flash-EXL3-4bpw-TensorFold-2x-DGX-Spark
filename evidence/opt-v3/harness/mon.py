"""Every 5 s on this node: CPU seconds used by everything but the engine, the busiest other process, reclaim and
compaction counters, block reads, hottest zone. One JSON line a sample, to correlate slow runs with the host."""
import glob
import json
import os
import sys
import time

HZ = os.sysconf("SC_CLK_TCK")


def cpu_total():
    f = open("/proc/stat").readline().split()[1:]
    return (int(f[0]) + int(f[1]) + int(f[2]) + int(f[5]) + int(f[6])) / HZ, int(f[4]) / HZ


def procs():
    out = {}
    for p in glob.glob("/proc/[0-9]*/stat"):
        try:
            s = open(p).read()
        except OSError:
            continue
        comm = s[s.index("(") + 1:s.rindex(")")]
        f = s[s.rindex(")") + 2:].split()
        out[(p.split("/")[2], comm)] = (int(f[11]) + int(f[12])) / HZ
    return out


def vm():
    keep = ("pgscan_kswapd", "pgscan_direct", "compact_stall", "pgmigrate_success", "pgpgin", "pgpgout", "pswpin", "pswpout")
    return {k: int(v) for k, v in (l.split() for l in open("/proc/vmstat")) if k in keep}


def hot():
    return max(int(open(z).read()) for z in glob.glob("/sys/class/thermal/thermal_zone*/temp")) / 1000


def gpu():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.current.graphics,power.draw,utilization.gpu,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4).stdout
        mhz, w, util, temp = [x.strip() for x in out.split(",")]
        return {"mhz": int(mhz), "w": float(w), "util": int(util), "gpu_c": int(temp)}
    except Exception:
        return {}


def capped_us():
    """Microseconds the driver has held the GPU clock down for power, ever (nvidia-smi's SW Power Capping counter)."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"], capture_output=True, text=True, timeout=4).stdout
        return next(int(l.split(":")[1].split()[0]) for l in out.splitlines() if "SW Power Capping" in l)
    except Exception:
        return 0


def main():
    c0, p0, v0, k0 = cpu_total(), procs(), vm(), capped_us()
    while True:
        time.sleep(5)
        c1, p1, v1 = cpu_total(), procs(), vm()
        d = {k: p1[k] - p0.get(k, 0) for k in p1}
        eng = sum(v for (pid, comm), v in d.items() if comm.startswith("tensorfold"))
        other = sorted(((v, comm) for (pid, comm), v in d.items() if not comm.startswith("tensorfold")), reverse=True)[:3]
        print(json.dumps({"t": time.strftime("%H:%M:%S"), "cpu_all": round(c1[0] - c0[0], 1), "engine": round(eng, 1),
                          "iowait": round(c1[1] - c0[1], 1), "top": [(round(v, 1), c) for v, c in other if v >= 0.3],
                          "vm": {k: v1[k] - v0[k] for k in v1 if v1[k] != v0[k]}, "hot": hot(), **gpu(), "capped_ms": round(((k1 := capped_us()) - k0) / 1000),
                          "free_gib": round(int(open("/proc/meminfo").readlines()[1].split()[1]) / 2**20, 1)}), flush=True)
        c0, p0, v0, k0 = c1, p1, v1, k1


if __name__ == "__main__":
    sys.exit(main())
