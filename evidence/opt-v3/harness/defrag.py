"""Before a load: fault this node's available memory in as huge pages, then free it. No root.

python3 tools/compact.py [--keep-gib 4]     (run.sh runs it on both nodes when COMPACT=1)

The load takes about 110 GiB of a Spark's 121, so whatever page cache is in the way goes during the load, while the
driver is asking for the GPU's memory. This takes it out first: a huge-page fault in a MADV_HUGEPAGE mapping makes the
kernel build each 2 MiB block (THP defrag = madvise), dropping or moving the page cache that sits there, and the blocks
come back whole when the mapping goes. It takes MemAvailable less --keep-gib and stops early if MemAvailable falls
under half of --keep-gib. Prints MemFree and the free memory in blocks of 2 MiB or more, before and after.
"""
import argparse
import mmap
import sys
import time

HUGE = 2 << 20
GIB = 1 << 30


def meminfo(key, path="/proc/meminfo"):
    with open(path) as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    raise KeyError(key)


def big_free(path="/proc/buddyinfo"):
    """Free bytes in buddy blocks of order 9 (2 MiB with 4 KiB pages) and up, all zones."""
    with open(path) as f:
        rows = [[int(x) for x in line.split()[4:]] for line in f]
    return sum(c * (mmap.PAGESIZE << order) for counts in rows for order, c in enumerate(counts) if order >= 9)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keep-gib", type=float, default=4.0, help="available memory left alone")
    a = ap.parse_args(argv)
    keep = int(a.keep_gib * GIB)
    before, free0 = big_free(), meminfo("MemFree")
    size = (meminfo("MemAvailable") - keep) // HUGE * HUGE
    if size <= 0:
        print(f"defrag: nothing to do, MemAvailable is under --keep-gib {a.keep_gib}")
        return 0
    try:  # if memory runs out under us, this process goes first
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    t0 = time.monotonic()
    m = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    m.madvise(mmap.MADV_HUGEPAGE)
    done = 0
    for off in range(0, size, HUGE):
        if off % GIB == 0 and meminfo("MemAvailable") < keep // 2:  # someone else took memory meanwhile
            break
        m[off] = 1
        done = off + HUGE
    m.close()
    time.sleep(1)
    print(f"defrag: faulted {done / GIB:.1f} GiB in {time.monotonic() - t0:.1f} s; free in blocks >= 2 MiB: "
          f"{before / GIB:.1f} -> {big_free() / GIB:.1f} GiB; MemFree {free0 / GIB:.1f} -> {meminfo('MemFree') / GIB:.1f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
