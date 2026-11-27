"""Opt-in memory probe. Set MEM_PROBE=1 to log RSS, child RSS and top allocation growth.

Off by default (tracemalloc slows Python). Output goes to the normal log, lines
prefixed [MemProbe], so it can be lined up with job phase log lines.
"""
import logging
import os
import threading
import time
import tracemalloc

logger = logging.getLogger(__name__)


def _rss_mb(pid: str = "self") -> float:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _children_rss_mb() -> float:
    total = 0.0
    try:
        me = os.getpid()
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            if ppid == me:
                total += _rss_mb(pid)
    except OSError:
        pass
    return total


def start_if_enabled(interval_s: int = 30) -> bool:
    if os.getenv("MEM_PROBE") != "1":
        return False
    tracemalloc.start(10)

    def loop():
        base = tracemalloc.take_snapshot()
        while True:
            time.sleep(interval_s)
            snap = tracemalloc.take_snapshot()
            top = snap.compare_to(base, "lineno")[:5]
            cur, peak = tracemalloc.get_traced_memory()
            print(
                f"[MemProbe] rss={_rss_mb():.0f}MB child_rss={_children_rss_mb():.0f}MB "
                f"py_traced={cur / 1e6:.0f}MB peak={peak / 1e6:.0f}MB",
                flush=True,
            )
            for stat in top:
                print(f"[MemProbe]   {stat}", flush=True)
            base = snap

    threading.Thread(target=loop, daemon=True, name="memprobe").start()
    return True
