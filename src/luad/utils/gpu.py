"""GPU power safety. The PC powered off once under full load at the default 600 W limit (likely a PSU trip),
so every GPU job checks the power limit first and logs power/temperature to a CSV (flushed per line)."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path


def gpu_query(fields: str) -> list[str]:
    out = subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                                  text=True)
    return [x.strip() for x in out.strip().splitlines()[0].split(",")]


def require_power_limit(max_w: float) -> float:
    limit = float(gpu_query("power.limit")[0])
    if limit > max_w:
        raise SystemExit(f"GPU power limit is {limit:.0f} W (> {max_w:.0f} W). "
                         f"Run in your own terminal: sudo nvidia-smi -pl {max_w:.0f}")
    return limit


def start_gpu_logger(log: Path, interval_ms: int = 1000) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    f = log.open("a", buffering=1)
    f.write(f"# run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,power.draw,temperature.gpu,clocks.sm,utilization.gpu,memory.used",
         "--format=csv,noheader", "-lms", str(interval_ms)], stdout=f, stderr=subprocess.STDOUT)
