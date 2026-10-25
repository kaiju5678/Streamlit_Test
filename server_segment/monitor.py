"""Device telemetry and per-request run records for the segmentation server.

One background sampler keeps the last few minutes of GPU (or CPU) readings in a ring buffer; a run record takes
the readings between its own start and end, so no thread is started per request.
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import contextmanager
from datetime import datetime

PERIOD_S = 0.5
NVSMI_FIELDS = ("memory.used", "memory.total", "utilization.gpu", "temperature.gpu", "power.draw")
NVSMI_KEYS = ("mem_used_mb", "mem_total_mb", "gpu_util", "temp_c", "power_w")


def _num(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:  # "[N/A]" on laptop GPUs that do not report power
        return None


class Telemetry:
    """nvidia-smi in loop mode (one long-lived process) when there is an NVIDIA GPU, psutil CPU/RAM otherwise."""

    def __init__(self, keep_s: float = 900) -> None:
        self.buffer: deque[tuple[float, dict]] = deque(maxlen=int(keep_s / PERIOD_S))
        self.device = socket.gethostname()
        self.kind = "none"
        if shutil.which("nvidia-smi"):
            self.kind = "nvidia-smi"
            try:
                name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                      capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
                self.device = name[0] if name else self.device
            except (OSError, subprocess.SubprocessError):
                pass
            threading.Thread(target=self._nvsmi_loop, daemon=True).start()
        else:
            try:
                import psutil  # noqa: F401
                self.kind = "psutil"
                threading.Thread(target=self._psutil_loop, daemon=True).start()
            except ImportError:
                pass

    def _nvsmi_loop(self) -> None:
        cmd = ["nvidia-smi", f"--query-gpu={','.join(NVSMI_FIELDS)}", "--format=csv,noheader,nounits",
               f"-lms={int(PERIOD_S * 1000)}"]
        while True:  # restart the reader if nvidia-smi ever exits
            try:
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) as proc:
                    assert proc.stdout
                    for line in proc.stdout:
                        vals = [_num(v.strip()) for v in line.split(",")]
                        if len(vals) == len(NVSMI_KEYS):
                            self.buffer.append((time.time(), dict(zip(NVSMI_KEYS, vals))))
            except OSError:
                pass
            time.sleep(5)

    def _psutil_loop(self) -> None:
        import psutil
        psutil.cpu_percent(interval=None)
        while True:
            time.sleep(PERIOD_S)
            vm = psutil.virtual_memory()
            self.buffer.append((time.time(), {"mem_used_mb": round(vm.used / 2**20), "mem_total_mb": round(vm.total / 2**20),
                                              "cpu_util": psutil.cpu_percent(interval=None)}))

    def latest(self) -> dict:
        return dict(self.buffer[-1][1]) if self.buffer else {}

    def between(self, t0: float, t1: float) -> list[dict]:
        return [{"t": round(t - t0, 2), **v} for t, v in list(self.buffer) if t0 <= t <= t1]


class RunRecord:
    """Log lines and stage timings of one /segment request; serialised by GET /runs/{id}."""

    def __init__(self, telemetry: Telemetry, model: str, filename: str, run_id: str | None = None) -> None:
        self.id = run_id or uuid.uuid4().hex[:12]
        self.telemetry = telemetry
        self.model, self.filename, self.bytes = model, filename, 0
        self.t0, self.t1 = time.time(), None
        self.status = "queued"
        self.stages: dict[str, float] = {}
        self.logs: list[dict] = []
        self._lock = threading.Lock()

    def log(self, level: str, tag: str, msg: str) -> None:
        line = {"t": datetime.now().isoformat(timespec="milliseconds"), "level": level, "tag": tag, "msg": msg}
        with self._lock:
            self.logs.append(line)
        print(f"[{self.id}] {level:<7} {tag:<11} {msg}", flush=True)

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        self.log("INFO", name.upper(), "started")
        try:
            yield
        except BaseException:
            self.stages[name] = round(time.perf_counter() - t, 2)
            self.log("ERROR", name.upper(), f"failed after {self.stages[name]:.2f}s")
            raise
        self.stages[name] = round(time.perf_counter() - t, 2)
        self.log("INFO", name.upper(), f"done in {self.stages[name]:.2f}s")

    def close(self, status: str) -> None:
        self.status, self.t1 = status, time.time()

    def as_dict(self) -> dict:
        end = self.t1 or time.time()
        with self._lock:
            logs = list(self.logs)
        return {"id": self.id, "filename": self.filename, "bytes": self.bytes, "model": self.model,
                "device": self.telemetry.device, "sampler": self.telemetry.kind,
                "started": datetime.fromtimestamp(self.t0).isoformat(timespec="seconds"),
                "wall_s": round(end - self.t0, 2), "status": self.status, "stages": dict(self.stages), "logs": logs,
                "samples": self.telemetry.between(self.t0, end + PERIOD_S)}


class RunBook(OrderedDict):
    """The most recent runs, in memory only."""

    def __init__(self, keep: int = 50) -> None:
        super().__init__()
        self.keep = keep

    def add(self, run: RunRecord) -> RunRecord:
        self[run.id] = run
        while len(self) > self.keep:
            self.popitem(last=False)
        return run
