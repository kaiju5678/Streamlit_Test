"""Segmentation server for the BodyComp Streamlit app (same API as ct-scan-test/inference/server.py).

Model: Mazurowski Lab muscle/fat nnU-Net (CT-Muscle-and-Fat-Segmentation/predict_muscle_fat.py),
labels 1 skeletal muscle · 2 SAT · 3 VAT · 4 intermuscular fat. TotalSegmentator then finds the
T12, L3 and L4 vertebrae: L3 is written into the mask as label 50 so the viewer sets the L3 slice
by itself, and the repo's 2D (L3) and 3D (T12–L4) body-composition metrics are computed with the
same formulas as predict_muscle_fat.py.

POST /segment      body: raw .nii / .nii.gz bytes; headers X-Filename, X-Run-Id (optional, 12 hex)
                   → 200 with a .nii.gz label mask on the CT grid.
                     Headers: X-Labels, X-Label-Scheme, X-Run-Id, X-Body-Composition
POST /convert      body: .zip of one DICOM study → .nii.gz of the largest CT series (no patient metadata)
GET  /runs/{id}    log lines, stage timings and resource samples of one run
GET  /device       device and GPU status
GET  /health

Every request works in its own OS temp folder, deleted after the reply. Runs are kept in memory only.

    python server.py                 # 127.0.0.1:8000, MODEL=mazurowski
    set MODEL=demo & python server.py   (HU thresholds; tests the pipeline without a GPU)
"""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote

import nibabel as nib
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

HERE = Path(__file__).resolve().parent
# Model output (tqdm bars) is echoed to this console; a Windows code page cannot print all of it.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
if os.name == "nt":
    os.system("")  # turn on ANSI colours in cmd.exe for the access log

MODEL = os.getenv("MODEL", "mazurowski").strip().lower()          # mazurowski | demo
SEG_REPO = Path(os.getenv("SEG_REPO", HERE.parent.parent / "CT-Muscle-and-Fat-Segmentation")).resolve()
SEG_PYTHON = os.getenv("SEG_PYTHON", sys.executable)
CHECKPOINT = os.getenv("SEG_CHECKPOINT", "final")                  # final | best
AUTO_VERTEBRAE = os.getenv("AUTO_VERTEBRAE", "1") == "1"
TOTALSEG_FAST = os.getenv("TOTALSEG_FAST", "0") == "1"
API_KEY = os.getenv("API_KEY", "")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
ACCESS_LOG = os.getenv("ACCESS_LOG", "1") == "1"                   # one line per request: status, method, path, time
TIMEOUT_S = int(os.getenv("SEG_TIMEOUT_S", "3600"))
MAX_BYTES = 1024**3
KEEP_RUNS = 50
L3_LABEL = 50
LABELS = {1: "muscle", 2: "sat", 3: "vat", 4: "muscle_fat"}        # dataset.json of the model folder

app = FastAPI(title="BodyComp segmentation (Mazurowski nnU-Net)")
GPU_LOCK = threading.Lock()          # one inference at a time on a small GPU


def status_text(code: int) -> str:
    """Status code coloured by class: 2xx green, 3xx cyan, 4xx yellow, 5xx red."""
    colour = {2: "32", 3: "36", 4: "33"}.get(code // 100, "31")
    return f"[{colour}m{code}[0m"


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    code = 500
    try:
        response = await call_next(request)
        code = response.status_code
        return response
    finally:
        if ACCESS_LOG:
            ms = (time.perf_counter() - t0) * 1000
            print(f"{datetime.now():%H:%M:%S}  {status_text(code)}  {request.method:<4} {request.url.path}  {ms:.0f} ms", flush=True)


def bin_dir() -> Path:
    """Scripts folder of the Python that runs the model (TotalSegmentator lives there)."""
    return Path(SEG_PYTHON).resolve().parent


def tool_env() -> dict:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    env["PATH"] = f"{bin_dir()}{os.pathsep}{env.get('PATH', '')}"
    return env


def totalseg_exe() -> str | None:
    return shutil.which("TotalSegmentator", path=str(bin_dir())) or shutil.which("TotalSegmentator")


# ---------------------------------------------------------------- resources
class Sampler:
    """GPU state from nvidia-smi (or tegrastats on Jetson); CPU/RAM from psutil otherwise."""

    def __init__(self) -> None:
        self.latest: dict = {}
        self.device = self._device_name()
        if shutil.which("tegrastats"):
            self.kind = "tegrastats"
            threading.Thread(target=self._tegra_loop, daemon=True).start()
        elif shutil.which("nvidia-smi"):
            self.kind = "nvidia-smi"
        else:
            try:
                import psutil  # noqa: F401
                self.kind = "psutil"
            except ImportError:
                self.kind = "none"

    @staticmethod
    def _device_name() -> str:
        p = Path("/proc/device-tree/model")
        if p.exists():
            return p.read_text(errors="ignore").strip("\x00 \n")
        if shutil.which("nvidia-smi"):
            try:
                return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                      capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
            except Exception:  # noqa: BLE001
                pass
        return socket.gethostname()

    def _tegra_loop(self) -> None:
        proc = subprocess.Popen(["tegrastats", "--interval", "500"], stdout=subprocess.PIPE, text=True)
        assert proc.stdout
        for line in proc.stdout:
            s: dict = {}
            if m := re.search(r"RAM (\d+)/(\d+)MB", line):
                s["mem_used_mb"], s["mem_total_mb"] = int(m[1]), int(m[2])
            if m := re.search(r"GR3D_FREQ (\d+)%", line):
                s["gpu_util"] = int(m[1])
            if m := re.search(r"(?:gpu|GPU)@([\d.]+)C", line):
                s["temp_c"] = float(m[1])
            if m := re.search(r"VDD_IN (\d+)mW", line):
                s["power_w"] = int(m[1]) / 1000
            self.latest = s

    def sample(self) -> dict:
        if self.kind == "tegrastats":
            return dict(self.latest)
        if self.kind == "nvidia-smi":
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
                v = [x.strip() for x in out.splitlines()[0].split(",")]
                num = lambda x: float(x) if re.match(r"^[\d.]+$", x) else None  # noqa: E731
                return {"mem_used_mb": num(v[0]), "mem_total_mb": num(v[1]), "gpu_util": num(v[2]),
                        "temp_c": num(v[3]), "power_w": num(v[4])}
            except Exception:  # noqa: BLE001
                return {}
        if self.kind == "psutil":
            import psutil
            vm = psutil.virtual_memory()
            return {"mem_used_mb": round(vm.used / 2**20), "mem_total_mb": round(vm.total / 2**20),
                    "cpu_util": psutil.cpu_percent(interval=None)}
        return {}


SAMPLER = Sampler()


# ---------------------------------------------------------------- runs
class Run:
    def __init__(self, filename: str, size: int, run_id: str | None = None) -> None:
        self.id = run_id or uuid.uuid4().hex[:12]
        self.filename = filename
        self.bytes = size
        self.model = MODEL
        self.started = time.time()
        self.finished: float | None = None
        self.status = "running"
        self.stages: dict[str, float] = {}
        self.logs: list[dict] = []
        self.samples: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._sample_loop, daemon=True).start()

    def log(self, level: str, tag: str, msg: str) -> None:
        with self._lock:
            self.logs.append({"t": datetime.now().isoformat(timespec="milliseconds"), "level": level, "tag": tag, "msg": msg})
        print(f"[{self.id}] {level} {tag} {msg}", flush=True)

    def stage(self, name: str):
        run = self

        class _S:
            def __enter__(self):
                self.t0 = time.time()
                run.log("INFO", name.upper(), "started")

            def __exit__(self, *exc):
                dt = time.time() - self.t0
                run.stages[name] = round(dt, 2)
                run.log("INFO" if not exc[0] else "ERROR", name.upper(), f"{'finished' if not exc[0] else 'failed'} in {dt:.2f}s")
                return False

        return _S()

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            s = SAMPLER.sample()
            if s:
                s["t"] = round(time.time() - self.started, 2)
                with self._lock:
                    self.samples.append(s)
            self._stop.wait(0.5)

    def finish(self, status: str) -> None:
        self._stop.set()
        s = SAMPLER.sample()  # closing sample, so even a short run has a start and an end point
        if s:
            s["t"] = round(time.time() - self.started, 2)
            self.samples.append(s)
        self.status = status
        self.finished = time.time()

    def to_dict(self) -> dict:
        with self._lock:
            return {"id": self.id, "filename": self.filename, "bytes": self.bytes, "model": self.model,
                    "device": SAMPLER.device, "sampler": SAMPLER.kind,
                    "started": datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
                    "wall_s": round((self.finished or time.time()) - self.started, 2),
                    "status": self.status, "stages": dict(self.stages), "logs": list(self.logs),
                    "samples": list(self.samples)}


RUNS: "OrderedDict[str, Run]" = OrderedDict()


def remember(run: Run) -> None:
    RUNS[run.id] = run
    while len(RUNS) > KEEP_RUNS:
        RUNS.popitem(last=False)


def check_key(key: str) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(401, "Wrong or missing X-Api-Key.")


# ---------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "repo_found": (SEG_REPO / "predict_muscle_fat.py").exists(),
            "totalsegmentator": bool(totalseg_exe()), "auto_vertebrae": AUTO_VERTEBRAE}


@app.get("/device")
def device(x_api_key: str = Header("")):
    check_key(x_api_key)
    busy = any(r.status == "running" for r in RUNS.values())
    return {"device": SAMPLER.device, "model": MODEL, "sampler": SAMPLER.kind, "busy": busy, **SAMPLER.sample()}


@app.get("/runs/{run_id}")
def get_run(run_id: str, x_api_key: str = Header("")):
    check_key(x_api_key)
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "Unknown run id.")
    return run.to_dict()


@app.post("/segment")
async def segment(request: Request, x_filename: str = Header("ct.nii.gz"), x_api_key: str = Header(""),
                  x_run_id: str = Header("")):
    check_key(x_api_key)
    name = Path(unquote(x_filename)).name
    work = Path(tempfile.mkdtemp(prefix="bodycomp_"))
    raw = work / "upload.bin"

    run = Run(name, 0, x_run_id if re.fullmatch(r"[0-9a-f]{12}", x_run_id or "") else None)
    remember(run)
    run.log("INFO", "PIPELINE", f"run {run.id} on {SAMPLER.device} with model '{MODEL}'")
    try:
        with run.stage("receive"):
            size = 0
            with raw.open("wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, "CT larger than 1 GB.")
                    f.write(chunk)
            if not size:
                raise HTTPException(400, "Empty upload.")
            run.bytes = size
            run.log("DEBUG", "RECEIVE", f"{name}: {size / 1e6:.1f} MB written to scratch")
        if GPU_LOCK.locked():
            run.log("INFO", "QUEUE", "waiting for the GPU (another run is in progress)")

        def work_fn():
            with GPU_LOCK:
                return run_model(run, raw, work)

        out, labels, body = await run_in_threadpool(work_fn)
    except HTTPException as e:
        run.log("ERROR", "PIPELINE", str(e.detail))
        run.finish("failed")
        shutil.rmtree(work, ignore_errors=True)
        raise
    except Exception as e:  # noqa: BLE001
        run.log("ERROR", "PIPELINE", str(e)[-400:])
        run.finish("failed")
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(500, f"Model failed: {str(e)[-400:]}")

    run.finish("completed")
    run.log("SUCCESS", "PIPELINE", f"completed in {run.to_dict()['wall_s']}s")
    headers = {"X-Label-Scheme": "generic", "X-Run-Id": run.id,
               "X-Labels": quote(json.dumps({str(k): v for k, v in labels.items()})),
               "X-Body-Composition": quote(json.dumps(body))}
    return FileResponse(out, media_type="application/octet-stream", headers=headers,
                        background=BackgroundTask(shutil.rmtree, work, ignore_errors=True))


@app.post("/convert")
async def convert(request: Request, x_api_key: str = Header(""), x_series_id: str = Header("")):
    """DICOM .zip → NIfTI of one series (the CT series with the most slices unless X-Series-Id picks another)."""
    check_key(x_api_key)
    try:
        import SimpleITK as sitk
    except ImportError:
        raise HTTPException(501, "DICOM conversion needs SimpleITK on the inference server: pip install SimpleITK")
    work = Path(tempfile.mkdtemp(prefix="bodycomp_dcm_"))
    try:
        zpath = work / "upload.zip"
        size = 0
        with zpath.open("wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_BYTES * 2:
                    raise HTTPException(413, "DICOM upload larger than 2 GB.")
                f.write(chunk)
        dicom_dir = work / "dicom"
        dicom_dir.mkdir()
        try:
            with zipfile.ZipFile(zpath) as z:
                # Flat folder with generated names: series are grouped by UID whatever the zip layout,
                # and no member name can escape the scratch folder.
                n = 0
                for m in z.infolist():
                    if m.is_dir() or m.file_size == 0:
                        continue
                    n += 1
                    with z.open(m) as src, (dicom_dir / f"f{n:06d}.dcm").open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        except zipfile.BadZipFile:
            raise HTTPException(400, "The upload is not a valid .zip file.")
        zpath.unlink()

        series: dict[str, dict] = {}
        for sid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir)) or []:
            series[sid] = {"files": list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), sid))}
        if not series:
            raise HTTPException(422, "No DICOM image series found in the upload.")
        for info in series.values():
            try:
                r = sitk.ImageFileReader()
                r.SetFileName(info["files"][0])
                r.ReadImageInformation()
                get = lambda tag: r.GetMetaData(tag).strip() if r.HasMetaDataKey(tag) else ""  # noqa: E731
                info["description"], info["modality"] = get("0008|103e"), get("0008|0060")
            except Exception:  # noqa: BLE001
                info["description"], info["modality"] = "", ""
        listing = [{"id": k, "slices": len(v["files"]), "description": v["description"], "modality": v["modality"]}
                   for k, v in sorted(series.items(), key=lambda kv: -len(kv[1]["files"]))]
        chosen = x_series_id if x_series_id in series else next(
            (s["id"] for s in listing if s["modality"] in ("CT", "") and s["slices"] >= 10), listing[0]["id"])

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series[chosen]["files"])
        img = reader.Execute()  # applies rescale slope/intercept, so values are HU
        img = sitk.Cast(sitk.Clamp(img, lowerBound=-32768, upperBound=32767), sitk.sitkInt16)
        out = work / "ct.nii.gz"
        sitk.WriteImage(img, str(out))  # NIfTI keeps geometry only: no names, IDs or dates
        print(f"converted series {chosen} ({len(series[chosen]['files'])} files, {size / 1e6:.1f} MB zip)", flush=True)
        return FileResponse(out, media_type="application/octet-stream",
                            headers={"X-Series": quote(json.dumps(listing)), "X-Series-Chosen": chosen},
                            background=BackgroundTask(shutil.rmtree, work, ignore_errors=True))
    except HTTPException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(500, f"DICOM conversion failed: {str(e)[-300:]}")


# ---------------------------------------------------------------- pipeline
def stream_cmd(run: Run, cmd: list[str], tag: str, cwd: Path | None = None) -> None:
    """Run a command (argument list, so paths with spaces are safe) and copy its output into the run log."""
    run.log("DEBUG", tag, " ".join(f'"{c}"' if " " in c else c for c in cmd))
    proc = subprocess.Popen(cmd, cwd=cwd, env=tool_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    start = time.time()
    tail: list[str] = []
    last_progress = ""
    assert proc.stdout
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if time.time() - start > TIMEOUT_S:
                raise RuntimeError(f"{tag} timed out after {TIMEOUT_S} s")
            tail = (tail + [line])[-20:]
            # tqdm redraws one bar many times; keep the log readable
            if m := re.search(r"(\d+)%\|", line):
                if m[1] == last_progress:
                    continue
                last_progress = m[1]
            level = "ERROR" if re.search(r"error|traceback", line, re.I) else "WARN" if re.search(r"warn", line, re.I) else "DEBUG"
            run.log(level, tag, line[:400])
    except BaseException:
        proc.kill()  # never leave a model process holding the GPU
        raise
    if proc.wait() != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} exited with code {proc.returncode}: " + " | ".join(tail[-3:]))


def as_nii_gz(src: Path, dst: Path) -> None:
    """predict_muscle_fat.py only accepts .nii.gz; gzip a plain .nii upload."""
    with src.open("rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        src.rename(dst)
        return
    with src.open("rb") as f, gzip.open(dst, "wb", compresslevel=1) as g:
        shutil.copyfileobj(f, g)
    src.unlink()


def canonical(path: Path, label: bool = False) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    arr = np.asanyarray(img.dataobj)
    if label:
        arr = np.rint(arr).astype(np.uint8)
    return arr, tuple(float(z) for z in img.header.get_zooms()[:3])


def peak_slice(path: Path) -> int | None:
    """Axial slice where a vertebra mask has the most voxels (the argmax rule of predict_muscle_fat.py)."""
    if not path.exists():
        return None
    v, _ = canonical(path, label=True)
    per = v.sum(axis=(0, 1), dtype=np.int64)
    return int(per.argmax()) if per.any() else None


def body_composition(ct_path: Path, seg_path: Path, vert_dir: Path | None) -> dict:
    """2D (L3) and 3D (T12–L4) metrics with the formulas of predict_muscle_fat.py, in RAS index space."""
    img, (sx, sy, sz) = canonical(ct_path)
    img = img.astype(np.float32)
    seg, _ = canonical(seg_path, label=True)
    t12 = l3 = l4 = None
    if vert_dir is not None:
        t12, l3, l4 = (peak_slice(vert_dir / f"vertebrae_{v}.nii.gz") for v in ("T12", "L3", "L4"))
    out: dict = {"slices": {"T12": t12, "L3": l3, "L4": l4}, "metrics_2d": None, "metrics_3d": None}
    if l3 is not None:
        im, sg = img[:, :, l3], seg[:, :, l3]
        px = sx * sy
        out["metrics_2d"] = {
            "slice_index": l3,
            "muscle_area_mm2": float((sg == 1).sum() * px),
            "muscle_density_hu": float(im[sg == 1].mean()) if (sg == 1).any() else None,
            "sfat_area_mm2": float((sg == 2).sum() * px),
            "vfat_area_mm2": float((sg == 3).sum() * px),
            "mfat_area_mm2": float((sg == 4).sum() * px),
            "total_fat_area_mm2": float(np.isin(sg, [2, 3, 4]).sum() * px),
            "body_area_mm2": float((im > -500).sum() * px),
        }
    if t12 is not None and l4 is not None:
        lo, hi = min(t12, l4), max(t12, l4)
        im, sg = img[:, :, lo:hi + 1], seg[:, :, lo:hi + 1]
        vox = sx * sy * sz
        out["metrics_3d"] = {
            "slice_range": [lo, hi],
            "muscle_volume_mm3": float((sg == 1).sum() * vox),
            "muscle_density_hu": float(im[sg == 1].mean()) if (sg == 1).any() else None,
            "sfat_volume_mm3": float((sg == 2).sum() * vox),
            "vfat_volume_mm3": float((sg == 3).sum() * vox),
            "mfat_volume_mm3": float((sg == 4).sum() * vox),
            "total_fat_volume_mm3": float(np.isin(sg, [2, 3, 4]).sum() * vox),
            "body_volume_mm3": float((im > -500).sum() * vox),
            "body_height_mm": float((hi - lo + 1) * sz),
        }
    return out


def merge_l3(run: Run, tissue: Path, l3: Path) -> None:
    """Write the L3 vertebra into the tissue mask as label 50 (bone does not overlap the soft-tissue labels)."""
    t_img = nib.load(str(tissue))
    t = np.asanyarray(t_img.dataobj).astype(np.uint8)
    v = np.asanyarray(nib.load(str(l3)).dataobj)
    if v.shape != t.shape:
        raise RuntimeError(f"L3 mask shape {v.shape} differs from tissue mask {t.shape}")
    sel = v > 0
    if not sel.any():
        raise RuntimeError("L3 is not inside the scanned range")
    t[sel] = L3_LABEL
    out = nib.Nifti1Image(t, t_img.affine, t_img.header)
    out.set_data_dtype(np.uint8)
    out.header.set_slope_inter(1, 0)
    nib.save(out, str(tissue))
    run.log("INFO", "L3", f"L3 vertebra found: {int(sel.sum())} voxels")


def run_model(run: Run, raw: Path, work: Path) -> tuple[Path, dict[int, str], dict]:
    ct = work / "ct.nii.gz"
    with run.stage("preprocess"):
        as_nii_gz(raw, ct)
        img = nib.load(str(ct))
        run.log("DEBUG", "PREPROCESS", f"grid {img.shape}, spacing {tuple(round(float(z), 3) for z in img.header.get_zooms()[:3])}, "
                                       f"orientation {''.join(nib.aff2axcodes(img.affine))}")

    if MODEL == "demo":
        with run.stage("inference"):
            hu = np.asanyarray(img.dataobj).astype(np.float32)
            lab = np.zeros(hu.shape, np.uint8)
            lab[(hu >= -29) & (hu <= 150)] = 1
            lab[(hu >= -190) & (hu <= -30)] = 3
            out = work / "demo_seg.nii.gz"
            seg = nib.Nifti1Image(lab, img.affine, img.header)
            seg.set_data_dtype(np.uint8)
            seg.header.set_slope_inter(1, 0)
            nib.save(seg, str(out))
        run.log("WARN", "PIPELINE", "demo model uses HU thresholds only; do not report these values")
        return out, {1: "muscle_hu_range", 3: "fat_hu_range"}, {"slices": {}, "metrics_2d": None, "metrics_3d": None}

    if MODEL != "mazurowski":
        raise RuntimeError(f"Unknown MODEL '{MODEL}'. Use mazurowski or demo.")
    script = SEG_REPO / "predict_muscle_fat.py"
    if not script.exists():
        raise RuntimeError(f"predict_muscle_fat.py not found in {SEG_REPO} (set SEG_REPO)")
    outdir = work / "out"
    outdir.mkdir()
    with run.stage("inference"):
        # nnU-Net only; the repo's own vertebra step uses os.system with unquoted paths, so it is run below instead.
        # cwd = repo, because the script loads its model folder by relative path.
        stream_cmd(run, [SEG_PYTHON, "-u", str(script), "--input", str(ct), "--output", str(outdir),
                         "--checkpoint_type", CHECKPOINT, "--body_composition_type", "None"], "NNUNET", cwd=SEG_REPO)
    out = outdir / "ct.nii.gz"
    if not out.exists():
        found = sorted(p for p in outdir.glob("*.nii*"))
        if not found:
            raise RuntimeError("nnU-Net produced no output file.")
        out = found[0]
    labels: dict[int, str] = dict(LABELS)

    vert_dir = None
    exe = totalseg_exe()
    if AUTO_VERTEBRAE and exe:
        try:
            vd = work / "vertebrae"
            cmd = [exe, "-i", str(ct), "-o", str(vd), "--roi_subset", "vertebrae_T12", "vertebrae_L3", "vertebrae_L4"]
            if TOTALSEG_FAST:
                cmd.append("--fast")
            with run.stage("vertebrae"):
                stream_cmd(run, cmd, "TOTALSEG")
            vert_dir = vd
        except Exception as e:  # noqa: BLE001  L3 is a convenience; keep the tissue mask if it fails
            run.log("WARN", "L3", f"automatic vertebra detection failed, set L3 by hand: {str(e)[-200:]}")
    elif AUTO_VERTEBRAE:
        run.log("WARN", "L3", "TotalSegmentator not found next to SEG_PYTHON; set L3 by hand")

    with run.stage("postprocess"):
        body = body_composition(ct, out, vert_dir)   # on labels 1–4 before L3 is merged in, as predict_muscle_fat.py does
        if vert_dir is not None:
            try:
                merge_l3(run, out, vert_dir / "vertebrae_L3.nii.gz")
                labels[L3_LABEL] = "vertebrae_L3"
            except Exception as e:  # noqa: BLE001
                run.log("WARN", "L3", f"L3 not merged, set it by hand: {str(e)[-200:]}")
        s = body["slices"]
        run.log("INFO", "METRICS", f"vertebra peak slices T12={s['T12']} L3={s['L3']} L4={s['L4']} (RAS index)")
        if body["metrics_3d"] is None:
            run.log("WARN", "METRICS", "T12 or L4 not in the scan: 3D (T12–L4) metrics not computed")
        counts = {int(k): int(v) for k, v in zip(*np.unique(np.asanyarray(nib.load(str(out)).dataobj), return_counts=True)) if k}
        run.log("DEBUG", "POSTPROCESS", f"label voxel counts {counts}")
    return out, labels, body


if __name__ == "__main__":
    import uvicorn

    print(f"BodyComp segmentation server  http://{HOST}:{PORT}")
    print(f"  model      : {MODEL}")
    if MODEL == "mazurowski":
        print(f"  repo       : {SEG_REPO}  ({'found' if (SEG_REPO / 'predict_muscle_fat.py').exists() else 'NOT FOUND'})")
        print(f"  python     : {SEG_PYTHON}")
        print(f"  checkpoint : {CHECKPOINT}")
        print(f"  vertebrae  : {'TotalSegmentator ' + (totalseg_exe() or 'NOT FOUND') if AUTO_VERTEBRAE else 'off'}")
    print(f"  device     : {SAMPLER.device} ({SAMPLER.kind})")
    print(f"  api key    : {'set' if API_KEY else 'off'}")
    print(f"  access log : {'on' if ACCESS_LOG else 'off'}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
