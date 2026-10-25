"""BodyComp segmentation server: the Mazurowski muscle/fat nnU-Net and TotalSegmentator vertebrae, resident on the GPU.

Models are loaded once at start-up (engine.py) and every CT is handled in memory; one worker thread runs the GPU
jobs in arrival order. The L3 vertebra is written into the mask as label 50 so the viewer sets the L3 slice itself,
and the 2D (L3) / 3D (T12–L4) metrics come from core/bodycomp.py, the same code the frontend uses.

POST /segment      body: .nii / .nii.gz bytes; headers X-Filename, X-Run-Id (optional, 12 hex)
                   -> .nii.gz label mask on the CT grid; headers X-Labels, X-Label-Scheme, X-Run-Id, X-Body-Composition
POST /convert      body: .zip of one DICOM study -> .nii.gz of one series; headers X-Series, X-Series-Chosen
GET  /runs/{id}    log lines, stage timings and resource samples of one run (kept in memory, last 50)
GET  /device       device, queue and GPU status
GET  /health       liveness; "ready" turns true once the models are loaded

    python server.py                       127.0.0.1:8000, Mazurowski model with TTA
    set MODEL=demo & python server.py      HU thresholds, no GPU (pipeline test)
    set SEG_TTA=0 & python server.py       no mirroring TTA: ~4x faster, IMAT less stable (see benchmark.py)

Run it with the Python environment that has nnunetv2 and TotalSegmentator (the models are imported, not launched).
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote

from fastapi import FastAPI, Header, HTTPException, Request, Response

import dicom_zip
from engine import Engine
from monitor import RunBook, RunRecord, Telemetry

HERE = Path(__file__).resolve().parent
for _stream in (sys.stdout, sys.stderr):   # nnU-Net / TotalSegmentator output on a Windows code page
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip() == "1"


MODEL = os.getenv("MODEL", "mazurowski").strip().lower()          # mazurowski | demo
SEG_REPO = Path(os.getenv("SEG_REPO", HERE.parent.parent / "CT-Muscle-and-Fat-Segmentation")).resolve()
MODEL_DIR = SEG_REPO / "nnUNetTrainer__nnUNetResEncUNetXLPlans__2d"
CHECKPOINT = os.getenv("SEG_CHECKPOINT", "final")                  # final | best
TTA = env_flag("SEG_TTA", "1")
WARMUP = env_flag("WARMUP", "1")
AUTO_VERTEBRAE = env_flag("AUTO_VERTEBRAE", "1")
TOTALSEG_FAST = env_flag("TOTALSEG_FAST", "0")
API_KEY = os.getenv("API_KEY", "")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
ACCESS_LOG = env_flag("ACCESS_LOG", "1")
MAX_CT_BYTES = 1 << 30
MAX_ZIP_BYTES = 2 << 30


class State:
    telemetry = Telemetry()
    runs = RunBook()
    gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")   # one model job at a time, FIFO
    engine: Engine | None = None
    load_error = ""
    pending = 0      # jobs submitted to the GPU worker and not finished
    active = 0       # 1 while the worker runs a CT


S = State()


def load_engine() -> None:
    t = time.perf_counter()
    try:
        if MODEL not in ("mazurowski", "demo"):
            raise RuntimeError(f"Unknown MODEL '{MODEL}'. Use mazurowski or demo.")
        if MODEL == "mazurowski" and not MODEL_DIR.exists():
            raise RuntimeError(f"model folder not found: {MODEL_DIR} (set SEG_REPO)")
        engine = Engine(MODEL, MODEL_DIR, CHECKPOINT, TTA, AUTO_VERTEBRAE, TOTALSEG_FAST)
        if WARMUP and engine.tissue is not None:
            engine.tissue.warm_up()
        S.engine = engine
        print(f"models ready in {time.perf_counter() - t:.1f}s", flush=True)
    except Exception as e:  # noqa: BLE001  keep serving /health and /device so the frontend can show the reason
        S.load_error = str(e)[-400:]
        print(f"model load failed: {S.load_error}", flush=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    S.gpu.submit(load_engine)   # first job of the GPU worker: requests that arrive meanwhile queue behind it
    yield
    S.gpu.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="BodyComp segmentation server", lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next):
    t = time.perf_counter()
    code = 500
    try:
        response = await call_next(request)
        code = response.status_code
        return response
    finally:
        if ACCESS_LOG:
            colour = {2: "32", 3: "36", 4: "33"}.get(code // 100, "31")
            print(f"{datetime.now():%H:%M:%S}  \x1b[{colour}m{code}\x1b[0m  {request.method:<4} {request.url.path}  "
                  f"{(time.perf_counter() - t) * 1000:.0f} ms", flush=True)


def require_key(key: str) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(401, "X-Api-Key does not match the server's API_KEY.")


async def read_body(request: Request, limit: int, what: str) -> bytes:
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > limit:
            raise HTTPException(413, f"{what} larger than {limit >> 30} GB.")
    if not buf:
        raise HTTPException(400, "Empty upload.")
    return bytes(buf)


def header_json(value) -> str:
    return quote(json.dumps(value))


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "ready": S.engine is not None, "error": S.load_error, "tta": TTA,
            "repo_found": MODEL_DIR.exists() or MODEL == "demo",
            "totalsegmentator": importlib.util.find_spec("totalsegmentator") is not None, "auto_vertebrae": AUTO_VERTEBRAE}


@app.get("/device")
def device(x_api_key: str = Header("")):
    require_key(x_api_key)
    return {"device": S.telemetry.device, "model": MODEL, "sampler": S.telemetry.kind, "ready": S.engine is not None,
            "busy": S.active > 0, "queued": max(0, S.pending - S.active), **S.telemetry.latest()}


@app.get("/runs/{run_id}")
def get_run(run_id: str, x_api_key: str = Header("")):
    require_key(x_api_key)
    if run_id not in S.runs:
        raise HTTPException(404, f"No run {run_id} in memory (the server keeps the last {S.runs.keep}).")
    return S.runs[run_id].as_dict()


def gpu_job(raw: bytes, run: RunRecord):
    S.active = 1
    try:
        if S.engine is None:
            raise RuntimeError(f"models are not loaded: {S.load_error or 'still loading'}")
        run.status = "running"
        run.log("INFO", "PIPELINE", f"run {run.id} on {S.telemetry.device}, model '{MODEL}', TTA {'on' if TTA else 'off'}")
        return S.engine.segment(raw, run)
    finally:
        S.active = 0


@app.post("/segment")
async def segment(request: Request, x_filename: str = Header("ct.nii.gz"), x_run_id: str = Header(""),
                  x_api_key: str = Header("")):
    require_key(x_api_key)
    run = S.runs.add(RunRecord(S.telemetry, MODEL, Path(unquote(x_filename)).name,
                               x_run_id if re.fullmatch(r"[0-9a-f]{12}", x_run_id or "") else None))
    try:
        with run.stage("receive"):
            raw = await read_body(request, MAX_CT_BYTES, "CT")
            run.bytes = len(raw)
            run.log("DEBUG", "RECEIVE", f"{run.filename}: {len(raw) / 1e6:.1f} MB in memory")
        if S.pending:
            run.log("INFO", "QUEUE", f"waiting for the GPU ({S.pending} job(s) ahead)")
        S.pending += 1
        try:
            result = await asyncio.wrap_future(S.gpu.submit(gpu_job, raw, run))
        finally:
            S.pending -= 1
    except HTTPException as e:
        run.log("ERROR", "PIPELINE", str(e.detail))
        run.close("failed")
        raise
    except Exception as e:  # noqa: BLE001
        run.log("ERROR", "PIPELINE", str(e)[-400:])
        run.close("failed")
        raise HTTPException(500, f"Segmentation failed: {str(e)[-400:]}")
    run.close("completed")
    run.log("SUCCESS", "PIPELINE", f"completed in {run.as_dict()['wall_s']}s")
    return Response(result.mask_gz, media_type="application/octet-stream", headers={
        "X-Label-Scheme": "generic", "X-Run-Id": run.id,
        "X-Labels": header_json({str(k): v for k, v in result.labels.items()}),
        "X-Body-Composition": header_json(result.body)})


@app.post("/convert")
async def convert(request: Request, x_series_id: str = Header(""), x_api_key: str = Header("")):
    require_key(x_api_key)
    if importlib.util.find_spec("SimpleITK") is None:
        raise HTTPException(501, "SimpleITK is not installed next to the server (pip install SimpleITK); DICOM cannot be read.")
    raw = await read_body(request, MAX_ZIP_BYTES, "DICOM upload")
    try:
        nii, listing, chosen = await asyncio.to_thread(dicom_zip.convert, raw, x_series_id)
    except dicom_zip.ConversionError as e:
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Could not convert the DICOM series: {str(e)[-300:]}")
    print(f"converted series {chosen} ({len(raw) / 1e6:.1f} MB zip)", flush=True)
    return Response(nii, media_type="application/octet-stream",
                    headers={"X-Series": header_json(listing), "X-Series-Chosen": chosen})


if __name__ == "__main__":
    import uvicorn

    if os.name == "nt":
        os.system("")   # ANSI colours in cmd.exe for the access log
    print(f"BodyComp segmentation server  http://{HOST}:{PORT}")
    print(f"  model      : {MODEL}" + (f"  ({MODEL_DIR.name}, checkpoint {CHECKPOINT}, TTA {'on' if TTA else 'off'})"
                                       if MODEL == "mazurowski" else ""))
    print(f"  vertebrae  : {'TotalSegmentator' + (' --fast' if TOTALSEG_FAST else '') if AUTO_VERTEBRAE else 'off'}")
    print(f"  device     : {S.telemetry.device} ({S.telemetry.kind})")
    print(f"  api key    : {'set' if API_KEY else 'off'}")
    print("  loading models in the background; /health reports ready=true when done")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
