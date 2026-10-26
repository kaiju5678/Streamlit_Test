# BodyComp (Streamlit): CT body composition workstation

The workflow of `ct-scan-test` (Next.js) rebuilt in Python/Streamlit, with the Mazurowski Lab muscle/fat
nnU-Net from `CT-Muscle-and-Fat-Segmentation` as the segmentation model.

| Tab | What it does |
|---|---|
| **Overview** | Muscle, visceral-fat and total-adipose volumes; L3 muscle attenuation; axial view with mask; 3D surface model; tissue composition; L3 indices against reference cut-offs; report (HTML → PDF) and CSV export |
| **Slice viewer** | Synced axial / coronal / sagittal views with crosshair (click to move), real-time scrolling in every view, zoom, caliper measurements |
| **Metrics table** | Per-label voxels, volume, mean and SD HU, slice and L3 areas; the repo's own 2D (L3) and 3D (T12–L4) metrics; cross-sectional area by slice (click to navigate); CSV |
| **AI report** | Draft from an OpenAI-compatible LLM (MedGemma) or rule-based fallback; flags numbers not in the measurements; editable; physician sign-off with Ed25519 signature |
| **Transmitted JSON** | The exact payload (the eight whitelisted biomarkers), its whitelist check, derived indices, LLM request, raw response and signature receipt |
| **Run log** | Server pipeline log merged with the session log; level filter; export; wall time, peak memory, throughput; GPU / power profile |

The sidebar has data input buttons (Open CT and Open mask open the file picker directly; a mask that does not
fit the open study opens its CT first (same NIfTI grid, searched in `SEG_REPO`/demo, `SEG_REPO`, `SEG_REPO`/results and
`CT_SEARCH_DIRS`), and the vertebrae_T12/L3/L4 that `predict_muscle_fat.py` wrote for that CT set L3 and give the
repo's 2D/3D metrics;
DICOM ▾ → folder or .zip),
Run segmentation with the label list, real-time axial scrub and mask opacity, W/L, L3 landmark, patient data, report
endpoint and device status. The slice views and the 3D model run in the browser (`ui/live.py`): every slice is sent
once, so scrolling, the scrub and the opacity redraw instantly in all views and the 3D slice plane follows; Python is
told the new position only when the movement stops. plotly.js for the 3D model is copied from the plotly package into
`static/` and served by Streamlit (works offline).

```
browser ─► Streamlit app.py (:8501) ─► server_segment/server.py (:8000) ─► Mazurowski nnU-Net, resident on the GPU (labels 1–4)
                                │                                  └► TotalSegmentator T12 / L3 / L4 (L3 → label 50)
                                ├─► OpenAI-compatible LLM (eight whitelisted numbers only)
                                └─► DATA_DIR/reports (signed reports, signing key)
```

## Run (Windows, venv in `intern2\venv`)

Open two `cmd` windows in `intern2` and activate the venv in each:

```cmd
venv\Scripts\activate.bat
cd ct-body-composition
```

Terminal 1, segmentation server:

```cmd
cd server_segment
python server.py
```

Terminal 2, frontend (opens http://localhost:8501):

```cmd
streamlit run app.py
```

Settings are in `.env` (copy of `.env.example`, same names as ct-scan-test). The app opens on a synthetic phantom,
so every tab works without patient data. To try the model, click **Open CT**, choose
`CT-Muscle-and-Fat-Segmentation/demo/example.nii.gz`, then **Run segmentation** (≈70 s on a 4 GB RTX 3050 laptop GPU).

Install (once): `pip install -r requirements.txt -r server_segment/requirements.txt` plus the repo's
`CT-Muscle-and-Fat-Segmentation/requirements.txt` (torch CUDA, nnunetv2, TotalSegmentator).

## Segmentation server (`server_segment/server.py`)

| Env | Default | |
|---|---|---|
| `MODEL` | `mazurowski` | `demo` = HU thresholds, no GPU (pipeline test only) |
| `SEG_REPO` | `../../CT-Muscle-and-Fat-Segmentation` | the cloned repo with the model folder |
| `SEG_CHECKPOINT` | `final` | `final` or `best` |
| `SEG_TTA` | `1` | mirroring test-time augmentation; `0` is ~4× faster on the network but IMAT less stable (`benchmark.py`) |
| `WARMUP` | `1` | one tiny forward pass at start-up so the first CT does not pay for cuDNN set-up |
| `AUTO_VERTEBRAE` / `TOTALSEG_FAST` | `1` / `0` | find T12/L3/L4; `--fast` 3 mm models |
| `API_KEY`, `HOST`, `PORT` | off, `127.0.0.1`, `8000` | set `INFERENCE_API_KEY` in the app's `.env` to match |

`POST /segment` returns the mask (`.nii.gz`, labels 1 muscle · 2 SAT · 3 VAT · 4 IMAT · 50 L3 vertebra) with headers
`X-Labels`, `X-Run-Id` and `X-Body-Composition` (vertebra slices plus the repo's 2D/3D metrics). Also
`/convert` (DICOM .zip → NIfTI, no patient tags), `/runs/{id}`, `/device` (also `busy`, `queued`), `/health` (`ready`
once the models are loaded).

The HTTP API (routes, headers, run-record fields) follows the inference server of `ct-scan-test` so the same
frontend contract applies; the implementation below is new.

Design (`engine.py`, `monitor.py`, `dicom_zip.py`):

- **Resident models.** The nnU-Net predictor is built once at start-up with the settings of `predict_muscle_fat.py`
  (fold 5, tile step 0.5, Gaussian, mirroring, mixed precision) and kept on the GPU; TotalSegmentator is called through
  its Python API in the same process. One worker thread runs GPU jobs in arrival order. Run the server with the Python
  environment that has torch, nnunetv2 and TotalSegmentator.
- **In memory.** The upload is decoded and segmented in memory and the mask is returned from memory; only DICOM
  conversion unpacks the zip into a temporary folder (GDCM reads files), removed before the reply.
- **One copy of the formulas.** The 2D (L3) / 3D (T12–L4) metrics come from `core/bodycomp.py`, the module the
  frontend uses, on RAS-reoriented arrays so slice numbers match the viewer.
- **Telemetry.** A single `nvidia-smi` loop (psutil without an NVIDIA GPU) fills a ring buffer; each run record takes
  the samples of its own time window.

Measured on the demo CT (512×512×53, RTX 3050 laptop 4 GB, TTA on) against the previous subprocess-per-request server:

| | previous | resident |
|---|---|---|
| nnU-Net stage | 57 s | 36 s |
| TotalSegmentator stage | 46 s | 31 s |
| total per CT | 105 s | 68 s |
| mask agreement | | 0.0016 % of voxels differ, Dice ≥ 0.9997 per label, L3 identical; 3D biomarkers within 0.003 % |

## Notes

- Reference cut-offs (`core/analysis.py`): SMI and muscle attenuation (Martin 2013), VAT area (Doyle 2013). Replace with
  values validated for your population.
- Research and decision-support use only; not a cleared medical device. Model licence CC BY-NC 4.0 (arXiv:2502.09779).

## Layout

```
app.py              page: header (live Seg time while segmenting), tabs
ui/state.py         session state and actions (load, segment, report, sign)
ui/sidebar.py       sidebar sections
ui/live.py          real-time slice viewers, sliders, file buttons, 3D model (st.components.v2)
ui/tabs.py          the six tabs
core/analysis.py    volumetrics, L3 indices, references, rule narrative, CSV
core/biomarkers.py  the eight transmitted biomarkers, whitelist check (blocks anything else), derived indices
core/render.py      MPR slices, Plotly viewers and charts, 3D surfaces
core/labels.py      label schemes, roles, colours, W/L presets
core/llm_client.py  OpenAI-compatible report drafting (JSON mode, retry, thinking-tag stripping)
core/signing.py     Ed25519 sign-off
core/seg_client.py  client for server_segment/server.py
server_segment/server.py segmentation server: routes, GPU job queue
server_segment/engine.py resident nnU-Net + TotalSegmentator, metrics via core/bodycomp.py
server_segment/monitor.py  GPU/CPU telemetry ring buffer, run records
server_segment/dicom_zip.py  DICOM .zip -> NIfTI
server_segment/benchmark.py  TTA on/off x AMP-FP16/FP32 speed and agreement
```
server_segment/select_cases.py  TotalSegmentator-dataset CTs that cover T12–L4 (cases.csv, selected.txt)
server_segment/fetch_totalseg.py  only the T12/L3/L4 masks (~20 MB) or chosen CTs out of the 23.6 GB Zenodo zip
