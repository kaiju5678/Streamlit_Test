# BodyComp (Streamlit): CT body composition workstation

The workflow of `ct-scan-test` (Next.js) rebuilt in Python/Streamlit, with the Mazurowski Lab muscle/fat
nnU-Net from `CT-Muscle-and-Fat-Segmentation` as the segmentation model.

| Tab | What it does |
|---|---|
| **Overview** | Muscle, visceral-fat and total-adipose volumes; L3 muscle attenuation; axial view with mask; 3D surface model; tissue composition; L3 indices against reference cut-offs; report (HTML → PDF) and CSV export |
| **Slice viewer** | Synced axial / coronal / sagittal views with crosshair (click to move), real-time scrolling in every view, zoom, caliper measurements |
| **Metrics table** | Per-label voxels, volume, mean and SD HU, slice and L3 areas; the repo's own 2D (L3) and 3D (T12–L4) metrics; cross-sectional area by slice (click to navigate); CSV |
| **AI report** | Draft from an OpenAI-compatible LLM (MedGemma) or rule-based fallback; flags numbers not in the measurements; editable; physician sign-off with Ed25519 signature |
| **Transmitted JSON** | The exact de-identified payload, LLM request, raw response and signature receipt |
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
browser ─► Streamlit app.py (:8501) ─► inference/server.py (:8000) ─► predict_muscle_fat.py (nnU-Net, labels 1–4)
                                │                                  └► TotalSegmentator T12 / L3 / L4 (L3 → label 50)
                                ├─► OpenAI-compatible LLM (de-identified numbers only)
                                └─► DATA_DIR/reports (signed reports, signing key)
```

## Run (Windows, venv in `intern2\venv`)

```cmd
ct-body-composition\run_server.bat       terminal 1  (segmentation server)
ct-body-composition\run_frontend.bat     terminal 2  ->  http://localhost:8501
```

Settings are in `.env` (copy of `.env.example`, same names as ct-scan-test). The app opens on a synthetic phantom,
so every tab works without patient data. To try the model, click **Open CT**, choose
`CT-Muscle-and-Fat-Segmentation/demo/example.nii.gz`, then **Run segmentation** (≈2–5 min on a 4 GB RTX 3050).

Install (once): `pip install -r requirements.txt -r inference/requirements.txt` plus the repo's
`CT-Muscle-and-Fat-Segmentation/requirements.txt` (torch CUDA, nnunetv2, TotalSegmentator).

## Segmentation server (`inference/server.py`)

| Env | Default | |
|---|---|---|
| `MODEL` | `mazurowski` | `demo` = HU thresholds, no GPU (pipeline test only) |
| `SEG_REPO` | `../../CT-Muscle-and-Fat-Segmentation` | the cloned repo with the model folder |
| `SEG_PYTHON` | current python | python with torch + nnunetv2 + TotalSegmentator |
| `SEG_CHECKPOINT` | `final` | `final` or `best` |
| `AUTO_VERTEBRAE` / `TOTALSEG_FAST` | `1` / `0` | find T12/L3/L4; `--fast` 3 mm models |
| `API_KEY`, `HOST`, `PORT` | off, `127.0.0.1`, `8000` | set `INFERENCE_API_KEY` in the app's `.env` to match |

`POST /segment` returns the mask (`.nii.gz`, labels 1 muscle · 2 SAT · 3 VAT · 4 IMAT · 50 L3 vertebra) with headers
`X-Labels`, `X-Run-Id` and `X-Body-Composition` (vertebra slices plus the repo's 2D/3D metrics). Also
`/convert` (DICOM .zip → NIfTI, no patient tags), `/runs/{id}`, `/device`, `/health`. Each request uses an OS temp folder
that is deleted after the reply.

The server runs `predict_muscle_fat.py --body_composition_type None` (nnU-Net only) and then TotalSegmentator itself:
the repo's vertebra step calls `os.system` with unquoted paths, which fails when a path contains a space ("Vs Code").
Metrics use the repo's formulas, computed on the RAS-reoriented volume so slice numbers match the viewer.

## Notes

- Reference cut-offs (`core/analysis.py`): SMI and muscle attenuation (Martin 2013), VAT area (Doyle 2013). Replace with
  values validated for your population.
- Research and decision-support use only; not a cleared medical device. Model licence CC BY-NC 4.0 (arXiv:2502.09779).

## Layout

```
app.py              page: header, live segmentation progress, tabs
ui/state.py         session state and actions (load, segment, report, sign)
ui/sidebar.py       sidebar sections
ui/live.py          real-time slice viewers, sliders, file buttons, 3D model (st.components.v2)
ui/tabs.py          the six tabs
core/analysis.py    volumetrics, L3 indices, references, payload, rule narrative, CSV
core/render.py      MPR slices, Plotly viewers and charts, 3D surfaces
core/labels.py      label schemes, roles, colours, W/L presets
core/llm_client.py  OpenAI-compatible report drafting (JSON mode, retry, thinking-tag stripping)
core/signing.py     Ed25519 sign-off
core/seg_client.py  client for inference/server.py
inference/server.py segmentation server
```
