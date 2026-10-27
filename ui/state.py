"""Session state and actions (the Workstation.tsx logic of ct-scan-test).

Everything lives in st.session_state for this browser tab. Nothing is written to disk except signed
reports (DATA_DIR/reports), as in the reference app.
"""
from __future__ import annotations

import copy
import io
import json
import re
import threading
import time
import uuid
import zipfile
from datetime import datetime

import numpy as np
import streamlit as st

import config
from core import analysis as an
from core import biomarkers, bodycomp
from core import labels as lb
from core import llm_client, nifti, phantom, seg_client, signing

ss = st.session_state

TABS = ["Overview", "Slice viewer", "Metrics table", "AI report", "Transmitted JSON", "Run log"]
NII = re.compile(r"\.nii(\.gz)?$", re.I)
MASK_NAME = re.compile(r"(seg|mask|label|lbl)", re.I)
VERTEBRA = re.compile(r"vertebrae_(T12|L3|L4)\.nii(\.gz)?$", re.I)
# label names of the Mazurowski model (its dataset.json), so a mask from file is named like a server result
MODEL_LABELS = {"1": "muscle", "2": "sat", "3": "vat", "4": "muscle_fat"}

DEFAULTS = dict(
    vol=None, mask=None, labels={}, scheme="generic", stats=None, ref="",
    cur={"i": 0, "j": 0, "k": 0}, wl={"w": 400, "l": 40}, opacity=0.45, tool="Cross",
    meas=[], pending=None, meas_id=1, l3=None,
    status={"text": "", "error": False}, report=None, signature=None, run=None, body=None, seg_secs=None,
    weblog=[], job=None, action=None, editing=False, confirming=False, sign_error="",
    mesh={}, hide_server=False, up_nonce=0, mask_conflict=None, last_click={}, tab_default=None, tab_nonce=0,
    p_sex="", p_age=None, p_h=None, p_w=None, dark=False,
)


def init() -> None:
    for k, v in DEFAULTS.items():
        if k not in ss:
            ss[k] = copy.deepcopy(v)
    if "booted" not in ss:
        ss.booted = True
        load_sample()


def now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def log(level: str, tag: str, msg: str) -> None:
    ss.weblog.append({"t": now(), "level": level, "tag": tag, "msg": msg, "origin": "web"})


def status(text: str, error: bool = False) -> None:
    ss.status = {"text": text, "error": error}


def patient() -> dict:
    return {"sex": ss.p_sex or "", "age": ss.p_age, "heightCm": ss.p_h, "weightKg": ss.p_w}


# ---------------------------------------------------------------- loading
def open_volume(v: dict, name: str, sample: bool, ct_bytes: bytes | None) -> None:
    hu = v["data"]
    ss.vol = {"hu": hu, "affine": v["affine"], "spacing": v["spacing"], "shape": hu.shape, "src": v["src"],
              "name": name, "sample": sample, "bytes": ct_bytes, "id": uuid.uuid4().hex[:8]}
    ss.ref = an.case_ref(name)
    for k in ("mask", "stats", "l3", "report", "signature", "run", "body", "pending", "seg_secs"):
        ss[k] = None
    ss.labels, ss.meas, ss.mesh = {}, [], {}
    ss.editing = ss.confirming = False
    nx, ny, nz = hu.shape
    ss.cur = {"i": nx // 2, "j": ny // 2, "k": nz // 2}


def apply_mask(data: np.ndarray, name: str, source: str, key: str, names: dict | None = None) -> dict:
    counts = np.bincount(data.ravel(), minlength=256)
    labels = lb.labels_from_names(counts, names) if names else lb.labels_for(counts, key)
    ss.mask = {"data": data, "name": name, "source": source, "counts": counts, "id": uuid.uuid4().hex[:8]}
    ss.labels = labels
    ss.scheme = key
    ss.stats = an.volume_stats(ss.vol["hu"], data)
    ss.mesh = {}
    via = "names from server" if names else f"{key} scheme"
    log("INFO", "MASK", f"{name}: {', '.join(d['name'] for d in labels.values())} ({source}, {via})")
    return labels


def read_label(raw: bytes, name: str) -> np.ndarray:
    """A label volume on the grid of the open CT."""
    if ss.vol is None:
        raise ValueError("Open the CT before its mask.")
    m = nifti.read(raw, as_label=True)["data"]
    if m.shape != ss.vol["shape"]:
        a, b = "×".join(map(str, m.shape)), "×".join(map(str, ss.vol["shape"]))
        which = "the sample phantom" if ss.vol["sample"] else ss.vol["name"]
        raise ValueError(f"{name} is {a} but the open study ({which}) is {b}, and the CT it was segmented from was not "
                         f"found in {', '.join(str(d) for d in config.CT_SEARCH_DIRS) or 'CT_SEARCH_DIRS'}. "
                         f"Choose the CT together with the mask (Open CT or Open mask, both files at once).")
    return m


class _MemFile:
    """An uploaded file kept in memory while the user decides about it (the uploader resets after each upload)."""

    def __init__(self, name: str, data: bytes):
        self.name, self._data = name, data

    def getvalue(self) -> bytes:
        return self._data


class _DiskFile:
    """A file found on disk, with the two methods of an uploaded file that load_results uses."""

    def __init__(self, path):
        self.path, self.name = path, path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


def find_ct(raw: bytes):
    """The CT a mask was segmented from: a NIfTI in CT_SEARCH_DIRS with the same voxel grid as the mask."""
    shape, affine = nifti.grid(raw)
    for d in config.CT_SEARCH_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.nii*")):
            if NII.search(p.name) and not MASK_NAME.search(p.name) and not VERTEBRA.search(p.name) \
                    and nifti.same_grid(p, shape, affine):
                return p
    return None


def find_vertebrae(ct_name: str) -> list:
    """vertebrae_T12/L3/L4 that predict_muscle_fat.py wrote for this CT (<stem>_vertebrae_segmentation/)."""
    folder = f"{NII.sub('', ct_name)}_vertebrae_segmentation"
    for d in config.CT_SEARCH_DIRS:
        if (d / folder).is_dir():
            return [_DiskFile(p) for p in sorted((d / folder).glob("vertebrae_*.nii*")) if VERTEBRA.search(p.name)]
    return []


def _grid_match(m: dict) -> str:
    """How a label volume (nifti.read) sits against the open CT: 'same', 'place' (same size, but a different position,
    spacing or orientation in space) or 'size'."""
    if ss.vol is None or m["data"].shape != ss.vol["shape"]:
        return "size"
    return "same" if np.allclose(m["affine"], ss.vol["affine"], atol=1e-2) else "place"


def _grid_text(shape, spacing) -> str:
    return f"{'×'.join(map(str, shape))} voxels at {' × '.join(f'{z:.2f}' for z in spacing)} mm"


def load_mask_bytes(raw: bytes, name: str, source: str, key: str, names: dict | None = None) -> int | None:
    return load_mask_array(read_label(raw, name), name, source, key, names)


def load_mask_array(m: np.ndarray, name: str, source: str, key: str, names: dict | None = None) -> int | None:
    labels = apply_mask(m, name, source, key, names)
    # Automatic L3: if the segmentation includes the L3 vertebra, use the middle of it.
    l3_label = next((L for L, d in labels.items() if d["name"] == "L3 vertebra"), None)
    if l3_label is not None:
        k = lb.middle_slice(m, l3_label)
        if k is not None:
            ss.l3 = k
            ss.cur = {**ss.cur, "k": k}
            log("INFO", "L3", f"L3 set automatically at axial slice {k + 1} (middle of the segmented L3 vertebra). Check it before reporting.")
            return k
    return None


def load_sample() -> None:
    t0 = time.time()
    ct, mask, affine, zooms = phantom.make()
    open_volume({"data": ct, "affine": affine, "spacing": zooms, "src": "RAS (synthetic)"},
                "sample_phantom_T12-L4.nii.gz", True, None)
    apply_mask(mask, "sample_phantom_seg.nii.gz", "sample", "generic")
    ss.l3 = ct.shape[2] // 2
    ss.cur = {**ss.cur, "k": ss.l3}
    ss.p_sex, ss.p_age, ss.p_h, ss.p_w = "M", 64, 172.0, 68.0
    status("Sample loaded. Example patient values filled in.")
    nx, ny, nz = ct.shape
    log("INFO", "SAMPLE", f"synthetic phantom {nx}×{ny}×{nz} built in {(time.time() - t0) * 1000:.0f} ms")


def load_ct(data: bytes, name: str) -> None:
    t0 = time.time()
    v = nifti.read(data)
    open_volume(v, name, False, nifti.gzipped(data))
    ss.p_sex, ss.p_age, ss.p_h, ss.p_w = "", None, None, None
    status(f"Loaded {name}")
    nx, ny, nz = v["data"].shape
    sx, sy, sz = v["spacing"]
    log("INFO", "LOAD", f"{name}: {len(data) / 1e6:.1f} MB, {nx}×{ny}×{nz} at {sx:.2f}×{sy:.2f}×{sz:.2f} mm ({v['src']}) "
                        f"in {(time.time() - t0) * 1000:.0f} ms")


def _safe(fn, tag: str, *args) -> None:
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001
        status(str(e), error=True)
        log("ERROR", tag, str(e))


# ---------------------------------------------------------------- callbacks (sidebar)
def load_results(files: list, mode: str = "check") -> None:
    """A finished segmentation: the tissue mask (labels 1–4), optionally with vertebrae_T12/L3/L4 from the same run
    (the output folder of predict_muscle_fat.py). L3 becomes label 50, as in a server result, and the repo's
    2D/3D metrics are computed.

    mode 'check': a mask that does not fit an open CT is not applied; ss.mask_conflict asks the user (sidebar dialog).
    With only the sample phantom or nothing open, the CT the mask came from is opened. 'open_ct': open that CT first.
    'keep': use the mask on the open CT although its position in space differs (same size only)."""
    nii = [f for f in files if NII.search(f.name)]
    vert = {VERTEBRA.search(f.name)[1].upper(): f for f in nii if VERTEBRA.search(f.name)}
    tissue = [f for f in nii if not VERTEBRA.search(f.name)]
    if not tissue:
        raise ValueError("No segmentation mask among the chosen files (e.g. example_segmentation.nii.gz).")
    tissue.sort(key=lambda f: not MASK_NAME.search(f.name))     # prefer *_segmentation.nii.gz
    f = tissue[0]
    if len(tissue) > 1:
        log("WARN", "MASK", f"{len(tissue)} masks chosen; using {f.name}")
    raw = f.getvalue()
    opened = None
    m0 = nifti.read(raw, as_label=True)
    fit = _grid_match(m0)
    if fit != "same" and mode != "keep":
        ct = find_ct(raw)
        if mode == "check" and ss.vol is not None and not ss.vol["sample"]:
            # A real CT is open: never replace it without asking
            ss.mask_conflict = {
                "files": [_MemFile(x.name, x.getvalue()) for x in files], "mask": f.name, "ct_name": ss.vol["name"],
                "mask_grid": _grid_text(m0["data"].shape, m0["spacing"]),
                "ct_grid": _grid_text(ss.vol["shape"], ss.vol["spacing"]),
                "why": "a different size" if fit == "size" else "the same size but a different position or orientation in space",
                "found": ct.name if ct else None, "can_keep": fit == "place"}
            status(f"{f.name} does not match the open CT ({ss.vol['name']}); it was not applied.", error=True)
            log("WARN", "MASK", f"{f.name} ({ss.mask_conflict['mask_grid']}) does not match {ss.vol['name']} "
                                f"({ss.mask_conflict['ct_grid']}); asking before anything changes")
            return
        # The sample phantom or nothing is open, or the user chose to: open the CT the mask was segmented from first.
        if ct is not None:
            load_ct(ct.read_bytes(), ct.name)
            opened = ct.name
            log("INFO", "MASK", f"{f.name} does not fit the open study; opened its CT {ct} first")
    seg = read_label(raw, f.name)
    if not vert:
        vert = {VERTEBRA.search(vf.name)[1].upper(): vf for vf in find_vertebrae(ss.vol["name"])}
    vols = {}
    for level, vf in vert.items():
        try:
            vols[level] = read_label(vf.getvalue(), vf.name)
        except ValueError as e:
            log("WARN", "L3", str(e))
    peaks = {level: bodycomp.peak_slice(vols.get(level)) for level in ("T12", "L3", "L4")}
    body = bodycomp.compute(ss.vol["hu"], seg, ss.vol["spacing"], peaks["T12"], peaks["L3"], peaks["L4"])
    names = dict(MODEL_LABELS)
    merged = seg
    if "L3" in vols and vols["L3"].any():
        merged = seg.copy()
        merged[vols["L3"] > 0] = 50
        names["50"] = "vertebrae_L3"
    k = load_mask_array(merged, f.name, "file", "generic", names)
    ss.body = body if vols else None
    ss.report = ss.signature = None
    found = ", ".join(level for level in ("T12", "L3", "L4") if level in vols)
    loaded = f"Loaded {opened} and {f.name}" if opened else f"Loaded {f.name}"
    if k is not None:
        status(f"{loaded} with {found}. L3 set at slice {k + 1}; metrics are on the Metrics table tab.")
    else:
        status(f"{loaded}. To set L3 automatically, also choose vertebrae_L3.nii.gz.")
    log("INFO", "MASK", f"{f.name} opened from file" + (f" with vertebrae {found}" if found else ""))


def _upload(key: str) -> list:
    f = ss.get(key)
    return [] if not f else f if isinstance(f, list) else [f]


def cb_upload_ct() -> None:
    files = _upload(f"up_ct_{ss.up_nonce}")
    ct = next((x for x in files if NII.search(x.name) and not MASK_NAME.search(x.name) and not VERTEBRA.search(x.name)), None)
    rest = [x for x in files if x is not ct]
    if ct:
        _safe(load_ct, "LOAD", ct.getvalue(), ct.name)
    if rest and ss.vol is not None:
        _safe(load_results, "MASK", rest)
    ss.up_nonce += 1


def cb_upload_mask() -> None:
    files = _upload(f"up_mask_files_{ss.up_nonce}")
    nii = [x for x in files if NII.search(x.name) and not VERTEBRA.search(x.name)]
    # The CT may be chosen together with the mask: the file whose name does not look like a mask.
    ct = next((x for x in nii if not MASK_NAME.search(x.name)), None) if len(nii) > 1 else None
    if ct:
        _safe(load_ct, "LOAD", ct.getvalue(), ct.name)
    rest = [x for x in files if x is not ct]
    if rest:
        _safe(load_results, "MASK", rest)
    if files:
        ss.up_nonce += 1


def cb_mask_conflict(choice: str) -> None:
    """The user's answer in the mask / CT mismatch dialog: 'open_ct', 'keep' or 'cancel'."""
    c, ss.mask_conflict = ss.mask_conflict, None
    if not c:
        return
    if choice == "cancel":
        status(f"{c['mask']} was not opened; {c['ct_name']} is still open.")
        log("INFO", "MASK", f"{c['mask']} not opened (does not match {c['ct_name']})")
        return
    if choice == "keep":
        log("WARN", "MASK", f"{c['mask']} used on {c['ct_name']} although its position in space differs")
    _safe(load_results, "MASK", c["files"], choice)


def cb_upload_dicom(kind: str) -> None:
    f = ss.get(f"up_dcm_{kind}_{ss.up_nonce}")
    if not f:
        return
    ss.action = ("dicom", f if isinstance(f, list) else [f])
    ss.up_nonce += 1


def cb_sample() -> None:
    load_sample()


def cb_theme() -> None:
    ss.dark = not ss.dark


def cb_set_l3() -> None:
    ss.l3 = ss.cur["k"]
    log("INFO", "L3", f"L3 set by hand at axial slice {ss.l3 + 1}")


def cb_go_l3() -> None:
    if ss.l3 is not None:
        ss.cur = {**ss.cur, "k": ss.l3}




def cb_wl_number() -> None:
    ss.wl = {"w": max(1, int(ss.w_ww or 1)), "l": int(ss.w_wl or 0)}




def cb_clear_meas() -> None:
    ss.meas, ss.pending = [], None




# Sidebar tool -> (measurement kind, points to click)
TOOLS = {"Caliper": ("line", 2), "ROI": ("roi", 2), "Angle": ("angle", 3), "HU": ("point", 1)}


def cb_tool() -> None:
    ss.pending = None


def apply_click(view: str, i: int, j: int, k: int) -> None:
    """Click on a slice: move the crosshair, or place a point of the selected measurement tool."""
    plane = {"axial": ("k", (i, j)), "coronal": ("j", (i, k)), "sagittal": ("i", (j, k))}[view]
    if ss.tool not in TOOLS:
        ss.cur = {"i": i, "j": j, "k": k}
        return
    kind, need = TOOLS[ss.tool]
    sl = ss.cur[plane[0]]
    p = ss.pending
    pts = (p["pts"] if p and p["view"] == view and p["slice"] == sl and p["kind"] == kind else []) + [plane[1]]
    if len(pts) < need:
        ss.pending = {"view": view, "slice": sl, "kind": kind, "pts": pts}
        return
    res = an.measure(kind, view, sl, pts, ss.vol["spacing"], ss.vol["hu"], ss.mask["data"] if ss.mask else None, ss.labels)
    ss.meas.append({"id": ss.meas_id, "kind": kind, "view": view, "slice": sl, "pts": pts, **res})
    ss.meas_id += 1
    ss.pending = None


def cb_go_meas(m: dict) -> None:
    axis = {"axial": "k", "coronal": "j", "sagittal": "i"}[m["view"]]
    ss.cur = {**ss.cur, axis: m["slice"]}


def cb_delete_meas(mid: int) -> None:
    ss.meas = [m for m in ss.meas if m["id"] != mid]


def cb_pick_slice(key: str) -> None:
    sel = (ss.get(key) or {}).get("selection") or {}
    pts = sel.get("points") or []
    if pts:
        ss.cur = {**ss.cur, "k": int(pts[0].get("x", ss.cur["k"]))}


def open_tab(name: str) -> None:
    """Select a tab from a callback; the tabs switch in the browser otherwise (app.py)."""
    ss.tab_default, ss.tab_nonce = name, ss.tab_nonce + 1


def cb_generate() -> None:
    ss.action = ("report",)
    open_tab("AI report")


def cb_toggle_edit() -> None:
    if not ss.editing and ss.report:
        n = ss.report["narrative"]
        for f in llm_client.NARRATIVE_KEYS:
            ss[f"ed_{f}"] = n[f]
        ss.ed_lifestyle = "\n".join(n["lifestyle"])
    ss.editing = not ss.editing


def cb_edit(field: str) -> None:
    if not ss.report:
        return
    value = ss[f"ed_{field}"]
    if field == "lifestyle":
        value = [x.strip() for x in value.split("\n") if x.strip()]
    ss.report = {**ss.report, "narrative": {**ss.report["narrative"], field: value}, "edited": True}


def cb_sign(payload: dict) -> None:
    ss.sign_error = ""
    try:
        s = signing.sign(payload, ss.report, ss.get("reader", ""), ss.get("remarks", ""), ss.ref, ss.vol["sample"])
        ss.signature = s
        ss.signed_remarks = ss.get("remarks", "")
        log("SUCCESS", "SIGN", f"signed by {s['reader']}; sha256 {s['sha256'][:16]}…; stored {s['file']}")
        status("Report signed and stored.")
    except Exception as e:  # noqa: BLE001
        ss.sign_error = str(e)
    ss.confirming = False


# ---------------------------------------------------------------- segmentation job (runs in a thread)
def cb_segment() -> None:
    if ss.job is not None:  # a second click that arrived before the rerun disabled the button
        return
    v = ss.vol
    if v is None or not v["bytes"]:
        status("Open a CT file first. The sample phantom already has a mask.", error=True)
        return
    run_id = uuid.uuid4().hex[:12]
    box: dict = {}
    data, name = v["bytes"], v["name"]

    def work():
        try:
            box["res"] = seg_client.segment(data, name if name.endswith(".gz") else name + ".gz", run_id)
        except Exception as e:  # noqa: BLE001
            box["err"] = str(e)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    ss.job = {"thread": t, "box": box, "run_id": run_id, "t0": time.time(), "vol_id": v["id"]}
    status(f"Sending {name} to the segmentation server…")
    log("INFO", "INFERENCE", f"uploading {name} ({len(data) / 1e6:.1f} MB), run {run_id}")


def finish_job() -> None:
    job, ss.job = ss.job, None
    box, run_id = job["box"], job["run_id"]
    ss.run = seg_client.run_record(run_id) or ss.run
    if job["vol_id"] != (ss.vol or {}).get("id"):
        log("WARN", "INFERENCE", f"run {run_id} finished for a study that is no longer open; result ignored")
        return
    if "err" in box:
        status(box["err"], error=True)
        log("ERROR", "INFERENCE", box["err"])
        return
    res = box["res"]
    secs = ss.seg_secs = time.time() - job["t0"]
    try:
        ss.body = res.get("body")
        stem = NII.sub("", ss.vol["name"])
        k = load_mask_bytes(res["mask_bytes"], f"{stem}_seg.nii.gz", "inference", res["scheme"], res.get("labels"))
        ss.report = ss.signature = None
        status(f"Segmentation complete. L3 set automatically at slice {k + 1}; please confirm it." if k is not None
               else "Segmentation complete. Check the mask before setting L3.")
        log("SUCCESS", "INFERENCE", f"mask back after {secs:.1f} s · run {run_id}")
    except Exception as e:  # noqa: BLE001
        status(str(e), error=True)
        log("ERROR", "INFERENCE", str(e))


# ---------------------------------------------------------------- long actions run in the main script
def run_action(action: tuple) -> None:
    kind = action[0]
    if kind == "dicom":
        _dicom(action[1])
    elif kind == "report":
        _report()


def _dicom(files: list) -> None:
    t0 = time.time()
    try:
        with st.spinner("Converting DICOM on the inference server…", show_time=True):
            if len(files) == 1 and files[0].name.lower().endswith(".zip"):
                body, label = files[0].getvalue(), re.sub(r"\.zip$", "", files[0].name, flags=re.I)
            else:
                buf = io.BytesIO()
                n = 0
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # DICOM pixel data barely compresses
                    for f in files:
                        if re.search(r"^\.|DICOMDIR$|\.(txt|xml|json|jpg|png|pdf|exe|inf|ini|html?)$", f.name, re.I):
                            continue
                        n += 1
                        z.writestr(f"f{n:06d}.dcm", f.getvalue())
                if not n:
                    raise ValueError("No DICOM files in that folder.")
                body, label = buf.getvalue(), "dicom_series"
                log("INFO", "DICOM", f"{n} files packed in {time.time() - t0:.1f} s")
            nii, listing, chosen = seg_client.convert_dicom(body)
        info = next((s for s in listing if s.get("id") == chosen), None)
        if len(listing) > 1:
            log("INFO", "DICOM", f"{len(listing)} series found; opened the largest CT series.")
        log("SUCCESS", "DICOM", f"converted {info['modality'] or 'series'} \"{info['description'] or 'no description'}\", "
                                f"{info['slices']} slices in {time.time() - t0:.1f} s; patient metadata not carried over"
            if info else f"converted in {time.time() - t0:.1f} s")
        load_ct(nii, re.sub(r"[^\w.\-]", "_", label) + ".nii.gz")
    except Exception as e:  # noqa: BLE001
        status(str(e), error=True)
        log("ERROR", "DICOM", str(e))


def _report() -> None:
    a = analysis()
    payload = a["payload"]
    if not payload:
        return
    key = a["payload_key"]
    ss.signature = None

    def rules(reason: str | None = None) -> dict:
        derived = biomarkers.derived(payload)
        return {"source": "rules", "generatedAt": now(), "narrative": an.rule_narrative(payload, derived),
                "derived": derived, "unverifiedNumbers": [], "fallbackReason": reason}

    def store(g: dict) -> None:  # session state only: no st call
        ss.report = {**g, "payloadKey": key, "edited": False}
        ss.editing = False
        status("Model unavailable; rule-based draft ready for review." if g.get("fallbackReason") else "Draft report ready for review.")

    if llm_client.configured():
        log("INFO", "REPORT", f"sending {len(key.encode())} bytes ({len(payload)} whitelisted biomarkers) to {config.LLM_MODEL}")
        # A rerun requested while the model is answering (a timer fragment, a click) stops this script at its next st
        # call, the spinner's exit, with a BaseException that skips the handlers below. So the draft is stored inside
        # the spinner, or the reply would be lost and the click would show nothing.
        with st.spinner(f"Generating report with {config.LLM_MODEL}…", show_time=True):
            try:
                g = llm_client.generate(payload)
                n_bad = len(g["unverifiedNumbers"])
                log("SUCCESS", "REPORT", f"draft received in {g['latencyMs']} ms{' after a retry' if g['attempts'] > 1 else ''}"
                                         f"{f'; {n_bad} unverified numbers' if n_bad else ''}")
            except biomarkers.PayloadBlocked as e:
                reason = str(e)
                g = rules(reason)
                log("ERROR", "PHI-GATE", f"{reason}; nothing was sent, used the rule-based draft instead")
            except Exception as e:  # noqa: BLE001  keep the workflow moving
                reason = str(e)
                g = rules(reason)
                log("WARN", "REPORT", f"{config.LLM_MODEL} failed ({reason[:200]}); used the rule-based draft instead")
            store(g)
    else:
        g = rules()
        log("INFO", "REPORT", "no language model set up; used the rule-based draft")
        store(g)


# ---------------------------------------------------------------- derived view of the state
def eight_biomarkers() -> tuple[dict | None, dict | None]:
    """The eight transmitted biomarkers and the range they cover (shown locally, never sent).
    T12–L4 from the vertebrae (server run or vertebrae_* files), as in predict_muscle_fat.py; without them the
    segmented range, computed with the same formulas so the payload has the same eight fields."""
    v, mask, stats = ss.vol, ss.mask, ss.stats
    if v is None or mask is None:
        return None, None
    m = (ss.body or {}).get("metrics_3d")
    if m:
        return biomarkers.from_metrics(m), {"source": "T12–L4", "slices": m["slice_range"]}
    if not stats or stats["k_min"] < 0:
        return None, None
    key = (v["id"], mask["id"])
    cache = ss.setdefault("bio_cache", {})
    if key not in cache:
        cache.clear()
        cache[key] = bodycomp.compute(v["hu"], mask["data"], v["spacing"], stats["k_min"], None, stats["k_max"])["metrics_3d"]
    m = cache[key]
    return biomarkers.from_metrics(m), {"source": "segmented range (T12/L4 not found)", "slices": m["slice_range"]}


def analysis() -> dict:
    v, mask, labels, stats = ss.vol, ss.mask, ss.labels, ss.stats
    p = patient()
    ev = None
    if v is not None and mask is not None and ss.l3 is not None:
        ev = an.evaluate(an.role_metrics(an.slice_stats(v["hu"], mask["data"], ss.l3), labels, v["spacing"]), p)
    payload, bio_range = eight_biomarkers()
    return {"vol": v, "mask": mask, "labels": labels, "stats": stats, "patient": p, "l3": ss.l3, "cur": ss.cur,
            "wl": ss.wl, "opacity": ss.opacity, "ev": ev, "payload": payload, "bio_range": bio_range,
            "payload_key": json.dumps(payload, sort_keys=True, ensure_ascii=False) if payload else "",
            "report": ss.report, "signature": ss.signature, "run": ss.run, "body": ss.body}
