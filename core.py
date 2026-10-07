"""Pure functions (no Streamlit) so they can be tested and reused by other teammates."""
import os, shlex, subprocess
import numpy as np
import nibabel as nib

LABELS = {  # id: (name, RGB)  -- matches nnU-Net label order in the training material
    1: ("Skeletal muscle", (66, 133, 244)),
    2: ("Subcutaneous fat", (244, 194, 66)),
    3: ("Visceral fat", (219, 90, 60)),
    4: ("Intramuscular fat", (140, 70, 210)),
}


def _canonical(path):
    return nib.as_closest_canonical(nib.load(path))


def load_volume(path):
    img = _canonical(path)
    return np.asarray(img.dataobj, dtype=np.float32), tuple(float(z) for z in img.header.get_zooms()[:3])


def load_mask(path):
    return np.asarray(_canonical(path).dataobj).astype(np.uint8)


def render_slice(vol, k, width=400, level=50, mask=None, opacity=0.4):
    """Axial slice -> float RGB (0-1) with window/level and optional mask overlay."""
    lo, hi = level - width / 2, level + width / 2
    g = np.clip((vol[:, :, k] - lo) / (hi - lo), 0, 1)
    rgb = np.stack([g] * 3, -1)
    if mask is not None:
        m = mask[:, :, k]
        for lab, (_, c) in LABELS.items():
            s = m == lab
            rgb[s] = (1 - opacity) * rgb[s] + opacity * np.array(c) / 255
    return np.ascontiguousarray(rgb.transpose(1, 0, 2)[::-1, ::-1])  # anterior up, radiological view


def mask_metrics(vol, mask, zooms):
    """Quick preview numbers only. The official 8-metric JSON comes from the biomarker module."""
    vox_cm3 = float(np.prod(zooms)) / 1000
    rows = {LABELS[l][0] + " (cm³)": round(float((mask == l).sum()) * vox_cm3, 1) for l in LABELS}
    mu = vol[mask == 1]
    rows["Muscle mean density (HU)"] = round(float(mu.mean()), 1) if mu.size else None
    return rows


def _mock_segment(inp, out, log):
    """PLACEHOLDER: HU thresholds only. Replace via SEGMENT_CMD once nnU-Net runs on the Jetson."""
    img = _canonical(inp)
    v = np.asarray(img.dataobj, dtype=np.float32)
    nx, ny = v.shape[:2]
    X, Y = np.meshgrid(np.arange(nx) - nx / 2, np.arange(ny) - ny / 2, indexing="ij")
    inner = (X / (nx * 0.22)) ** 2 + (Y / (ny * 0.18)) ** 2 < 1
    inner = inner[:, :, None]
    m = np.zeros(v.shape, np.uint8)
    m[(v >= -29) & (v <= 150)] = 1
    fat = (v >= -190) & (v <= -30)
    m[fat & ~inner] = 2
    m[fat & inner] = 3
    nib.save(nib.Nifti1Image(m, img.affine), out)
    log("MOCK segmentation done (not nnU-Net).")


def run_segmentation(inp, out, log):
    """If env SEGMENT_CMD is set, e.g. 'python seg.py {input} {output}', run it; else use the mock."""
    cmd = os.environ.get("SEGMENT_CMD")
    if not cmd:
        return _mock_segment(inp, out, log)
    log(f"Running: {cmd}")
    p = subprocess.run(shlex.split(cmd.format(input=inp, output=out)), capture_output=True, text=True)
    log((p.stdout + p.stderr).strip() or "(no output)")
    if p.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"Segmentation failed (exit {p.returncode})")
