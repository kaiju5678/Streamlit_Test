"""Volumetrics, L3 indices, reference cut-offs, the de-identified payload and the rule-based narrative.

Port of ct-scan-test/lib/{volumetrics,metrics,reference,payload,csv}.ts. Arrays are RAS-canonical
(x → right, y → anterior, z = k → superior); slice k is the axial plane.
"""
from __future__ import annotations

import hashlib
import io
import csv
import math
import re
from datetime import datetime, timezone

import numpy as np

from core.labels import ROLES

MARTIN = "Martin L, et al. J Clin Oncol 2013;31:1539–47"
DOYLE = "Doyle SL, et al. Nutr Res 2013;33:171–9 (measured at L3–L4)"


def r1(v):
    return None if v is None or not math.isfinite(v) else round(v * 10) / 10


# ---------------------------------------------------------------- volume statistics
def volume_stats(hu: np.ndarray, mask: np.ndarray) -> dict:
    """One pass: per-label voxel count, HU sum and sum of squares, and per-slice counts."""
    nz = mask.shape[2]
    flat_m = mask.ravel()
    flat_h = hu.ravel().astype(np.float64)
    n = np.bincount(flat_m, minlength=256).astype(np.float64)
    s = np.bincount(flat_m, weights=flat_h, minlength=256)
    q = np.bincount(flat_m, weights=flat_h * flat_h, minlength=256)
    slice_n = np.zeros((256, nz), np.uint32)
    for k in range(nz):
        slice_n[:, k] = np.bincount(mask[:, :, k].ravel(), minlength=256)
    n[0] = s[0] = q[0] = 0
    slice_n[0] = 0
    any_k = np.nonzero(slice_n.sum(axis=0))[0]
    return {"nz": nz, "n": n, "sum": s, "sumsq": q, "slice_n": slice_n,
            "k_min": int(any_k[0]) if any_k.size else -1, "k_max": int(any_k[-1]) if any_k.size else -1}


def _tissue(n: float, s: float, q: float, vox_mm3: float) -> dict:
    mean = s / n if n else None
    sd = math.sqrt(max(0.0, q / n - mean * mean)) if n > 1 and mean is not None else None
    return {"voxels": int(n), "cm3": n * vox_mm3 / 1000, "mean_hu": mean, "sd_hu": sd}


def label_volume(st: dict, L: int, spacing) -> dict:
    return _tissue(st["n"][L], st["sum"][L], st["sumsq"][L], float(np.prod(spacing)))


def role_volumes(st: dict, labels: dict, spacing) -> dict:
    acc = {r: [0.0, 0.0, 0.0] for r in ROLES}
    for L, d in labels.items():
        a = acc[d["role"]]
        a[0] += st["n"][L]
        a[1] += st["sum"][L]
        a[2] += st["sumsq"][L]
    vox = float(np.prod(spacing))
    return {r: _tissue(*a, vox) for r, a in acc.items()}


def role_profile(st: dict, labels: dict, spacing, role: str) -> np.ndarray:
    """Cross-sectional area (cm²) per axial slice for one tissue role."""
    out = np.zeros(st["nz"], np.float64)
    px = spacing[0] * spacing[1] / 100
    for L, d in labels.items():
        if d["role"] == role:
            out += st["slice_n"][L] * px
    return out


def slice_stats(hu: np.ndarray, mask: np.ndarray, k: int) -> dict:
    m = mask[:, :, k].ravel()
    h = hu[:, :, k].ravel().astype(np.float64)
    n = np.bincount(m, minlength=256).astype(np.float64)
    s = np.bincount(m, weights=h, minlength=256)
    n[0] = s[0] = 0
    return {"n": n, "sum": s}


# ---------------------------------------------------------------- slice measurements (sidebar tools)
# Points are plane coordinates (column axis, row axis) of a view: axial (i, j) at k, coronal (i, k) at j, sagittal (j, k) at i
def _plane(vol: np.ndarray, view: str, sl: int) -> np.ndarray:
    return {"axial": lambda: vol[:, :, sl], "coronal": lambda: vol[:, sl, :], "sagittal": lambda: vol[sl, :, :]}[view]()


def _plane_spacing(view: str, spacing) -> tuple[float, float]:
    return {"axial": (spacing[0], spacing[1]), "coronal": (spacing[0], spacing[2]), "sagittal": (spacing[1], spacing[2])}[view]


def measure(kind: str, view: str, sl: int, pts: list, spacing, hu: np.ndarray, mask: np.ndarray | None, labels: dict) -> dict:
    """One measurement on a slice: 'line' (2 points), 'roi' (centre, edge), 'angle' (3 points, vertex second), 'point'.

    Returns the short text drawn on the slice and a detail line for the list."""
    cs, rs = _plane_spacing(view, spacing)
    mm = [(p[0] * cs, p[1] * rs) for p in pts]
    plane = _plane(hu, view, sl)
    lab = _plane(mask, view, sl) if mask is not None else None
    name = lambda L: labels[L]["name"] if L in labels else "no label"
    if kind == "line":
        d = math.dist(*mm)
        return {"text": f"{d:.1f} mm", "detail": ""}
    if kind == "angle":
        (ax, ay), (vx, vy), (cx, cy) = mm
        a1, a2 = math.atan2(ay - vy, ax - vx), math.atan2(cy - vy, cx - vx)
        deg = abs(math.degrees(a1 - a2)) % 360
        deg = 360 - deg if deg > 180 else deg
        return {"text": f"{deg:.1f}°", "detail": ""}
    if kind == "point":
        c, r = pts[0]
        L = int(lab[c, r]) if lab is not None else 0
        return {"text": f"{int(plane[c, r])} HU", "detail": name(L) if lab is not None else ""}
    # roi: a circle in millimetres (an ellipse in voxels when the pixels are not square)
    r_mm = math.dist(*mm)
    (c0, r0) = pts[0]
    n0, n1 = plane.shape
    a0, a1 = max(0, int(c0 - r_mm / cs) - 1), min(n0, int(c0 + r_mm / cs) + 2)
    b0, b1 = max(0, int(r0 - r_mm / rs) - 1), min(n1, int(r0 + r_mm / rs) + 2)
    ii, jj = np.meshgrid((np.arange(a0, a1) - c0) * cs, (np.arange(b0, b1) - r0) * rs, indexing="ij")
    inside = ii ** 2 + jj ** 2 <= r_mm ** 2
    vals = plane[a0:a1, b0:b1][inside].astype(np.float64)
    if not vals.size:
        return {"text": "empty ROI", "detail": "", "r_mm": r_mm}
    area = vals.size * cs * rs / 100
    detail = f"min {int(vals.min())} · max {int(vals.max())} HU"
    if lab is not None:
        ls, cnt = np.unique(lab[a0:a1, b0:b1][inside], return_counts=True)
        top = sorted(zip(cnt, ls), reverse=True)[:3]
        detail += " · " + ", ".join(f"{c / vals.size * 100:.0f}% {name(int(L))}" for c, L in top)
    return {"text": f"{vals.mean():.0f} ± {vals.std():.0f} HU · {area:.2f} cm²", "detail": detail, "r_mm": r_mm}


def role_metrics(st: dict, labels: dict, spacing) -> dict:
    px_cm2 = spacing[0] * spacing[1] / 100
    acc = {r: [0.0, 0.0] for r in ROLES}
    for L, d in labels.items():
        acc[d["role"]][0] += st["n"][L]
        acc[d["role"]][1] += st["sum"][L]
    return {r: {"area": n * px_cm2, "hu": s / n if n else None} for r, (n, s) in acc.items()}


# ---------------------------------------------------------------- reference cut-offs (lib/reference.ts)
def bmi_of(p: dict) -> float | None:
    try:
        h, w = float(p.get("heightCm") or 0), float(p.get("weightKg") or 0)
    except ValueError:
        return None
    return w / (h / 100) ** 2 if h > 0 and w > 0 else None


def ref_smi(p: dict) -> dict | None:
    """Skeletal muscle index at L3, cm²/m²."""
    b = bmi_of(p)
    if not p.get("sex") or b is None:
        return None
    return {"value": 41 if p["sex"] == "F" else 43 if b < 25 else 53, "flag_when": "below",
            "label": "Low muscle mass", "source": MARTIN}


def ref_smra(p: dict) -> dict | None:
    """Mean skeletal muscle attenuation at L3, HU."""
    b = bmi_of(p)
    if b is None:
        return None
    return {"value": 41 if b < 25 else 33, "flag_when": "below", "label": "Low muscle attenuation (myosteatosis)", "source": MARTIN}


def ref_vat_area(p: dict) -> dict | None:
    """Visceral fat area at L3, cm²."""
    if not p.get("sex"):
        return None
    return {"value": 163.8 if p["sex"] == "M" else 80.1, "flag_when": "above", "label": "Visceral obesity", "source": DOYLE}


def is_flagged(value, c) -> bool | None:
    if value is None or not c or not math.isfinite(value):
        return None
    return bool(value < c["value"] if c["flag_when"] == "below" else value > c["value"])


def evaluate(rm: dict, p: dict) -> dict:
    try:
        h = float(p.get("heightCm") or 0) / 100
    except ValueError:
        h = 0
    smi = rm["muscle"]["area"] / (h * h) if h > 0 and rm["muscle"]["area"] > 0 else None
    c_smi, c_hu, c_vat = ref_smi(p), ref_smra(p), ref_vat_area(p)
    return {
        "sma": rm["muscle"]["area"], "smi": smi, "muscle_hu": rm["muscle"]["hu"],
        "vat": rm["vat"]["area"], "sat": rm["sat"]["area"], "imat": rm["imat"]["area"],
        "vat_sat": rm["vat"]["area"] / rm["sat"]["area"] if rm["sat"]["area"] > 0 else None,
        "smi_cut": c_smi["value"] if smi is not None and c_smi else None,
        "low_muscle": is_flagged(smi, c_smi),
        "hu_cut": c_hu["value"] if rm["muscle"]["hu"] is not None and c_hu else None,
        "myosteatosis": is_flagged(rm["muscle"]["hu"], c_hu),
        "vat_cut": c_vat["value"] if c_vat else None,
        "visceral_obesity": is_flagged(rm["vat"]["area"], c_vat) if rm["vat"]["area"] > 0 else None,
    }


# ---------------------------------------------------------------- payload (lib/payload.ts)
def case_ref(case_id: str) -> str:
    return "case-" + hashlib.sha256(case_id.encode()).hexdigest()[:12]


def rule_narrative(pl: dict, derived: dict) -> dict:
    """Deterministic narrative used when the language model is not configured or fails. Uses only the eight
    biomarkers and the indices derived from them, like the model."""
    cm3 = lambda v: "not available" if v is None else f"{v / 1000:,.1f} cm³"  # noqa: E731
    val = lambda v, u="": "not available" if v is None else f"{v} {u}".strip()  # noqa: E731
    d, hu = derived, pl["muscle_density_hu"]
    myo = d["myosteatosis_indication"]
    vs = d["vat_sat_ratio"]

    summary = (f"Over the analysed segment ({val(pl['body_height_mm'], 'mm')} craniocaudal), skeletal muscle volume is "
               f"{cm3(pl['muscle_volume_mm3'])} with a mean density of {val(hu, 'HU')}, and total fat volume is "
               f"{cm3(pl['total_fat_volume_mm3'])} ({val(d['fat_share_of_body_pct'], '%')} of the body volume).")
    muscle = (f"Muscle volume {cm3(pl['muscle_volume_mm3'])}, {val(d['muscle_share_of_body_pct'], '%')} of the body volume; "
              f"volume-based muscle index (mean cross-section) {val(d['muscle_volume_index_cm2'], 'cm²')}. "
              f"Mean muscle density {val(hu, 'HU')}: myosteatosis {myo or 'not assessable'}.")
    fat = (f"Subcutaneous fat {cm3(pl['sfat_volume_mm3'])}, visceral fat {cm3(pl['vfat_volume_mm3'])} "
           f"({val(d['visceral_share_of_fat_pct'], '%')} of total fat), intramuscular fat {cm3(pl['mfat_volume_mm3'])} "
           f"({val(d['intramuscular_fat_to_muscle_pct'], '%')} of the muscle volume). VAT/SAT ratio {val(vs)}.")
    metabolic = ("A visceral-predominant fat distribution (VAT/SAT ratio above 1) is associated with higher cardiometabolic risk; "
                 "metabolic syndrome risk may be raised." if vs is not None and vs > 1 else
                 "Fat is predominantly subcutaneous (VAT/SAT ratio at or below 1), a less adverse distribution; "
                 "metabolic syndrome risk is not suggested by the fat distribution alone." if vs is not None else
                 "The fat distribution cannot be assessed.")
    likely = bool(myo and myo.startswith(("likely", "possible")))
    sarcopenia = ("Low muscle density suggests reduced muscle quality (myosteatosis), which is associated with sarcopenia; "
                  "muscle mass cannot be graded without height and sex-specific references." if likely else
                  "Muscle density does not indicate myosteatosis; muscle mass cannot be graded without height and "
                  "sex-specific references." if myo else "Muscle quality cannot be assessed.")
    lifestyle = ["Regular resistance and aerobic exercise to maintain muscle mass and reduce visceral fat."]
    if likely:
        lifestyle.append("Adequate protein intake and a nutritional assessment to support muscle quality.")
    if vs is not None and vs > 1:
        lifestyle.append("Calorie-appropriate diet and cardiometabolic check-up (blood pressure, glucose, lipids).")
    lifestyle.append("Repeat body-composition measurement on follow-up imaging to track change.")
    return {
        "summary": summary, "muscle": muscle, "fat": fat, "metabolic": metabolic, "sarcopenia": sarcopenia,
        "lifestyle": lifestyle,
        "limitations": ("Rule-based draft. Automated segmentation must be reviewed by a qualified reader. Age, sex, height "
                        "and weight are not part of the transmitted data, so volume cut-offs are not applied; the "
                        "myosteatosis cut-offs were defined on a single L3 slice, here applied to the segment mean."),
    }


def unverified_numbers(text: str, pl: dict) -> list[str]:
    """Numbers in text that do not appear in the payload (tolerating rounding)."""
    known: list[float] = []

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            known.append(float(o))
        elif isinstance(o, str):  # e.g. the cut-offs quoted in myosteatosis_indication
            known.extend(float(s) for s in re.findall(r"\d+(?:\.\d+)?", o))
        elif isinstance(o, list):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for x in o.values():
                walk(x)

    walk(pl)
    out: list[str] = []
    # "1,472,960.8" is one number; "T12" / "L3" are vertebra names, not values
    for s in re.findall(r"(?<![A-Za-z\d.,])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![A-Za-z\d.,])-?\d+(?:\.\d+)?", text):
        x = float(s.replace(",", ""))
        if abs(x) <= 10 and x.is_integer():
            continue  # counts, list numbers, "L3", "6 months"
        ok = any(abs(k - x) <= max(0.051, abs(k) * 0.005) or abs(round(k) - x) < 0.01 for k in known)
        if not ok and s not in out:
            out.append(s)
    return out


# ---------------------------------------------------------------- CSV (lib/csv.ts)
def metrics_csv(*, ref: str, shape, spacing, mask: dict, labels: dict, stats: dict, patient: dict,
                l3: int | None, hu: np.ndarray, body: dict | None) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["# BodyComp metrics", ref, datetime.now(timezone.utc).isoformat(timespec="seconds")])
    w.writerow(["# grid", *shape, "spacing_mm", *[f"{s:.4f}" for s in spacing]])
    w.writerow(["# patient", patient.get("sex") or "", patient.get("age") or "", patient.get("heightCm") or "",
                patient.get("weightKg") or "", "l3_slice", "" if l3 is None else l3 + 1])
    l3s = slice_stats(hu, mask["data"], l3) if l3 is not None else None
    px = spacing[0] * spacing[1] / 100
    w.writerow(["label", "name", "role", "voxels", "volume_cm3", "mean_hu", "sd_hu", "l3_area_cm2", "l3_mean_hu"])
    for L, d in labels.items():
        v = label_volume(stats, L, spacing)
        n3 = l3s["n"][L] if l3s else 0
        w.writerow([L, d["name"], d["role"], v["voxels"], _f(v["cm3"]), _f(v["mean_hu"]), _f(v["sd_hu"]),
                    _f(n3 * px) if l3s else "", _f(l3s["sum"][L] / n3) if l3s and n3 else ""])
    if body and (body.get("metrics_3d") or body.get("metrics_2d")):
        w.writerow([])
        w.writerow(["# predict_muscle_fat.py metrics", "vertebra slices", json_slices(body)])
        for kind in ("metrics_2d", "metrics_3d"):
            for k, val in (body.get(kind) or {}).items():
                w.writerow([kind, k, val])
    return buf.getvalue()


def json_slices(body: dict) -> str:
    s = body.get("slices") or {}
    return " ".join(f"{k}={'' if v is None else v + 1}" for k, v in s.items())


def _f(v):
    return "" if v is None or not math.isfinite(v) else f"{v:.2f}"
