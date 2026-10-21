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


def r2(v):
    return None if v is None or not math.isfinite(v) else round(v * 100) / 100


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


def build_payload(*, ref: str, shape, spacing, mask: dict | None, labels: dict, scheme: str, stats: dict | None,
                  patient: dict, l3: int | None, sample: bool, ev: dict | None) -> dict:
    """De-identified metrics: the only data sent to the language model."""
    p = patient
    b = bmi_of(p)
    rv = role_volumes(stats, labels, spacing) if stats else None
    adipose = rv["sat"]["cm3"] + rv["vat"]["cm3"] + rv["imat"]["cm3"] if rv else None
    tissue = adipose + rv["muscle"]["cm3"] if rv else None

    refs = []

    def push(metric, value, c):
        if c:
            refs.append({"metric": metric, "cutoff": c["value"], "flag_when": c["flag_when"],
                         "flagged": is_flagged(value, c), "source": c["source"]})

    if ev:
        push("skeletal_muscle_index_cm2_m2", ev["smi"], ref_smi(p))
        push("muscle_mean_hu", ev["muscle_hu"], ref_smra(p))
        push("vat_area_cm2", ev["vat"], ref_vat_area(p))

    flags = []
    if sample:
        flags.append("synthetic phantom, not patient data")
    if not mask:
        flags.append("no segmentation loaded")
    if mask and l3 is None:
        flags.append("L3 slice not set")
    if not p.get("sex") or b is None:
        flags.append("sex, height or weight missing: index cut-offs not applied")

    try:
        age = float(p.get("age") or 0)
    except ValueError:
        age = 0
    num = lambda v: float(v) if v not in (None, "") and _is_num(v) and float(v) else None  # noqa: E731
    st = stats
    return {
        "schema": "bodycomp.metrics/v1",
        "case_ref": ref,
        "synthetic_sample": sample,
        "scan": {
            "axial_slices": int(shape[2]),
            "spacing_mm": [r2(float(s)) for s in spacing],
            "segmented_slices": [st["k_min"], st["k_max"]] if st and st["k_min"] >= 0 else None,
            "segmented_extent_mm": r1((st["k_max"] - st["k_min"] + 1) * float(spacing[2])) if st and st["k_min"] >= 0 else None,
        },
        "segmentation": {"source": mask["source"] if mask else "none",
                         "scheme": scheme},
        "patient": {"sex": p.get("sex") or None, "age_years": age if age > 0 else None,
                    "height_cm": num(p.get("heightCm")), "weight_kg": num(p.get("weightKg")), "bmi": r1(b)},
        "l3": {
            "slice_index": l3,
            "skeletal_muscle_area_cm2": r1(ev["sma"]),
            "skeletal_muscle_index_cm2_m2": r1(ev["smi"]),
            "muscle_mean_hu": r1(ev["muscle_hu"]),
            "vat_area_cm2": r1(ev["vat"]),
            "sat_area_cm2": r1(ev["sat"]),
            "imat_area_cm2": r1(ev["imat"]),
            "vat_sat_ratio": r2(ev["vat_sat"]),
        } if ev else None,
        "volumes_segmented_range": {
            "muscle_cm3": r1(rv["muscle"]["cm3"]),
            "muscle_mean_hu": r1(rv["muscle"]["mean_hu"]),
            "sat_cm3": r1(rv["sat"]["cm3"]),
            "vat_cm3": r1(rv["vat"]["cm3"]),
            "imat_cm3": r1(rv["imat"]["cm3"]),
            "adipose_total_cm3": r1(adipose),
            "adipose_share_pct": r1(adipose / tissue * 100) if tissue else None,
        } if rv else None,
        "references": refs,
        "quality_flags": flags,
    }


def _is_num(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def rule_narrative(pl: dict) -> dict:
    """Deterministic narrative used when no language model is configured. Uses only payload values."""
    l, v = pl["l3"], pl["volumes_segmented_range"]
    ref = lambda m: next((r for r in pl["references"] if r["metric"] == m), None)  # noqa: E731
    fmt = lambda x, u: "not available" if x is None else f"{x} {u}".strip()  # noqa: E731
    smi, hu, vat = ref("skeletal_muscle_index_cm2_m2"), ref("muscle_mean_hu"), ref("vat_area_cm2")

    findings = []
    if smi and smi["flagged"]:
        findings.append("low skeletal muscle index")
    if hu and hu["flagged"]:
        findings.append("low muscle attenuation")
    if vat and vat["flagged"]:
        findings.append("visceral fat area above the reference cut-off")

    if not l:
        summary = "L3 metrics are not available yet. Load a segmentation and set the L3 slice."
    elif findings:
        summary = f"At L3 the measurements show {', '.join(findings)}. " + (
            "Low muscle mass together with visceral obesity is consistent with a sarcopenic obesity pattern; clinical correlation is required."
            if smi and smi["flagged"] and vat and vat["flagged"] else "Clinical correlation is required.")
    else:
        summary = "At L3 the measured values are within the configured reference cut-offs."

    sc, seg = pl["scan"], pl["segmentation"]
    spacing = " × ".join(str(x) for x in sc["spacing_mm"])
    examination = f"{sc['axial_slices']} axial slices at {spacing} mm. Segmentation source: {seg['source']} ({seg['scheme']} labels)"
    if sc["segmented_extent_mm"]:
        examination += f", covering {sc['segmented_extent_mm']} mm craniocaudally"
    examination += "."
    if l:
        examination += f" L3 analysed on axial slice {l['slice_index'] + 1}."

    def cut(c, word_yes, word_no):
        if not c:
            return ""
        sym = "<" if c["flag_when"] == "below" else ">"
        return f" (cut-off {sym}{c['cutoff']}; {word_yes if c['flagged'] else word_no})"

    if l:
        muscle = (f"Skeletal muscle area {fmt(l['skeletal_muscle_area_cm2'], 'cm²')}, skeletal muscle index "
                  f"{fmt(l['skeletal_muscle_index_cm2_m2'], 'cm²/m²')}{cut(smi, 'below', 'not below')}. "
                  f"Mean muscle attenuation {fmt(l['muscle_mean_hu'], 'HU')}{cut(hu, 'below', 'not below')}.")
        if v and v["muscle_cm3"] is not None:
            muscle += f" Muscle volume over the segmented range {v['muscle_cm3']} cm³."
        adipose = (f"Visceral fat area {fmt(l['vat_area_cm2'], 'cm²')}{cut(vat, 'above', 'not above')}, subcutaneous fat area "
                   f"{fmt(l['sat_area_cm2'], 'cm²')}, intermuscular fat area {fmt(l['imat_area_cm2'], 'cm²')}, "
                   f"VAT/SAT ratio {fmt(l['vat_sat_ratio'], '')}.")
        if v and v["adipose_total_cm3"] is not None:
            adipose += (f" Total adipose volume over the segmented range {v['adipose_total_cm3']} cm³ "
                        f"({v['adipose_share_pct']}% of segmented tissue).")
    else:
        muscle = adipose = "Not available."

    recs = []
    if (smi and smi["flagged"]) or (hu and hu["flagged"]):
        recs.append("Consider nutritional and physical-function assessment for low muscle mass or quality.")
    if vat and vat["flagged"]:
        recs.append("Consider cardiometabolic risk assessment in the context of visceral obesity.")
    if l:
        recs.append("Repeat body-composition measurement on follow-up imaging to track change.")

    return {
        "summary": summary, "examination": examination, "muscle": muscle, "adipose": adipose,
        "recommendations": recs,
        "limitations": " ".join(["Automated segmentation must be reviewed by a qualified reader."]
                                + [f"Note: {f}." for f in pl["quality_flags"]]),
    }


def unverified_numbers(text: str, pl: dict) -> list[str]:
    """Numbers in text that do not appear in the payload (tolerating rounding)."""
    known: list[float] = []

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            known.append(float(o))
        elif isinstance(o, list):
            for x in o:
                walk(x)
        elif isinstance(o, dict):
            for x in o.values():
                walk(x)

    walk(pl)
    if pl.get("l3"):
        known += [pl["l3"]["slice_index"] + 1, 3]
    out: list[str] = []
    for s in re.findall(r"-?\d+(?:\.\d+)?", text):
        x = float(s)
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
