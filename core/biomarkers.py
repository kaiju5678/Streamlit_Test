"""The eight de-identified biomarkers: the only data that leaves the app (project briefing, Stage 2).

- FIELDS is the whitelist. `check()` runs right before transmission and blocks a payload with any other field,
  a missing field, or a value that is not a plain number (a string could carry a name or an ID).
- `derived()` computes the interpretation indices of Stage 3 (VAT/SAT ratio, myosteatosis indication, volume-based
  muscle index, ...) from the eight values alone, in the prompt stage, so they add no information about the patient.
"""
from __future__ import annotations

import math

FIELDS = {
    "muscle_volume_mm3": "Skeletal muscle volume",
    "muscle_density_hu": "Mean skeletal muscle density (HU)",
    "sfat_volume_mm3": "Subcutaneous fat volume",
    "vfat_volume_mm3": "Visceral fat volume",
    "mfat_volume_mm3": "Intramuscular fat volume",
    "total_fat_volume_mm3": "Total fat volume (subcutaneous + visceral + intramuscular)",
    "body_volume_mm3": "Total body volume of the analysed segment",
    "body_height_mm": "Craniocaudal height of the analysed segment",
}

# Martin et al., J Clin Oncol 2013;31:1539-47: low muscle attenuation below 41 HU (BMI < 25) or 33 HU (BMI >= 25).
# BMI is not one of the eight values, so the band between the two cut-offs is reported as "possible".
MYO_LOW, MYO_HIGH = 33.0, 41.0
MYO_SOURCE = "Martin et al., J Clin Oncol 2013 (41 HU if BMI < 25, 33 HU if BMI >= 25; defined at L3)"


class PayloadBlocked(ValueError):
    """The payload failed the whitelist check and must not be transmitted."""


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def violations(payload) -> list[str]:
    """Everything that stops a payload from being sent; empty when it may go."""
    if not isinstance(payload, dict):
        return ["payload is not a JSON object"]
    out = [f"field not in the whitelist: {k!r}" for k in payload if k not in FIELDS]
    out += [f"missing field: {k}" for k in FIELDS if k not in payload]
    out += [f"{k} is not a number or null: {type(v).__name__}" for k, v in payload.items()
            if k in FIELDS and v is not None and not _num(v)]
    return out


def check(payload) -> dict:
    bad = violations(payload)
    if bad:
        raise PayloadBlocked("Transmission blocked by the whitelist check: " + "; ".join(bad))
    return payload


def from_metrics(m: dict | None) -> dict | None:
    """The eight values out of bodycomp.compute()['metrics_3d'] (which also holds the slice range, not sent)."""
    if not m:
        return None
    return {k: (round(float(m[k]), 1) if m.get(k) is not None else None) for k in FIELDS}


def _ratio(a, b, scale: float = 1.0, nd: int = 1):
    return round(a / b * scale, nd) if _num(a) and _num(b) and b else None


def myosteatosis(hu) -> str | None:
    if not _num(hu):
        return None
    if hu < MYO_LOW:
        return f"likely (below {MYO_LOW:g} HU, the lower cut-off)"
    if hu < MYO_HIGH:
        return f"possible ({MYO_LOW:g}-{MYO_HIGH:g} HU: below the cut-off for BMI < 25 only)"
    return f"not indicated (at or above {MYO_HIGH:g} HU)"


def derived(p: dict) -> dict:
    """Indices computed from the eight values only."""
    mv, sf, vf, mf = p["muscle_volume_mm3"], p["sfat_volume_mm3"], p["vfat_volume_mm3"], p["mfat_volume_mm3"]
    tf, bv, h = p["total_fat_volume_mm3"], p["body_volume_mm3"], p["body_height_mm"]
    return {
        "vat_sat_ratio": _ratio(vf, sf, nd=2),
        "visceral_share_of_fat_pct": _ratio(vf, tf, 100),
        "fat_share_of_body_pct": _ratio(tf, bv, 100),
        "muscle_share_of_body_pct": _ratio(mv, bv, 100),
        "intramuscular_fat_to_muscle_pct": _ratio(mf, mv, 100),
        # volume-based muscle mass index: muscle volume per mm of segment = mean muscle cross-section (mm² -> cm²)
        "muscle_volume_index_cm2": _ratio(mv, h, 0.01),
        "myosteatosis_indication": myosteatosis(p["muscle_density_hu"]),
        "myosteatosis_reference": MYO_SOURCE,
    }
