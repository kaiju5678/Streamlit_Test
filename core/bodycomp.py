"""Body-composition metrics with the formulas of predict_muscle_fat.py (CT-Muscle-and-Fat-Segmentation).

Same calculation as server_segment/server.py, for masks opened from a results folder. Arrays are RAS-canonical
(axis 2 = axial slice); labels 1 muscle · 2 SAT · 3 VAT · 4 intermuscular fat.
"""
from __future__ import annotations

import numpy as np


def peak_slice(vertebra: np.ndarray | None) -> int | None:
    """Axial slice where a vertebra mask has the most voxels (the argmax rule of predict_muscle_fat.py)."""
    if vertebra is None:
        return None
    per = (vertebra > 0).sum(axis=(0, 1), dtype=np.int64)
    return int(per.argmax()) if per.any() else None


def compute(hu: np.ndarray, seg: np.ndarray, spacing, t12: int | None, l3: int | None, l4: int | None) -> dict:
    sx, sy, sz = (float(s) for s in spacing)
    out: dict = {"slices": {"T12": t12, "L3": l3, "L4": l4}, "metrics_2d": None, "metrics_3d": None}
    if l3 is not None:
        im, sg = hu[:, :, l3].astype(np.float32), seg[:, :, l3]
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
        im, sg = hu[:, :, lo:hi + 1].astype(np.float32), seg[:, :, lo:hi + 1]
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
