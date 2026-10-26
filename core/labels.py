"""Labels of the Mazurowski model, tissue roles, colours and window presets (port of ct-scan-test/lib/labels.ts)."""
from __future__ import annotations

import re

import numpy as np

ROLES = {
    "muscle": "Skeletal muscle",
    "sat": "Subcutaneous fat",
    "vat": "Visceral fat",
    "imat": "Intermuscular fat",
    "bone": "Bone",
    "other": "Other",
}

ROLE_COLOR = {
    "muscle": (226, 92, 92),
    "sat": (240, 198, 80),
    "vat": (64, 190, 174),
    "imat": (172, 128, 255),
    "bone": (200, 206, 224),
    "other": (236, 150, 64),
}

EXTRA = [(96, 160, 255), (255, 120, 180), (140, 220, 90), (255, 170, 90), (90, 220, 230), (220, 220, 90), (180, 140, 100)]

SCHEMES = {
    "generic": {1: ("Skeletal muscle", "muscle"), 2: ("Subcutaneous fat", "sat"), 3: ("Visceral fat", "vat"),
                4: ("Intermuscular fat", "imat"), 5: ("Bone", "bone")},
}

PRESETS = [("Soft tissue", 400, 40), ("Abdomen", 350, 40), ("Liver", 150, 60),
           ("Fat", 300, -80), ("Bone", 1500, 300), ("Lung", 1500, -600)]

VIEW_COLOR = {"axial": "#f0b44c", "coronal": "#4cc3e0", "sagittal": "#e273c8"}


def rgb(c) -> str:
    return f"rgb({c[0]},{c[1]},{c[2]})"


def labels_for(counts: np.ndarray, key: str = "generic") -> dict[int, dict]:
    sch = SCHEMES.get(key, SCHEMES["generic"])
    out: dict[int, dict] = {}
    e = 0
    for L in np.nonzero(counts)[0]:
        L = int(L)
        if L == 0:
            continue
        d = sch.get(L)
        role = d[1] if d else "other"
        if d:
            color = ROLE_COLOR[role]
        else:
            color, e = EXTRA[e % len(EXTRA)], e + 1
        out[L] = {"name": d[0] if d else f"Label {L}", "role": role, "color": color, "visible": role != "bone"}
    return out


# Known label names (Mazurowski dataset.json, TotalSegmentator, ...) → display name and role.
# Names are normalised first: lower case, spaces and hyphens → "_".
NAME_ROLES = [
    (r"^torso_fat$", "Trunk fat (stands in for VAT)", "vat"),
    (r"^(sat|s_?fat|subcutaneous(_adipose)?(_fat|_tissue)?)$", "Subcutaneous fat", "sat"),
    (r"^(vat|v_?fat|visceral(_adipose)?(_fat|_tissue)?)$", "Visceral fat", "vat"),
    (r"^(imat|m_?fat|muscle_fat|muscular_fat|intra_?muscular(_adipose)?(_fat|_tissue)?|inter_?muscular(_adipose)?(_fat|_tissue)?)$",
     "Intermuscular fat", "imat"),
    (r"^(skeletal_)?muscles?$", "Skeletal muscle", "muscle"),
    (r"^vertebrae_l3$", "L3 vertebra", "bone"),
    (r"^vertebrae_", "", "bone"),
    (r"^muscle_hu_range$", "Demo: muscle HU band", "muscle"),
    (r"^fat_hu_range$", "Demo: fat HU band", "vat"),
]


def labels_from_names(counts: np.ndarray, names: dict[str, str]) -> dict[int, dict]:
    """Labels from names sent by the segmentation server, so numbering never has to be guessed."""
    out: dict[int, dict] = {}
    e = 0
    for L in np.nonzero(counts)[0]:
        L = int(L)
        if L == 0:
            continue
        raw = names.get(str(L))
        key = re.sub(r"[\s-]+", "_", raw.strip().lower()) if raw else None
        hit = next((h for h in NAME_ROLES if key and re.search(h[0], key)), None)
        role = hit[2] if hit else "other"
        name = (hit[1] if hit and hit[1] else None) or (raw.replace("_", " ").capitalize() if raw else f"Label {L}")
        if hit:
            color = ROLE_COLOR[role]
        else:
            color, e = EXTRA[e % len(EXTRA)], e + 1
        # the L3 vertebra is a landmark: keep it visible so the level can be checked
        out[L] = {"name": name, "role": role, "color": color, "visible": role != "bone" or "L3" in name}
    return out


def middle_slice(mask: np.ndarray, label: int) -> int | None:
    """Middle axial slice of a label (k = superior axis), or None if absent."""
    ks = np.nonzero((mask == label).any(axis=(0, 1)))[0]
    return None if ks.size == 0 else int(round((ks[0] + ks[-1]) / 2))
