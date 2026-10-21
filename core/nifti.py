"""NIfTI handling in memory."""
from __future__ import annotations

import gzip

import nibabel as nib
import numpy as np


def read(data: bytes, as_label: bool = False) -> dict:
    """Parse .nii / .nii.gz bytes and reorient to RAS+ (x → right, y → anterior, z → superior)."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    try:
        img = nib.Nifti1Image.from_bytes(data)
    except Exception:  # noqa: BLE001  (NIfTI-2)
        img = nib.Nifti2Image.from_bytes(data)
    src = "".join(nib.aff2axcodes(img.affine))
    img = nib.as_closest_canonical(img)
    arr = np.squeeze(np.asanyarray(img.dataobj))
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {arr.shape}.")
    if as_label:
        arr = np.rint(arr).astype(np.uint8)
    else:
        arr = np.clip(np.rint(arr), -32768, 32767).astype(np.int16)
    return {"data": arr, "affine": img.affine, "spacing": tuple(float(z) for z in img.header.get_zooms()[:3]),
            "src": f"{src} → RAS"}


def grid(data: bytes) -> tuple[tuple[int, ...], np.ndarray]:
    """Shape and affine as stored in the file (header only, no reorientation)."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    try:
        img = nib.Nifti1Image.from_bytes(data)
    except Exception:  # noqa: BLE001  (NIfTI-2)
        img = nib.Nifti2Image.from_bytes(data)
    return tuple(int(n) for n in img.shape[:3]), img.affine


def same_grid(path, shape: tuple[int, ...], affine: np.ndarray) -> bool:
    """True if the NIfTI file at `path` has this voxel grid (reads the header only)."""
    try:
        img = nib.load(str(path))
    except Exception:  # noqa: BLE001
        return False
    return tuple(int(n) for n in img.shape[:3]) == shape and np.allclose(img.affine, affine, atol=1e-3)


def to_bytes(arr: np.ndarray, affine: np.ndarray) -> bytes:
    """Serialize an array as .nii.gz bytes."""
    img = nib.Nifti1Image(arr, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    img.header.set_xyzt_units("mm")
    return gzip.compress(img.to_bytes(), compresslevel=5)


def gzipped(data: bytes) -> bytes:
    return data if data[:2] == b"\x1f\x8b" else gzip.compress(data, compresslevel=1)
