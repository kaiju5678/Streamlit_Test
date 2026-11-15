"""DICOM study (.zip) to one NIfTI volume in Hounsfield units, with no patient metadata.

Fast path, all in memory: pydicom reads each file from the zip, files are grouped by series UID, the chosen
series is ordered by position along the slice normal, and its pixels are stacked and rescaled to HU. Only geometry
(position, orientation, spacing) goes into the NIfTI; of the other tags, only series description and modality are
read, to describe the series to the user.

Fallback, for what the fast path cannot handle (compressed transfer syntaxes without a pydicom decoder, mixed
orientations, uneven slice gaps): GDCM through SimpleITK, which needs real files, so the series is unpacked into a
TemporaryDirectory that is removed before the function returns. Writing the files costs a few seconds on Windows.
"""
from __future__ import annotations

import io
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

MIN_SLICES = 10


class ConversionError(ValueError):
    pass


@dataclass
class Series:
    uid: str
    files: list[bytes] = field(default_factory=list)
    headers: list = field(default_factory=list)
    description: str = ""
    modality: str = ""

    def info(self) -> dict:
        return {"id": self.uid, "slices": len(self.files), "description": self.description, "modality": self.modality}


def _group(zip_bytes: bytes) -> list[Series]:
    import pydicom

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as e:
        raise ConversionError("The upload is not a valid .zip file.") from e
    groups: dict[str, Series] = {}
    with zf:
        members = [m for m in zf.infolist() if not m.is_dir() and m.file_size]
        if not members:
            raise ConversionError("The zip is empty.")
        for m in members:
            raw = zf.read(m)
            try:
                ds = pydicom.dcmread(io.BytesIO(raw), force=True)   # pixels are decoded only for the chosen series
            except Exception:  # noqa: BLE001  (not DICOM: DICOMDIR, notes, images)
                continue
            uid = str(ds.get("SeriesInstanceUID", "") or "")
            if not uid or "ImagePositionPatient" not in ds:
                continue
            s = groups.setdefault(uid, Series(uid, description=str(ds.get("SeriesDescription", "") or "").strip(),
                                              modality=str(ds.get("Modality", "") or "").strip()))
            s.files.append(raw)
            s.headers.append(ds)
    if not groups:
        raise ConversionError("No DICOM image series found in the upload.")
    return sorted(groups.values(), key=lambda s: -len(s.files))


def _in_memory(s: Series) -> bytes | None:
    """NIfTI bytes of one series, or None when its geometry needs GDCM."""
    try:
        iop = np.array([[float(x) for x in h.ImageOrientationPatient] for h in s.headers])
        if not np.allclose(iop, iop[0], atol=1e-4):
            return None
        row, col = iop[0, :3], iop[0, 3:]
        normal = np.cross(row, col)
        pos = np.array([[float(x) for x in h.ImagePositionPatient] for h in s.headers])
        order = np.argsort(pos @ normal)
        pos = pos[order]
        steps = np.diff(pos @ normal)
        if len(pos) < 2 or steps.min() <= 1e-3 or steps.max() - steps.min() > 0.01 * steps.mean() + 1e-3:
            return None   # duplicate positions or uneven gaps
        dy, dx = (float(x) for x in s.headers[0].PixelSpacing)   # row spacing, column spacing
        slices = []
        for i in order:
            ds = s.headers[i]
            px = ds.pixel_array.astype(np.float32)
            if px.ndim != 2 or (slices and px.shape != slices[0].shape):
                return None
            slices.append(px * float(ds.get("RescaleSlope", 1) or 1) + float(ds.get("RescaleIntercept", 0) or 0))
    except Exception:  # noqa: BLE001  (missing tags, no decoder for the transfer syntax)
        return None
    # [row j, column i, slice k] -> voxel [i, j, k]; DICOM positions are LPS, NIfTI wants RAS
    hu = np.clip(np.rint(np.stack(slices, axis=-1).transpose(1, 0, 2)), -32768, 32767).astype(np.int16)
    affine = np.eye(4)
    affine[:3, 0] = row * dx
    affine[:3, 1] = col * dy
    affine[:3, 2] = (pos[-1] - pos[0]) / (len(pos) - 1)
    affine[:3, 3] = pos[0]
    affine[:2] *= -1
    img = nib.Nifti1Image(hu, affine)   # geometry only: no names, IDs or dates
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    img.header.set_xyzt_units("mm")
    return img.to_bytes()


def _with_gdcm(s: Series) -> bytes:
    import SimpleITK as sitk

    with tempfile.TemporaryDirectory(prefix="bodycomp_dcm_") as tmp:
        folder = Path(tmp)
        for n, raw in enumerate(s.files):
            (folder / f"{n:06d}.dcm").write_bytes(raw)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), s.uid)
                            or sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder)))
        vol = reader.Execute()   # rescale slope / intercept applied: values are HU
        vol = sitk.Cast(sitk.Clamp(vol, lowerBound=-32768, upperBound=32767), sitk.sitkInt16)
        out = folder / "series.nii"
        sitk.WriteImage(vol, str(out))
        return out.read_bytes()


def convert(zip_bytes: bytes, want: str = "") -> tuple[bytes, list[dict], str, str]:
    """Returns (uncompressed .nii bytes, series listing, chosen series UID, reader used).
    Default choice: the CT series with the most slices."""
    series = _group(zip_bytes)
    pick = next((s for s in series if s.uid == want), None) or next(
        (s for s in series if s.modality in ("CT", "") and len(s.files) >= MIN_SLICES), series[0])
    nii = _in_memory(pick)
    how = "memory"
    if nii is None:
        nii, how = _with_gdcm(pick), "gdcm"
    return nii, [s.info() for s in series], pick.uid, how
