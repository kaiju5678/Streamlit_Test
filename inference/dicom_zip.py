"""DICOM study (.zip) to one NIfTI volume in Hounsfield units, with no patient metadata.

GDCM (through SimpleITK) groups the files by series UID and orders them by position, so it needs real files: the
zip is unpacked into a TemporaryDirectory that is removed before the function returns. Only two tags are read,
series description and modality, to describe the series to the user.
"""
from __future__ import annotations

import io
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

MIN_SLICES = 10


class ConversionError(ValueError):
    pass


@dataclass
class Series:
    uid: str
    files: list[str]
    description: str = ""
    modality: str = ""

    def info(self) -> dict:
        return {"id": self.uid, "slices": len(self.files), "description": self.description, "modality": self.modality}


def _unpack(zip_bytes: bytes, into: Path) -> int:
    """Members are written under generated names: series are found by UID whatever the folder layout, and no
    member path can point outside the temporary folder."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as e:
        raise ConversionError("The upload is not a valid .zip file.") from e
    n = 0
    with zf:
        for member in zf.infolist():
            if member.is_dir() or not member.file_size:
                continue
            n += 1
            (into / f"{n:06d}.dcm").write_bytes(zf.read(member))
    return n


def _describe(sitk, s: Series) -> None:
    reader = sitk.ImageFileReader()
    reader.SetFileName(s.files[0])
    try:
        reader.ReadImageInformation()
    except RuntimeError:
        return
    tag = lambda k: reader.GetMetaData(k).strip() if reader.HasMetaDataKey(k) else ""  # noqa: E731
    s.description, s.modality = tag("0008|103e"), tag("0008|0060")


def convert(zip_bytes: bytes, want: str = "") -> tuple[bytes, list[dict], str]:
    """Returns (nii.gz bytes, series listing, chosen series UID). Default choice: the CT series with the most slices."""
    import SimpleITK as sitk

    with tempfile.TemporaryDirectory(prefix="bodycomp_dcm_") as tmp:
        folder = Path(tmp)
        if not _unpack(zip_bytes, folder):
            raise ConversionError("The zip is empty.")
        uids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or []
        series = [Series(uid, list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), uid))) for uid in uids]
        if not series:
            raise ConversionError("No DICOM image series found in the upload.")
        for s in series:
            _describe(sitk, s)
        series.sort(key=lambda s: -len(s.files))
        pick = next((s for s in series if s.uid == want), None) or next(
            (s for s in series if s.modality in ("CT", "") and len(s.files) >= MIN_SLICES), series[0])

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(pick.files)
        vol = reader.Execute()   # rescale slope / intercept applied: values are HU
        vol = sitk.Cast(sitk.Clamp(vol, lowerBound=-32768, upperBound=32767), sitk.sitkInt16)
        out = folder / "series.nii.gz"
        sitk.WriteImage(vol, str(out))   # NIfTI carries geometry only: no names, IDs or dates
        return out.read_bytes(), [s.info() for s in series], pick.uid
