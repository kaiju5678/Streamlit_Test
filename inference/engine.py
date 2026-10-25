"""Resident segmentation models.

The Mazurowski nnU-Net (CT-Muscle-and-Fat-Segmentation, labels 1 muscle · 2 SAT · 3 VAT · 4 intermuscular fat) is
loaded once and stays on the GPU between requests, instead of a new predict_muscle_fat.py process per CT. Predictor
settings are those of predict_muscle_fat.py (fold 5, tile step 0.5, Gaussian, mirroring TTA, mixed precision).
TotalSegmentator finds T12 / L3 / L4 through its Python API in the same process. Everything stays in memory.
"""
from __future__ import annotations

import gzip
import sys
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.bodycomp import compute, peak_slice  # noqa: E402  same formulas as the frontend, one copy

TISSUE_LABELS = {1: "muscle", 2: "sat", 3: "vat", 4: "muscle_fat"}   # dataset.json of the model folder
L3_LABEL = 50
VERTEBRAE = ("T12", "L3", "L4")


@dataclass
class Result:
    mask_gz: bytes
    labels: dict[int, str]
    body: dict
    counts: dict[int, int] = field(default_factory=dict)


def read_nifti(raw: bytes) -> nib.Nifti1Image:
    """Upload bytes (.nii or .nii.gz) to an image, without touching the disk."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return nib.Nifti1Image.from_bytes(raw)


def label_image(lab: np.ndarray, like: nib.Nifti1Image) -> nib.Nifti1Image:
    out = nib.Nifti1Image(lab.astype(np.uint8), like.affine, like.header)
    out.set_data_dtype(np.uint8)
    out.header.set_slope_inter(1, 0)
    return out


class RAS:
    """Reorients arrays of one image to RAS-canonical (axis 2 = axial, superior up), as the frontend and
    core.bodycomp expect: the same transform nib.as_closest_canonical applies, worked out once per CT."""

    def __init__(self, img: nib.Nifti1Image) -> None:
        self.ornt = nib.orientations.io_orientation(img.affine)
        zooms = img.header.get_zooms()[:3]
        self.spacing = [0.0, 0.0, 0.0]
        for axis, (to, _) in enumerate(self.ornt):
            self.spacing[int(to)] = float(zooms[axis])

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return nib.orientations.apply_orientation(arr, self.ornt)


class TissueModel:
    def __init__(self, model_dir: Path, checkpoint: str, tta: bool) -> None:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        self.torch = torch
        self.predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=tta,
                                         perform_everything_on_device=True, device=torch.device("cuda", 0),
                                         verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        self.predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=(5,),
                                                            checkpoint_name=f"checkpoint_{checkpoint}.pth")
        p = self.predictor
        self.labels_mgr = p.plans_manager.get_label_manager(p.dataset_json)
        self.preprocessor = p.configuration_manager.preprocessor_class(verbose=False)

    def warm_up(self) -> None:
        """One tiny forward pass so cuDNN picks its kernels before the first real CT."""
        patch = self.predictor.configuration_manager.patch_size
        self.predictor.predict_logits_from_preprocessed_data(self.torch.zeros((1, 1, *patch)))

    def predict(self, img: nib.Nifti1Image) -> np.ndarray:
        """Label map on the CT's own voxel grid (same axis order as the NIfTI)."""
        from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

        p = self.predictor
        # nnU-Net was trained on SimpleITK arrays: (channel, z, y, x) of the stored grid, spacing reversed to match
        vol = np.asanyarray(img.dataobj, dtype=np.float32).transpose(2, 1, 0)[None]
        props = {"spacing": [float(s) for s in img.header.get_zooms()[:3][::-1]]}
        data, _, props = self.preprocessor.run_case_npy(vol, None, props, p.plans_manager, p.configuration_manager,
                                                        p.dataset_json)
        logits = p.predict_logits_from_preprocessed_data(self.torch.from_numpy(np.ascontiguousarray(data)).float())
        seg = convert_predicted_logits_to_segmentation_with_correct_shape(logits.cpu(), p.plans_manager,
                                                                          p.configuration_manager, self.labels_mgr, props)
        del logits
        return np.asarray(seg, dtype=np.uint8).transpose(2, 1, 0)


class VertebraModel:
    def __init__(self, fast: bool) -> None:
        import totalsegmentator.python_api as ts_api
        from totalsegmentator.map_to_binary import class_map

        # TotalSegmentator posts usage statistics (machine id, run counter, versions) to stats.totalsegmentator.com
        # after every prediction unless its config says otherwise; this server sends nothing out.
        ts_api.send_usage_stats = lambda *args, **kwargs: None
        self.run = ts_api.totalsegmentator
        self.fast = fast
        ids = {name: k for k, name in class_map["total"].items()}
        self.ids = {v: ids[f"vertebrae_{v}"] for v in VERTEBRAE}

    def find(self, img: nib.Nifti1Image) -> dict[str, np.ndarray]:
        """Binary mask per vertebra on the CT grid; a vertebra outside the scan comes back empty."""
        seg = self.run(img, None, task="total", roi_subset=[f"vertebrae_{v}" for v in VERTEBRAE], fast=self.fast,
                       quiet=True, nr_thr_resamp=1, nr_thr_saving=1)
        lab = np.asanyarray(seg.dataobj)
        return {v: lab == k for v, k in self.ids.items()}


class Engine:
    """Tissue mask, vertebra levels and body-composition metrics for one CT."""

    def __init__(self, model: str, model_dir: Path, checkpoint: str, tta: bool, vertebrae: bool, fast_vertebrae: bool) -> None:
        self.model = model
        self.tissue = TissueModel(model_dir, checkpoint, tta) if model == "mazurowski" else None
        self.vertebrae = None
        self.vertebrae_error = ""
        if vertebrae and self.tissue is not None:
            try:
                self.vertebrae = VertebraModel(fast_vertebrae)
            except ImportError as e:
                self.vertebrae_error = f"TotalSegmentator not importable: {e}"

    def segment(self, raw: bytes, run) -> Result:
        with run.stage("preprocess"):
            img = read_nifti(raw)
            run.log("DEBUG", "PREPROCESS", f"grid {img.shape}, spacing {tuple(round(float(z), 3) for z in img.header.get_zooms()[:3])}, "
                                           f"orientation {''.join(nib.aff2axcodes(img.affine))}")

        if self.tissue is None:  # MODEL=demo: HU thresholds, a pipeline check without a GPU
            with run.stage("inference"):
                hu = np.asanyarray(img.dataobj, dtype=np.float32)
                lab = np.zeros(hu.shape, np.uint8)
                lab[(hu >= -29) & (hu <= 150)] = 1
                lab[(hu >= -190) & (hu <= -30)] = 3
            run.log("WARN", "PIPELINE", "MODEL=demo: masks are HU windows, not a segmentation; never report them")
            return Result(gzip.compress(label_image(lab, img).to_bytes(), 1), {1: "muscle_hu_range", 3: "fat_hu_range"},
                          {"slices": {}, "metrics_2d": None, "metrics_3d": None})

        with run.stage("inference"):
            lab = self.tissue.predict(img)
        labels = dict(TISSUE_LABELS)

        verts: dict[str, np.ndarray] = {}
        if self.vertebrae is not None:
            self.tissue.torch.cuda.empty_cache()   # give TotalSegmentator the GPU memory nnU-Net no longer needs
            try:
                with run.stage("vertebrae"):
                    verts = self.vertebrae.find(img)
            except Exception as e:  # noqa: BLE001  L3 is a convenience: keep the tissue mask and let the reader set it
                run.log("WARN", "L3", f"automatic vertebra detection failed, set L3 by hand: {str(e)[-200:]}")
        elif self.vertebrae_error:
            run.log("WARN", "L3", f"{self.vertebrae_error}; set L3 by hand")

        with run.stage("postprocess"):
            ras = RAS(img)
            levels = {v: peak_slice(ras(verts[v])) if v in verts else None for v in VERTEBRAE}
            # metrics on labels 1-4, before L3 is written into the mask (the order of predict_muscle_fat.py)
            body = compute(ras(np.asanyarray(img.dataobj, dtype=np.float32)), ras(lab), ras.spacing,
                           levels["T12"], levels["L3"], levels["L4"])
            if "L3" in verts and verts["L3"].any():
                lab[verts["L3"]] = L3_LABEL   # bone does not overlap the soft-tissue labels
                labels[L3_LABEL] = "vertebrae_L3"
                run.log("INFO", "L3", f"L3 vertebra found: {int(verts['L3'].sum())} voxels")
            elif verts:
                run.log("WARN", "L3", "L3 is not inside the scanned range; set it by hand")
            run.log("INFO", "METRICS", f"vertebra peak slices T12={levels['T12']} L3={levels['L3']} L4={levels['L4']} (RAS index)")
            if body["metrics_3d"] is None:
                run.log("WARN", "METRICS", "T12 or L4 not in the scan: 3D (T12–L4) metrics not computed")
            counts = {int(k): int(n) for k, n in zip(*np.unique(lab, return_counts=True)) if k}
            run.log("DEBUG", "POSTPROCESS", f"voxels per label {counts}")
            mask_gz = gzip.compress(label_image(lab, img).to_bytes(), 1)
        return Result(mask_gz, labels, body, counts)
