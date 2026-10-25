"""Speed / agreement benchmark of the Mazurowski nnU-Net: test-time augmentation (mirroring) on/off x AMP (FP16) / FP32.

    python benchmark.py --input path/to/ct.nii.gz [--input another.nii.gz] [--repeats 3] [--out bench_out]

Uses the same model and predictor settings as predict_muscle_fat.py (2D ResEnc XL, fold 5, tile step 0.5, Gaussian),
with the model loaded once and each CT preprocessed once, so the timings compare only the network part:
  gpu_s     sliding-window inference on the GPU (what TTA and precision change)
  export_s  logits -> label map at the original grid
nnU-Net runs under torch.autocast (FP16 mixed precision) by default; "fp32" turns that off. Agreement is measured
against the configuration the app uses (TTA on + AMP): Dice per label and volume difference.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

import nnunetv2.inference.predict_from_raw_data as nnp
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

HERE = Path(__file__).resolve().parent
SEG_REPO = Path(os.getenv("SEG_REPO", HERE.parent.parent / "CT-Muscle-and-Fat-Segmentation")).resolve()
MODEL_DIR = SEG_REPO / "nnUNetTrainer__nnUNetResEncUNetXLPlans__2d"
LABELS = {1: "muscle", 2: "SAT", 3: "VAT", 4: "IMAT"}
CONFIGS = [  # (name, use_mirroring, amp); the first is the reference = what the app runs
    ("tta+amp", True, True),
    ("tta+fp32", True, False),
    ("notta+amp", False, True),
    ("notta+fp32", False, False),
]


class _Torch:
    """Stand-in for the torch module inside nnU-Net's predictor so its autocast can be switched off (FP32)."""

    def __init__(self, amp: bool) -> None:
        self.amp = amp

    def __getattr__(self, name):
        return getattr(torch, name)

    def autocast(self, device_type, enabled=True, **kw):
        return torch.autocast(device_type, enabled=enabled and self.amp, **kw)


def dice(a: np.ndarray, b: np.ndarray) -> float | None:
    s = int(a.sum()) + int(b.sum())
    return None if s == 0 else 2 * int(np.logical_and(a, b).sum()) / s


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required (predict_muscle_fat.py runs on cuda:0).")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU {gpu} · torch {torch.__version__} · model {MODEL_DIR.name} · checkpoint {args.checkpoint}")

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=True,
                                device=torch.device("cuda", 0), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(MODEL_DIR), use_folds=(5,),
                                                   checkpoint_name=f"checkpoint_{args.checkpoint}.pth")
    pm, cm, dj = predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json
    lm = pm.get_label_manager(dj)
    rw = pm.image_reader_writer_class()

    rows, agree = [], []
    for path in args.input:
        name = Path(path).name.replace(".nii.gz", "").replace(".nii", "")
        img, props = rw.read_images([path])
        t = time.perf_counter()
        data, _, dprops = cm.preprocessor_class(verbose=False).run_case_npy(img, None, props, pm, cm, dj)
        pre_s = time.perf_counter() - t
        data = torch.from_numpy(np.ascontiguousarray(data)).float()
        voxel_ml = float(np.prod(props["spacing"])) / 1000
        print(f"\n{name}: shape {img.shape[1:]} · spacing {tuple(round(float(s), 2) for s in props['spacing'])} · "
              f"preprocess {pre_s:.1f} s")

        segs = {}
        for cfg, mirror, amp in CONFIGS:
            predictor.use_mirroring = mirror
            nnp.torch = _Torch(amp)
            times, exports = [], []
            for r in range(args.warmup + args.repeats):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                logits = predictor.predict_logits_from_preprocessed_data(data)
                torch.cuda.synchronize()
                g = time.perf_counter() - t
                t = time.perf_counter()
                seg = convert_predicted_logits_to_segmentation_with_correct_shape(logits.cpu(), pm, cm, lm, dprops)
                e = time.perf_counter() - t
                del logits
                if r >= args.warmup:
                    times.append(g)
                    exports.append(e)
            peak = torch.cuda.max_memory_allocated() / 2**20
            segs[cfg] = np.asarray(seg)
            if args.save_masks:
                rw.write_seg(segs[cfg], str(out / f"{name}_{cfg}.nii.gz"), props)
            vols = {LABELS[k]: round(int((segs[cfg] == k).sum()) * voxel_ml, 1) for k in LABELS}
            row = {"case": name, "config": cfg, "tta": mirror, "precision": "amp-fp16" if amp else "fp32",
                   "gpu_s_mean": round(statistics.mean(times), 2), "gpu_s_sd": round(statistics.stdev(times), 2) if len(times) > 1 else 0.0,
                   "export_s": round(statistics.mean(exports), 2), "preprocess_s": round(pre_s, 2),
                   "peak_vram_mb": round(peak), "runs": len(times), **{f"{k}_ml": v for k, v in vols.items()}}
            rows.append(row)
            print(f"  {cfg:11s} gpu {row['gpu_s_mean']:6.2f} ± {row['gpu_s_sd']:.2f} s · export {row['export_s']:.2f} s · "
                  f"VRAM {row['peak_vram_mb']} MB · " + " ".join(f"{k} {v}" for k, v in vols.items()))

        ref_name = CONFIGS[0][0]
        ref = segs[ref_name]
        for cfg, *_ in CONFIGS[1:]:
            a = {"case": name, "config": cfg, "vs": ref_name,
                 "voxels_changed_pct": round(float((segs[cfg] != ref).mean() * 100), 4)}
            for k, lab in LABELS.items():
                d = dice(segs[cfg] == k, ref == k)
                rv, cv = int((ref == k).sum()), int((segs[cfg] == k).sum())
                a[f"dice_{lab}"] = None if d is None else round(d, 4)
                a[f"vol_diff_{lab}_pct"] = None if rv == 0 else round((cv - rv) / rv * 100, 2)
            agree.append(a)
            print(f"  {cfg:11s} vs {ref_name}: " + " ".join(f"Dice {lab} {a[f'dice_{lab}']}" for lab in LABELS.values())
                  + f" · {a['voxels_changed_pct']}% voxels differ")

    meta = {"gpu": gpu, "torch": torch.__version__, "model": MODEL_DIR.name, "checkpoint": args.checkpoint,
            "repeats": args.repeats, "warmup": args.warmup, "date": time.strftime("%Y-%m-%d %H:%M")}
    (out / "benchmark.json").write_text(json.dumps({"meta": meta, "timing": rows, "agreement": agree}, indent=2), encoding="utf-8")
    for fname, data_rows in (("timing.csv", rows), ("agreement.csv", agree)):
        with open(out / fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data_rows[0]))
            w.writeheader()
            w.writerows(data_rows)
    print(f"\nwritten: {out / 'benchmark.json'}, timing.csv, agreement.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", action="append", required=True, help="CT .nii.gz (repeat for several cases)")
    p.add_argument("--repeats", type=int, default=3, help="timed runs per configuration (default 3)")
    p.add_argument("--warmup", type=int, default=1, help="untimed runs before timing each configuration (default 1)")
    p.add_argument("--checkpoint", default=os.getenv("SEG_CHECKPOINT", "final"), choices=("final", "best"))
    p.add_argument("--out", default=str(HERE / "bench_out"))
    p.add_argument("--save_masks", action="store_true", help="also write each configuration's mask as NIfTI")
    main(p.parse_args())
