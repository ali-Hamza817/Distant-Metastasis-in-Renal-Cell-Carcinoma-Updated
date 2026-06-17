# -*- coding: utf-8 -*-
"""
Re-cache TCIA DICOM volumes with patient_id-based naming.
Creates: e:/rcc/pretrained_weights/dicom_cache_v2/TCGA-XX-XXXX.pt
So that patient_id -> cache_file mapping is trivial.
Only processes 267 TCGA-KIRC patient folders.
Skips if already cached (idempotent).
"""
import os, sys, glob, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import torch
import SimpleITK as sitk
from monai.transforms import Resize

TCIA_DIR  = "e:/rcc/TCIA_TCGA-KIRC_09-16-2015 (2)/tcga_kirc"
CACHE_DIR = "e:/rcc/pretrained_weights/dicom_cache_v2"
IMG_SIZE  = (96, 96, 96)

os.makedirs(CACHE_DIR, exist_ok=True)

def load_and_cache(patient_id, patient_path):
    cache_file = os.path.join(CACHE_DIR, f"{patient_id}.pt")
    if os.path.exists(cache_file):
        return True, "already_cached"

    # Find DICOM series recursively — take deepest with most files
    best_files = []
    reader = sitk.ImageSeriesReader()
    for root, dirs, files in os.walk(patient_path):
        try:
            series_files = reader.GetGDCMSeriesFileNames(root)
            if len(series_files) > len(best_files):
                best_files = series_files
        except Exception:
            pass

    if len(best_files) < 5:
        return False, f"too few slices ({len(best_files)})"

    try:
        reader.SetFileNames(best_files)
        image = reader.Execute()

        # Resample to 1mm isotropic
        orig_size    = image.GetSize()
        orig_spacing = image.GetSpacing()
        new_size = [int(round(s * sp / 1.0)) for s, sp in zip(orig_size, orig_spacing)]
        new_size = [max(1, s) for s in new_size]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing((1.0, 1.0, 1.0))
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(-1024)
        resampler.SetInterpolator(sitk.sitkLinear)
        image = resampler.Execute(image)

        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        arr = np.clip(arr, -200, 300)
        arr = (arr - 50.0) / 250.0

        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, D, H, W)
        resize_fn = Resize(spatial_size=IMG_SIZE, mode='trilinear')
        tensor = resize_fn(tensor)
        tensor = tensor.float()

        torch.save(tensor, cache_file)
        return True, f"ok ({tensor.shape})"
    except Exception as e:
        return False, str(e)[:80]


if __name__ == '__main__':
    patient_dirs = sorted([
        d for d in os.listdir(TCIA_DIR)
        if os.path.isdir(os.path.join(TCIA_DIR, d)) and d.startswith('TCGA-')
    ])
    print(f"Found {len(patient_dirs)} patient folders")
    print(f"Output: {CACHE_DIR}")
    print("-" * 70)

    success = 0
    already = 0
    failed  = 0
    t0 = time.time()

    for i, pid in enumerate(patient_dirs):
        patient_path = os.path.join(TCIA_DIR, pid)
        ok, msg = load_and_cache(pid, patient_path)
        if ok and msg == "already_cached":
            already += 1
        elif ok:
            success += 1
            elapsed = time.time() - t0
            rate = (success + already) / max(elapsed, 1)
            remaining = (len(patient_dirs) - i - 1) / max(rate, 0.001)
            print(f"[{i+1:3d}/{len(patient_dirs)}] {pid}: {msg} | ETA: {remaining/60:.1f}min")
        else:
            failed += 1
            print(f"[{i+1:3d}/{len(patient_dirs)}] {pid}: FAILED - {msg}")

        sys.stdout.flush()

    print("\n" + "=" * 70)
    print(f"Done! Success: {success} | Already cached: {already} | Failed: {failed}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
    print(f"Cache files: {len(glob.glob(os.path.join(CACHE_DIR, '*.pt')))}")
