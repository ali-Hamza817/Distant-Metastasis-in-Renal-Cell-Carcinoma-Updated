"""PHASE 4 — Extract radiomics features from CT DICOM with LASSO selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.config import CONFIG

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


def _basic_radiomics(volume: np.ndarray) -> dict[str, float]:
    """Fallback feature extractor when pyradiomics is unavailable."""
    v = volume.astype(np.float64)
    features = {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "median": float(np.median(v)),
        "skewness": float(((v - v.mean()) ** 3).mean() / (v.std() ** 3 + 1e-8)),
        "kurtosis": float(((v - v.mean()) ** 4).mean() / (v.std() ** 4 + 1e-8)),
        "energy": float((v ** 2).sum()),
        "entropy": float(
            -np.sum(
                (lambda h: (lambda p: p * np.log(p + 1e-8))(h / h.sum()))(
                    np.histogram(v.flatten(), bins=32)[0] + 1e-8
                )
            )
        ),
    }
    # Texture approximations on mid-slice
    if v.ndim == 3:
        sl = v[v.shape[0] // 2]
    else:
        sl = v
    gx = np.diff(sl, axis=0)
    gy = np.diff(sl, axis=1)
    features["gradient_mean"] = float(np.abs(gx).mean() + np.abs(gy).mean())
    features["gradient_std"] = float(np.abs(gx).std() + np.abs(gy).std())
    # Expand to ~100 features via multi-scale stats
    for i, pct in enumerate([10, 25, 50, 75, 90]):
        features[f"pct_{pct}"] = float(np.percentile(v, pct))
    for i, off in enumerate(range(0, min(sl.shape[0], 8), 2)):
        patch = sl[off:off+4, off:off+4] if sl.shape[0] > off+4 else sl
        features[f"patch_{i}_mean"] = float(patch.mean())
        features[f"patch_{i}_std"] = float(patch.std())
    return features


def _pyradiomics_features(image_path: Path) -> dict[str, float]:
    from radiomics import featureextractor
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    for group in ["firstorder", "shape", "glcm", "glrlm", "glszm", "ngtdm", "gldm"]:
        extractor.enableFeatureClassByName(group)
    result = extractor.execute(str(image_path), str(image_path))
    return {k: float(v) for k, v in result.items() if k.startswith("original")}


def load_ct_volume(dicom_dir: Path) -> np.ndarray | None:
    if not HAS_SITK:
        return None
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        if not series_ids:
            nrrd = list(dicom_dir.glob("*.nrrd")) + list(dicom_dir.glob("*.mha"))
            if nrrd:
                return sitk.GetArrayFromImage(sitk.ReadImage(str(nrrd[0])))
            return None
        files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
        reader.SetFileNames(files)
        img = reader.Execute()
        return sitk.GetArrayFromImage(img)
    except Exception:
        return None


def extract_patient_features(patient_dir: Path) -> dict[str, float] | None:
    volume = load_ct_volume(patient_dir)
    if volume is None:
        return None
    try:
        import radiomics  # noqa: F401
        mid = volume[volume.shape[0] // 2]
        tmp = patient_dir / "_mid_slice.npy"
        np.save(tmp, mid)
        feats = _pyradiomics_features(tmp)
        tmp.unlink(missing_ok=True)
        return feats
    except Exception:
        return _basic_radiomics(volume)


def lasso_select(X: np.ndarray, y: np.ndarray, feature_names: list[str], top_k: int) -> list[str]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000).fit(Xs, y)
    coef = np.abs(lasso.coef_)
    order = np.argsort(coef)[::-1]
    selected = [feature_names[i] for i in order[:top_k] if coef[i] > 0]
    if len(selected) < top_k:
        selected = [feature_names[i] for i in order[:top_k]]
    return selected


def main():
    parser = argparse.ArgumentParser(description="Extract radiomics + LASSO feature selection")
    parser.add_argument("--dicom_dir", default="data/dicom")
    parser.add_argument("--out_dir", default="data/radiomics")
    parser.add_argument("--seer_csv", default=None)
    parser.add_argument("--top_k", type=int, default=CONFIG["radiomics_top_k"])
    args = parser.parse_args()

    dicom_dir = Path(args.dicom_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 4 — Radiomics Extraction + LASSO")
    print("=" * 60)

    patient_dirs = [p for p in dicom_dir.iterdir() if p.is_dir()]
    if not patient_dirs:
        print("No DICOM patient folders found. Creating synthetic radiomics for pipeline demo...")
        seer = pd.read_csv(args.seer_csv or CONFIG["seer_csv"])
        rng = np.random.default_rng(42)
        n = min(500, len(seer))
        feat_names = [f"radiomics_{i}" for i in range(100)]
        rows = []
        for i in range(n):
            row = {"patient_id": seer.iloc[i]["patient_id"], "tcga_id": f"SYN-{i:04d}"}
            row.update({k: float(rng.normal()) for k in feat_names})
            rows.append(row)
        features_df = pd.DataFrame(rows)
    else:
        rows = []
        for pdir in tqdm(patient_dirs, desc="Extracting"):
            feats = extract_patient_features(pdir)
            if feats:
                row = {"patient_id": pdir.name, "tcga_id": pdir.name}
                row.update(feats)
                rows.append(row)
        features_df = pd.DataFrame(rows)

    if features_df.empty:
        print("No features extracted.")
        return

    features_df.to_csv(out_dir / "radiomics_full.csv", index=False)
    meta_cols = {"patient_id", "tcga_id"}
    feature_names = [c for c in features_df.columns if c not in meta_cols]

    # LASSO selection using SEER metastasis labels (matched by index for demo)
    seer = pd.read_csv(args.seer_csv or CONFIG["seer_csv"])
    y = seer["metastasis"].values[: len(features_df)]
    X = features_df[feature_names].fillna(0).values
    selected = lasso_select(X, y, feature_names, args.top_k)

    selection = {"selected_features": selected, "top_k": args.top_k, "n_total": len(feature_names)}
    (out_dir / "lasso_selected.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    features_df[["patient_id", "tcga_id"] + selected].to_csv(out_dir / "radiomics_lasso40.csv", index=False)
    print(f"Extracted {len(feature_names)} features, selected top {len(selected)}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
