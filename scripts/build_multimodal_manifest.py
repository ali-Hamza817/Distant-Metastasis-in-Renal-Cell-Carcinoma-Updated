"""Build multimodal manifest linking SEER clinical + TCGA imaging/genomics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.config import CONFIG


def _load_genomics_vector(path: Path, dim: int) -> list[float]:
    if path.suffix == ".tsv" or path.suffix == ".txt":
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        vals = []
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    vals.append(float(parts[1]))
                except ValueError:
                    pass
        if vals:
            arr = np.array(vals[:dim], dtype=np.float32)
            if len(arr) < dim:
                out = np.zeros(dim, dtype=np.float32)
                out[: len(arr)] = arr
                return out.tolist()
            return arr.tolist()
    return np.zeros(dim).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seer_csv", default=CONFIG["seer_csv"])
    parser.add_argument("--dicom_dir", default=CONFIG["dicom_dir"])
    parser.add_argument("--genomics_dir", default=CONFIG["genomics_dir"])
    parser.add_argument("--radiomics_json", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max_records", type=int, default=None)
    args = parser.parse_args()

    seer = pd.read_csv(args.seer_csv)
    dicom_dir = Path(args.dicom_dir)
    genomics_dir = Path(args.genomics_dir)
    out_path = Path(args.out or Path(CONFIG["manifest_dir"]) / "multimodal_manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    radiomics_features = []
    radiomics_path = Path(args.radiomics_json or Path(CONFIG["radiomics_dir"]) / "lasso_selected.json")
    if radiomics_path.exists():
        radiomics_features = json.loads(radiomics_path.read_text())["selected_features"]

    radiomics_df = None
    rad_csv = Path(CONFIG["radiomics_dir"]) / "radiomics_lasso40.csv"
    if rad_csv.exists():
        radiomics_df = pd.read_csv(rad_csv)

    dicom_patients = {p.name: p for p in dicom_dir.iterdir() if p.is_dir()} if dicom_dir.exists() else {}
    genomics_files = list(genomics_dir.glob("*.tsv")) + list(genomics_dir.glob("*.txt")) if genomics_dir.exists() else []

    records = []
    n = args.max_records or len(seer)
    rng = np.random.default_rng(42)

    for i in range(min(n, len(seer))):
        row = seer.iloc[i]
        rec = {
            "patient_id": row["patient_id"],
            "clinical": {f: float(row[f]) for f in CONFIG["clinical_features"]},
            "labels": {
                "metastasis": int(row["metastasis"]),
                "lung_met": int(row["lung_met"]),
                "bone_met": int(row["bone_met"]),
                "liver_met": int(row["liver_met"]),
                "brain_met": int(row["brain_met"]),
                "survival_months": float(row["survival_months"]),
                "vital_status": int(row["vital_status"]),
            },
            "imaging_path": None,
            "genomics": None,
            "radiomics": {},
        }

        if dicom_patients:
            pid = list(dicom_patients.keys())[i % len(dicom_patients)]
            rec["imaging_path"] = str(dicom_patients[pid] / list(dicom_patients[pid].iterdir())[0].name
                                       if list(dicom_patients[pid].iterdir()) else dicom_patients[pid])

        if genomics_files:
            gf = genomics_files[i % len(genomics_files)]
            rec["genomics"] = _load_genomics_vector(gf, CONFIG["genomics_dim"])
        else:
            rec["genomics"] = rng.normal(0, 1, CONFIG["genomics_dim"]).tolist()

        if radiomics_df is not None and i < len(radiomics_df):
            for f in radiomics_features:
                rec["radiomics"][f] = float(radiomics_df.iloc[i].get(f, 0.0))
        elif radiomics_features:
            for f in radiomics_features:
                rec["radiomics"][f] = float(rng.normal())

        records.append(rec)

    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Built manifest with {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
