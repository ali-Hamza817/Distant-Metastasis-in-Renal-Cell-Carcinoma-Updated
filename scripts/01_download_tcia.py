"""PHASE 3 — Download TCGA-KIRC CT scans from TCIA."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

TCIA_BASE = "https://services.cancerimagingarchive.net/services/v4"
COLLECTION = "TCGA-KIRC"


def get_patient_list(max_patients: int | None = None) -> list[str]:
    url = f"{TCIA_BASE}/TCIA/query/getPatient?Collection={COLLECTION}"
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            import time
            time.sleep(5 * (attempt + 1))
    patients = [p["PatientID"] for p in resp.json()]
    if max_patients:
        patients = patients[:max_patients]
    return patients


def download_series(patient_id: str, out_dir: Path) -> dict:
    """Download one patient's CT series via TCIA NBIA API."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_url = f"{TCIA_BASE}/TCIA/query/getSeries?Collection={COLLECTION}&PatientID={patient_id}"
    resp = requests.get(meta_url, timeout=60)
    if resp.status_code != 200:
        return {"patient_id": patient_id, "status": "failed", "error": resp.text[:200]}

    series_list = resp.json()
    ct_series = [s for s in series_list if "CT" in s.get("Modality", "")]
    if not ct_series:
        ct_series = series_list[:1]

    saved = []
    for series in ct_series[:1]:
        series_uid = series["SeriesInstanceUID"]
        patient_dir = out_dir / patient_id
        patient_dir.mkdir(exist_ok=True)

        # TCIA getImage download (returns zip of DICOM)
        img_url = (
            f"{TCIA_BASE}/TCIA/query/getImage"
            f"?SeriesInstanceUID={series_uid}"
        )
        zip_path = patient_dir / f"{series_uid}.zip"
        if zip_path.exists():
            saved.append(str(zip_path))
            continue

        try:
            r = requests.get(img_url, timeout=300, stream=True)
            if r.status_code == 200:
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(patient_dir / series_uid)
                saved.append(str(patient_dir / series_uid))
        except Exception as e:
            return {"patient_id": patient_id, "status": "error", "error": str(e)}

    return {"patient_id": patient_id, "status": "ok", "paths": saved}


def main():
    parser = argparse.ArgumentParser(description="Download TCGA-KIRC CT from TCIA")
    parser.add_argument("--out_dir", default="data/dicom", help="Output directory")
    parser.add_argument("--max_patients", type=int, default=10, help="Limit patients (10 for test)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 3a — TCIA CT Download (TCGA-KIRC)")
    print("=" * 60)

    try:
        patients = get_patient_list(args.max_patients)
    except requests.RequestException as e:
        print(f"TCIA API unavailable ({e}). Writing offline stub manifest.")
        manifest = [{"patient_id": f"TCGA-KIRC-STUB-{i:03d}", "status": "stub"} for i in range(args.max_patients)]
        manifest_path = out_dir / "download_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Stub manifest: {manifest_path}")
        return

    print(f"Downloading {len(patients)} patients...")

    manifest = []
    for pid in tqdm(patients):
        result = download_series(pid, out_dir)
        manifest.append(result)

    manifest_path = out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok = sum(1 for m in manifest if m["status"] == "ok")
    print(f"\nDone: {ok}/{len(manifest)} patients downloaded")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
