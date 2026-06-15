"""PHASE 3 — Download TCGA-KIRC RNA-seq from GDC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from tqdm import tqdm

GDC_API = "https://api.gdc.cancer.gov"
PROJECT = "TCGA-KIRC"


def query_rna_files() -> list[dict]:
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": [PROJECT]}},
            {"op": "=", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "=", "content": {"field": "files.analysis.workflow_type", "value": ["STAR - Counts"]}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id",
        "format": "JSON",
        "size": "500",
    }
    resp = requests.get(f"{GDC_API}/files", params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()["data"]["hits"]


def download_file(file_id: str, file_name: str, out_dir: Path) -> str:
    out_path = out_dir / file_name
    if out_path.exists():
        return str(out_path)
    url = f"{GDC_API}/data/{file_id}"
    resp = requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Download TCGA-KIRC RNA-seq from GDC")
    parser.add_argument("--out_dir", default="data/genomics", help="Output directory")
    parser.add_argument("--max_files", type=int, default=None, help="Limit files for testing")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 3b — GDC RNA-seq Download (TCGA-KIRC)")
    print("=" * 60)

    files = query_rna_files()
    if args.max_files:
        files = files[: args.max_files]
    print(f"Found {len(files)} RNA-seq files")

    manifest = []
    for hit in tqdm(files):
        fid = hit["file_id"]
        fname = hit["file_name"]
        case_ids = []
        for c in hit.get("cases", []):
            case_ids.append(c.get("submitter_id", ""))
        try:
            path = download_file(fid, fname, out_dir)
            manifest.append({
                "file_id": fid,
                "file_name": fname,
                "case_id": case_ids[0] if case_ids else "",
                "path": path,
                "status": "ok",
            })
        except Exception as e:
            manifest.append({"file_id": fid, "file_name": fname, "status": "error", "error": str(e)})

    manifest_path = out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok = sum(1 for m in manifest if m.get("status") == "ok")
    print(f"\nDone: {ok}/{len(manifest)} files downloaded")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
