"""Aggregate all phase results into Table 1 LaTeX + JSON."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main():
    rows = []

    baseline = load_json(RUNS / "baseline" / "baseline_results.json")
    if baseline:
        rows.append({
            "method": "LightGBM (clinical)",
            "auc": baseline["holdout_auc"],
            "cv_auc": f"{baseline['cv_auc_mean']:.3f} ± {baseline['cv_auc_std']:.3f}",
        })

    clinical = load_json(RUNS / "clinical_only" / "final_results.json")
    if clinical:
        m = clinical["test_metrics"]
        rows.append({
            "method": "MMRCCNet (clinical-only)",
            "auc": m["auc"],
            "cv_auc": "see statistical_validation.json",
            "site_auc": m.get("mean_site_auc"),
            "c_index": m.get("c_index"),
        })

    full = load_json(RUNS / "full" / "final_results.json")
    if full:
        m = full["test_metrics"]
        rows.append({
            "method": "MMRCCNet (full multimodal)",
            "auc": m["auc"],
            "site_auc": m.get("mean_site_auc"),
            "c_index": m.get("c_index"),
        })

    stats = load_json(RUNS / "clinical_only" / "statistical_validation.json")

    latex = ["\\begin{tabular}{lccc}", "\\toprule",
             "Method & AUC & Site AUC & C-index \\\\", "\\midrule"]
    for r in rows:
        site = f"{r.get('site_auc', 0):.3f}" if r.get("site_auc") else "---"
        cidx = f"{r.get('c_index', 0):.3f}" if r.get("c_index") else "---"
        latex.append(f"{r['method']} & {r['auc']:.3f} & {site} & {cidx} \\\\")
    latex += ["\\bottomrule", "\\end{tabular}"]

    out = {"table1": rows, "statistical_validation": stats}
    (RUNS / "table1_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (RUNS / "table1.tex").write_text("\n".join(latex), encoding="utf-8")
    print("Table 1 saved to runs/table1.tex and runs/table1_comparison.json")
    for r in rows:
        print(f"  {r['method']}: AUC={r['auc']:.4f}")


if __name__ == "__main__":
    main()
