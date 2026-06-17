"""
Fetch real TCGA-KIRC clinical labels from the NCI GDC public API.
No authentication required. Saves: e:/rcc/data/tcga_kirc_clinical.csv
"""
import json, urllib.request, time, os
import pandas as pd

GDC_CASES = "https://api.gdc.cancer.gov/cases"
OUT_CSV   = "e:/rcc/data/tcga_kirc_clinical.csv"
META_CSV  = "E:/rcc/TCIA_TCGA-KIRC_09-16-2015 (2)/metadata/metadata.csv"

FIELDS = ",".join([
    "submitter_id",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.days_to_last_follow_up",
    "demographic.days_to_death",
    "demographic.vital_status",
    "demographic.age_at_index",
    "demographic.gender",
])

FILTERS = json.dumps({
    "op": "in",
    "content": {
        "field": "project.project_id",
        "value": ["TCGA-KIRC"]
    }
})

def fetch_all():
    records = []
    size = 500
    offset = 0
    while True:
        params = f"?filters={urllib.parse.quote(FILTERS)}&fields={FIELDS}&size={size}&from={offset}&format=json"
        url = GDC_CASES + params
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            hits = data["data"]["hits"]
            if not hits:
                break
            records.extend(hits)
            total = data["data"]["pagination"]["total"]
            print(f"  Fetched {len(records)}/{total} cases...")
            if len(records) >= total:
                break
            offset += size
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error fetching offset {offset}: {e}")
            break
    return records

import urllib.parse

print("Fetching TCGA-KIRC clinical data from NCI GDC API...")
cases = fetch_all()
print(f"Total cases fetched: {len(cases)}")

rows = []
for c in cases:
    sid = c.get("submitter_id", "")
    demo = c.get("demographic", [{}])
    if isinstance(demo, list):
        demo = demo[0] if demo else {}
    diag = c.get("diagnoses", [{}])
    if isinstance(diag, list):
        diag = diag[0] if diag else {}

    vital = demo.get("vital_status", "unknown")
    age   = demo.get("age_at_index", None)
    sex   = demo.get("gender", "unknown")
    days_death  = demo.get("days_to_death", None)
    days_follow = diag.get("days_to_last_follow_up", None)

    m_stage = diag.get("ajcc_pathologic_m", "unknown")
    t_stage = diag.get("ajcc_pathologic_t", "unknown")
    n_stage = diag.get("ajcc_pathologic_n", "unknown")

    # Binary metastasis label from M-stage
    if m_stage and "m1" in str(m_stage).lower():
        metastasis = 1
    elif m_stage and "m0" in str(m_stage).lower():
        metastasis = 0
    else:
        metastasis = None  # unknown

    # Survival months
    if days_death and str(days_death).isdigit():
        survival_months = float(days_death) / 30.44
    elif days_follow and str(days_follow).replace('.','').isdigit():
        survival_months = float(days_follow) / 30.44
    else:
        survival_months = None

    rows.append({
        "patient_id":       sid,
        "age":              age,
        "sex":              1 if str(sex).lower() == "male" else 0,
        "t_stage_raw":      t_stage,
        "n_stage_raw":      n_stage,
        "m_stage_raw":      m_stage,
        "metastasis":       metastasis,
        "survival_months":  survival_months,
        "vital_status":     vital,
    })

df = pd.DataFrame(rows)
print(f"\nTotal TCGA-KIRC records: {len(df)}")
print(f"M-stage distribution:\n{df['m_stage_raw'].value_counts()}")
print(f"Metastasis labels available: {df['metastasis'].notna().sum()}")
print(f"  -> M1 (positive): {(df['metastasis']==1).sum()}")
print(f"  -> M0 (negative): {(df['metastasis']==0).sum()}")
print(f"  -> Unknown:       {df['metastasis'].isna().sum()}")

# Now cross-reference with the cached DICOM patient IDs
meta = pd.read_csv(META_CSV)
dicom_patients = set(meta["PatientID"].unique())
print(f"\nPatients with DICOM data: {len(dicom_patients)}")

df["has_dicom"] = df["patient_id"].isin(dicom_patients)
matched = df[df["has_dicom"] & df["metastasis"].notna()]
print(f"Patients with BOTH DICOM + M-stage label: {len(matched)}")
print(f"  -> M1: {(matched['metastasis']==1).sum()}")
print(f"  -> M0: {(matched['metastasis']==0).sum()}")

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved to: {OUT_CSV}")
