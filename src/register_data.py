"""Step 1 - Data registration.

Validates that the raw dataset in data/ has the expected columns and basic integrity,
prints a summary and writes a small registration manifest (data/data_manifest.json).
Exits with a non-zero status when validation fails so the CI pipeline stops early.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
    "Product_Id", "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
    "Product_Type", "Product_MRP", "Store_Id", "Store_Establishment_Year", "Store_Size",
    "Store_Location_City_Type", "Store_Type", "Product_Store_Sales_Total",
]


def validate(path: Path) -> dict:
    """Validate the dataset and return a registration manifest."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty")

    print("=" * 60)
    print("DATASET REGISTRATION SUMMARY")
    print("=" * 60)
    print(f"File           : {path.name}")
    print(f"Rows x Columns : {df.shape[0]} x {df.shape[1]}")
    print(f"Extra columns  : {extra if extra else 'none'}")
    print(f"Missing values : {int(df.isnull().sum().sum())}")
    print(f"Duplicate rows : {int(df.duplicated().sum())}")
    print("\nColumn data types:")
    print(df.dtypes.to_string())
    print("\nNumerical summary:")
    print(df.describe().T.round(3).to_string())
    print("\nAll expected columns are present. Validation PASSED.")

    return {
        "file": path.name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/SuperKart.csv")
    parser.add_argument("--manifest", default="data/data_manifest.json")
    args = parser.parse_args()
    try:
        manifest = validate(Path(args.data))
    except Exception as exc:  # report any validation failure and fail the job
        print(f"VALIDATION FAILED: {exc}")
        sys.exit(1)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest saved to {args.manifest}")
