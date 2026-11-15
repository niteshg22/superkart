"""Step 2 - Data preparation.

Loads the dataset from the repository data/ folder, cleans it, engineers features,
drops unnecessary columns, splits into train/test and saves the splits locally in
`artifacts/` (uploaded as a workflow artifact by GitHub Actions for the next job).
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Product_Store_Sales_Total"
REFERENCE_YEAR = 2025          # fixed so that Store_Age is reproducible
RANDOM_STATE = 42


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data and create model features."""
    df = df.copy()
    df = df.drop_duplicates()

    # Standardise text columns (strip spaces) and unify sugar-content labels
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip()
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})

    # Feature engineering
    # Product_Id starts with FD (food), DR (drink) or NC (non-consumable)
    df["Product_Category"] = df["Product_Id"].str[:2]
    df["Store_Age"] = REFERENCE_YEAR - df["Store_Establishment_Year"]

    # Drop identifiers / redundant columns (Store_Id is fully described by the store attributes)
    df = df.drop(columns=["Product_Id", "Store_Id", "Store_Establishment_Year"])

    # Impute any gaps (none expected) - median for numbers, mode for categories
    for c in df.columns:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median() if df[c].dtype != "object" else df[c].mode()[0])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/SuperKart.csv")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    raw = pd.read_csv(args.data)
    print("Raw shape:", raw.shape)
    df = clean(raw)
    print("Cleaned shape:", df.shape)
    print("Columns kept:", list(df.columns))

    train, test = train_test_split(df, test_size=args.test_size, random_state=RANDOM_STATE)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(f"Saved train ({train.shape}) and test ({test.shape}) to '{out}/'")
