from pathlib import Path
import json

import numpy as np
import pandas as pd

BASE_DIR = Path(
    r"C:\CN_PROJECT\IntelligentNetworkTrafficAnalysis"
)

DATA_DIR = BASE_DIR / "data" / "processed" / "split"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(
    MODEL_DIR / "feature_columns.json",
    encoding="utf-8"
) as file:
    features = json.load(file)

for split_name in ["train", "test"]:
    file_path = DATA_DIR / f"{split_name}.csv"

    df = pd.read_csv(file_path, low_memory=False)

    X = df[features].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace([np.inf, -np.inf], np.nan)

    negative_counts = (X < 0).sum()
    negative_counts = negative_counts[
        negative_counts > 0
    ].sort_values(ascending=False)

    print("\n" + "=" * 65)
    print(f"{split_name.upper()} DATA AUDIT")
    print("=" * 65)

    print("Records:", f"{len(df):,}")
    print("Features:", len(features))
    print("Missing feature values:", int(X.isna().sum().sum()))

    print("\nFeatures with negative values:")
    if negative_counts.empty:
        print("None")
    else:
        print(negative_counts.to_string())

        for feature in negative_counts.index:
            print(f"\nSmallest values: {feature}")
            print(X.loc[X[feature] < 0, feature].nsmallest(5))

    # Save records containing at least one negative feature
    mask = (X < 0).any(axis=1)

    if mask.any():
        columns = [
            col for col in ["Target", "Original_Label"]
            if col in df.columns
        ]

        anomalies = df.loc[mask, columns].copy()
        anomalies.insert(0, "row_index", df.index[mask])

        for feature in negative_counts.index:
            anomalies[feature] = X.loc[mask, feature]

        output_file = (
            REPORT_DIR /
            f"negative_feature_rows_{split_name}.csv"
        )

        anomalies.to_csv(output_file, index=False)

        print("\nRecords with negative features:", int(mask.sum()))
        print("Anomaly records saved to:", output_file)
    else:
        print("\nNo rows with negative feature values.")