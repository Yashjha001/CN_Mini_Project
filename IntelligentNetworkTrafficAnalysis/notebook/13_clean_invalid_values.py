from pathlib import Path
import json

import numpy as np
import pandas as pd

BASE_DIR = Path(
    r"C:\CN_PROJECT\IntelligentNetworkTrafficAnalysis"
)

SOURCE_DIR = BASE_DIR / "data" / "processed" / "split"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "split_cleaned"

FEATURE_FILE = BASE_DIR / "models" / "feature_columns.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(FEATURE_FILE, encoding="utf-8") as file:
    feature_columns = json.load(file)

# Preserve these existing dataset values for now.
# The CIC-IDS2017 data contains -1 in initial-window fields.
SENTINEL_FEATURES = {
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
}

for split_name in ["train", "test"]:
    input_file = SOURCE_DIR / f"{split_name}.csv"
    output_file = OUTPUT_DIR / f"{split_name}.csv"

    print("\n" + "=" * 60)
    print(f"Processing {split_name.upper()} dataset")
    print("=" * 60)

    df = pd.read_csv(input_file, low_memory=False)

    if not set(feature_columns).issubset(df.columns):
        raise ValueError(
            f"Missing expected model features in {input_file}"
        )

    X = df[feature_columns].copy()

    # Convert features to numeric and replace infinities.
    for column in feature_columns:
        X[column] = pd.to_numeric(
            X[column], errors="coerce"
        )

    X = X.replace([np.inf, -np.inf], np.nan)

    invalid_counts = {}

    for column in feature_columns:
        if column in SENTINEL_FEATURES:
            # Do not reinterpret the existing -1 encoding.
            continue

        invalid_mask = X[column] < 0
        count = int(invalid_mask.sum())

        if count:
            X.loc[invalid_mask, column] = np.nan
            invalid_counts[column] = count

    print("Rows preserved:", f"{len(df):,}")
    print("Invalid values replaced:", sum(invalid_counts.values()))
    print("Missing feature values after cleaning:",
          int(X.isna().sum().sum()))

    if invalid_counts:
        print("\nColumns with invalid negative values:")
        for column, count in sorted(invalid_counts.items()):
            print(f"  {column}: {count}")

    # Preserve Target and Original_Label without modifying them.
    for column in feature_columns:
        df[column] = X[column]

    df.to_csv(output_file, index=False)
    print("\nSaved:", output_file)

print("\nCleaning completed.")
print("Original train/test files remain unchanged.")