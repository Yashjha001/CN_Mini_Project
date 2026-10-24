from pathlib import Path
import json

import numpy as np
import pandas as pd

BASE_DIR = Path(
    r"C:\CN_PROJECT\IntelligentNetworkTrafficAnalysis"
)

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "split_cleaned"
    / "train.csv"
)
FEATURE_FILE = BASE_DIR / "models" / "feature_columns.json"
LIVE_FILE = BASE_DIR / "reports" / "live_flow_features_aligned.csv"
REPORT_FILE = BASE_DIR / "reports" / "live_feature_validation.csv"

with open(FEATURE_FILE, encoding="utf-8") as file:
    features = json.load(file)

print("Loading training features...")
train = pd.read_csv(
    TRAIN_FILE,
    usecols=features,
    low_memory=False,
)

print("Loading live-flow features...")
live = pd.read_csv(LIVE_FILE, low_memory=False)

if list(live.columns) != features:
    raise ValueError(
        "Live feature names or order do not match the model."
    )

rows = []

for feature in features:
    training_values = pd.to_numeric(
        train[feature], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    live_values = pd.to_numeric(
        live[feature], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    live_valid = live_values.dropna()

    if training_values.empty:
        continue

    train_min = training_values.min()
    train_max = training_values.max()

    missing_count = int(live_values.isna().sum())

    if live_valid.empty:
        outside_range = len(live_values)
        outside_percentile = len(live_values)
        live_min = np.nan
        live_max = np.nan
    else:
        outside_range = int(
            (
                (live_valid < train_min)
                | (live_valid > train_max)
            ).sum()
        )

        lower = training_values.quantile(0.01)
        upper = training_values.quantile(0.99)

        outside_percentile = int(
            (
                (live_valid < lower)
                | (live_valid > upper)
            ).sum()
        )

        live_min = live_valid.min()
        live_max = live_valid.max()

    count = len(live_values)

    rows.append({
        "Feature": feature,
        "Train_Min": train_min,
        "Train_Max": train_max,
        "Live_Min": live_min,
        "Live_Max": live_max,
        "Live_Missing": missing_count,
        "Live_Missing_Pct": round(
            missing_count / max(count, 1) * 100, 2
        ),
        "Outside_Train_Range": outside_range,
        "Outside_Train_Range_Pct": round(
            outside_range / max(len(live_valid), 1) * 100, 2
        ),
        "Outside_Train_P01_P99_Pct": round(
            outside_percentile / max(len(live_valid), 1) * 100, 2
        ),
    })

report = pd.DataFrame(rows)

report.to_csv(REPORT_FILE, index=False)

print("\nVALIDATION SUMMARY")
print("=" * 65)
print("Live flow records:", len(live))
print("Model features:", len(features))
print(
    "Features with missing values:",
    int((report["Live_Missing"] > 0).sum())
)
print(
    "Features with values outside training min/max:",
    int((report["Outside_Train_Range"] > 0).sum())
)

print("\nFeatures with the most out-of-range live values:")
print(
    report.sort_values(
        "Outside_Train_Range_Pct",
        ascending=False
    ).head(15).to_string(index=False)
)

print("\nFull report saved to:")
print(REPORT_FILE)