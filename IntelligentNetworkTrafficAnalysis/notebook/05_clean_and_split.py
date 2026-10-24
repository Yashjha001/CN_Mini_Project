from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(
    r"C:\CN_PROJECT\IntelligentNetworkTrafficAnalysis"
)

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "binary_traffic_sample.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("Loading sampled dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)
df.columns = df.columns.str.strip()

required_columns = {"Target", "Original_Label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Missing required columns: "
        f"{required_columns - set(df.columns)}"
    )

df["Target"] = df["Target"].astype("string").str.strip()

# Retain only valid binary target labels
valid_targets = df["Target"].isin(["Normal", "Attack"])
invalid_count = int((~valid_targets).sum())

if invalid_count:
    print("Removing records with invalid targets:", invalid_count)
    df = df.loc[valid_targets].copy()

print("Records loaded:", f"{len(df):,}")

# --------------------------------------------------
# 3. Separate metadata from model features
# --------------------------------------------------

excluded_columns = {"Target", "Original_Label"}

feature_columns = [
    col for col in df.columns
    if col not in excluded_columns
]

# Convert feature columns to numeric values
for col in feature_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace positive and negative infinity with NaN
df[feature_columns] = df[feature_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

# --------------------------------------------------
# 4. Remove exact duplicate records
# --------------------------------------------------

before = len(df)

df = df.drop_duplicates().reset_index(drop=True)

duplicates_removed = before - len(df)

print("Duplicate records removed:", f"{duplicates_removed:,}")
print("Records after deduplication:", f"{len(df):,}")

# --------------------------------------------------
# 5. Remove the eight constant features
# --------------------------------------------------

constant_columns = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]

constant_columns = [
    col for col in constant_columns
    if col in df.columns
]

df = df.drop(columns=constant_columns)

feature_columns = [
    col for col in df.columns
    if col not in excluded_columns
]

print("Constant features removed:", len(constant_columns))
print("Remaining model features:", len(feature_columns))

# --------------------------------------------------
# 6. Create X and y
# --------------------------------------------------

X = df[feature_columns]
y = df["Target"]

if y.nunique() != 2:
    raise ValueError("Expected exactly two target classes.")

# --------------------------------------------------
# 7. Stratified train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Retain original labels for analysis only.
# Original_Label must NOT be used as a model feature.

train_df = X_train.copy()
train_df["Target"] = y_train
train_df["Original_Label"] = df.loc[
    X_train.index, "Original_Label"
]

test_df = X_test.copy()
test_df["Target"] = y_test
test_df["Original_Label"] = df.loc[
    X_test.index, "Original_Label"
]

# --------------------------------------------------
# 8. Save the two datasets
# --------------------------------------------------

train_file = OUTPUT_DIR / "train.csv"
test_file = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

# --------------------------------------------------
# 9. Print summary
# --------------------------------------------------

print("\n" + "=" * 55)
print("CLEANING AND SPLITTING COMPLETED")
print("=" * 55)

print(f"Training records: {len(train_df):,}")
print(f"Testing records:  {len(test_df):,}")

print("\nTraining target distribution:")
print(y_train.value_counts())
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nTesting target distribution:")
print(y_test.value_counts())
print((y_test.value_counts(normalize=True) * 100).round(2))

print("\nTraining file:", train_file)
print("Testing file: ", test_file)

print("\nNote: Missing and infinite feature values remain as NaN.")
print("They will be imputed using training data only.")