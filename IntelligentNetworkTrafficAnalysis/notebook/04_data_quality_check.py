from pathlib import Path
import pandas as pd
import numpy as np

DATA_FILE = Path(
    r"C:\CN_PROJECT\IntelligentNetworkTrafficAnalysis"
    r"\data\processed\binary_traffic_sample.csv"
)

print("Loading sampled dataset...")

df = pd.read_csv(DATA_FILE, low_memory=False)
df.columns = df.columns.str.strip()

print("\n1. DATASET OVERVIEW")
print("-" * 50)
print("Rows:", f"{len(df):,}")
print("Columns:", len(df.columns))
print("Memory usage (MB):",
      round(df.memory_usage(deep=True).sum() / 1024**2, 2))

print("\n2. TARGET DISTRIBUTION")
print("-" * 50)
print(df["Target"].value_counts(dropna=False))
print("\nTarget percentages:")
print((df["Target"].value_counts(normalize=True) * 100).round(2))

print("\n3. ORIGINAL ATTACK LABELS")
print("-" * 50)
print(df["Original_Label"].value_counts(dropna=False))

# Exclude both label columns from feature analysis
excluded = {"Target", "Original_Label"}
feature_cols = [c for c in df.columns if c not in excluded]
X_raw = df[feature_cols]

print("\n4. MISSING VALUES")
print("-" * 50)
missing = X_raw.isna().sum()
print(missing[missing > 0].sort_values(ascending=False))
print("Total missing feature values:", int(missing.sum()))

print("\n5. NUMERIC FEATURE CHECK")
print("-" * 50)

# Identify values that cannot be converted to numbers
X_num = X_raw.apply(pd.to_numeric, errors="coerce")

invalid = X_num.isna() & X_raw.notna()
invalid_counts = invalid.sum()
print("Non-numeric values by feature:")
print(invalid_counts[invalid_counts > 0])

print("\n6. INFINITE VALUES")
print("-" * 50)
infinite = pd.Series(
    np.isinf(X_num.to_numpy()).sum(axis=0),
    index=X_num.columns
)
print(infinite[infinite > 0].sort_values(ascending=False))
print("Total infinite feature values:", int(infinite.sum()))

print("\n7. CONSTANT FEATURES")
print("-" * 50)
constant_cols = [
    c for c in X_num.columns
    if X_num[c].nunique(dropna=False) <= 1
]
print(constant_cols if constant_cols else "None found")

print("\n8. DUPLICATE ROWS")
print("-" * 50)
print("Exact duplicate rows:", f"{int(df.duplicated().sum()):,}")

print("\n9. DUPLICATE COLUMN NAMES")
print("-" * 50)
duplicates = df.columns[df.columns.duplicated()].tolist()
print(duplicates if duplicates else "None")

print("\nData quality inspection completed.")