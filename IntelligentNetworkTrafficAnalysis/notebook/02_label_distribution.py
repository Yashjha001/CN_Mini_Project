from pathlib import Path
from collections import Counter
import pandas as pd

DATA_DIR = Path(r"C:\CN_PROJECT\MachineLearningCVE")

files = sorted(DATA_DIR.glob("*.csv"))

overall_counts = Counter()

for file in files:
    print("\n" + "=" * 70)
    print(f"File: {file.name}")

    file_counts = Counter()
    total_rows = 0

    try:
        # Read large files in chunks to reduce memory usage
        for chunk in pd.read_csv(
            file,
            chunksize=200_000,
            low_memory=False
        ):
            # Remove spaces around column names
            chunk.columns = chunk.columns.str.strip()

            if "Label" not in chunk.columns:
                print("ERROR: Label column not found.")
                break

            # Normalize label text
            labels = chunk["Label"].astype("string").str.strip()

            counts = labels.value_counts(dropna=False)

            file_counts.update({
                str(label): int(count)
                for label, count in counts.items()
            })

            total_rows += len(chunk)

        print("Total records:", f"{total_rows:,}")
        print("Label distribution:")

        for label, count in sorted(file_counts.items()):
            print(f"  {label}: {count:,}")

        overall_counts.update(file_counts)

    except Exception as error:
        print("Error processing file:", error)

print("\n" + "=" * 70)
print("OVERALL LABEL DISTRIBUTION")

for label, count in sorted(overall_counts.items()):
    print(f"{label}: {count:,}")

print("\nTotal records:", f"{sum(overall_counts.values()):,}")