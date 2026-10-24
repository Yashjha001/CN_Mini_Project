from pathlib import Path
from collections import Counter
import pandas as pd

# Dataset and output paths
DATA_DIR = Path(r"C:\CN_PROJECT\MachineLearningCVE")
OUTPUT_DIR = Path(
    r"C:\CN_PROJECT\IntelligentNetworkTrafficAnalysis\data\processed"
)
OUTPUT_FILE = OUTPUT_DIR / "binary_traffic_sample.csv"

CHUNK_SIZE = 100_000
SAMPLE_FRACTION = 0.05

files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Recreate the generated output file on each run.
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

first_chunk = True
total_input = 0
total_saved = 0
target_counts = Counter()

for file_index, file in enumerate(files):
    print(f"\nProcessing: {file.name}")

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            low_memory=False
        )
    ):
        # Standardize column names
        chunk.columns = chunk.columns.str.strip()

        label_columns = [
            col for col in chunk.columns
            if col.casefold() == "label"
        ]

        if not label_columns:
            raise ValueError(f"Label column missing in {file.name}")

        label_column = label_columns[0]

        # Remove records without a valid label
        labels = chunk[label_column].astype("string").str.strip()
        valid = labels.notna() & labels.ne("")

        chunk = chunk.loc[valid].copy()
        labels = labels.loc[valid]

        # Preserve the original label for later analysis
        chunk = chunk.drop(columns=[label_column])
        chunk["Original_Label"] = labels.to_numpy()

        # Convert the problem to binary classification
        chunk["Target"] = [
            "Normal" if label.upper() == "BENIGN" else "Attack"
            for label in labels
        ]

        total_input += len(chunk)

        # Take a reproducible sample from each chunk
        seed = 42 + file_index * 10_000 + chunk_index

        sampled = chunk.sample(
            frac=SAMPLE_FRACTION,
            random_state=seed
        )

        # Append samples without loading the entire dataset
        sampled.to_csv(
            OUTPUT_FILE,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk
        )

        first_chunk = False
        total_saved += len(sampled)
        target_counts.update(sampled["Target"].value_counts().to_dict())

    print("Completed:", file.name)

if first_chunk:
    raise ValueError("No valid labeled records were found.")

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)
print(f"Input records processed: {total_input:,}")
print(f"Sampled records saved:   {total_saved:,}")
print(f"Sampling percentage:     {SAMPLE_FRACTION:.0%}")
print(f"Output file:              {OUTPUT_FILE}")

print("\nSampled target distribution:")
for label, count in sorted(target_counts.items()):
    print(f"{label}: {count:,}")

print("\nNext step: inspect the sampled dataset before training.")