from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"C:\CN_PROJECT\MachineLearningCVE")

files = sorted(DATA_DIR.glob("*.csv"))

if not files:
    print("No CSV files found. Check the folder and file extensions.")
else:
    print(f"Found {len(files)} CSV files\n")

    for file in files:
        print("=" * 70)
        print("File:", file.name)

        try:
            df = pd.read_csv(file, nrows=5)

            print("Columns:", len(df.columns))
            print("Sample shape:", df.shape)
            print("Column names:")
            print(df.columns.tolist())

            label_col = next(
                (col for col in df.columns
                 if col.strip().lower() == "label"),
                None
            )

            if label_col:
                print("\nSample labels:")
                print(df[label_col].astype(str).str.strip().tolist())
            else:
                print("\nLabel column not found.")

        except Exception as error:
            print("Error:", error)