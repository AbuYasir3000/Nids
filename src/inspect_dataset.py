
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

def main():
    csv_files = sorted(DATA_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {DATA_DIR}")
        print("Place the extracted CIC-IDS2017 CSV files in data/raw.")
        return

    print(f"Found {len(csv_files)} CSV file(s).")

    for file in csv_files:
        print(f"\n{'=' * 60}")
        print(f"File: {file.name}")

        try:
            # Read only a small sample initially.
            df = pd.read_csv(file, nrows=5, low_memory=False)

            print(f"Sample shape: {df.shape}")
            print("\nColumns:")
            print(df.columns.tolist())

            label_columns = [
                col for col in df.columns
                if col.strip().lower() == "label"
            ]

            if label_columns:
                print("\nSample labels:")
                print(df[label_columns[0]].value_counts(dropna=False))

            print("\nFirst 5 rows:")
            print(df.head().to_string(index=False))

        except Exception as exc:
            print(f"Could not read this file: {exc}")

if __name__ == "__main__":
    main()
