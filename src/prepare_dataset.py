
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "cicids2017_cleaned.csv"


def main():
    csv_files = sorted(DATA_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframes = []
    total_rows = 0

    print(f"Found {len(csv_files)} CSV files\n")

    for file in csv_files:
        print(f"Reading: {file.name}")

        try:
            df = pd.read_csv(
                file,
                encoding="cp1252",
                low_memory=False
            )

            # Standardize column names.
            df.columns = df.columns.str.strip()

            if "Label" not in df.columns:
                print("  WARNING: Label column missing. Skipping file.")
                continue

            rows_before = len(df)
            total_rows += rows_before

            # Normalize label text and remove missing labels.
            df["Label"] = df["Label"].astype("string").str.strip()
            df = df.dropna(subset=["Label"])
            df = df[df["Label"] != ""]

            # Remove duplicate rows within this file.
            df = df.drop_duplicates()

            # Replace infinite numeric values with missing values.
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].replace(
                [np.inf, -np.inf], np.nan
            )

            dataframes.append(df)

            print(
                f"  Original rows: {rows_before:,} | "
                f"Rows retained: {len(df):,}"
            )

        except Exception as error:
            print(f"  ERROR: {error}")

    if not dataframes:
        print("\nNo usable CSV data was loaded.")
        return

    print("\nCombining datasets...")
    combined = pd.concat(dataframes, ignore_index=True, sort=False)

    # Remove duplicate records across the combined dataset.
    before_duplicates = len(combined)
    combined = combined.drop_duplicates()
    duplicates_removed = before_duplicates - len(combined)

    combined.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Original rows read: {total_rows:,}")
    print(f"Combined rows before final deduplication: {before_duplicates:,}")
    print(f"Cross-file duplicates removed: {duplicates_removed:,}")
    print(f"Final rows saved: {len(combined):,}")
    print(f"Features/columns: {len(combined.columns)}")

    print("\nFinal label distribution:")
    print(combined["Label"].value_counts().to_string())

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
