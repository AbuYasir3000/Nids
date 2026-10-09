
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def read_label_column(file):
    """Read only the Label column, handling different file encodings."""
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(
                file,
                usecols=lambda col: col.strip().lower() == "label",
                encoding=encoding,
                low_memory=False
            )
        except UnicodeDecodeError:
            continue

    raise ValueError("Could not read the file using supported encodings.")


def main():
    csv_files = sorted(DATA_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {DATA_DIR}")
        print("Check that your CIC-IDS2017 CSV files are inside data/raw.")
        return

    print(f"Found {len(csv_files)} CSV files\n")

    overall_counts = {}

    for file in csv_files:
        print("=" * 70)
        print(f"File: {file.name}")

        try:
            df = read_label_column(file)

            if df.empty or len(df.columns) == 0:
                print("No Label column found.")
                continue

            label_col = df.columns[0]
            labels = df[label_col].astype("string").str.strip()

            print(f"Rows: {len(labels):,}")
            print("\nAttack label counts:")
            print(labels.value_counts(dropna=False).to_string())

            counts = labels.value_counts(dropna=False)
            for label, count in counts.items():
                label = "<MISSING>" if pd.isna(label) else str(label)
                overall_counts[label] = (
                    overall_counts.get(label, 0) + int(count)
                )

        except Exception as error:
            print(f"Error processing file: {error}")

    print("\n" + "=" * 70)
    print("OVERALL ATTACK LABEL COUNTS")
    print("=" * 70)

    if overall_counts:
        summary = pd.Series(overall_counts, name="Count")
        print(summary.sort_values(ascending=False).to_string())
        print(f"\nTotal records counted: {sum(overall_counts.values()):,}")
        print(f"Distinct labels: {len(overall_counts)}")
    else:
        print("No labels could be counted.")


if __name__ == "__main__":
    main()
