
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    files = list(DATA_DIR.rglob("*WebAttacks*.csv"))

    if not files:
        print("WebAttacks CSV file not found.")
        return

    file = files[0]
    print(f"Checking: {file.name}")

    df = None

    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(
                file,
                usecols=lambda col: col.strip().lower() == "label",
                encoding=encoding,
                low_memory=False
            )
            print(f"Encoding used: {encoding}")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        print("Could not read the label column.")
        return

    label_col = df.columns[0]
    labels = df[label_col].astype("string").str.strip()

    print(f"\nTotal rows: {len(labels):,}")
    print(f"Missing labels: {labels.isna().sum():,}")
    print(f"Blank labels: {(labels == '').sum():,}")

    print("\nLabel counts, including missing values:")
    print(labels.value_counts(dropna=False).to_string())

    print("\nFirst 20 rows with missing labels:")
    missing_rows = df[labels.isna() | (labels == "")]
    print(missing_rows.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
