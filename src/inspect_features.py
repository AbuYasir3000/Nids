
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cicids2017_cleaned.csv"
)


def main():
    print("Loading column information...")

    df = pd.read_csv(DATA_FILE, nrows=5, low_memory=False)

    print(f"\nTotal columns: {len(df.columns)}")

    print("\nAll column names:")
    for column in df.columns:
        print(repr(column))

    identifiers = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
    ]

    print("\nColumns to exclude from model inputs:")
    for column in identifiers:
        print(f"{column}: {'found' if column in df.columns else 'not found'}")

    print("\nTarget column:")
    print("Label" if "Label" in df.columns else "Label column missing")


if __name__ == "__main__":
    main()
