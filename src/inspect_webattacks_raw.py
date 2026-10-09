
from pathlib import Path
import csv
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    files = list(DATA_DIR.rglob("*WebAttacks*.csv"))

    if not files:
        print("WebAttacks CSV file not found.")
        return

    file = files[0]
    print(f"File: {file.name}\n")

    counts = Counter()
    examples = []

    with file.open("r", encoding="cp1252", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        label_index = next(
            i for i, name in enumerate(header)
            if name.strip().lower() == "label"
        )

        print(f"Label column index: {label_index}")
        print(f"Total columns: {len(header)}")

        for line_number, row in enumerate(reader, start=2):
            label = row[label_index].strip()
            counts[label] += 1

            if not label and len(examples) < 10:
                examples.append((line_number, row[-5:]))

    print("\nRaw label counts:")
    for label, count in counts.most_common():
        print(f"{label!r}: {count:,}")

    print("\nRows with empty labels:")
    for line_number, ending in examples:
        print(f"Line {line_number}: {ending!r}")


if __name__ == "__main__":
    main()
