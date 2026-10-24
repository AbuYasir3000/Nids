
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DDOS_FILENAME_PART = "friday-workinghours-afternoon-ddos"

EXCLUDED = {
    "flow id", "source ip", "destination ip", "timestamp", "label"
}


def main():
    matches = [
        p for p in DATA_ROOT.rglob("*.csv")
        if DDOS_FILENAME_PART in p.name.lower()
        and "pcap_iscx.csv" in p.name.lower()
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one DDoS capture CSV; found {len(matches)}."
        )

    path = matches[0]
    print(f"Reading {path.name}...")

    try:
        df = pd.read_csv(path, encoding="cp1252", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)

    df.columns = df.columns.astype(str).str.strip()

    label_col = next(
        (c for c in df.columns if c.lower() == "label"), None
    )
    if label_col is None:
        raise ValueError("Label column not found.")

    labels = df[label_col].astype(str).str.strip()
    y = (labels.str.upper() != "BENIGN").astype("int8")

    excluded = {
        c for c in df.columns if c.strip().lower() in EXCLUDED
    }
    X = df.drop(columns=list(excluded), errors="ignore")

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all").astype("float32")

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Test rows:     {len(X_test):,}")

    # Exact duplicate comparison on the numeric model features.
    # Missing values are normalized so NaNs hash consistently.
    def row_hashes(frame):
        normalized = frame.copy()
        normalized = normalized.fillna(np.nan)
        return pd.util.hash_pandas_object(
            normalized, index=False
        ).to_numpy(dtype="uint64")

    train_hash = row_hashes(X_train)
    test_hash = row_hashes(X_test)

    train_hash_set = set(train_hash.tolist())
    matches_train = np.fromiter(
        (h in train_hash_set for h in test_hash),
        dtype=bool,
        count=len(test_hash),
    )

    print("\n=== Exact feature-row overlap ===")
    print(f"Test rows matching a training feature row: {matches_train.sum():,}")
    print(f"Percentage of test rows: {matches_train.mean() * 100:.4f}%")

    # Compare labels for test rows whose feature hash exists in training.
    train_pairs = pd.DataFrame({
        "hash": train_hash,
        "label": y_train.to_numpy(),
    }).drop_duplicates()

    test_pairs = pd.DataFrame({
        "hash": test_hash,
        "label": y_test.to_numpy(),
        "matched": matches_train,
    })

    train_label_sets = train_pairs.groupby("hash")["label"].agg(
        lambda s: set(s.tolist())
    ).to_dict()

    overlapping = test_pairs[test_pairs["matched"]].copy()
    overlapping["training_labels"] = overlapping["hash"].map(
        train_label_sets
    )

    conflicting = overlapping[
        overlapping.apply(
            lambda row: row["label"] not in row["training_labels"],
            axis=1,
        )
    ]

    print(
        "Matched test rows with a conflicting training label: "
        f"{len(conflicting):,}"
    )

    # Duplicate rows within each side.
    train_duplicate_rows = int(X_train.duplicated().sum())
    test_duplicate_rows = int(X_test.duplicated().sum())

    print("\n=== Duplicates within each split ===")
    print(f"Duplicate training feature rows: {train_duplicate_rows:,}")
    print(f"Duplicate test feature rows:     {test_duplicate_rows:,}")

    # Save a compact report.
    report = pd.DataFrame([{
        "capture": path.name,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "test_rows_matching_training_features": int(matches_train.sum()),
        "test_match_percentage": float(matches_train.mean() * 100),
        "conflicting_label_matches": len(conflicting),
        "duplicate_training_rows": train_duplicate_rows,
        "duplicate_test_rows": test_duplicate_rows,
        "numeric_features": X.shape[1],
    }])

    out = REPORT_DIR / "ddos_split_duplicate_audit.csv"
    report.to_csv(out, index=False)

    print(f"\nSaved report: {out}")
    print("\nNote: exact feature matches are a warning signal, not proof")
    print("of leakage. Near-duplicates and temporal correlation need")
    print("separate checks.")


if __name__ == "__main__":
    main()
