
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# Project folders
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Entire PortScan capture will be excluded from training.
HELDOUT_KEY = "friday-workinghours-afternoon-portscan"

# Columns that identify a flow rather than describe its traffic.
NON_FEATURE_COLUMNS = {
    "flow id",
    "source ip",
    "destination ip",
    "timestamp",
    "label",
}

def find_csv_files():
    files = [
        p for p in DATA_DIR.rglob("*.csv")
        if "friday-workinghours-afternoon-portscan" not in p.name.lower()
        or "portscan" in p.name.lower()
    ]

    # Use the eight original CIC-IDS2017 capture files.
    files = [
        p for p in files
        if p.name.lower().endswith("pcap_iscx.csv")
    ]

    heldout = [
        p for p in files
        if HELDOUT_KEY in p.name.lower()
    ]

    if len(heldout) != 1:
        raise FileNotFoundError(
            "Could not uniquely locate the Friday PortScan CSV. "
            "Check the filename and data folder."
        )

    return sorted(files), heldout[0]


def load_capture(path):
    print(f"Reading: {path.name}", flush=True)

    # CIC-IDS2017 files may contain encoding issues.
    df = pd.read_csv(
        path,
        encoding="cp1252",
        low_memory=False,
    )

    # Remove whitespace around column names.
    df.columns = df.columns.astype(str).str.strip()

    label_col = next(
        (c for c in df.columns if c.lower() == "label"),
        None,
    )
    if label_col is None:
        raise ValueError(f"No Label column found in {path.name}")

    labels = df[label_col].astype(str).str.strip().str.upper()

    # Binary target: BENIGN = 0; every attack = 1.
    y = (labels != "BENIGN").astype("int8")

    excluded = {
        c for c in df.columns
        if c.strip().lower() in NON_FEATURE_COLUMNS
    }

    X = df.drop(columns=list(excluded), errors="ignore")

    # Convert features to numeric and handle infinite values.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove columns with no usable numeric values.
    X = X.dropna(axis=1, how="all")

    # Remove rows with missing targets, if any.
    valid = labels.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    # Keep capture name for reporting outside the feature matrix.
    return X, y


def main():
    files, heldout_path = find_csv_files()

    print("\nHeld-out capture:", heldout_path.name)
    print("Other capture files:", len(files) - 1)

    train_frames = []
    train_labels = []

    for path in files:
        if path.resolve() == heldout_path.resolve():
            continue

        X, y = load_capture(path)
        train_frames.append(X)
        train_labels.append(y)

        print(
            f"  Training source: {path.name} | "
            f"rows={len(X):,} | attacks={int(y.sum()):,}",
            flush=True,
        )

    X_test, y_test = load_capture(heldout_path)

    # Align feature columns across different CSV captures.
    feature_names = sorted(
        set().union(*(set(x.columns) for x in train_frames))
        & set(X_test.columns)
    )

    if not feature_names:
        raise ValueError("No common numeric features found.")

    X_train = pd.concat(
        [x.reindex(columns=feature_names) for x in train_frames],
        ignore_index=True,
    )
    y_train = pd.concat(train_labels, ignore_index=True)
    X_test = X_test.reindex(columns=feature_names)

    # Remove features that are completely missing in training.
    usable = X_train.notna().any(axis=0)
    X_train = X_train.loc[:, usable]
    X_test = X_test.loc[:, usable]

    print("\n=== Dataset summary ===")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows:     {len(X_test):,}")
    print(f"Features:      {X_train.shape[1]}")
    print(f"Training attacks: {int(y_train.sum()):,}")
    print(f"Test attacks:     {int(y_test.sum()):,}")
    print(f"Test benign:      {int((y_test == 0).sum()):,}")

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Training and test data must each contain both "
            "BENIGN and ATTACK rows for this evaluation."
        )

    # Fit preprocessing on training data only.
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining model...", flush=True)
    start = time.time()
    model.fit(X_train, y_train)
    print(f"Training time: {time.time() - start:.1f} seconds")

    print("\nTesting on the entire held-out capture...", flush=True)
    predictions = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    results = {
        "heldout_capture": heldout_path.name,
        "training_rows": len(y_train),
        "test_rows": len(y_test),
        "features": X_train.shape[1],
        "accuracy_pct": accuracy_score(y_test, predictions) * 100,
        "balanced_accuracy_pct": (
            balanced_accuracy_score(y_test, predictions) * 100
        ),
        "macro_f1_pct": (
            f1_score(y_test, predictions, average="macro") * 100
        ),
        "attack_precision_pct": (
            precision_score(y_test, predictions, zero_division=0) * 100
        ),
        "attack_recall_pct": recall_score(
            y_test, predictions, zero_division=0
        ) * 100,
        "benign_fpr_pct": (
            fp / (tn + fp) * 100 if (tn + fp) else 0.0
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print("\n=== HELD-OUT PORTSCAN RESULTS ===")
    for key, value in results.items():
        if key.endswith("_pct"):
            print(f"{key}: {value:.4f}%")
        else:
            print(f"{key}: {value}")

    print("\n=== Classification report ===")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            zero_division=0,
            digits=4,
        )
    )

    output = REPORT_DIR / "heldout_portscan_evaluation.csv"
    pd.DataFrame([results]).to_csv(output, index=False)

    print(f"\nSaved results to: {output}")


if __name__ == "__main__":
    main()
