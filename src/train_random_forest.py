
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cicids2017_cleaned.csv"
)
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

EXCLUDED_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label",
]


def main():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        print("Run src/prepare_dataset.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        print("ERROR: Label column not found.")
        return

    y = df["Label"].astype("string").str.strip()

    # Remove identifiers and separate features from labels.
    X = df.drop(
        columns=[c for c in EXCLUDED_COLUMNS if c in df.columns]
    )

    # Convert feature values to numbers.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    # Remove rows without valid labels.
    valid = y.notna() & y.ne("") & y.ne("nan")
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    # Remove classes with fewer than two records because stratified
    # splitting cannot place them in both train and test sets.
    counts = y.value_counts()
    eligible = y.map(counts) >= 2
    X = X.loc[eligible].reset_index(drop=True)
    y = y.loc[eligible].reset_index(drop=True)

    # Save preprocessing information for future predictions.
    feature_columns = X.columns.tolist()
    medians = X.median().fillna(0)

    # Fill invalid values consistently.
    X = X.fillna(medians).astype("float32")

    print(f"Rows: {len(X):,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Classes: {y.nunique()}")

    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")

    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    start = time.time()
    model.fit(X_train, y_train)
    print(f"Training time: {time.time() - start:.1f} seconds")

    print("\nEvaluating...")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    report = classification_report(
        y_test, predictions, zero_division=0
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification report:\n")
    print(report)

    # Create output folders.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the model and the preprocessing details together.
    model_path = MODEL_DIR / "random_forest.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "medians": medians.to_dict(),
            "excluded_columns": EXCLUDED_COLUMNS,
        },
        model_path,
        compress=3,
    )

    # Save evaluation results.
    report_path = REPORT_DIR / "random_forest_report.txt"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("Random Forest Baseline Evaluation\n\n")
        f.write(f"Dataset: {DATA_FILE.name}\n")
        f.write(f"Training rows: {len(X_train):,}\n")
        f.write(f"Testing rows: {len(X_test):,}\n")
        f.write(f"Features: {len(feature_columns)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced accuracy: {balanced_accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(str(confusion_matrix(y_test, predictions)))

    print("\nSaved successfully:")
    print(f"Model and preprocessing: {model_path}")
    print(f"Evaluation report: {report_path}")


if __name__ == "__main__":
    main()
