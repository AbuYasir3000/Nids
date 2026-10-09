
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

RANDOM_STATE = 42
BENIGN_SAMPLE_SIZE = 100_000
ATTACK_SAMPLE_SIZE = 100_000


def main():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        print("ERROR: Label column not found.")
        return

    labels = df["Label"].astype("string").str.strip()

    excluded = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "Label",
    ]

    X = df.drop(
        columns=[col for col in excluded if col in df.columns]
    )

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    # Match the same feature columns and order used by Random Forest.
    saved_rf = joblib.load(MODEL_DIR / "random_forest.joblib")
    feature_columns = saved_rf["feature_columns"]
    medians = pd.Series(saved_rf["medians"])

    X = X.reindex(columns=feature_columns)
    X = X.fillna(medians).fillna(0).astype("float32")

    valid_labels = labels.notna() & labels.ne("")
    X = X.loc[valid_labels].reset_index(drop=True)
    labels = labels.loc[valid_labels].reset_index(drop=True)

    benign_indices = labels[labels == "BENIGN"].index
    attack_indices = labels[labels != "BENIGN"].index

    # Sample traffic to keep memory and training time manageable.
    rng = np.random.RandomState(RANDOM_STATE)

    benign_count = min(BENIGN_SAMPLE_SIZE, len(benign_indices))
    attack_count = min(ATTACK_SAMPLE_SIZE, len(attack_indices))

    benign_sample = rng.choice(
        benign_indices.to_numpy(),
        size=benign_count,
        replace=False,
    )
    attack_sample = rng.choice(
        attack_indices.to_numpy(),
        size=attack_count,
        replace=False,
    )

    benign_X = X.iloc[benign_sample].reset_index(drop=True)
    attack_X = X.iloc[attack_sample].reset_index(drop=True)

    # Hold out benign examples for evaluation.
    benign_train, benign_test = train_test_split(
        benign_X,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    # Fit preprocessing using benign training traffic only.
    scaler = StandardScaler()
    benign_train_scaled = scaler.fit_transform(benign_train)
    benign_test_scaled = scaler.transform(benign_test)
    attack_scaled = scaler.transform(attack_X)

    print(f"Benign training rows: {len(benign_train):,}")
    print(f"Benign test rows: {len(benign_test):,}")
    print(f"Attack evaluation rows: {len(attack_X):,}")

    print("\nTraining Isolation Forest...")
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(benign_train_scaled)
    print(f"Training time: {time.time() - start:.1f} seconds")

    # Isolation Forest: 1 = normal, -1 = anomaly.
    benign_predictions = model.predict(benign_test_scaled)
    attack_predictions = model.predict(attack_scaled)

    y_true = np.concatenate([
        np.zeros(len(benign_test), dtype=int),
        np.ones(len(attack_X), dtype=int),
    ])

    y_pred = np.concatenate([
        (benign_predictions == -1).astype(int),
        (attack_predictions == -1).astype(int),
    ])

    print("\nEvaluation (0 = benign, 1 = attack):")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print(f"Attack detection rate: {recall_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "isolation_forest.joblib"
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "medians": medians.to_dict(),
        "threshold": float(model.offset_),
    }, model_path, compress=3)

    report_path = REPORT_DIR / "isolation_forest_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Isolation Forest evaluation\n\n")
        f.write("0 = benign, 1 = attack/anomaly\n\n")
        f.write(classification_report(
            y_true,
            y_pred,
            target_names=["BENIGN", "ATTACK"],
            zero_division=0,
        ))
        f.write("\nConfusion matrix:\n")
        f.write(str(confusion_matrix(y_true, y_pred)))

    print(f"\nModel saved: {model_path}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()

