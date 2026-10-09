
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cicids2017_cleaned.csv"
)
MODEL_FILE = PROJECT_ROOT / "models" / "isolation_forest.joblib"
RF_FILE = PROJECT_ROOT / "models" / "random_forest.joblib"


def main():
    print("Loading saved models and dataset...")

    if not all(path.exists() for path in [DATA_FILE, MODEL_FILE, RF_FILE]):
        print("Required dataset or model file is missing.")
        return

    saved_if = joblib.load(MODEL_FILE)
    saved_rf = joblib.load(RF_FILE)

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df.columns = df.columns.str.strip()

    labels = df["Label"].astype("string").str.strip()

    excluded = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "Label",
    ]

    X = df.drop(columns=[c for c in excluded if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    X = X.reindex(columns=saved_if["feature_columns"])
    X = X.fillna(pd.Series(saved_if["medians"])).fillna(0)

    valid = labels.notna() & labels.ne("")
    X = X.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].reset_index(drop=True)

    # Reproduce the same deterministic sampling as the baseline.
    rng = np.random.RandomState(42)

    benign_indices = labels[labels == "BENIGN"].index.to_numpy()
    attack_indices = labels[labels != "BENIGN"].index.to_numpy()

    benign_indices = rng.choice(
        benign_indices, size=100_000, replace=False
    )
    attack_indices = rng.choice(
        attack_indices, size=100_000, replace=False
    )

    benign_X = X.iloc[benign_indices].reset_index(drop=True)
    attack_X = X.iloc[attack_indices].reset_index(drop=True)

    benign_train, benign_test = train_test_split(
        benign_X, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    scaler.fit(benign_train)

    benign_test_scaled = scaler.transform(benign_test)
    attack_scaled = scaler.transform(attack_X)

    model = saved_if["model"]

    benign_scores = model.decision_function(benign_test_scaled)
    attack_scores = model.decision_function(attack_scaled)

    print("\nThreshold analysis")
    print("Lower decision scores are more anomalous.\n")
    print(
        f"{'Threshold':>12} {'Benign FPR':>12} "
        f"{'Attack recall':>15} {'Attack precision':>18}"
    )

    # A prediction is anomalous when decision_function < threshold.
    for threshold in [-0.20, -0.10, -0.05, 0.00, 0.02, 0.05, 0.10]:
        false_positives = np.sum(benign_scores < threshold)
        true_positives = np.sum(attack_scores < threshold)

        fpr = false_positives / len(benign_scores)
        recall = true_positives / len(attack_scores)

        predicted_anomalies = false_positives + true_positives
        precision = (
            true_positives / predicted_anomalies
            if predicted_anomalies
            else 0.0
        )

        print(
            f"{threshold:12.2f} {fpr:11.2%} "
            f"{recall:14.2%} {precision:17.2%}"
        )


if __name__ == "__main__":
    main()

