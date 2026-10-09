
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

RF_PATH = MODEL_DIR / "random_forest.joblib"
IF_PATH = MODEL_DIR / "isolation_forest.joblib"


class HybridNIDS:
    def __init__(self):
        if not RF_PATH.exists():
            raise FileNotFoundError(f"Random Forest not found: {RF_PATH}")

        if not IF_PATH.exists():
            raise FileNotFoundError(f"Isolation Forest not found: {IF_PATH}")

        print("Loading Random Forest...")
        rf_data = joblib.load(RF_PATH)
        self.rf_model = rf_data["model"]
        self.feature_columns = rf_data["feature_columns"]
        self.medians = pd.Series(rf_data["medians"])

        print("Loading Isolation Forest...")
        if_data = joblib.load(IF_PATH)
        self.if_model = if_data["model"]
        self.if_scaler = if_data["scaler"]

        print("Hybrid NIDS loaded successfully.")

    def prepare_features(self, data):
        """Align incoming numeric flow data to the model's features."""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            data = data.to_frame().T
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a dict, Series, or DataFrame.")

        data = data.copy()
        data.columns = data.columns.str.strip()

        excluded = [
            "Flow ID",
            "Source IP",
            "Destination IP",
            "Timestamp",
            "Label",
        ]

        data = data.drop(
            columns=[c for c in excluded if c in data.columns],
            errors="ignore",
        )

        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.reindex(columns=self.feature_columns)
        data = data.fillna(self.medians).fillna(0)

        return data.astype("float32")

    def predict(self, data):
        """Return the known-class prediction and anomaly flag."""
        X = self.prepare_features(data)

        rf_predictions = self.rf_model.predict(X)

        # Use the scaler saved with Isolation Forest.
        X_scaled = self.if_scaler.transform(X)
        if_predictions = self.if_model.predict(X_scaled)
        anomaly_scores = self.if_model.decision_function(X_scaled)

        results = pd.DataFrame({
            "predicted_class": rf_predictions,
            "is_anomaly": if_predictions == -1,
            "anomaly_score": anomaly_scores,
        })

        results["hybrid_status"] = np.where(
            results["is_anomaly"],
            "Suspicious - anomaly detected",
            "No anomaly detected",
        )

        return results


def main():
    detector = HybridNIDS()

    # Demonstrate loading and inference using real dataset rows.
    data_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "cicids2017_cleaned.csv"
    )

    print("\nLoading 5 sample flows...")
    sample = pd.read_csv(data_file, nrows=5, low_memory=False)

    # Keep labels separately for demonstration only.
    actual_labels = sample["Label"].astype(str).tolist()

    results = detector.predict(sample)

    results.insert(0, "actual_class_for_demo", actual_labels)

    print("\nHybrid detection results:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
