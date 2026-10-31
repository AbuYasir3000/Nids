from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .traffic_features import (
        high_rate_multiport_flags, source_flows_per_minute, target_flood_analysis,
    )
except ImportError:
    from traffic_features import (
        high_rate_multiport_flags, source_flows_per_minute, target_flood_analysis,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

# Prefer the newest retrained Random Forest when present: v3 (adds UDPLag), then
# v2 (adds a scan capture), then the original. Delete or rename the v3/v2 files
# to fall back to an older model.
def _pick_rf_path():
    for name in ("random_forest_v3.joblib", "random_forest_v2.joblib"):
        if (MODEL_DIR / name).exists():
            return MODEL_DIR / name
    return MODEL_DIR / "random_forest.joblib"


RF_PATH = _pick_rf_path()
IF_PATH = MODEL_DIR / "isolation_forest.joblib"
GENERALIZED_PATH = MODEL_DIR / "novel_attack_generalizer.joblib"


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
        self._validate_artifact_schema("Random Forest", self.feature_columns)
        self._validate_feature_count("Random Forest", self.rf_model, self.feature_columns)

        print("Loading Isolation Forest...")
        if_data = joblib.load(IF_PATH)
        self.if_model = if_data["model"]
        self.if_scaler = if_data["scaler"]
        self.if_feature_columns = list(if_data["feature_columns"])
        self.if_medians = pd.Series(if_data["medians"])
        self._validate_artifact_schema("Isolation Forest", self.if_feature_columns)
        self._validate_feature_count("Isolation Forest scaler", self.if_scaler, self.if_feature_columns)
        self._validate_feature_count("Isolation Forest", self.if_model, self.if_feature_columns)
        if self.if_feature_columns != list(self.feature_columns):
            raise ValueError(
                "Isolation Forest feature order differs from Random Forest; "
                "separate feature alignment is required."
            )

        self.generalized_model = None
        self.generalized_imputer = None
        self.generalized_feature_columns = []
        self.generalized_threshold = None
        self.source_burst_threshold = None
        if GENERALIZED_PATH.exists():
            print("Loading cross-capture attack model...")
            generalized_data = joblib.load(GENERALIZED_PATH)
            self.generalized_model = generalized_data["model"]
            self.generalized_imputer = generalized_data["imputer"]
            self.generalized_feature_columns = list(generalized_data["feature_columns"])
            self._validate_artifact_schema(
                "Learned binary classifier", self.generalized_feature_columns
            )
            self._validate_feature_count(
                "Learned binary classifier", self.generalized_model,
                self.generalized_feature_columns,
            )
            self._validate_feature_count(
                "Learned binary classifier imputer", self.generalized_imputer,
                self.generalized_feature_columns,
            )
            self.generalized_threshold = float(
                generalized_data["decision_threshold"]
            )
            burst_threshold = generalized_data.get("source_burst_threshold")
            self.source_burst_threshold = (
                int(burst_threshold) if burst_threshold is not None else None
            )

        print("Hybrid NIDS loaded successfully.")

    @property
    def generalized_model_ready(self):
        return (
            self.generalized_model is not None
            and self.generalized_threshold is not None
        )

    @staticmethod
    def _as_frame(data):
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            data = data.to_frame().T
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a dict, Series, or DataFrame.")
        data = data.copy()
        data.columns = data.columns.astype(str).str.strip()
        if data.columns.duplicated().any():
            duplicates = data.columns[data.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate input feature columns: {duplicates}")
        return data

    @staticmethod
    def _validate_artifact_schema(name, feature_columns):
        normalized = [str(column).strip() for column in feature_columns]
        if normalized != list(feature_columns) or len(normalized) != len(set(normalized)):
            raise ValueError(f"{name} artifact has invalid or duplicate feature names.")
        forbidden = {
            "label", "flow id", "source ip", "destination ip", "timestamp",
        }
        leaked = [column for column in normalized if column.casefold() in forbidden]
        if leaked:
            raise ValueError(f"{name} artifact includes target/identifier features: {leaked}")

    @staticmethod
    def _positive_class_probability(model, features):
        """Return probability for the learned ATTACK class (label 1).

        Never assume that the positive class is in probability column 1: the
        order is defined by ``model.classes_`` and can differ across estimators.
        """
        if not hasattr(model, "predict_proba"):
            raise TypeError("Learned binary classifier must implement predict_proba().")
        classes = np.asarray(getattr(model, "classes_", []))
        positive_indices = [
            index for index, label in enumerate(classes)
            if str(label).strip().casefold() in {"1", "attack", "true"}
        ]
        if len(positive_indices) != 1:
            raise ValueError(
                "Cannot identify the learned ATTACK class in classifier classes_ "
                f"{classes.tolist()}. Expected exactly one class labeled 1/ATTACK."
            )
        probabilities = np.asarray(model.predict_proba(features))
        if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
            raise ValueError(
                "Classifier probability output does not match its classes_ schema."
            )
        return probabilities[:, positive_indices[0]]

    @staticmethod
    def _validate_feature_count(name, component, feature_columns):
        count = getattr(component, "n_features_in_", len(feature_columns))
        if int(count) != len(feature_columns):
            raise ValueError(
                f"{name} expects {count} features but its artifact schema lists "
                f"{len(feature_columns)}."
            )

    @staticmethod
    def _prepare_named_features(data, feature_columns, medians, model_name):
        missing = [column for column in feature_columns if column not in data.columns]
        if missing:
            raise ValueError(
                f"{model_name} input is missing required features: {missing}"
            )
        raw = data.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
        raw = raw.replace([np.inf, -np.inf], np.nan)
        all_missing = raw.columns[raw.isna().all()].tolist()
        if all_missing:
            raise ValueError(
                f"{model_name} input features contain no usable numeric values: "
                f"{all_missing}"
            )
        saved_medians = pd.Series(medians).reindex(feature_columns)
        invalid_medians = saved_medians.index[
            ~np.isfinite(saved_medians.to_numpy(dtype=float))
        ].tolist()
        if invalid_medians:
            raise ValueError(
                f"{model_name} artifact has no finite training medians for: "
                f"{invalid_medians}"
            )
        imputed_counts = raw.isna().sum(axis=1).to_numpy()
        return raw.fillna(saved_medians).astype("float32"), imputed_counts

    def prepare_features(self, data):
        """Align incoming numeric flow data to the Random Forest schema."""
        data = self._as_frame(data)
        features, _ = self._prepare_named_features(
            data, self.feature_columns, self.medians, "Random Forest"
        )
        return features

    def predict(self, data):
        """Return known-class, anomaly, and optional generalized attack signals."""
        original = self._as_frame(data)
        X, imputed_feature_count = self._prepare_named_features(
            original, self.feature_columns, self.medians, "Random Forest"
        )

        rf_predictions = self.rf_model.predict(X)
        if_features, _ = self._prepare_named_features(
            original, self.if_feature_columns, self.if_medians, "Isolation Forest"
        )
        X_scaled = self.if_scaler.transform(if_features)
        if_predictions = self.if_model.predict(X_scaled)
        anomaly_scores = self.if_model.decision_function(X_scaled)

        is_anomaly = if_predictions == -1
        source_counts = source_flows_per_minute(original)
        results = pd.DataFrame({
            "predicted_class": rf_predictions,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_scores,
            "imputed_feature_count": imputed_feature_count,
        })

        # Keep flow context alongside model results. Row identifiers are stable
        # within this capture and avoid inventing network identifiers.
        metadata_aliases = {
            "flow_id": ("Flow ID", "flow_id"),
            "source_ip": ("Source IP", "src_ip"),
            "destination_ip": ("Destination IP", "dst_ip"),
            "source_port": ("Source Port", "src_port"),
            "destination_port": ("Destination Port", "dst_port"),
            "protocol": ("Protocol", "protocol"),
            "timestamp": ("Timestamp", "timestamp"),
        }
        results["flow_identifier"] = [f"flow-{index + 1:06d}" for index in range(len(original))]
        for output_column, candidates in metadata_aliases.items():
            source_column = next((name for name in candidates if name in original.columns), None)
            if source_column is not None:
                results[output_column] = original[source_column].reset_index(drop=True)
        if "flow_id" in results.columns:
            results["flow_identifier"] = results["flow_id"].fillna(results["flow_identifier"])

        behavior = _source_minute_indicators(original)
        results["unique_destination_ports_per_source_minute"] = behavior["ports"]
        results["unique_destination_ips_per_source_minute"] = behavior["destinations"]
        if self.source_burst_threshold is not None:
            is_high_rate_multiport = high_rate_multiport_flags(
                source_counts, behavior["ports"], self.source_burst_threshold
            )
        else:
            is_high_rate_multiport = np.zeros(len(X), dtype=bool)
        results["is_high_rate_multiport_activity"] = is_high_rate_multiport
        is_target_flood, _ = target_flood_analysis(original)
        results["is_target_flood_activity"] = is_target_flood

        if self.generalized_model is not None:
            general_features, _ = self._prepare_named_features(
                original,
                self.generalized_feature_columns,
                pd.Series(self.generalized_imputer.statistics_, index=self.generalized_feature_columns),
                "Learned binary classifier",
            )
            general_features = self.generalized_imputer.transform(
                general_features
            ).astype("float32")
            attack_probability = self._positive_class_probability(
                self.generalized_model, general_features
            )
            is_model_generalized_attack = (
                attack_probability > self.generalized_threshold
            )
            is_source_burst_alert = (
                source_counts > self.source_burst_threshold
                if self.source_burst_threshold is not None
                else np.zeros(len(X), dtype=bool)
            )
            # Keep a learned attack prediction separate from the source-rate
            # heuristic. A busy source is only a behavioral signal; it is not
            # enough to label every flow in its time window as an attack.
            is_generalized_attack = is_model_generalized_attack
        else:
            attack_probability = np.full(len(X), np.nan, dtype="float32")
            is_model_generalized_attack = np.zeros(len(X), dtype=bool)
            is_source_burst_alert = np.zeros(len(X), dtype=bool)
            is_generalized_attack = np.zeros(len(X), dtype=bool)

        known_attack = (
            pd.Series(rf_predictions).astype(str).str.strip().str.upper()
            .ne("BENIGN").to_numpy()
        )
        results["generalized_attack_probability"] = attack_probability
        results["binary_classifier_score"] = attack_probability
        results["binary_classifier_result"] = np.select(
            [is_model_generalized_attack], ["ATTACK"], default="BENIGN"
        ) if self.generalized_model is not None else np.full(len(X), "UNAVAILABLE", dtype=object)
        results["is_binary_classifier_attack"] = is_model_generalized_attack
        results["source_flows_per_minute"] = source_counts
        results["is_model_generalized_attack"] = is_model_generalized_attack
        results["is_source_burst_alert"] = is_source_burst_alert
        results["is_generalized_attack"] = is_generalized_attack
        results["behavioral_indicators"] = [
            "; ".join(
                indicator for indicator, condition in (
                    ("high destination-port diversity", is_high_rate_multiport[index]),
                    ("high source flow rate", is_source_burst_alert[index]),
                    ("flood of unanswered flows to one target", is_target_flood[index]),
                    ("multiple destination ports in source-minute window", behavior["ports"][index] > 1),
                    ("multiple destination IPs in source-minute window", behavior["destinations"][index] > 1),
                ) if condition
            ) or "none"
            for index in range(len(X))
        ]
        results["decision_label"] = np.select(
            [known_attack, is_model_generalized_attack, is_high_rate_multiport, is_source_burst_alert, is_target_flood, is_anomaly],
            [
                np.array([f"Known attack: {name}" for name in rf_predictions], dtype=object),
                "Suspicious - binary attack signal; family unknown",
                "Behavioral signal - high-rate multiport activity",
                "Behavioral signal - high source flow rate",
                "Behavioral signal - flood of unanswered flows to one target",
                "Anomalous - attack type unknown",
            ],
            default="Insufficient evidence",
        )
        results["decision_explanation"] = np.select(
            [known_attack, is_model_generalized_attack, is_high_rate_multiport, is_source_burst_alert, is_target_flood, is_anomaly],
            [
                "Random Forest predicted a supported non-benign class.",
                "The calibrated binary classifier exceeded its saved decision threshold.",
                "Source rate and destination-port diversity exceeded a CIC-calibrated heuristic; not a model prediction.",
                "Source flows per minute exceeded a CIC-calibrated heuristic; not a model prediction.",
                "Over 1,000 flows reached one target in a minute and 90% or more got no reply (uncalibrated development rule; not a model prediction).",
                "Isolation Forest flagged this flow; no attack family is inferred.",
            ],
            default="No configured model or calibrated rule flagged this flow.",
        )
        results["severity"] = np.select(
            [known_attack | is_model_generalized_attack, is_source_burst_alert | is_target_flood, is_anomaly],
            ["high", "medium", "low"],
            default="none",
        )
        results["hybrid_status"] = results["decision_label"]
        return results


def _source_minute_indicators(data):
    """Return observable port/IP diversity for each source-minute group."""
    n = len(data)
    ports = np.ones(n, dtype="int64")
    destinations = np.ones(n, dtype="int64")
    normalized = {str(column).strip().lower(): column for column in data.columns}
    source_col = normalized.get("source ip") or normalized.get("src_ip")
    timestamp_col = normalized.get("timestamp")
    port_col = normalized.get("destination port") or normalized.get("dst_port")
    destination_col = normalized.get("destination ip") or normalized.get("dst_ip")
    if source_col is None or timestamp_col is None or n == 0:
        return {"ports": ports, "destinations": destinations}
    keys = pd.DataFrame(index=data.index)
    keys["source"] = data[source_col].astype("string").str.strip()
    keys["window"] = pd.to_datetime(data[timestamp_col], errors="coerce").dt.floor("min")
    valid = keys["source"].notna() & keys["source"].ne("") & keys["window"].notna()
    if valid.any():
        groupers = [keys.loc[valid, "source"], keys.loc[valid, "window"]]
        positions = np.flatnonzero(valid.to_numpy())
        if port_col is not None:
            counts = data.loc[valid].assign(_source=groupers[0].to_numpy(), _window=groupers[1].to_numpy()).groupby(["_source", "_window"])[port_col].transform("nunique").fillna(0).to_numpy()
            ports[positions] = counts.astype("int64")
        if destination_col is not None:
            counts = data.loc[valid].assign(_source=groupers[0].to_numpy(), _window=groupers[1].to_numpy()).groupby(["_source", "_window"])[destination_col].transform("nunique").fillna(0).to_numpy()
            destinations[positions] = counts.astype("int64")
    return {"ports": ports, "destinations": destinations}


def main():
    detector = HybridNIDS()
    data_file = PROJECT_ROOT / "data" / "processed" / "cicids2017_cleaned.csv"
    print("\nLoading 5 sample flows...")
    sample = pd.read_csv(data_file, nrows=5, low_memory=False)
    actual_labels = sample["Label"].astype(str).tolist()
    results = detector.predict(sample)
    results.insert(0, "actual_class_for_demo", actual_labels)
    print("\nHybrid detection results:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
