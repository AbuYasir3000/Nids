from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score
from traffic_features import source_flows_per_minute

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
OUTPUT_PATH = MODEL_DIR / "novel_attack_generalizer.joblib"
REPORT_PATH = REPORT_DIR / "novel_attack_generalizer_training.txt"

PORTSCAN_CAPTURE = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
CALIBRATION_CAPTURE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
CALIBRATION_FPR = 0.001
MAX_ROWS_PER_CLASS_PER_CAPTURE = 20_000
RANDOM_STATE = 42
EXCLUDED_COLUMNS = {"flow id", "source ip", "destination ip", "timestamp", "label"}


def load_capture(filename, include_source_burst_counts=False):
    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required CIC-IDS2017 capture not found: {path}")

    df = pd.read_csv(path, encoding="cp1252", low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    label_column = next(
        (column for column in df.columns if column.strip().lower() == "label"),
        None,
    )
    if label_column is None:
        raise ValueError(f"No Label column found in {filename}")

    source_burst_counts = (
        source_flows_per_minute(df) if include_source_burst_counts else None
    )
    labels = df[label_column].astype("string").str.strip().str.upper()
    valid = labels.notna() & labels.ne("") & labels.ne("NAN")
    y = labels.loc[valid].ne("BENIGN").astype("int8").reset_index(drop=True)
    feature_columns = [
        column for column in df.columns
        if column.strip().lower() not in EXCLUDED_COLUMNS
    ]
    X = df.loc[valid, feature_columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all").reset_index(drop=True)
    if include_source_burst_counts:
        return X, y, source_burst_counts[valid.to_numpy()]
    return X, y


def sample_capture(X, y, rng):
    parts = []
    targets = []
    for label in (0, 1):
        indices = np.flatnonzero(y.to_numpy() == label)
        count = min(MAX_ROWS_PER_CLASS_PER_CAPTURE, len(indices))
        if count:
            selected = rng.choice(indices, size=count, replace=False)
            parts.append(X.iloc[selected])
            targets.append(y.iloc[selected])
    return (
        pd.concat(parts, ignore_index=True, sort=False),
        pd.concat(targets, ignore_index=True),
    )


def main():
    start = time.time()
    training_paths = sorted(
        path for path in DATA_DIR.glob("*pcap_ISCX.csv")
        if path.name not in {PORTSCAN_CAPTURE, CALIBRATION_CAPTURE}
    )
    if not training_paths:
        raise FileNotFoundError(f"No training captures found in {DATA_DIR}")

    rng = np.random.default_rng(RANDOM_STATE)
    samples = []
    targets = []
    capture_counts = {}
    for path in training_paths:
        print(f"Reading training capture: {path.name}", flush=True)
        X, y = load_capture(path.name)
        X, y = sample_capture(X, y, rng)
        samples.append(X)
        targets.append(y)
        capture_counts[path.name] = {
            "rows": int(len(y)),
            "benign": int((y == 0).sum()),
            "attacks": int((y == 1).sum()),
        }
        print(
            f"  sampled {len(y):,} rows; {int(y.sum()):,} attacks",
            flush=True,
        )

    X_train = pd.concat(samples, ignore_index=True, sort=False)
    y_train = pd.concat(targets, ignore_index=True).astype("int8")
    feature_columns = sorted(X_train.columns.astype(str))
    X_train = X_train.reindex(columns=feature_columns)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train).astype("float32")

    model = HistGradientBoostingClassifier(
        max_iter=150,
        max_leaf_nodes=31,
        learning_rate=0.1,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    print(f"Training generalized attack classifier on {X_train.shape}...", flush=True)
    model.fit(X_train, y_train)

    calibration_X, calibration_y, calibration_bursts = load_capture(
        CALIBRATION_CAPTURE, include_source_burst_counts=True
    )
    calibration_X = calibration_X.reindex(columns=feature_columns)
    calibration_X = imputer.transform(calibration_X).astype("float32")
    calibration_scores = model.predict_proba(calibration_X)[:, 1]
    benign_scores = calibration_scores[calibration_y.to_numpy() == 0]
    attack_scores = calibration_scores[calibration_y.to_numpy() == 1]
    if not len(benign_scores) or not len(attack_scores):
        raise ValueError("The calibration capture must contain benign and attack rows.")

    # A threshold is selected from benign calibration traffic only.
    threshold = float(
        np.quantile(benign_scores, 1 - CALIBRATION_FPR, method="higher")
    )
    calibration_flags = calibration_scores > threshold
    observed_fpr = float(calibration_flags[calibration_y.to_numpy() == 0].mean())
    calibration_attack_recall = float(
        recall_score(
            calibration_y.to_numpy()[calibration_y.to_numpy() == 1],
            calibration_flags[calibration_y.to_numpy() == 1],
            zero_division=0,
        )
    )

    benign_mask = calibration_y.to_numpy() == 0
    source_burst_threshold = int(
        np.quantile(
            calibration_bursts[benign_mask],
            1 - CALIBRATION_FPR,
            method="higher",
        )
    )
    source_burst_flags = calibration_bursts > source_burst_threshold
    source_burst_calibration_fpr = float(
        source_burst_flags[benign_mask].mean()
    )
    source_burst_attack_recall = float(
        source_burst_flags[~benign_mask].mean()
    )
    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "imputer": imputer,
        "decision_threshold": threshold,
        "source_burst_threshold": source_burst_threshold,
        "source_burst_calibration_fpr": source_burst_calibration_fpr,
        "source_burst_attack_recall": source_burst_attack_recall,
        "calibration_target_fpr": CALIBRATION_FPR,
        "calibration_observed_fpr": observed_fpr,
        "calibration_capture": CALIBRATION_CAPTURE,
        "training_captures": [path.name for path in training_paths],
        "training_rows": int(len(y_train)),
        "training_capture_counts": capture_counts,
        "random_state": RANDOM_STATE,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, OUTPUT_PATH, compress=3)

    report = [
        "Cross-capture generalized attack model training",
        "=================================================",
        f"Model: HistGradientBoostingClassifier",
        f"Training rows: {len(y_train):,}",
        f"Training features: {len(feature_columns)}",
        f"Training captures: {', '.join(path.name for path in training_paths)}",
        f"PortScan capture excluded from training and calibration: {PORTSCAN_CAPTURE}",
        f"Calibration capture: {CALIBRATION_CAPTURE}",
        f"Target benign false-positive rate: {CALIBRATION_FPR:.3%}",
        f"Calibrated decision threshold: {threshold:.10f}",
        f"Observed calibration benign false-positive rate: {observed_fpr:.4%}",
        f"Source-flow burst threshold (flows/source/minute): {source_burst_threshold}",
        f"Source-flow burst calibration false-positive rate: {source_burst_calibration_fpr:.4%}",
        f"Source-flow burst DDoS-capture attack recall (diagnostic): {source_burst_attack_recall:.4%}",
        f"Calibration-capture attack recall (diagnostic): {calibration_attack_recall:.4%}",
        "The threshold is selected using benign rows only.",
        "PortScan must be treated as an exploratory benchmark after model selection.",
        f"Model artifact: {OUTPUT_PATH}",
        f"Elapsed seconds: {time.time() - start:.1f}",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
