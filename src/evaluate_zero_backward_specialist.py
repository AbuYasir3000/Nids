
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEST_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

TRAIN_FILES = [
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

BACKWARD_COL = "Total Backward Packets"
RANDOM_STATE = 42

THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


# ============================================================
# DATA LOADING
# ============================================================

def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find CSV:\n{path}\n\n"
            "Check that your CIC-IDS2017 CSV files are inside data/raw."
        )

    df = None
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"Could not read CSV: {path}")

    df.columns = df.columns.astype(str).str.strip()

    if "Label" not in df.columns:
        raise ValueError(
            f"Label column is missing from {filename}.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df["Label"] = df["Label"].astype(str).str.strip()
    return df


def prepare_features(df):
    identifier_names = {
        "flow id",
        "source ip",
        "destination ip",
        "timestamp",
        "label",
    }

    drop_columns = [
        col for col in df.columns
        if col.strip().lower() in identifier_names
    ]

    X = df.drop(columns=drop_columns, errors="ignore").copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    return X.dropna(axis=1, how="all")


def binary_labels(labels):
    return (
        labels.astype(str)
        .str.strip()
        .str.upper()
        .ne("BENIGN")
        .astype(np.int8)
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    return {
        "rows": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "attack_precision": tp / (tp + fp) if tp + fp else 0.0,
        "attack_recall": tp / (tp + fn) if tp + fn else 0.0,
        "benign_fpr": fp / (fp + tn) if fp + tn else 0.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def print_metrics(name, y_true, y_pred):
    result = calculate_metrics(y_true, y_pred)

    print(f"\n--- {name} ---")
    print(f"Rows:               {result['rows']:,}")
    print(f"Accuracy:           {result['accuracy']:.4%}")
    print(f"Balanced accuracy:  {result['balanced_accuracy']:.4%}")
    print(f"Macro F1:           {result['macro_f1']:.4f}")
    print(f"Attack precision:   {result['attack_precision']:.4%}")
    print(f"Attack recall:      {result['attack_recall']:.4%}")
    print(f"Benign false alarms:{result['benign_fpr']:.4%}")
    print(f"True negatives:     {result['TN']:,}")
    print(f"False positives:    {result['FP']:,}")
    print(f"False negatives:    {result['FN']:,}")
    print(f"True positives:     {result['TP']:,}")

    return result


# ============================================================
# PROBABILITY ANALYSIS
# ============================================================

def print_probability_distribution(name, scores):
    if len(scores) == 0:
        print(f"\n{name}: no rows")
        return

    print(f"\n{name} (n={len(scores):,})")
    print(f"  Mean attack score:   {np.mean(scores):.4f}")
    print(f"  Median attack score: {np.median(scores):.4f}")
    print(f"  90th percentile:     {np.quantile(scores, 0.90):.4f}")
    print(f"  95th percentile:     {np.quantile(scores, 0.95):.4f}")
    print(f"  99th percentile:     {np.quantile(scores, 0.99):.4f}")

    for threshold in THRESHOLDS:
        flagged = scores >= threshold
        print(
            f"  Threshold {threshold:.2f}: "
            f"{flagged.sum():,}/{len(scores):,} flagged "
            f"({flagged.mean():.2%})"
        )


def threshold_table(y_true, scores, group_name):
    rows = []

    for threshold in THRESHOLDS:
        predictions = (scores >= threshold).astype(np.int8)
        result = calculate_metrics(y_true, predictions)

        rows.append({
            "group": group_name,
            "threshold": threshold,
            **result,
        })

    return rows


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    print("=" * 70)
    print("SENTINELAI - ZERO-BACKWARD AND PROBABILITY EXPERIMENT")
    print("Production models will not be changed.")
    print("=" * 70)

    training_frames = []
    training_targets = []

    # Load the seven training captures, excluding the held-out DDoS file.
    for filename in TRAIN_FILES:
        print(f"\nLoading training capture: {filename}")

        df = load_csv(filename)
        X = prepare_features(df)
        y = binary_labels(df["Label"])

        print(f"  Rows: {len(df):,}")
        print(f"  BENIGN: {int((y == 0).sum()):,}")
        print(f"  ATTACK: {int((y == 1).sum()):,}")

        training_frames.append(X)
        training_targets.append(y)

    # Use numeric features shared by all training captures.
    common_features = set(training_frames[0].columns)

    for frame in training_frames[1:]:
        common_features.intersection_update(frame.columns)

    features = sorted(common_features)

    if BACKWARD_COL not in features:
        raise ValueError(
            f"Required feature '{BACKWARD_COL}' was not found."
        )

    X_train_raw = pd.concat(
        [frame[features] for frame in training_frames],
        ignore_index=True,
    )
    y_train = pd.concat(
        training_targets, ignore_index=True
    ).to_numpy()

    del training_frames, training_targets

    print(f"\nTraining rows: {len(X_train_raw):,}")
    print(f"Common numeric features: {len(features)}")

    # Load the held-out DDoS capture.
    print(f"\nLoading held-out capture: {TEST_FILE}")
    test_df = load_csv(TEST_FILE)

    y_test = binary_labels(test_df["Label"]).to_numpy()
    X_test_raw = prepare_features(test_df).reindex(
        columns=features
    )

    # Calculate subgroup membership before imputing missing values.
    train_backward = pd.to_numeric(
        X_train_raw[BACKWARD_COL], errors="coerce"
    )
    test_backward = pd.to_numeric(
        X_test_raw[BACKWARD_COL], errors="coerce"
    )

    train_zero = train_backward.eq(0).fillna(False).to_numpy()
    test_zero = test_backward.eq(0).fillna(False).to_numpy()
    test_nonzero = ~test_zero

    print("\n" + "=" * 70)
    print("HELD-OUT DATA VERIFICATION")
    print("=" * 70)
    print(f"Test rows: {len(y_test):,}")
    print(f"BENIGN rows: {int((y_test == 0).sum()):,}")
    print(f"ATTACK rows: {int((y_test == 1).sum()):,}")
    print(f"Zero-backward rows: {int(test_zero.sum()):,}")
    print(
        "Zero-backward BENIGN:",
        f"{int(((y_test == 0) & test_zero).sum()):,}",
    )
    print(
        "Zero-backward ATTACK:",
        f"{int(((y_test == 1) & test_zero).sum()):,}",
    )

    # Fit imputer only on training data.
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    # --------------------------------------------------------
    # GENERAL BASELINE MODEL
    # --------------------------------------------------------

    print("\nTraining general Random Forest ...")

    general_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    general_model.fit(X_train, y_train)

    attack_column = list(general_model.classes_).index(1)
    attack_scores = general_model.predict_proba(X_test)[:, attack_column]

    baseline_predictions = (attack_scores >= 0.50).astype(np.int8)

    results = []
    results.append(
        {
            "model": "general_baseline_threshold_0.50",
            **print_metrics(
                "GENERAL BASELINE (threshold=0.50)",
                y_test,
                baseline_predictions,
            ),
        }
    )

    # --------------------------------------------------------
    # ATTACK SCORE DISTRIBUTIONS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ATTACK PROBABILITY ANALYSIS")
    print("=" * 70)

    groups = [
        ("Zero-backward ATTACK", test_zero & (y_test == 1)),
        ("Zero-backward BENIGN", test_zero & (y_test == 0)),
        ("Nonzero-backward ATTACK", test_nonzero & (y_test == 1)),
        ("Nonzero-backward BENIGN", test_nonzero & (y_test == 0)),
    ]

    probability_rows = []

    for group_name, mask in groups:
        scores = attack_scores[mask]
        print_probability_distribution(group_name, scores)

        if len(scores):
            probability_rows.append({
                "group": group_name,
                "rows": len(scores),
                "mean_score": float(np.mean(scores)),
                "median_score": float(np.median(scores)),
                "p90_score": float(np.quantile(scores, 0.90)),
                "p95_score": float(np.quantile(scores, 0.95)),
                "p99_score": float(np.quantile(scores, 0.99)),
            })

    # --------------------------------------------------------
    # THRESHOLD COMPARISON
    # --------------------------------------------------------

    threshold_rows = []

    for threshold in THRESHOLDS:
        predictions = (attack_scores >= threshold).astype(np.int8)

        result = calculate_metrics(y_test, predictions)
        threshold_rows.append({
            "group": "full_capture",
            "threshold": threshold,
            **result,
        })

        # Also calculate performance for the two backward-traffic groups.
        for group_name, mask in [
            ("zero_backward", test_zero),
            ("nonzero_backward", test_nonzero),
        ]:
            if mask.any():
                group_result = calculate_metrics(
                    y_test[mask],
                    predictions[mask],
                )
                threshold_rows.append({
                    "group": group_name,
                    "threshold": threshold,
                    **group_result,
                })

    threshold_df = pd.DataFrame(threshold_rows)

    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON - FULL CAPTURE")
    print("=" * 70)

    full_table = threshold_df[
        threshold_df["group"] == "full_capture"
    ]

    print(
        full_table[
            [
                "threshold",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "attack_precision",
                "attack_recall",
                "benign_fpr",
                "FP",
                "FN",
            ]
        ].to_string(index=False, formatters={
            "accuracy": "{:.4%}".format,
            "balanced_accuracy": "{:.4%}".format,
            "macro_f1": "{:.4f}".format,
            "attack_precision": "{:.4%}".format,
            "attack_recall": "{:.4%}".format,
            "benign_fpr": "{:.4%}".format,
        })
    )

    # --------------------------------------------------------
    # ZERO-BACKWARD SPECIALIST COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ZERO-BACKWARD SPECIALIST COMPARISON")
    print("=" * 70)

    X_special_raw = X_train_raw.loc[train_zero]
    y_special = y_train[train_zero]

    print(f"Specialist training rows: {len(y_special):,}")
    print(f"Specialist BENIGN: {int((y_special == 0).sum()):,}")
    print(f"Specialist ATTACK: {int((y_special == 1).sum()):,}")

    specialist_imputer = SimpleImputer(strategy="median")
    X_special = specialist_imputer.fit_transform(X_special_raw)
    X_zero_test = specialist_imputer.transform(
        X_test_raw.loc[test_zero]
    )

    X_special = np.asarray(X_special, dtype=np.float32)
    X_zero_test = np.asarray(X_zero_test, dtype=np.float32)

    specialist_model = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    print("Training specialist model ...")
    specialist_model.fit(X_special, y_special)

    specialist_class_index = list(specialist_model.classes_).index(1)
    specialist_scores = specialist_model.predict_proba(
        X_zero_test
    )[:, specialist_class_index]

    specialist_predictions = (
        specialist_scores >= 0.50
    ).astype(np.int8)

    results.append({
        "model": "zero_backward_specialist_threshold_0.50",
        **print_metrics(
            "SPECIALIST (zero-backward subset, threshold=0.50)",
            y_test[test_zero],
            specialist_predictions,
        ),
    })

    # Hybrid routing result: specialist for zero-backward rows,
    # general model for all other rows.
    hybrid_predictions = baseline_predictions.copy()
    hybrid_predictions[test_zero] = specialist_predictions

    results.append({
        "model": "hybrid_routed_threshold_0.50",
        **print_metrics(
            "HYBRID ROUTING (full capture)",
            y_test,
            hybrid_predictions,
        ),
    })

    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    comparison_path = REPORT_DIR / "probability_model_comparison.csv"
    threshold_path = REPORT_DIR / "probability_threshold_analysis.csv"
    probability_path = REPORT_DIR / "attack_probability_distributions.csv"

    pd.DataFrame(results).to_csv(comparison_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    pd.DataFrame(probability_rows).to_csv(
        probability_path, index=False
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Model comparison: {comparison_path}")
    print(f"Threshold analysis: {threshold_path}")
    print(f"Probability distributions: {probability_path}")
    print("\nProduction models were not modified.")


if __name__ == "__main__":
    main()
