
from pathlib import Path
import sys
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
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------
# PATHS AND SETTINGS
# ---------------------------------------------------------

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

LABEL_COLUMN = " Label"
BACKWARD_COLUMN = " Total Backward Packets"
VARIANCE_COLUMN = " Packet Length Variance"

ID_COLUMNS = [
    "Flow ID",
    " Source IP",
    " Destination IP",
    " Timestamp",
    "Label",
    " Label",
]

RANDOM_STATE = 42

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------


def read_csv_safely(path):
    """Read CIC-IDS2017 CSVs, handling common encoding variations."""
    if not path.exists():
        raise FileNotFoundError(
            f"\nCould not find:\n{path}\n"
            "Check that your CIC-IDS2017 CSV files are inside data/raw."
        )

    last_error = None

    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Could not decode {path}: {last_error}")

    # Remove accidental whitespace from column names.
    df.columns = df.columns.astype(str).str.strip()

    # Normalize label-column name.
    if "Label" not in df.columns:
        candidates = [
            col for col in df.columns
            if col.strip().lower() == "label"
        ]
        if candidates:
            df = df.rename(columns={candidates[0]: "Label"})

    if "Label" not in df.columns:
        raise ValueError(
            f"No Label column found in {path.name}.\n"
            f"Columns found: {df.columns.tolist()[-10:]}"
        )

    # Normalize common feature-column names.
    for col in list(df.columns):
        stripped = col.strip()
        if stripped != col:
            df = df.rename(columns={col: stripped})

    # The previous strip may have changed the label name.
    if "Label" not in df.columns:
        label_matches = [
            col for col in df.columns
            if col.strip().lower() == "label"
        ]
        if label_matches:
            df = df.rename(columns={label_matches[0]: "Label"})

    df["Label"] = df["Label"].astype(str).str.strip()
    return df


def make_binary_labels(labels):
    """BENIGN = 0; every other label = 1."""
    return (
        labels.astype(str)
        .str.strip()
        .str.upper()
        .ne("BENIGN")
        .astype(np.int8)
    )


def prepare_numeric_features(df):
    """Remove identifiers/labels and retain numeric model features."""
    drop_cols = [
        col for col in df.columns
        if col.strip().lower() in {
            "flow id",
            "source ip",
            "destination ip",
            "timestamp",
            "label",
        }
    ]

    X = df.drop(columns=drop_cols, errors="ignore").copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove columns that have no numeric values at all.
    X = X.dropna(axis=1, how="all")

    return X


def load_capture(filename):
    path = DATA_DIR / filename
    df = read_csv_safely(path)

    original_labels = df["Label"].copy()
    y = make_binary_labels(original_labels)
    X = prepare_numeric_features(df)

    return X, y, original_labels


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------


def calculate_metrics(y_true, y_pred):
    matrix = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    attack_precision = tp / (tp + fp) if tp + fp else 0.0
    attack_recall = tp / (tp + fn) if tp + fn else 0.0
    benign_fpr = fp / (fp + tn) if fp + tn else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "attack_precision": attack_precision,
        "attack_recall": attack_recall,
        "benign_fpr": benign_fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_metrics(name, y_true, y_pred):
    metrics = calculate_metrics(y_true, y_pred)

    print(f"\n{'=' * 65}")
    print(name)
    print("=" * 65)
    print(f"Accuracy:          {metrics['accuracy']:.4%}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4%}")
    print(f"Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"Attack precision:  {metrics['attack_precision']:.4%}")
    print(f"Attack recall:     {metrics['attack_recall']:.4%}")
    print(f"Benign false alarm rate: {metrics['benign_fpr']:.4%}")
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("                Predicted BENIGN   Predicted ATTACK")
    print(f"Actual BENIGN   {metrics['tn']:>15,}   {metrics['fp']:>16,}")
    print(f"Actual ATTACK   {metrics['fn']:>15,}   {metrics['tp']:>16,}")

    return metrics


# ---------------------------------------------------------
# LOAD TRAINING CAPTURES
# ---------------------------------------------------------


def main():
    print("=" * 65)
    print("SENTINELAI - HELD-OUT DDOS MODEL EXPERIMENT")
    print("=" * 65)
    print("Production model files will not be changed.")

    training_frames = []
    training_labels = []

    for filename in TRAIN_FILES:
        print(f"\nLoading training capture: {filename}")

        X_file, y_file, labels_file = load_capture(filename)

        print(f"  Rows: {len(X_file):,}")
        print(
            "  BENIGN:",
            f"{int((labels_file.str.upper() == 'BENIGN').sum()):,}",
        )
        print(
            "  ATTACK:",
            f"{int((labels_file.str.upper() != 'BENIGN').sum()):,}",
        )

        training_frames.append(X_file)
        training_labels.append(y_file)

    # Keep only features present in every training capture.
    common_features = set(training_frames[0].columns)
    for frame in training_frames[1:]:
        common_features.intersection_update(frame.columns)

    common_features = sorted(common_features)

    if not common_features:
        raise ValueError("No common numeric features across training files.")

    print(f"\nCommon numeric features: {len(common_features)}")

    X_train_raw = pd.concat(
        [frame[common_features] for frame in training_frames],
        ignore_index=True,
    )
    y_train = pd.concat(training_labels, ignore_index=True)

    del training_frames, training_labels

    # Drop rows with invalid target values, if any.
    valid = y_train.notna()
    X_train_raw = X_train_raw.loc[valid].reset_index(drop=True)
    y_train = y_train.loc[valid].reset_index(drop=True)

    # -----------------------------------------------------
    # LOAD HELD-OUT DDOS CAPTURE
    # -----------------------------------------------------

    print(f"\nLoading held-out capture: {TEST_FILE}")

    X_test_raw, y_test, original_labels = load_capture(TEST_FILE)

    # Use the same features and feature order as the training data.
    X_test_raw = X_test_raw.reindex(columns=common_features)

    # -----------------------------------------------------
    # DEBUG VERIFICATION
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("DEBUG: HELD-OUT CAPTURE VERIFICATION")
    print("=" * 65)

    backward_col = next(
        (c for c in common_features
         if c.strip().lower() == "total backward packets"),
        None,
    )
    variance_col = next(
        (c for c in common_features
         if c.strip().lower() == "packet length variance"),
        None,
    )

    print("Rows loaded:", f"{len(X_test_raw):,}")
    print("Actual BENIGN labels:", f"{int((y_test == 0).sum()):,}")
    print("Actual ATTACK labels:", f"{int((y_test == 1).sum()):,}")
    print(
        "Actual DDoS labels:",
        f"{int(original_labels.str.upper().eq('DDOS').sum()):,}",
    )

    if backward_col:
        backward_values = pd.to_numeric(
            X_test_raw[backward_col], errors="coerce"
        )
        zero_backward = backward_values.eq(0).fillna(False)
        print(
            "Zero-backward rows (BENIGN + ATTACK):",
            f"{int(zero_backward.sum()):,}",
        )
        print(
            "Zero-backward BENIGN rows:",
            f"{int(((y_test == 0) & zero_backward).sum()):,}",
        )
        print(
            "Zero-backward ATTACK rows:",
            f"{int(((y_test == 1) & zero_backward).sum()):,}",
        )
    else:
        raise ValueError(
            f"Feature '{BACKWARD_COLUMN}' not found among common features."
        )

    if variance_col:
        variance_values = pd.to_numeric(
            X_test_raw[variance_col], errors="coerce"
        )
        print(
            "Zero packet-length variance rows:",
            f"{int(variance_values.eq(0).fillna(False).sum()):,}",
        )
    else:
        print("Warning: Packet Length Variance feature is unavailable.")

    print("=" * 65)

    print("\nTraining rows:", f"{len(X_train_raw):,}")
    print("Training BENIGN:", f"{int((y_train == 0).sum()):,}")
    print("Training ATTACK:", f"{int((y_train == 1).sum()):,}")

    # -----------------------------------------------------
    # IMPUTE USING TRAINING DATA ONLY
    # -----------------------------------------------------

    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    # -----------------------------------------------------
    # TRAIN TWO EXPERIMENTAL MODELS
    # -----------------------------------------------------

    experiment_configs = [
        (
            "baseline_balanced_subsample",
            "balanced_subsample",
        ),
        (
            "candidate_balanced",
            "balanced",
        ),
    ]

    result_rows = []
    subgroup_rows = []

    zero_backward_mask = (
        pd.to_numeric(X_test_raw[backward_col], errors="coerce")
        .eq(0)
        .fillna(False)
        .to_numpy()
    )

    nonzero_backward_mask = ~zero_backward_mask

    for experiment_name, class_weight in experiment_configs:
        print(f"\nTraining {experiment_name} ...")

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=class_weight,
            min_samples_leaf=2,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = print_metrics(
            experiment_name, y_test, predictions
        )

        result_rows.append({
            "model": experiment_name,
            **metrics,
        })

        # Evaluate both groups separately.
        for subgroup_name, mask in [
            ("zero_backward", zero_backward_mask),
            ("nonzero_backward", nonzero_backward_mask),
        ]:
            if not mask.any():
                continue

            subgroup_metrics = calculate_metrics(
                y_test.to_numpy()[mask],
                predictions[mask],
            )

            subgroup_rows.append({
                "model": experiment_name,
                "subgroup": subgroup_name,
                "rows": int(mask.sum()),
                **subgroup_metrics,
            })

            true_group = y_test.to_numpy()[mask]
            pred_group = predictions[mask]

            actual_attacks = true_group == 1
            attack_count = int(actual_attacks.sum())
            detected_attacks = int(
                ((pred_group == 1) & actual_attacks).sum()
            )

            print(f"\n  Subgroup: {subgroup_name}")
            print(f"  Total rows: {int(mask.sum()):,}")
            print(f"  Actual attacks: {attack_count:,}")
            print(f"  Detected attacks: {detected_attacks:,}")
            print(
                "  Attack recall:",
                f"{detected_attacks / attack_count:.4%}"
                if attack_count else "N/A",
            )
            print(
                "  Benign false alarm rate:",
                f"{subgroup_metrics['benign_fpr']:.4%}",
            )

    # -----------------------------------------------------
    # SAVE EXPERIMENT REPORTS
    # -----------------------------------------------------

    comparison_path = REPORT_DIR / "ddos_model_comparison.csv"
    subgroup_path = REPORT_DIR / "ddos_subgroup_comparison.csv"

    pd.DataFrame(result_rows).to_csv(
        comparison_path, index=False
    )
    pd.DataFrame(subgroup_rows).to_csv(
        subgroup_path, index=False
    )

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETE")
    print("=" * 65)
    print("Model comparison saved to:", comparison_path)
    print("Subgroup comparison saved to:", subgroup_path)
    print(
        "\nNote: these are experimental models trained in memory. "
        "Your production model files have not been overwritten."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
