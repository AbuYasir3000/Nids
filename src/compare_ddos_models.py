
from pathlib import Path
import gc
import time

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
from sklearn.model_selection import train_test_split


# ============================================================
# SENTINELAI: BASELINE VS DDoS-AUGMENTED MODEL
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 100

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_COLUMNS = {
    "flow id",
    "source ip",
    "destination ip",
    "timestamp",
    "label",
}

DDOS_FILENAME_PART = "friday-workinghours-afternoon-ddos"


def find_capture_files():
    """Find the original CIC-IDS2017 capture CSVs."""
    files = [
        p for p in DATA_ROOT.rglob("*.csv")
        if "pcap_iscx.csv" in p.name.lower()
    ]

    ddos_files = [
        p for p in files
        if DDOS_FILENAME_PART in p.name.lower()
    ]

    if len(ddos_files) != 1:
        raise FileNotFoundError(
            "Could not uniquely find the original DDoS capture CSV.\n"
            f"Found {len(ddos_files)} matching files.\n"
            f"Search location: {DATA_ROOT}\n"
            "Check that the original CIC-IDS2017 CSVs are inside data/."
        )

    ddos_file = ddos_files[0]
    other_files = [p for p in files if p != ddos_file]

    if len(other_files) != 7:
        raise RuntimeError(
            f"Expected 7 other CIC-IDS2017 captures, found "
            f"{len(other_files)}.\n"
            "Check the files inside your data directory."
        )

    return sorted(other_files), ddos_file


def read_capture(path):
    """Read one capture and normalize its labels and feature names."""
    print(f"Reading: {path.name}")

    try:
        df = pd.read_csv(path, encoding="cp1252", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)

    # CIC-IDS2017 CSVs can contain spaces in column names.
    df.columns = df.columns.astype(str).str.strip()

    label_column = next(
        (c for c in df.columns if c.lower() == "label"),
        None,
    )

    if label_column is None:
        raise ValueError(f"No Label column found in {path.name}")

    labels = df[label_column].astype(str).str.strip()

    # Binary classification: BENIGN = 0, every attack = 1.
    y = (labels.str.upper() != "BENIGN").astype("int8")

    excluded = {
        c for c in df.columns
        if c.strip().lower() in EXCLUDED_COLUMNS
    }

    feature_df = df.drop(
        columns=list(excluded | {label_column}),
        errors="ignore",
    )

    # Force feature columns to numeric. Invalid values become NaN.
    for column in feature_df.columns:
        feature_df[column] = pd.to_numeric(
            feature_df[column], errors="coerce"
        )

    feature_df = feature_df.replace(
        [np.inf, -np.inf], np.nan
    )

    # Remove columns that contain no usable numeric values.
    feature_df = feature_df.dropna(axis=1, how="all")

    # float32 reduces memory use compared with float64.
    feature_df = feature_df.astype("float32")

    print(
        f"  Rows: {len(feature_df):,} | "
        f"Benign: {(y == 0).sum():,} | "
        f"Attack: {(y == 1).sum():,} | "
        f"Numeric features: {feature_df.shape[1]}"
    )

    return feature_df, y


def align_features(frame, feature_columns):
    """Align feature columns in a consistent order."""
    frame = frame.reindex(columns=feature_columns)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.astype("float32")


def report_model(name, model, imputer, X_test, y_test, groups):
    """Evaluate the model overall and on traffic subgroups."""
    probabilities = model.predict_proba(
        imputer.transform(X_test)
    )[:, 1]

    # Use the standard 0.50 threshold for the main comparison.
    predictions = (probabilities >= 0.50).astype("int8")

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    results = {
        "model": name,
        "accuracy_pct": accuracy_score(y_test, predictions) * 100,
        "balanced_accuracy_pct": (
            balanced_accuracy_score(y_test, predictions) * 100
        ),
        "macro_f1_pct": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ) * 100,
        "attack_precision_pct": precision_score(
            y_test, predictions, zero_division=0
        ) * 100,
        "attack_recall_pct": recall_score(
            y_test, predictions, zero_division=0
        ) * 100,
        "benign_fpr_pct": (
            fp / (tn + fp) * 100 if (tn + fp) else 0
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)
    print(f"Accuracy:           {results['accuracy_pct']:.4f}%")
    print(
        f"Balanced accuracy:  "
        f"{results['balanced_accuracy_pct']:.4f}%"
    )
    print(f"Macro F1:            {results['macro_f1_pct']:.4f}%")
    print(
        f"Attack precision:   "
        f"{results['attack_precision_pct']:.4f}%"
    )
    print(
        f"Attack recall:      "
        f"{results['attack_recall_pct']:.4f}%"
    )
    print(
        f"Benign false alarm: "
        f"{results['benign_fpr_pct']:.4f}%"
    )
    print(f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["BENIGN", "ATTACK"],
            digits=4,
            zero_division=0,
        )
    )

    # Groups are based on the original, un-imputed test features.
    subgroup_results = []

    for group_name, mask in groups.items():
        if mask.sum() == 0:
            continue

        group_y = y_test.loc[mask]
        group_pred = pd.Series(
            predictions, index=y_test.index
        ).loc[mask]

        attacks = group_y == 1
        benign = group_y == 0

        attack_recall = (
            (group_pred[attacks] == 1).mean() * 100
            if attacks.any() else np.nan
        )
        benign_fpr = (
            (group_pred[benign] == 1).mean() * 100
            if benign.any() else np.nan
        )

        subgroup_results.append({
            "model": name,
            "subgroup": group_name,
            "rows": int(mask.sum()),
            "actual_attacks": int(attacks.sum()),
            "actual_benign": int(benign.sum()),
            "attack_recall_pct": attack_recall,
            "benign_fpr_pct": benign_fpr,
        })

        print(
            f"{group_name}: rows={mask.sum():,}, "
            f"attack recall={attack_recall:.4f}%, "
            f"benign FPR={benign_fpr:.4f}%"
        )

    return results, subgroup_results


def main():
    started = time.time()

    print("=" * 65)
    print("SENTINELAI DDoS MODEL COMPARISON")
    print("=" * 65)

    other_paths, ddos_path = find_capture_files()

    print("\nTraining captures:")
    for path in other_paths:
        print(f"  {path.name}")

    print(f"\nDDoS capture: {ddos_path.name}")

    # --------------------------------------------------------
    # 1. Read the DDoS capture and make a stratified split.
    # --------------------------------------------------------
    X_ddos_all, y_ddos_all = read_capture(ddos_path)

    all_indices = np.arange(len(y_ddos_all))

    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_ddos_all,
    )

    X_ddos_train = X_ddos_all.iloc[train_idx].copy()
    y_ddos_train = y_ddos_all.iloc[train_idx].copy()

    X_test = X_ddos_all.iloc[test_idx].copy()
    y_test = y_ddos_all.iloc[test_idx].copy()

    # Detect flows with no backward traffic using the original
    # feature values before imputation.
    backward_columns = [
        c for c in X_test.columns
        if c.strip().lower() in {
            "total backward packets",
            "total length of bwd packets",
        }
    ]

    if not backward_columns:
        raise ValueError(
            "Could not find Total Backward Packets or "
            "Total Length of Bwd Packets in the DDoS CSV."
        )

    backward_values = X_test[backward_columns].fillna(0)
    zero_backward_mask = backward_values.eq(0).all(axis=1)
    nonzero_backward_mask = ~zero_backward_mask

    groups = {
        "zero_backward": zero_backward_mask,
        "nonzero_backward": nonzero_backward_mask,
    }

    print("\nDDoS split:")
    print(f"  DDoS training rows: {len(X_ddos_train):,}")
    print(f"  DDoS test rows:     {len(X_test):,}")
    print(f"  Test attacks:       {(y_test == 1).sum():,}")
    print(f"  Test benign:        {(y_test == 0).sum():,}")
    print(
        f"  Zero-backward test rows: "
        f"{zero_backward_mask.sum():,}"
    )
    print(
        f"  Nonzero-backward test rows: "
        f"{nonzero_backward_mask.sum():,}"
    )

    del X_ddos_all, y_ddos_all, all_indices, train_idx, test_idx
    gc.collect()

    # --------------------------------------------------------
    # 2. Load the other seven captures.
    # --------------------------------------------------------
    train_frames = []
    train_labels = []

    for path in other_paths:
        X_part, y_part = read_capture(path)
        train_frames.append(X_part)
        train_labels.append(y_part)

    # Use only features present in every training capture and
    # in the DDoS capture. This avoids mismatched feature sets.
    common_features = set(X_ddos_train.columns)

    for frame in train_frames:
        common_features.intersection_update(frame.columns)

    feature_columns = sorted(common_features)

    if len(feature_columns) < 20:
        raise RuntimeError(
            f"Only {len(feature_columns)} common features found. "
            "Check the source CSV columns."
        )

    print(f"\nCommon numeric features: {len(feature_columns)}")

    X_test = align_features(X_test, feature_columns)
    X_ddos_train = align_features(X_ddos_train, feature_columns)

    X_other = pd.concat(
        [
            align_features(frame, feature_columns)
            for frame in train_frames
        ],
        ignore_index=True,
    )

    y_other = pd.concat(train_labels, ignore_index=True)

    # Free intermediate frames to reduce memory pressure.
    del train_frames, train_labels
    gc.collect()

    print(f"Other-capture training rows: {len(X_other):,}")

    # --------------------------------------------------------
    # 3. Prepare the two training sets.
    # --------------------------------------------------------
    # Baseline: seven non-DDoS captures only.
    X_baseline_train = X_other
    y_baseline_train = y_other

    # Candidate: the same seven captures + DDoS training split.
    X_candidate_train = pd.concat(
        [X_other, X_ddos_train],
        ignore_index=True,
    )
    y_candidate_train = pd.concat(
        [y_other, y_ddos_train],
        ignore_index=True,
    )

    print(
        f"Baseline training rows:  {len(X_baseline_train):,}"
    )
    print(
        f"Candidate training rows: {len(X_candidate_train):,}"
    )

    # Fit imputation using training data only.
    # Both models use the same feature order and test set.
    print("\nFitting baseline imputer...")
    baseline_imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    X_baseline_ready = baseline_imputer.fit_transform(
        X_baseline_train
    )

    print("Training baseline Random Forest...")
    baseline = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    baseline_start = time.time()
    baseline.fit(X_baseline_ready, y_baseline_train)
    print(f"Baseline training time: {time.time() - baseline_start:.1f}s")

    del X_baseline_ready
    gc.collect()

    # Candidate imputer is fitted on the larger candidate training set.
    print("\nFitting candidate imputer...")
    candidate_imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    X_candidate_ready = candidate_imputer.fit_transform(
        X_candidate_train
    )

    print("Training candidate Random Forest...")
    candidate = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    candidate_start = time.time()
    candidate.fit(X_candidate_ready, y_candidate_train)
    print(
        f"Candidate training time: "
        f"{time.time() - candidate_start:.1f}s"
    )

    del X_candidate_ready
    del X_baseline_train, y_baseline_train
    del X_candidate_train, y_candidate_train
    del X_other, y_other, X_ddos_train, y_ddos_train
    gc.collect()

    # --------------------------------------------------------
    # 4. Evaluate both models on exactly the same test rows.
    # --------------------------------------------------------
    baseline_results, baseline_subgroups = report_model(
        "BASELINE: seven non-DDoS captures",
        baseline,
        baseline_imputer,
        X_test,
        y_test,
        groups,
    )

    candidate_results, candidate_subgroups = report_model(
        "CANDIDATE: seven captures + DDoS training split",
        candidate,
        candidate_imputer,
        X_test,
        y_test,
        groups,
    )

    comparison = pd.DataFrame([
        baseline_results,
        candidate_results,
    ])

    comparison_path = REPORT_DIR / "ddos_augmented_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    subgroup_comparison = pd.DataFrame(
        baseline_subgroups + candidate_subgroups
    )

    subgroup_path = REPORT_DIR / "ddos_augmented_subgroup_comparison.csv"
    subgroup_comparison.to_csv(subgroup_path, index=False)

    # --------------------------------------------------------
    # 5. Threshold analysis on the held-out test split.
    # This is diagnostic; use a separate validation set to
    # choose a threshold for production.
    # --------------------------------------------------------
    threshold_rows = []

    for name, model, imputer in [
        ("baseline", baseline, baseline_imputer),
        ("candidate", candidate, candidate_imputer),
    ]:
        scores = model.predict_proba(
            imputer.transform(X_test)
        )[:, 1]

        for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
            pred = (scores >= threshold).astype("int8")
            tn, fp, fn, tp = confusion_matrix(
                y_test, pred, labels=[0, 1]
            ).ravel()

            threshold_rows.append({
                "model": name,
                "threshold": threshold,
                "accuracy_pct": accuracy_score(
                    y_test, pred
                ) * 100,
                "macro_f1_pct": f1_score(
                    y_test, pred, average="macro", zero_division=0
                ) * 100,
                "attack_recall_pct": (
                    tp / (tp + fn) * 100 if tp + fn else 0
                ),
                "benign_fpr_pct": (
                    fp / (fp + tn) * 100 if fp + tn else 0
                ),
                "attack_precision_pct": (
                    tp / (tp + fp) * 100 if tp + fp else 0
                ),
                "FP": int(fp),
                "FN": int(fn),
            })

    threshold_path = REPORT_DIR / "ddos_augmented_threshold_analysis.csv"
    pd.DataFrame(threshold_rows).to_csv(
        threshold_path, index=False
    )

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETE")
    print("=" * 65)
    print(f"Model comparison:  {comparison_path}")
    print(f"Subgroup results:  {subgroup_path}")
    print(f"Threshold results: {threshold_path}")
    print(f"Total elapsed: {time.time() - started:.1f}s")
    print("\nNo production model files were changed.")


if __name__ == "__main__":
    main()
