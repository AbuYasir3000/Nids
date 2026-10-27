"""Capture-separated diagnostics for the saved binary model and CIC rules.

This script never writes or replaces model artifacts. It trains one temporary
in-memory candidate with a capture-separated train/calibration/validation
split, evaluates the saved model's score distributions, and audits frozen
behavioral thresholds across the local CIC-IDS2017 captures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hybrid_detector import GENERALIZED_PATH
from train_generalized_attack_model import (
    CALIBRATION_FPR,
    CALIBRATION_CAPTURE,
    DATA_DIR,
    MAX_ROWS_PER_CLASS_PER_CAPTURE,
    PORTSCAN_CAPTURE,
    RANDOM_STATE,
    load_capture,
    sample_capture,
)
from traffic_features import (
    PORT_DIVERSITY_THRESHOLD,
    source_flows_per_minute,
)

VALIDATION_CAPTURE = "Wednesday-workingHours.pcap_ISCX.csv"
CANDIDATE_CALIBRATION_CAPTURE = "Friday-WorkingHours-Morning.pcap_ISCX.csv"
OUTPUT_DIR = ROOT / "reports"
REPORT_PATH = OUTPUT_DIR / "generalized_model_diagnostics.txt"
PR_PATH = OUTPUT_DIR / "generalized_validation_precision_recall.csv"
RULE_PATH = OUTPUT_DIR / "behavioral_capture_diagnostics.csv"


def binary_metrics(y_true, y_score, threshold):
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_score) > float(threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision_attack = precision_score(y_true, y_pred, zero_division=0)
    recall_attack = recall_score(y_true, y_pred, zero_division=0)
    f1_attack = f1_score(y_true, y_pred, zero_division=0)
    precision_benign = float(tn / (tn + fn)) if tn + fn else 0.0
    recall_benign = float(tn / (tn + fp)) if tn + fp else 0.0
    f1_benign = (
        2 * precision_benign * recall_benign / (precision_benign + recall_benign)
        if precision_benign + recall_benign else 0.0
    )
    return {
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision_attack),
        "recall": float(recall_attack),
        "f1": float(f1_attack),
        "macro_f1": float((f1_benign + f1_attack) / 2),
        "benign_precision": precision_benign,
        "benign_recall": recall_benign,
        "benign_f1": f1_benign,
        "benign_support": int((y_true == 0).sum()),
        "attack_support": int((y_true == 1).sum()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "benign_fpr": float(fp / (tn + fp)) if tn + fp else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def probability_summary(name, scores, y):
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=np.int8)
    lines = [f"{name}: rows={len(y):,}; attacks={int(y.sum()):,}; benign={int((y == 0).sum()):,}"]
    for label, mask in (("BENIGN", y == 0), ("ATTACK", y == 1)):
        if not mask.any():
            continue
        values = scores[mask]
        quantiles = np.quantile(values, [0, .5, .9, .99, .999, 1])
        lines.append(
            f"  {label} probability min/median/p90/p99/p99.9/max="
            + "/".join(f"{value:.8g}" for value in quantiles)
        )
    return lines


def load_training_capture(name):
    X, y = load_capture(name)
    if "Label" in X.columns:
        raise ValueError(f"Target leakage: Label remained in features for {name}")
    return X, y.astype("int8")


def group_port_diversity(frame):
    columns = {str(c).strip().casefold(): c for c in frame.columns}
    src = columns.get("source ip")
    ts = columns.get("timestamp")
    dport = columns.get("destination port")
    counts = np.ones(len(frame), dtype=np.int64)
    if src is None or ts is None or dport is None or frame.empty:
        return counts
    source = frame[src].astype("string").str.strip()
    minute = pd.to_datetime(frame[ts], errors="coerce").dt.floor("min")
    port = pd.to_numeric(frame[dport], errors="coerce")
    valid = source.notna() & source.ne("") & minute.notna()
    if valid.any():
        grouped = pd.DataFrame({"src": source[valid], "minute": minute[valid], "port": port[valid]})
        diversity = grouped.groupby(["src", "minute"], sort=False)["port"].transform("nunique")
        counts[np.flatnonzero(valid.to_numpy())] = diversity.to_numpy(dtype=np.int64)
    return counts


def read_capture_labels(path):
    frame = pd.read_csv(path, encoding="cp1252", low_memory=False)
    frame.columns = frame.columns.astype(str).str.strip()
    label_col = next((c for c in frame.columns if c.casefold() == "label"), None)
    if label_col is None:
        raise ValueError(f"No Label column in {path.name}")
    labels = frame[label_col].astype("string").str.strip().str.upper()
    valid = labels.notna() & labels.ne("") & labels.ne("NAN")
    y = labels.loc[valid].ne("BENIGN").astype("int8").to_numpy()
    clean = frame.loc[valid].reset_index(drop=True)
    return clean, y


def evaluate_behavioral_capture(path, source_threshold):
    frame, y = read_capture_labels(path)
    source_counts = source_flows_per_minute(frame)
    port_counts = group_port_diversity(frame)
    masks = {
        "source_burst": source_counts > source_threshold,
        "high_rate_multiport": (source_counts > source_threshold)
        & (port_counts > PORT_DIVERSITY_THRESHOLD),
    }
    output = {"capture": path.name, "rows": len(y), "benign": int((y == 0).sum()), "attack": int(y.sum())}
    for name, flags in masks.items():
        tn, fp, fn, tp = confusion_matrix(y, flags.astype(np.int8), labels=[0, 1]).ravel()
        has_both_classes = np.unique(y).size == 2
        output.update({
            f"{name}_tn": int(tn), f"{name}_fp": int(fp),
            f"{name}_fn": int(fn), f"{name}_tp": int(tp),
            f"{name}_precision": float(precision_score(y, flags, zero_division=0)) if has_both_classes else None,
            f"{name}_recall": float(recall_score(y, flags, zero_division=0)) if has_both_classes else None,
            f"{name}_f1": float(f1_score(y, flags, zero_division=0)) if has_both_classes else None,
            f"{name}_balanced_accuracy": float(balanced_accuracy_score(y, flags)) if has_both_classes else None,
            f"{name}_benign_fpr": float(fp / (tn + fp)) if tn + fp else 0.0,
            f"{name}_flow_flags": int(flags.sum()),
        })
    output["source_minute_events"] = int(
        pd.DataFrame({"source": frame.get("Source IP"), "minute": pd.to_datetime(frame.get("Timestamp"), errors="coerce").dt.floor("min"), "flag": masks["source_burst"]})
        .loc[lambda x: x["flag"] & x["source"].notna() & x["minute"].notna()]
        .groupby(["source", "minute"], sort=False).ngroups
    )
    return output


def main(udplag_csv: Path | None = None):
    artifact = joblib.load(GENERALIZED_PATH)
    prod_model = artifact["model"]
    prod_columns = list(artifact["feature_columns"])
    prod_threshold = float(artifact["decision_threshold"])
    imputer = artifact["imputer"]
    lines = [
        "Saved binary model diagnostics (no production artifact modified)",
        "===============================================================",
        f"Artifact: {GENERALIZED_PATH.name}; threshold={prod_threshold:.10f}; features={len(prod_columns)}",
        f"Training captures: {', '.join(artifact['training_captures'])}",
        f"Calibration capture: {artifact['calibration_capture']}; target FPR={artifact['calibration_target_fpr']:.3%}; observed FPR={artifact['calibration_observed_fpr']:.4%}",
        "Training preprocessing: fitted SimpleImputer(strategy=median), then float32; HistGradientBoostingClassifier(class_weight=balanced).",
        "Labels are collapsed to BENIGN=0 and every non-BENIGN CIC label=1; UDP-lag/WebDDoS are not represented in this training label taxonomy.",
        "The threshold is the 'higher' quantile of benign scores at 99.9%, then predictions use strict score > threshold. This is deliberately conservative to target low benign FPR.",
        "",
    ]

    # Explain the saved model's scores on CIC calibration and prior PortScan holdout.
    for name in (CALIBRATION_CAPTURE, PORTSCAN_CAPTURE):
        path = DATA_DIR / name
        frame, y = read_capture_labels(path)
        X = frame.loc[:, prod_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        score = prod_model.predict_proba(imputer.transform(X).astype("float32"))[:, 1]
        lines.append(f"Saved model score distribution on {name} (not a new independent test):")
        lines.extend(probability_summary("  scores", score, y))
        lines.append("  " + str(binary_metrics(y, score, prod_threshold)))
        lines.append("")

    if udplag_csv is not None:
        if not udplag_csv.is_file():
            raise FileNotFoundError(f"UDPLag CSV not found: {udplag_csv}")
        frame = pd.read_csv(udplag_csv, low_memory=False)
        frame.columns = frame.columns.astype(str).str.strip()
        labels = frame["Label"].astype("string").str.strip().str.upper()
        valid = labels.notna() & labels.ne("") & labels.ne("NAN")
        y = labels.loc[valid].ne("BENIGN").astype("int8").to_numpy()
        X = frame.loc[valid, prod_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        score = prod_model.predict_proba(imputer.transform(X).astype("float32"))[:, 1]
        lines.append("Saved model score distribution on UDPLag.csv (diagnostic only; not used for tuning):")
        lines.extend(probability_summary("  scores", score, y))
        lines.append("  " + str(binary_metrics(y, score, prod_threshold)))
        duration = pd.to_numeric(frame.loc[valid, "Flow Duration"], errors="coerce").to_numpy()
        zero_duration = duration == 0
        lines.append(
            f"  Zero-duration rows={int(zero_duration.sum()):,}; "
            f"invalid Flow Bytes/s cells={int(X['Flow Bytes/s'].isna().sum()):,}; "
            f"invalid Flow Packets/s cells={int(X['Flow Packets/s'].isna().sum()):,}."
        )
        for label in sorted(labels.loc[valid].unique()):
            label_mask = labels.loc[valid].eq(label).to_numpy()
            lines.append(
                f"  {label}: zero-duration={int((zero_duration & label_mask).sum()):,}/"
                f"{int(label_mask.sum()):,}."
            )
        lines.extend(probability_summary("  zero-duration rows", score[zero_duration], y[zero_duration]))
        lines.extend(probability_summary("  non-zero-duration rows", score[~zero_duration], y[~zero_duration]))
        lines.append("")
    else:
        lines.append(
            "UDPLag-specific diagnostics skipped; pass --udplag-csv PATH to include a labeled CSV."
        )
        lines.append("")

    # Train a disposable candidate with strict capture separation. The candidate
    # is never serialized and is not proposed as a UDP-lag-capable model.
    validation_name = VALIDATION_CAPTURE
    calibration_name = CANDIDATE_CALIBRATION_CAPTURE
    excluded = {validation_name, calibration_name, CALIBRATION_CAPTURE, PORTSCAN_CAPTURE}
    training_names = sorted(
        p.name for p in DATA_DIR.glob("*pcap_ISCX.csv") if p.name not in excluded
    )
    if not training_names:
        raise RuntimeError("No candidate training captures available after split exclusions.")
    rng = np.random.default_rng(RANDOM_STATE)
    train_parts, target_parts = [], []
    reference_features = None
    for name in training_names:
        X, y = load_training_capture(name)
        if reference_features is None:
            reference_features = set(X.columns)
        elif set(X.columns) != reference_features:
            raise ValueError(
                f"Training capture schema mismatch for {name}: "
                f"missing={sorted(reference_features - set(X.columns))}; "
                f"extra={sorted(set(X.columns) - reference_features)}"
            )
        X, y = sample_capture(X, y, rng)
        train_parts.append(X)
        target_parts.append(y)
    train_frame = pd.concat(train_parts, ignore_index=True, sort=False)
    train_y = pd.concat(target_parts, ignore_index=True).astype("int8").to_numpy()
    candidate_features = sorted(train_frame.columns.astype(str))
    if set(candidate_features) != reference_features:
        raise RuntimeError("Candidate feature set changed while concatenating captures.")
    candidate_imputer = SimpleImputer(strategy="median")
    train_matrix = candidate_imputer.fit_transform(train_frame.loc[:, candidate_features]).astype("float32")
    candidate = HistGradientBoostingClassifier(
        max_iter=150, max_leaf_nodes=31, learning_rate=0.1,
        l2_regularization=1.0, class_weight="balanced", random_state=RANDOM_STATE,
    )
    candidate.fit(train_matrix, train_y)

    calibration_X, calibration_y = load_training_capture(calibration_name)
    validation_X, validation_y = load_training_capture(validation_name)
    for capture_name, capture_X in ((calibration_name, calibration_X), (validation_name, validation_X)):
        missing = [column for column in candidate_features if column not in capture_X]
        if missing:
            raise ValueError(f"{capture_name} lacks candidate features: {missing}")
    calibration_scores = candidate.predict_proba(candidate_imputer.transform(calibration_X.loc[:, candidate_features]).astype("float32"))[:, 1]
    validation_scores = candidate.predict_proba(candidate_imputer.transform(validation_X.loc[:, candidate_features]).astype("float32"))[:, 1]
    calibration_y_arr = calibration_y.to_numpy(dtype=np.int8)
    validation_y_arr = validation_y.to_numpy(dtype=np.int8)
    benign_calibration = calibration_scores[calibration_y_arr == 0]
    if not benign_calibration.size:
        raise ValueError(f"No benign rows in calibration capture {calibration_name}")
    candidate_threshold = float(np.quantile(benign_calibration, 1 - CALIBRATION_FPR, method="higher"))
    precision, recall, thresholds = precision_recall_curve(validation_y_arr, validation_scores)
    pr = pd.DataFrame({
        "threshold": np.append(thresholds, np.nan),
        "precision": precision,
        "recall": recall,
    })
    pr.to_csv(PR_PATH, index=False)
    lines.extend([
        "Capture-separated validation experiment (temporary in-memory candidate; no artifact written)",
        f"  Train captures: {', '.join(training_names)}",
        f"  Calibration capture: {calibration_name} (benign-only threshold quantile; target FPR={CALIBRATION_FPR:.3%})",
        f"  Validation capture: {validation_name} (precision-recall curve and fixed-threshold comparisons)",
        f"  Candidate sampled training rows={len(train_y):,}; train benign={int((train_y == 0).sum()):,}; train attack={int(train_y.sum()):,}",
        f"  Calibration rows={len(calibration_y_arr):,}; validation rows={len(validation_y_arr):,}",
        f"  Candidate threshold from calibration benign scores={candidate_threshold:.10f}",
        f"  Validation average precision={average_precision_score(validation_y_arr, validation_scores):.6f}",
        f"  PR curve points={len(pr):,}; saved to {PR_PATH.relative_to(ROOT)}",
        "  Threshold comparisons below are on validation only; threshold selection used calibration capture, not validation or UDPLag.",
    ])
    for title, threshold in (
        ("Default 0.5 diagnostic", 0.5),
        ("Saved production threshold (cross-model comparison only)", prod_threshold),
        ("Candidate threshold calibrated on separate capture", candidate_threshold),
    ):
        lines.append(f"  {title}: {binary_metrics(validation_y_arr, validation_scores, threshold)}")
    lines.extend([
        "  Limitation: this is CIC-IDS2017 cross-capture validation of a temporary binary CIC detector, not UDP-lag validation and not an independent final test.",
        "",
    ])

    # Frozen rule audit over every available labeled CIC capture. These are
    # diagnostics only: CIC captures participated in prior threshold or model
    # development, so none is claimed as untouched final evaluation.
    source_threshold = int(artifact.get("source_burst_threshold", 0))
    rule_rows = []
    for path in sorted(DATA_DIR.glob("*pcap_ISCX.csv")):
        print(f"Evaluating frozen behavioral rules: {path.name}", flush=True)
        rule_rows.append(evaluate_behavioral_capture(path, source_threshold))
    rule_frame = pd.DataFrame(rule_rows)
    rule_frame.to_csv(RULE_PATH, index=False)
    lines.extend([
        "Frozen behavioral-rule capture audit (flow-level; all local CIC captures have prior development/calibration exposure)",
        f"  Source-flow threshold: >{source_threshold:,} flows/source/minute.",
        f"  Multiport condition: >{PORT_DIVERSITY_THRESHOLD} unique destination ports in the source-minute, plus source-flow threshold.",
        f"  Per-capture metrics saved to {RULE_PATH.relative_to(ROOT)}; source-burst event counts are separate from flow-flag counts.",
        "  Precision/recall/F1/balanced accuracy are blank for captures with only one ground-truth class; their confusion counts and benign FPR remain available.",
        "  These rows are not an untouched final-performance estimate because source-rate/port-diversity thresholds were derived using CIC captures and the project's previous reports examined DDoS/PortScan captures.",
        "",
        "No production model, threshold, or model artifact was modified or replaced.",
    ])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Report saved to {REPORT_PATH}")
    print(f"PR curve saved to {PR_PATH}")
    print(f"Behavioral capture table saved to {RULE_PATH}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--udplag-csv",
        type=Path,
        help="Optional labeled UDP-lag CSV for diagnostic scoring; never used for tuning.",
    )
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.udplag_csv))
