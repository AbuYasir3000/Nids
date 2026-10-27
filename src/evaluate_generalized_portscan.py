from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from traffic_features import source_flows_per_minute
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
MODEL_PATH = ROOT / "models" / "novel_attack_generalizer.joblib"
REPORT_PATH = ROOT / "reports" / "generalized_portscan_evaluation.txt"
EXCLUDED_COLUMNS = {"flow id", "source ip", "destination ip", "timestamp", "label"}


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Held-out PortScan capture not found: {DATA_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}. "
            "Run src/train_generalized_attack_model.py first."
        )

    artifact = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, encoding="cp1252", low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    label_column = next(
        (column for column in df.columns if column.strip().lower() == "label"),
        None,
    )
    if label_column is None:
        raise ValueError("No Label column found in held-out capture.")

    labels = df[label_column].astype("string").str.strip().str.upper()
    burst_counts = source_flows_per_minute(df)
    valid = labels.notna() & labels.ne("") & labels.ne("NAN")
    y_true = labels.loc[valid].ne("BENIGN").astype("int8").to_numpy()
    columns = [
        column for column in df.columns
        if column.strip().lower() not in EXCLUDED_COLUMNS
    ]
    X = df.loc[valid, columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.reindex(columns=artifact["feature_columns"])
    X = artifact["imputer"].transform(X).astype("float32")

    scores = artifact["model"].predict_proba(X)[:, 1]
    model_flags = scores > float(artifact["decision_threshold"])
    burst_flags = burst_counts > int(artifact["source_burst_threshold"])
    y_pred = model_flags | burst_flags
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    results = {
        "capture": DATA_PATH.name,
        "training_rows": artifact["training_rows"],
        "features": len(artifact["feature_columns"]),
        "rows": len(y_true),
        "calibration_capture": artifact["calibration_capture"],
        "calibration_target_fpr": artifact["calibration_target_fpr"],
        "calibration_observed_fpr": artifact["calibration_observed_fpr"],
        "decision_threshold": artifact["decision_threshold"],
        "source_burst_threshold": artifact["source_burst_threshold"],
        "model_only_attack_recall": recall_score(
            y_true, model_flags, zero_division=0
        ),
        "source_burst_attack_recall": recall_score(
            y_true, burst_flags, zero_division=0
        ),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "attack_precision": precision_score(y_true, y_pred, zero_division=0),
        "attack_recall": recall_score(y_true, y_pred, zero_division=0),
        "benign_fpr": fp / (fp + tn) if fp + tn else 0.0,
        "roc_auc": roc_auc_score(y_true, scores),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    report = [
        "Cross-capture generalized model: held-out PortScan evaluation",
        "==============================================================",
        f"Capture: {results['capture']}",
        f"Training rows: {results['training_rows']:,}",
        f"Features: {results['features']}",
        f"Evaluation rows: {results['rows']:,}",
        f"Threshold calibrated on: {results['calibration_capture']} benign rows",
        f"Calibration target false-positive rate: {results['calibration_target_fpr']:.3%}",
        f"Calibration observed false-positive rate: {results['calibration_observed_fpr']:.4%}",
        f"Model decision threshold: {results['decision_threshold']:.10f}",
        f"Source flow burst threshold (per source per minute): {results['source_burst_threshold']}",
        f"Model-only attack recall: {results['model_only_attack_recall']:.4%}",
        f"Burst-rule attack recall: {results['source_burst_attack_recall']:.4%}",
        "",
        f"Accuracy: {results['accuracy']:.4%}",
        f"Balanced accuracy: {results['balanced_accuracy']:.4%}",
        f"Macro F1: {results['macro_f1']:.4f}",
        f"Attack precision: {results['attack_precision']:.4%}",
        f"Attack recall: {results['attack_recall']:.4%}",
        f"Benign false-positive rate: {results['benign_fpr']:.4%}",
        f"ROC AUC: {results['roc_auc']:.4f}",
        f"Confusion matrix TN/FP/FN/TP: {tn}/{fp}/{fn}/{tp}",
        "",
        "The PortScan capture is excluded from training and threshold calibration.",
        "The combined alert is the OR of the calibrated model and source-flow burst rule.",
        "Treat this as a development result; confirm generalization on an independent dataset.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
