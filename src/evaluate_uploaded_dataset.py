"""Evaluate the saved Argus detectors against one labeled flow CSV.

This script is read-only with respect to input data and model artifacts. It
reports the saved models' predictions without fitting, calibrating, or tuning.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_detector import (
    GENERALIZED_PATH,
    IF_PATH,
    MODEL_DIR,
    RF_PATH,
    _source_minute_indicators,
)
from src.traffic_features import (
    PORT_DIVERSITY_THRESHOLD,
    source_flows_per_minute,
)


DEFAULT_OUTPUT = Path("reports") / "uploaded_dataset_evaluation.txt"
TRAINING_REFERENCE = PROJECT_ROOT / "data" / "processed" / "cicids2017_cleaned.csv"
IDENTIFIER_COLUMNS = {
    "flow id", "source ip", "destination ip", "timestamp", "label",
    "unnamed: 0",
}


def read_labeled_csv(path: Path) -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    else:
        raise ValueError("Could not parse input CSV: " + " | ".join(errors))

    frame.columns = frame.columns.map(lambda column: str(column).strip())
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names after trimming: {duplicates}")
    label_columns = [
        column for column in frame.columns if column.casefold() == "label"
    ]
    if len(label_columns) != 1:
        raise ValueError(f"Expected one Label column, found {label_columns}.")
    return frame


def validate_schema(name: str, feature_columns: list[str], frame: pd.DataFrame) -> list[str]:
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError(f"{name} artifact has duplicate feature names.")
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is incompatible; missing exact features: {missing}")
    return feature_columns


def inspect_training_reference(feature_columns: list[str]) -> dict | None:
    """Read training-source schema and selected feature-domain evidence in chunks."""
    if not TRAINING_REFERENCE.is_file():
        return None
    header = pd.read_csv(TRAINING_REFERENCE, nrows=0, encoding="cp1252").columns
    header = [str(column).strip() for column in header]
    missing = [column for column in feature_columns if column not in header]
    selected = {
        "Label", "Flow Duration", "Flow Bytes/s", "Flow Packets/s",
        "Fwd Header Length", "Fwd Header Length.1",
    }
    summaries = {
        "rows": 0,
        "duration_min": np.inf,
        "duration_max": -np.inf,
        "duration_zero": 0,
        "rate_invalid": Counter(),
        "duplicate_header_mismatch": 0,
    }
    for chunk in pd.read_csv(
        TRAINING_REFERENCE,
        usecols=lambda column: column.strip() in selected,
        chunksize=200_000,
        low_memory=False,
        encoding="cp1252",
    ):
        chunk.columns = chunk.columns.map(lambda column: str(column).strip())
        summaries["rows"] += len(chunk)
        if "Flow Duration" in chunk:
            duration = pd.to_numeric(chunk["Flow Duration"], errors="coerce")
            summaries["duration_min"] = min(summaries["duration_min"], float(duration.min()))
            summaries["duration_max"] = max(summaries["duration_max"], float(duration.max()))
            summaries["duration_zero"] += int(duration.eq(0).sum())
        for column in ("Flow Bytes/s", "Flow Packets/s"):
            if column in chunk:
                values = pd.to_numeric(chunk[column], errors="coerce")
                invalid = values.isna() | ~np.isfinite(values.to_numpy(dtype=float))
                summaries["rate_invalid"][column] += int(invalid.sum())
        if {"Fwd Header Length", "Fwd Header Length.1"}.issubset(chunk.columns):
            first = pd.to_numeric(chunk["Fwd Header Length"], errors="coerce")
            duplicate = pd.to_numeric(chunk["Fwd Header Length.1"], errors="coerce")
            equal = first.eq(duplicate) | (first.isna() & duplicate.isna())
            summaries["duplicate_header_mismatch"] += int((~equal).sum())
    return {
        "columns": len(header),
        "missing_features": missing,
        "summary": summaries,
    }


def numeric_matrix(frame: pd.DataFrame, columns: list[str], name: str):
    raw = frame.loc[:, columns]
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    missing_by_column = numeric.isna().sum().sort_values(ascending=False)
    missing_cells = int(missing_by_column.sum())
    all_missing = numeric.columns[numeric.isna().all()].tolist()
    if all_missing:
        raise ValueError(
            f"{name} cannot be evaluated: entire required feature columns are "
            f"non-numeric or missing: {all_missing}"
        )
    return numeric, missing_by_column


def binary_metrics(y_true, y_pred) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "confusion_matrix": matrix.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_attack": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_attack": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_attack": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "benign_false_positive_rate": float(fp / (tn + fp)) if tn + fp else None,
    }


def evaluate_flow_rule(y_true, flags) -> dict:
    return binary_metrics(np.asarray(y_true, dtype=np.int8), np.asarray(flags, dtype=np.int8))


def build_event_table(frame, labels, source_counts, port_counts, destination_counts, burst_threshold):
    source_col = next(
        (column for column in frame.columns if column.casefold() == "source ip"),
        None,
    )
    timestamp_col = next(
        (column for column in frame.columns if column.casefold() == "timestamp"),
        None,
    )
    if source_col is None or timestamp_col is None:
        return pd.DataFrame(), 0

    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
    sources = frame[source_col].astype("string").str.strip()
    valid = sources.notna() & sources.ne("") & timestamps.notna()
    excluded = int((~valid).sum())
    if not valid.any():
        return pd.DataFrame(), excluded

    grouped = pd.DataFrame({
        "source_ip": sources.loc[valid].to_numpy(),
        "window_start": timestamps.loc[valid].dt.floor("min").to_numpy(),
        "observed_timestamp": timestamps.loc[valid].to_numpy(),
        "label": labels.loc[valid].to_numpy(),
        "source_flow_count": np.asarray(source_counts)[valid.to_numpy()],
        "unique_destination_ports": np.asarray(port_counts)[valid.to_numpy()],
        "unique_destination_ips": np.asarray(destination_counts)[valid.to_numpy()],
    })
    destination_column = next(
        (column for column in frame.columns if column.casefold() == "destination ip"),
        None,
    )
    port_column = next(
        (column for column in frame.columns if column.casefold() == "destination port"),
        None,
    )
    if destination_column is not None:
        grouped["destination_ip"] = frame.loc[valid, destination_column].astype("string").to_numpy()
    if port_column is not None:
        grouped["destination_port"] = frame.loc[valid, port_column].to_numpy()
    # All rows in a source-minute have the same group features. Reduce them to
    # one source-minute event before evaluating or reporting rule alerts.
    events = grouped.groupby(["source_ip", "window_start"], sort=True).agg(
        flow_count=("label", "size"),
        window_end=("observed_timestamp", "max"),
        unique_destination_ports=("unique_destination_ports", "max"),
        unique_destination_ips=("unique_destination_ips", "max"),
        labels=("label", lambda values: ", ".join(sorted(set(values)))),
        attack_flow_count=("label", lambda values: int((values.str.upper() != "BENIGN").sum())),
        benign_flow_count=("label", lambda values: int((values.str.upper() == "BENIGN").sum())),
    ).reset_index()
    if "destination_ip" in grouped:
        top_destination_ips = grouped.groupby(["source_ip", "window_start"])["destination_ip"].agg(
            lambda values: ", ".join(
                f"{value} ({count:,})"
                for value, count in Counter(values.dropna().astype(str)).most_common(3)
            )
        )
        events["top_destination_ips"] = [
            top_destination_ips.loc[(row.source_ip, row.window_start)]
            for row in events.itertuples()
        ]
    else:
        events["top_destination_ips"] = "unavailable"
    if "destination_port" in grouped:
        top_destination_ports = grouped.groupby(["source_ip", "window_start"])["destination_port"].agg(
            lambda values: ", ".join(
                f"{value} ({count:,})"
                for value, count in Counter(values.dropna().astype(str)).most_common(3)
            )
        )
        events["top_destination_ports"] = [
            top_destination_ports.loc[(row.source_ip, row.window_start)]
            for row in events.itertuples()
        ]
    else:
        events["top_destination_ports"] = "unavailable"
    events["source_flows_per_minute"] = events["flow_count"]
    events["source_burst_alert"] = events["source_flows_per_minute"] > burst_threshold
    events["high_rate_multiport_alert"] = (
        events["source_burst_alert"]
        & (events["unique_destination_ports"] > PORT_DIVERSITY_THRESHOLD)
    )
    events["actual_attack_event"] = events["attack_flow_count"] > 0
    return events, excluded


def summarize_binary(name: str, metrics: dict) -> list[str]:
    return [
        name,
        f"  Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}",
        f"  Accuracy={metrics['accuracy']:.6f}; balanced_accuracy={metrics['balanced_accuracy']:.6f}; macro_F1={metrics['macro_f1']:.6f}",
        f"  Attack precision={metrics['precision_attack']:.6f}; recall={metrics['recall_attack']:.6f}; F1={metrics['f1_attack']:.6f}",
        f"  BENIGN false-positive rate={metrics['benign_false_positive_rate']}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Labeled flow CSV to evaluate")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_path = args.csv.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"CSV does not exist: {input_path}")

    frame = read_labeled_csv(input_path)
    label_column = next(column for column in frame.columns if column.casefold() == "label")
    labels = frame[label_column].astype("string").str.strip()
    valid_label = labels.notna() & labels.ne("") & labels.str.upper().ne("NAN")
    valid_rows = frame.loc[valid_label].reset_index(drop=True)
    labels = labels.loc[valid_label].reset_index(drop=True)
    y_attack = labels.str.upper().ne("BENIGN").astype(np.int8).to_numpy()

    rf_artifact = joblib.load(RF_PATH)
    if_artifact = joblib.load(IF_PATH)
    generalized_artifact = joblib.load(GENERALIZED_PATH) if GENERALIZED_PATH.exists() else None
    rf_columns = list(rf_artifact["feature_columns"])
    if_columns = list(if_artifact["feature_columns"])
    generalized_columns = (
        list(generalized_artifact["feature_columns"])
        if generalized_artifact is not None else []
    )
    training_reference = inspect_training_reference(rf_columns)

    schema_status = {}
    for name, columns in (
        ("Random Forest", rf_columns),
        ("Isolation Forest", if_columns),
        ("Learned binary classifier", generalized_columns),
    ):
        if not columns:
            schema_status[name] = "artifact unavailable"
            continue
        try:
            validate_schema(name, columns, valid_rows)
            schema_status[name] = f"compatible by exact feature names ({len(columns)})"
        except ValueError as exc:
            schema_status[name] = str(exc)

    # RF and Isolation Forest were trained against the same CIC schema. Check
    # the actual serialized order before sharing their model input matrix.
    same_rf_if_order = rf_columns == if_columns
    rf_if_schema_note = (
        "same names and order" if same_rf_if_order
        else "different feature order; each artifact is evaluated separately"
    )

    output_counts = {}
    imputation_counts = {}
    predictions = {}
    model_classes = list(rf_artifact["model"].classes_)
    report_lines = [
        "Argus labeled CSV evaluation",
        "==================================",
        f"Input: {input_path}",
        f"Rows: {len(frame):,}; labeled rows: {len(valid_rows):,}; excluded unlabeled rows: {len(frame) - len(valid_rows):,}",
        f"Original columns: {len(frame.columns)}",
        "",
        "Ground-truth label distribution:",
    ]
    report_lines.extend(
        f"  {label}: {int(count):,}"
        for label, count in labels.value_counts(dropna=False).items()
    )
    report_lines.extend(["", "Feature/schema checks:"])
    report_lines.extend(f"  {name}: {status}" for name, status in schema_status.items())
    report_lines.append(f"  RF vs IF schema order: {rf_if_schema_note}")
    report_lines.append(
        "  Semantic feature definitions/units: feature names are checked against "
        "the local cleaned training CSV below; the CSV does not record extractor "
        "version or a full feature-definition contract."
    )
    report_lines.append(
        "  Model input excludes Label, Flow ID, Source IP, Destination IP, "
        "Timestamp, and Unnamed: 0; extra columns are not selected by any artifact."
    )
    if training_reference is None:
        report_lines.append("  CIC-IDS2017 cleaned training reference: unavailable.")
    else:
        train_summary = training_reference["summary"]
        report_lines.extend([
            f"  CIC-IDS2017 cleaned training reference: {train_summary['rows']:,} rows, {training_reference['columns']} columns; missing expected RF features={training_reference['missing_features']}.",
            f"  Flow Duration training range: {train_summary['duration_min']} to {train_summary['duration_max']}; zero-duration rows={train_summary['duration_zero']:,}.",
            "  Training rate cells made invalid by cleaning: " + repr(dict(train_summary["rate_invalid"])),
            f"  Fwd Header Length duplicate mismatches in training reference: {train_summary['duplicate_header_mismatch']:,}.",
        ])

    # Multiclass RF: do not invent labels for attack families absent in training.
    if schema_status["Random Forest"].startswith("compatible"):
        numeric, missing_by_column = numeric_matrix(valid_rows, rf_columns, "Random Forest")
        missing_cells = int(missing_by_column.sum())
        medians = pd.Series(rf_artifact["medians"]).reindex(rf_columns)
        bad_medians = medians.index[~np.isfinite(medians.to_numpy(dtype=float))].tolist()
        if bad_medians:
            raise ValueError(f"Random Forest artifact has invalid medians: {bad_medians}")
        rf_input = numeric.fillna(medians).astype("float32")
        rf_pred = rf_artifact["model"].predict(rf_input)
        predictions["Random Forest"] = pd.Series(rf_pred)
        output_counts["Random Forest"] = Counter(map(str, rf_pred))
        imputation_counts["Random Forest"] = missing_cells
        supported = labels.isin(model_classes).to_numpy()
        unsupported_labels = labels.loc[~supported].value_counts().to_dict()
        strict_classes = sorted(set(labels.astype(str)) | set(map(str, rf_pred)))
        strict_matrix = confusion_matrix(
            labels.astype(str), pd.Series(rf_pred).astype(str), labels=strict_classes
        )
        report_lines.extend([
            "",
            "Random Forest multiclass results:",
            f"  Trained output classes: {', '.join(map(str, model_classes))}",
            f"  Unsupported actual-label rows (excluded from supported-class metrics): {int((~supported).sum()):,}",
            f"  Unsupported actual labels: {unsupported_labels}",
            f"  Invalid/missing feature cells imputed from saved RF training medians: {missing_cells:,}",
            "  Feature columns requiring imputation: " + repr({str(k): int(v) for k, v in missing_by_column[missing_by_column > 0].items()}),
            "  Prediction counts:",
        ])
        report_lines.extend(f"    {name}: {count:,}" for name, count in output_counts["Random Forest"].most_common())
        report_lines.extend([
            f"  Strict exact-label agreement across all rows: {accuracy_score(labels.astype(str), pd.Series(rf_pred).astype(str)):.6f} (unsupported actual classes count as misses; this is not supported-class multiclass performance).",
            "  Strict confusion matrix (rows=true, columns=predicted; "
            f"labels={strict_classes!r}):",
            str(strict_matrix),
            "  Strict per-class report (unsupported labels remain explicit):",
            classification_report(
                labels.astype(str),
                pd.Series(rf_pred).astype(str),
                labels=strict_classes,
                zero_division=0,
            ),
        ])
        report_lines.append("  Actual-label support and RF prediction counts:")
        for actual_class in sorted(labels.unique()):
            actual_mask = labels.eq(actual_class).to_numpy()
            report_lines.append(
                f"    {actual_class}: support={int((labels == actual_class).sum()):,}; "
                f"model_prediction_count={output_counts['Random Forest'].get(str(actual_class), 0):,}; "
                f"correct_exact_matches={int((pd.Series(rf_pred).astype(str).to_numpy()[actual_mask] == str(actual_class)).sum()):,}; "
                f"trained_class={'yes' if actual_class in model_classes else 'no'}"
            )
        if supported.any():
            y_supported = labels.loc[supported].astype(str)
            p_supported = pd.Series(rf_pred).loc[supported].astype(str)
            supported_label_set = sorted(set(y_supported) | set(p_supported))
            if len(supported_label_set) == 1:
                supported_matrix = np.array([[
                    int((y_supported.astype(str) == supported_label_set[0]).sum())
                ]])
            else:
                supported_matrix = confusion_matrix(
                    y_supported, p_supported, labels=supported_label_set
                )
            report_lines.extend([
                "  Supported-label confusion matrix (rows=true, columns=predicted; "
                "labels=" + repr(supported_label_set) + "):",
                str(supported_matrix),
            ])
            if len(set(y_supported)) > 1:
                report_lines.extend([
                    "  Supported-label report:",
                    classification_report(y_supported, p_supported, labels=supported_label_set, zero_division=0),
                ])
            else:
                report_lines.append(
                    "  No multiclass performance report: BENIGN is the only "
                    "ground-truth class supported by this artifact."
                )
        report_lines.append(
            "  Interpretation: UDP-lag/WebDDoS are not RF classes. Their exact-label "
            "detection rate is zero by model design; no attack-family mapping was invented."
        )
    else:
        report_lines.extend(["", f"Random Forest skipped: {schema_status['Random Forest']}"])

    # Binary attack task: BENIGN=0, all known non-BENIGN labels=1.
    if generalized_artifact is not None and schema_status["Learned binary classifier"].startswith("compatible"):
        numeric, missing_by_column = numeric_matrix(valid_rows, generalized_columns, "Learned binary classifier")
        missing_cells = int(missing_by_column.sum())
        imputer = generalized_artifact["imputer"]
        binary_input = imputer.transform(numeric).astype("float32")
        generalized_model = generalized_artifact["model"]
        scores = generalized_model.predict_proba(binary_input)[:, 1]
        threshold = float(generalized_artifact["decision_threshold"])
        binary_pred = (scores > threshold).astype(np.int8)
        predictions["Learned binary classifier"] = pd.Series(binary_pred)
        output_counts["Learned binary classifier"] = Counter({
            "ATTACK": int(binary_pred.sum()),
            "BENIGN": int((binary_pred == 0).sum()),
        })
        imputation_counts["Learned binary classifier"] = missing_cells
        metrics = binary_metrics(y_attack, binary_pred)
        report_lines.extend([
            "",
            "Learned binary classifier (positive=ATTACK, negative=BENIGN):",
            f"  Calibration capture: {generalized_artifact.get('calibration_capture')}",
            f"  Saved threshold: {threshold:.10f}; invalid/missing feature cells handled by saved fitted imputer: {missing_cells:,}",
            "  Feature columns requiring imputation: " + repr({str(k): int(v) for k, v in missing_by_column[missing_by_column > 0].items()}),
            *summarize_binary("  Flow-level metrics:", metrics),
        ])
        report_lines.append("  Detection rate by actual class:")
        for class_name, support in labels.value_counts().items():
            mask = labels.eq(class_name).to_numpy()
            detected = int(binary_pred[mask].sum())
            report_lines.append(
                f"    {class_name}: support={int(support):,}; predicted_attack={detected:,}; "
                f"detection_rate={detected / int(support):.6f}"
            )
        report_lines.append(
            "  Interpretation: this is a valid exact-capture binary comparison "
            "(BENIGN vs all labeled attacks), but one capture does not establish "
            "generalization to other UDP-lag deployments."
        )
    elif generalized_artifact is None:
        report_lines.extend(["", "Learned binary classifier skipped: artifact absent."])
    else:
        report_lines.extend(["", f"Learned binary classifier skipped: {schema_status['Learned binary classifier']}"])

    # Isolation Forest uses its saved scaler and its own saved feature schema.
    if schema_status["Isolation Forest"].startswith("compatible"):
        numeric, missing_by_column = numeric_matrix(valid_rows, if_columns, "Isolation Forest")
        missing_cells = int(missing_by_column.sum())
        medians = pd.Series(if_artifact["medians"]).reindex(if_columns)
        if_input = numeric.fillna(medians).astype("float32")
        scaled = if_artifact["scaler"].transform(if_input)
        anomaly_pred = (if_artifact["model"].predict(scaled) == -1).astype(np.int8)
        predictions["Isolation Forest"] = pd.Series(anomaly_pred)
        output_counts["Isolation Forest"] = Counter({
            "ANOMALY": int(anomaly_pred.sum()),
            "NOT_ANOMALY": int((anomaly_pred == 0).sum()),
        })
        imputation_counts["Isolation Forest"] = missing_cells
        anomaly_binary = binary_metrics(y_attack, anomaly_pred)
        report_lines.extend([
            "",
            "Isolation Forest anomaly output (anomaly is not a confirmed attack):",
            f"  Saved score threshold/offset: {if_artifact.get('threshold')}; invalid/missing feature cells imputed from saved medians: {missing_cells:,}",
            "  Feature columns requiring imputation: " + repr({str(k): int(v) for k, v in missing_by_column[missing_by_column > 0].items()}),
            *summarize_binary("  Attack-vs-benign diagnostic only:", anomaly_binary),
            "  Anomalies by actual class:",
        ])
        for class_name, support in labels.value_counts().items():
            mask = labels.eq(class_name).to_numpy()
            found = int(anomaly_pred[mask].sum())
            report_lines.append(
                f"    {class_name}: support={int(support):,}; anomalies={found:,}; "
                f"rate={found / int(support):.6f}"
            )
    else:
        report_lines.extend(["", f"Isolation Forest skipped: {schema_status['Isolation Forest']}"])

    # Rule comparisons are explicitly separate from learned models.
    label_for_rules = labels.reset_index(drop=True)
    source_counts = source_flows_per_minute(valid_rows)
    behavior = _source_minute_indicators(valid_rows)
    burst_threshold = int(generalized_artifact.get("source_burst_threshold", 0)) if generalized_artifact else 0
    burst_flow_flags = source_counts > burst_threshold
    multiport_flow_flags = (
        burst_flow_flags
        & (behavior["ports"] > PORT_DIVERSITY_THRESHOLD)
    )
    burst_metrics = evaluate_flow_rule(y_attack, burst_flow_flags)
    multiport_metrics = evaluate_flow_rule(y_attack, multiport_flow_flags)
    events, invalid_event_rows = build_event_table(
        valid_rows,
        label_for_rules,
        source_counts,
        behavior["ports"],
        behavior["destinations"],
        burst_threshold,
    )
    report_lines.extend([
        "",
        "Behavioral rules (heuristics; not model predictions):",
        f"  Source-flow threshold: >{burst_threshold:,} rows per source/minute; port-diversity threshold: >{PORT_DIVERSITY_THRESHOLD} unique destination ports per source/minute.",
        f"  Flow rows matched source-burst rule: {int(burst_flow_flags.sum()):,}",
        *summarize_binary("  Source-burst flow-level comparison:", burst_metrics),
        f"  Flow rows matched combined high-rate/multiport heuristic: {int(multiport_flow_flags.sum()):,}",
        *summarize_binary("  High-rate/multiport flow-level comparison:", multiport_metrics),
            f"  Source-minute groups: {len(events):,}; rows excluded from event grouping for missing source/time: {invalid_event_rows:,}",
    ])
    if not events.empty:
        alert_events = events.loc[
            events["source_burst_alert"] | events["high_rate_multiport_alert"]
        ]
        report_lines.extend([
            f"  Deduplicated alert events: {len(alert_events):,}.",
            "  Event-level precision/recall not reported: the dataset labels flows, "
            "not source-minute events or incidents.",
            "  Alert event evidence:",
        ])
        if alert_events.empty:
            report_lines.append("    none")
        for _, event in alert_events.iterrows():
            reasons = []
            if event["source_burst_alert"]:
                reasons.append("source-rate threshold exceeded")
            if event["high_rate_multiport_alert"]:
                reasons.append("destination-port-diversity threshold exceeded")
            report_lines.append(
                f"    source={event['source_ip']}; window={event['window_start']} to {event['window_end']}; "
                f"flows={int(event['flow_count']):,}; "
                f"unique_dst_ports={int(event['unique_destination_ports']):,}; "
                f"unique_dst_ips={int(event['unique_destination_ips']):,}; "
                f"top_dst={event['top_destination_ips']}; top_ports={event['top_destination_ports']}; "
                f"labels={event['labels']}; reason={'; '.join(reasons)}"
            )
    report_lines.append(
        "  These single-capture heuristic metrics describe this labeled file only; "
        "they do not calibrate or validate thresholds for other data."
    )

    report_lines.extend([
        "",
        "Count consistency:",
        f"  Rows evaluated by each available detector: {len(valid_rows):,}",
        f"  Label rows excluded (missing/blank/NaN): {len(frame) - len(valid_rows):,}",
        "  Prediction counts above sum to the rows evaluated for that detector; "
        "heuristic event counts are deduplicated source-minute groups and must not "
        "be added to flow-level model/anomaly counts.",
        "",
        "Limitations:",
        "  The local cleaned CIC-IDS2017 reference confirms matching feature names, "
        "similar duration scale, and an identical duplicate header, but does not "
        "record extractor version or prove every feature definition is identical.",
        "  No model was trained, tuned, or overwritten by this evaluation.",
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nReport saved to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    raise SystemExit(main())
