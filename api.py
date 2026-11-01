
import os
import logging
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.hybrid_detector import HybridNIDS
from src.pcap_converter import convert_pcap_to_csv, normalize_flow_columns
from src.traffic_features import (
    PORT_DIVERSITY_THRESHOLD,
    source_minute_event_table,
    target_flood_analysis,
)


# 20 GiB upload limit. Converted flow CSVs can be larger than their captures.
MAX_FILE_SIZE = 20 * 1024 * 1024 * 1024
MAX_CONVERTED_FILE_SIZE = 40 * 1024 * 1024 * 1024
PREVIEW_ROWS = 500
PCAP_SUFFIXES = {".pcap", ".pcapng", ".pcap10"}

RESULTS_DIR = Path(tempfile.gettempdir()) / "sentinelai_nids_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Argus NIDS API",
    description="Hybrid AI Network Intrusion Detection System",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

detector = None
detector_error = None

try:
    detector = HybridNIDS()
    print("Argus models loaded successfully.")
except Exception as exc:
    detector_error = f"{type(exc).__name__}: {exc}"
    print(f"Model loading failed: {detector_error}")


@app.get("/")
def home():
    return {
        "name": "Argus NIDS API",
        "status": "online",
        "detector_ready": detector is not None,
        "generalized_model_ready": bool(getattr(detector, "generalized_model_ready", False)),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "max_rows_per_analysis": None,
        "preview_rows": PREVIEW_ROWS,
        "detector_error": detector_error,
    }


@app.get("/health")
def health():
    return {
        "api_status": "online",
        "detector_ready": detector is not None,
        "generalized_model_ready": bool(getattr(detector, "generalized_model_ready", False)),
        "detector_error": detector_error,
    }


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".csv" and suffix not in PCAP_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV or a PCAP/PCAPNG capture file.",
        )

    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=f"Detection models unavailable: {detector_error}",
        )

    input_path = None
    conversion_dir = None

    try:
        # Stream the upload to disk and enforce the file-size limit.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            delete=False,
        ) as temp:
            input_path = Path(temp.name)
            total_bytes = 0

            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File is too large. Maximum size is 20 GiB.",
                    )

                temp.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        data_path = input_path
        if suffix in PCAP_SUFFIXES:
            conversion_dir = Path(tempfile.mkdtemp(prefix="sentinelai_pcap_"))
            data_path = conversion_dir / "flows.csv"
            try:
                convert_pcap_to_csv(input_path, data_path)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except TimeoutError as exc:
                raise HTTPException(status_code=408, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if data_path.stat().st_size > MAX_CONVERTED_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail="Converted flow data exceeds the 40 GiB processing limit.",
                )

        try:
            df = pd.read_csv(
                data_path,
                encoding_errors="replace",
                low_memory=False,
            )
        except Exception as exc:
            source_type = "converted flow CSV" if suffix != ".csv" else "CSV"
            raise HTTPException(
                status_code=400,
                detail=f"Could not read {source_type}: {exc}",
            ) from exc

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The file contains no flow records.",
            )

        df = normalize_flow_columns(
            df,
            detector.feature_columns,
            convert_time_units=suffix in PCAP_SUFFIXES,
        )
        missing_features = [
            feature for feature in detector.feature_columns if feature not in df.columns
        ]
        if missing_features:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Input is missing {len(missing_features)} required model "
                    f"features: {', '.join(missing_features[:8])}. Provide a "
                    "CIC-IDS-compatible flow CSV or PCAP."
                ),
            )

        # Preserve labels separately; never pass them as prediction targets
        # or features to the detector.
        label_column = next(
            (
                col for col in df.columns
                if str(col).strip().lower() == "label"
            ),
            None,
        )

        actual_labels = None

        if label_column is not None:
            actual_labels = (
                df[label_column]
                .astype("string")
                .str.strip()
                .reset_index(drop=True)
            )

        model_features = df.drop(columns=[label_column], errors="ignore")
        predictions = detector.predict(model_features)

        if not isinstance(predictions, pd.DataFrame):
            predictions = pd.DataFrame(predictions)

        results_df = predictions.copy().reset_index(drop=True)

        if len(results_df) != len(df):
            raise RuntimeError(
                f"Expected {len(df)} predictions, got {len(results_df)}."
            )

        if "predicted_class" not in results_df.columns:
            raise RuntimeError(
                "Detector output does not contain 'predicted_class'."
            )

        predicted_labels = (
            results_df["predicted_class"]
            .astype("string")
            .str.strip()
        )

        # Optional evaluation against labels in the uploaded CSV.
        evaluation = None

        if actual_labels is not None:
            valid = (
                actual_labels.notna()
                & actual_labels.ne("")
                & predicted_labels.notna()
            )

            y_true = actual_labels[valid].astype(str)
            y_pred = predicted_labels[valid].astype(str)

            if len(y_true) > 0:
                classes = sorted(set(y_true) | set(y_pred))
                rf_classes = set(
                    map(str, getattr(getattr(detector, "rf_model", None), "classes_", []))
                )
                actual_support = y_true.value_counts().to_dict()
                unsupported_actual = {
                    str(name): int(count)
                    for name, count in actual_support.items()
                    if str(name) not in rf_classes
                }

                evaluation = {
                    "evaluated_rows": int(len(y_true)),
                    "accuracy": float(
                        accuracy_score(y_true, y_pred)
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_true, y_pred)
                    ),
                    "precision_macro": float(
                        precision_score(
                            y_true,
                            y_pred,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "recall_macro": float(
                        recall_score(
                            y_true,
                            y_pred,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "f1_macro": float(
                        f1_score(
                            y_true,
                            y_pred,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "classes": classes,
                    "unsupported_actual_label_counts": unsupported_actual,
                    "supported_actual_rows": int(
                        sum(count for name, count in actual_support.items() if str(name) in rf_classes)
                    ),
                    "confusion_matrix": confusion_matrix(
                        y_true,
                        y_pred,
                        labels=classes,
                    ).tolist(),
                    "classification_report": classification_report(
                        y_true,
                        y_pred,
                        labels=classes,
                        output_dict=True,
                        zero_division=0,
                    ),
                    "note": (
                        "Strict label comparison. Actual labels absent from the "
                        "Random Forest training classes are counted as unmatched, "
                        "not mapped to another family. This is an evaluation of "
                        "this uploaded capture, not independent performance if it "
                        "was used to train or tune the model."
                    ),
                }

        generalization_evaluation = None
        if (
            actual_labels is not None
            and getattr(detector, "generalized_model_ready", False)
            and "is_generalized_attack" in results_df.columns
        ):
            valid_generalization = (
                actual_labels.notna()
                & actual_labels.ne("")
                & results_df["is_generalized_attack"].notna()
            )
            if valid_generalization.any():
                y_true_attack = (
                    actual_labels[valid_generalization]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .ne("BENIGN")
                    .astype(int)
                )
                y_pred_attack = (
                    results_df.loc[
                        valid_generalization, "is_generalized_attack"
                    ]
                    .astype(bool)
                    .astype(int)
                )
                generalization_evaluation = {
                    "evaluated_rows": int(len(y_true_attack)),
                    "accuracy": float(accuracy_score(y_true_attack, y_pred_attack)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_true_attack, y_pred_attack)
                    ),
                    "precision_attack": float(
                        precision_score(y_true_attack, y_pred_attack, zero_division=0)
                    ),
                    "recall_attack": float(
                        recall_score(y_true_attack, y_pred_attack, zero_division=0)
                    ),
                    "f1_attack": float(
                        f1_score(y_true_attack, y_pred_attack, zero_division=0)
                    ),
                    "confusion_matrix": confusion_matrix(
                        y_true_attack, y_pred_attack, labels=[0, 1]
                    ).tolist(),
                    "note": (
                        "Binary BENIGN-versus-ATTACK results for the learned "
                        "generalized classifier only. The source-burst heuristic "
                        "is reported separately and is not counted as a model "
                        "prediction. Do not treat metrics as independent if this "
                        "capture was used for training, calibration, or model selection."
                    ),
                }
        if actual_labels is not None:
            results_df["actual_class_for_evaluation"] = actual_labels

        class_counts = {
            str(name): int(count)
            for name, count in predicted_labels.value_counts().items()
        }

        # Log aggregate diagnostics only; never include flow values or identifiers.
        imputed_rows = 0
        imputed_feature_cells = 0
        if "imputed_feature_count" in results_df.columns:
            imputation_counts = pd.to_numeric(
                results_df["imputed_feature_count"], errors="coerce"
            ).fillna(0)
            imputed_rows = int(imputation_counts.gt(0).sum())
            imputed_feature_cells = int(imputation_counts.sum())

        anomaly_count = 0

        if "is_anomaly" in results_df.columns:
            anomaly_count = int(
                results_df["is_anomaly"]
                .fillna(False)
                .astype(bool)
                .sum()
            )

        logger.info(
            "CSV analysis complete: rows=%d predicted_classes=%s anomalies=%d "
            "imputed_rows=%d imputed_feature_cells=%d",
            len(df),
            class_counts,
            anomaly_count,
            imputed_rows,
            imputed_feature_cells,
        )

        model_generalized_attack_count = 0
        if "is_model_generalized_attack" in results_df.columns:
            model_generalized_attack_count = int(
                results_df["is_model_generalized_attack"]
                .fillna(False)
                .astype(bool)
                .sum()
            )
        elif "is_generalized_attack" in results_df.columns:
            model_generalized_attack_count = int(
                results_df["is_generalized_attack"]
                .fillna(False)
                .astype(bool)
                .sum()
            )
        source_burst_alert_count = 0
        if "is_source_burst_alert" in results_df.columns:
            source_burst_alert_count = int(
                results_df["is_source_burst_alert"]
                .fillna(False)
                .astype(bool)
                .sum()
            )
        known_attack_flags = predicted_labels.str.upper().ne("BENIGN").fillna(False)
        any_flow_flag = known_attack_flags.copy()
        for flag_column in ("is_generalized_attack", "is_anomaly"):
            if flag_column in results_df.columns:
                any_flow_flag |= (
                    results_df[flag_column].fillna(False).astype(bool)
                )
        flagged_flow_count = int(any_flow_flag.sum())
        high_rate_multiport_flow_count = int(
            results_df.get(
                "is_high_rate_multiport_activity",
                pd.Series(False, index=results_df.index),
            )
            .fillna(False)
            .astype(bool)
            .sum()
        )
        source_burst_flow_count = source_burst_alert_count
        behavioral_events = source_minute_event_table(
            df,
            results_df.get(
                "unique_destination_ports_per_source_minute",
                pd.Series(1, index=df.index),
            ).to_numpy(),
            results_df.get(
                "unique_destination_ips_per_source_minute",
                pd.Series(1, index=df.index),
            ).to_numpy(),
            getattr(detector, "source_burst_threshold", None),
        )
        event_source_burst_flows = int(
            behavioral_events.loc[
                behavioral_events.get(
                    "source_burst_rule", pd.Series(False, index=behavioral_events.index)
                ),
                "flow_count",
            ].sum()
        ) if not behavioral_events.empty else 0
        event_multiport_flows = int(
            behavioral_events.loc[
                behavioral_events.get(
                    "high_rate_multiport_rule",
                    pd.Series(False, index=behavioral_events.index),
                ),
                "flow_count",
            ].sum()
        ) if not behavioral_events.empty else 0
        if event_source_burst_flows != source_burst_flow_count:
            raise RuntimeError(
                "Source-burst event flow totals do not match per-row detector flags."
            )
        if event_multiport_flows != high_rate_multiport_flow_count:
            raise RuntimeError(
                "Multiport event flow totals do not match per-row detector flags."
            )
        source_burst_event_count = int(
            behavioral_events.get(
                "source_burst_rule", pd.Series(dtype=bool)
            ).sum()
        )
        high_rate_multiport_event_count = int(
            behavioral_events.get(
                "high_rate_multiport_rule", pd.Series(dtype=bool)
            ).sum()
        )
        behavioral_alerts = behavioral_events.to_dict(orient="records")
        # Separate rule: floods of unanswered flows aimed at one target.
        target_flood_flags, target_flood_events = target_flood_analysis(df)
        target_flood_flow_count = int(target_flood_flags.sum())
        target_flood_alerts = target_flood_events.to_dict(orient="records")
        behavioral_findings = {
            "classification": (
                "Source-minute behavioral signals observed"
                if behavioral_alerts
                else (
                    "Flood of unanswered flows to one target observed"
                    if target_flood_flow_count
                    else "No source-minute behavioral signals observed"
                )
            ),
            "target_flood_flow_count": target_flood_flow_count,
            "target_flood_events": target_flood_alerts,
            "source_burst_event_count": source_burst_event_count,
            "source_burst_flow_count": source_burst_flow_count,
            "high_rate_multiport_event_count": high_rate_multiport_event_count,
            "high_rate_multiport_flow_count": high_rate_multiport_flow_count,
            "destination_port_diversity_threshold": PORT_DIVERSITY_THRESHOLD,
            "source_flow_burst_threshold": getattr(
                detector, "source_burst_threshold", None
            ),
            "events": behavioral_alerts,
            "note": (
                "One event is one source-IP/minute key; flow_count reports the "
                "associated rows and is not an incident count. Rules use source "
                "rate and destination-port diversity, do not identify an attack "
                "family, and are not included in learned-model counts. Thresholds "
                "were calibrated on CIC-IDS2017 and are not validated for every "
                "uploaded dataset."
            ),
        }
        # Store complete results on disk. Return only a small preview in JSON.
        analysis_id = str(uuid4())
        results_path = RESULTS_DIR / f"{analysis_id}.csv"

        results_df.to_csv(
            results_path,
            index=False,
            na_rep="",
        )

        preview_df = results_df.head(PREVIEW_ROWS).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        preview_df = preview_df.astype(object)
        preview_df = preview_df.where(pd.notna(preview_df), None)

        return {
            "analysis_id": analysis_id,
            "filename": Path(file.filename).name,
            "total_flows": int(len(df)),
            "anomaly_count": anomaly_count,
            "generalized_attack_count": model_generalized_attack_count,
            "model_generalized_attack_count": model_generalized_attack_count,
            "source_burst_event_count": source_burst_event_count,
            "source_burst_flow_count": source_burst_flow_count,
            "high_rate_multiport_event_count": high_rate_multiport_event_count,
            "high_rate_multiport_flow_count": high_rate_multiport_flow_count,
            "target_flood_flow_count": target_flood_flow_count,
            "flagged_flow_count": flagged_flow_count,
            "behavioral_findings": behavioral_findings,
            "counting_note": (
                "All counts are flow-level. flagged_flow_count is the union of "
                "known-class model flags, generalized model flags, and anomalies. "
                "Source-rate and port-diversity heuristics are separate and are "
                "not included. These are not deduplicated incidents."
            ),
            "class_counts": class_counts,
            "columns": list(results_df.columns),
            "results": preview_df.to_dict(orient="records"),
            "preview_rows": min(PREVIEW_ROWS, len(results_df)),
            "download_url": f"/download/{analysis_id}",
            "evaluation": evaluation,
            "generalization_evaluation": generalization_evaluation,
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Input is incompatible with the saved model schemas: {exc}",
        ) from exc
    except Exception as exc:
        print(f"Analysis error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        if input_path is not None:
            input_path.unlink(missing_ok=True)
        if conversion_dir is not None:
            shutil.rmtree(conversion_dir, ignore_errors=True)

        await file.close()


@app.get("/download/{analysis_id}")
def download_results(analysis_id: str):
    try:
        safe_id = str(uuid4()) if False else str(__import__("uuid").UUID(analysis_id))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid analysis ID.",
        )

    results_path = RESULTS_DIR / f"{safe_id}.csv"

    if not results_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Results file not found. Please analyze the CSV again.",
        )

    return FileResponse(
        path=results_path,
        media_type="text/csv",
        filename="sentinelai_detection_results.csv",
    )
