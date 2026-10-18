
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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


BASE_DIR = Path(__file__).resolve().parent

MAX_FILE_SIZE = 500 * 1024 * 1024
MAX_ROWS = 300_000

app = FastAPI(
    title="SentinelAI NIDS API",
    description="Hybrid AI Network Intrusion Detection System",
    version="1.1.0",
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
    print("SentinelAI models loaded successfully.")
except Exception as exc:
    detector_error = f"{type(exc).__name__}: {exc}"
    print(f"Model loading failed: {detector_error}")


@app.get("/")
def home():
    return {
        "name": "SentinelAI NIDS API",
        "status": "online",
        "detector_ready": detector is not None,
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "max_rows_per_analysis": MAX_ROWS,
        "detector_error": detector_error,
    }


@app.get("/health")
def health():
    return {
        "api_status": "online",
        "detector_ready": detector is not None,
        "detector_error": detector_error,
    }


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=f"Detection models unavailable: {detector_error}",
        )

    try:
        contents = await file.read(MAX_FILE_SIZE + 1)

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum size is 500 MB.",
            )

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV is empty.",
            )

        try:
            df = pd.read_csv(
                BytesIO(contents),
                encoding_errors="replace",
                low_memory=False,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read CSV: {exc}",
            ) from exc

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The CSV contains no data rows.",
            )

        if len(df) > MAX_ROWS:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Maximum {MAX_ROWS:,} rows per analysis. "
                    f"Your file contains {len(df):,} rows."
                ),
            )

        # Save labels separately. They are never passed to the detector.
        label_column = next(
            (column for column in df.columns
             if str(column).strip().lower() == "label"),
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

        # Inference happens before any comparison with actual labels.
        predictions = detector.predict(df)

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
            .astype(str)
            .str.strip()
        )

        # Evaluation is optional and only available when labels exist.
        evaluation = None

        if actual_labels is not None:
            valid = (
                actual_labels.notna()
                & actual_labels.ne("")
                & predicted_labels.notna()
            )

            y_true = actual_labels[valid].astype(str)
            y_pred = predicted_labels[valid]

            if len(y_true) > 0:
                # Include all actual and predicted classes in the matrix.
                classes = sorted(set(y_true) | set(y_pred))

                evaluation = {
                    "evaluated_rows": int(len(y_true)),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_true, y_pred)
                    ),
                    "precision_macro": float(
                        precision_score(
                            y_true, y_pred,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "recall_macro": float(
                        recall_score(
                            y_true, y_pred,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "f1_macro": float(
                        f1_score(
                            y_true, y_pred,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "classes": classes,
                    "confusion_matrix": confusion_matrix(
                        y_true, y_pred, labels=classes
                    ).tolist(),
                    "classification_report": classification_report(
                        y_true,
                        y_pred,
                        labels=classes,
                        output_dict=True,
                        zero_division=0,
                    ),
                    "note": (
                        "These metrics compare predicted classes with the "
                        "CSV labels. They are only an independent performance "
                        "estimate if this file was not used to train or tune "
                        "the model."
                    ),
                }

        # Include the true label in exported results only after inference.
        if actual_labels is not None:
            results_df["actual_class_for_evaluation"] = actual_labels

        class_counts = (
            predicted_labels
            .value_counts()
            .to_dict()
        )

        anomaly_count = 0

        if "is_anomaly" in results_df.columns:
            anomaly_count = int(
                results_df["is_anomaly"]
                .fillna(False)
                .astype(bool)
                .sum()
            )

        # Convert NaN and infinite values to JSON-safe nulls.
        results_df = results_df.replace([np.inf, -np.inf], np.nan)
        results_df = results_df.astype(object)
        results_df = results_df.where(pd.notna(results_df), None)

        return {
            "filename": file.filename,
            "total_flows": int(len(df)),
            "anomaly_count": anomaly_count,
            "class_counts": {
                str(name): int(count)
                for name, count in class_counts.items()
            },
            "columns": list(results_df.columns),
            "results": results_df.to_dict(orient="records"),
            "evaluation": evaluation,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Analysis error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        await file.close()
