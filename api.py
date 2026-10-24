
import os
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


MAX_FILE_SIZE = 500 * 1024 * 1024
MAX_ROWS = 300_000
PREVIEW_ROWS = 500

RESULTS_DIR = Path(tempfile.gettempdir()) / "sentinelai_nids_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="SentinelAI NIDS API",
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
        "preview_rows": PREVIEW_ROWS,
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

    input_path = None

    try:
        # Stream the upload to disk and enforce the file-size limit.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".csv",
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
                        detail="File is too large. Maximum size is 500 MB.",
                    )

                temp.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV is empty.",
            )

        try:
            df = pd.read_csv(
                input_path,
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
                        "Metrics compare predictions with CSV labels. "
                        "They estimate independent performance only if "
                        "this data was not used to train or tune the model."
                    ),
                }

        if actual_labels is not None:
            results_df["actual_class_for_evaluation"] = actual_labels

        class_counts = {
            str(name): int(count)
            for name, count in predicted_labels.value_counts().items()
        }

        anomaly_count = 0

        if "is_anomaly" in results_df.columns:
            anomaly_count = int(
                results_df["is_anomaly"]
                .fillna(False)
                .astype(bool)
                .sum()
            )

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
            "class_counts": class_counts,
            "columns": list(results_df.columns),
            "results": preview_df.to_dict(orient="records"),
            "preview_rows": min(PREVIEW_ROWS, len(results_df)),
            "download_url": f"/download/{analysis_id}",
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
        if input_path is not None:
            input_path.unlink(missing_ok=True)

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
