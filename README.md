# Hybrid AI Network Intrusion Detection System

A final year project that explores supervised and anomaly based detection on network flow records from the CIC-IDS2017 dataset. The project contains a Random Forest classifier, an Isolation Forest anomaly detector, offline evaluation scripts, a FastAPI service, a Streamlit demo, and a React dashboard.

## What it does

- Predicts a known traffic class with a trained Random Forest.
- Flags unusual flows with a trained Isolation Forest.
- Accepts CIC-IDS2017 compatible CSV files through the dashboard/API.
- Shows flow totals, predicted class counts, anomaly counts, and downloadable predictions.
- Includes capture-level evaluation scripts to examine performance on held-out attack captures.

The class prediction and anomaly flag are separate signals. An anomaly is not proof of an attack, and a predicted class is not a verified incident.

## Repository layout

- `src/hybrid_detector.py`: model loading, feature alignment, and inference.
- `src/train_*.py`: training scripts for the saved models.
- `src/evaluate_*.py`: held-out capture evaluation scripts.
- `api.py`: FastAPI upload and analysis service.
- `app.py`: Streamlit demo.
- `frontend/`: React + Vite dashboard.
- `data/`: local dataset files (not committed).
- `models/`: locally trained model files (not committed).
- `reports/`: generated evaluation and analysis outputs.

## Setup

Use Python 3.10 or newer and Node.js 20.19+ or 22.12+ for the frontend (Vite 8 requirement).

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm ci
cd ..
```

The trained model files and CIC-IDS2017 dataset are intentionally not tracked by Git. Put the expected model artifacts in `models/` and dataset CSVs under `data/raw/` or `data/processed/` as required by each script. See the CIC-IDS2017 source and license/terms before redistribution: https://www.unb.ca/cic/datasets/ids-2017.html

## Run the applications

Start the API from the repository root:

```powershell
uvicorn api:app --reload
```

Start the React dashboard in another terminal:

```powershell
cd frontend
npm run dev
```

The Streamlit demo can also be started from the repository root:

```powershell
streamlit run app.py
```

The API expects CSV headers compatible with the features used to train the saved model. Missing numeric features are imputed from training medians; extra columns and identifier/label columns are excluded.

## Checks

Run the model-agnostic unit tests:

```powershell
python -m unittest discover -s tests -v
```

Check the dashboard:

```powershell
cd frontend
npm run lint
npm run build
```

Run evaluation scripts only after placing the required CIC-IDS2017 files in the expected data directory. These scripts may take substantial time and write reports under `reports/`.

## Evaluation and limitations

The repository includes capture-level results, which are more informative about generalization than a random row split because flows from the same capture can be highly similar. The checked-in reports show materially different performance by evaluation design and attack capture:

| Evaluation | Rows | Accuracy | Balanced accuracy | Macro F1 | Attack recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random row split, multiclass | 566,108 | 99.94% | 89.14% | 91.22% | varies by class |
| Held-out DDoS capture, binary | 225,745 | 79.28% | 81.73% | 79.17% | 63.48% |
| Held-out PortScan capture, binary | 286,467 | 44.42% | 49.88% | 30.84% | 9.63% |

These rows represent different tasks and are not directly comparable. The random row split can place highly similar flows from the same capture in both train and test sets, so its high accuracy does not demonstrate generalization to a new capture. The capture-level results show that performance does not reliably transfer to unseen attack traffic. The current model is an academic prototype and should not be presented as a reliable operational detector.

Report the dataset version, capture split, preprocessing, class balance, per-class precision/recall/F1, confusion matrix, and threshold-selection procedure with every final result. Do not tune thresholds on the final held-out capture. Re-run the evaluation scripts from the raw data before treating checked-in reports as reproducible results.

## Defensive-use note

This project is for academic research and defensive cybersecurity education. Predictions require human review and should not be used alone to block traffic or make security decisions.
