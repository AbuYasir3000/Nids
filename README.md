# Hybrid AI Network Intrusion Detection System

A final year project that explores supervised and anomaly based detection on network flow records from the CIC-IDS2017 dataset. The project contains a Random Forest classifier, an Isolation Forest anomaly detector, offline evaluation scripts, a FastAPI service, a Streamlit demo, and a React dashboard.

## What it does

- Predicts a known traffic class with a trained Random Forest.
- Flags unusual flows with a trained Isolation Forest.
- Accepts CIC-IDS2017 compatible CSV files through the dashboard/API.
- Accepts PCAP, PCAPNG, and `.pcap10` captures after optional CICFlowMeter conversion.
- Shows flow totals, predicted class counts, anomaly counts, and downloadable predictions.
- Includes capture-level evaluation scripts to examine performance on held-out attack captures.

The Random Forest class, generalized binary classifier, source-burst heuristic, and Isolation Forest anomaly flag are separate signals. Learned model counts exclude source-burst and port-diversity heuristics; heuristics are shown separately because they can match benign high-volume traffic and do not identify an attack family. The source-rate threshold was calibrated on CIC-IDS2017 and is not validated for every dataset. Counts are flow-level, not deduplicated incidents. Calibration and development-capture results are documented in `reports/port_diversity_rule_evaluation.txt`.

## Repository layout

- `src/hybrid_detector.py`: model loading, feature alignment, and inference.
- `src/train_*.py`: training scripts for the saved models.
- `src/evaluate_*.py`: held-out capture evaluation scripts.
- `api.py`: FastAPI upload and analysis service.
- `src/pcap_converter.py`: PCAP-to-flow conversion and feature-column alignment.
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

PCAP, PCAPNG, and `.pcap10` uploads need the optional CICFlowMeter converter. Use Python 3.12 or newer for this add-on:

```powershell
python -m pip install -r requirements-pcap.txt
```

Restart the API after installing it. CSV uploads continue to work without this optional package. PCAP conversion runs on the API server, uses CICFlowMeter to create flow records, and has a 4-hour conversion timeout per capture. Classic PCAP files using Linux cooked capture v2 are adapted to Ethernet framing before extraction.

Uploads may be up to 20 GiB; converted flow CSV output may be up to 40 GiB. There is no fixed flow-count limit. Large captures need substantial free disk space in the system temporary directory and enough memory to hold the flow data and analysis results. The Streamlit upload limit is configured in `.streamlit/config.toml`.

The trained model files and CIC-IDS2017 dataset are intentionally not tracked by Git. Put the expected model artifacts in `models/` and dataset CSVs under `data/raw/` or `data/processed/` as required by each script. See the CIC-IDS2017 source and license/terms before redistribution: https://www.unb.ca/cic/datasets/ids-2017.html

## Run the applications

To enable the cross-capture attack alert, train its optional local artifact first. This takes several minutes and requires the CIC-IDS2017 CSV files under `data/raw/`:

```powershell
python src/train_generalized_attack_model.py
```

The trainer excludes the DDoS calibration capture and the PortScan evaluation capture from fitting. It chooses its alert threshold from benign rows in the DDoS capture, targeting a 0.1% calibration false-positive rate. Evaluate the saved artifact with the frozen threshold using:

```powershell
python src/evaluate_generalized_portscan.py
```

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

CSV and PCAP uploads must provide all 80 saved model features. CIC CSV time fields are kept in their existing microsecond units. PCAP conversion maps Python CICFlowMeter's seconds to the model's microseconds for duration, IAT, active, and idle fields; rates remain per second. The duplicate `Fwd Header Length.1` training column is copied from its verified identical `Fwd Header Length` value when the extractor emits it only once. Unsupported schemas are rejected instead of silently filling entire missing columns with training medians. Individual invalid or missing values are handled by the saved training medians, and each result row reports how many feature values were imputed. PCAP flows have no ground-truth labels, so uploads do not produce label-based accuracy metrics.

The saved artifacts do not record an extractor version or a complete feature-definition manifest. CIC-IDS2017's official documentation says its flow features were generated with CICFlowMeter; that tool creates bidirectional flows and defines forward direction from the first packet. Therefore matching headers alone does not establish exact semantic equivalence for another CSV's flow timeouts, packet direction, preprocessing, or zero-duration rate policy. The training CSVs contain zero-duration rows whose `Flow Bytes/s`/`Flow Packets/s` values became invalid and were median-imputed. A separate capture with many zero-duration rows can therefore have a materially different imputation pattern. Do not replace these values with zero or alter model thresholds without a separately trained and validated artifact.

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
| Held-out PortScan capture, binary baseline | 286,467 | 44.42% | 49.88% | 30.84% | 0.10% |
| Generalized classifier alone, PortScan development evaluation | 286,467 | 44.51% | 49.96% | 31.00% | 0.24% |
| Historical classifier-plus-source-burst union, PortScan development evaluation (retired; not a current alert metric) | 286,467 | 98.72% | 98.82% | 98.71% | 97.96% |

These rows represent different tasks and are not directly comparable. The random row split can place highly similar flows from the same capture in both train and test sets, so its high accuracy does not demonstrate generalization to a new capture. The classifier alone transfers poorly to PortScan. The source-burst rule counts flows by source IP and minute; its threshold is calibrated using benign rows from the separate DDoS capture. The classifier-plus-rule union above is a historical development experiment and is retired from live combined alert counts; current outputs keep model flags and deduplicated behavioral events separate. Its PortScan result is exploratory, not an independent generalization result or evidence of operational readiness, because this capture informed development. The model-only threshold targeted a 0.1% calibration false-positive rate; observed DDoS calibration false-positive rate was 0.093% and DDoS attack recall was 20.70%.

Report the dataset version, capture split, preprocessing, class balance, per-class precision/recall/F1, confusion matrix, and threshold-selection procedure with every final result. Do not tune thresholds on the final held-out capture. The available local data contains only CIC-IDS2017 captures, all of which are used in development or calibration; independent validation therefore remains outstanding. Re-run the evaluation scripts from the raw data before treating checked-in reports as reproducible results.

`python src/diagnose_generalized_model.py` reproduces the saved binary model's probability summaries, a temporary capture-separated CIC train/calibration/validation experiment with a precision-recall curve, and frozen behavioral-rule diagnostics for all local CIC captures. Add `--udplag-csv PATH` to include an optional labeled UDP-lag CSV in score diagnostics; it is never used for tuning. The script writes reports/CSV diagnostics only and never writes model artifacts. The chosen validation capture is not an independent final test, and the historical DDoS/PortScan reports mean no available capture is treated as untouched final evidence.

## Defensive-use note

This project is for academic research and defensive cybersecurity education. Predictions require human review and should not be used alone to block traffic or make security decisions.
