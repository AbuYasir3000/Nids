# Hybrid AI Network Intrusion Detection System

A final year project that explores supervised and anomaly based detection on network flow records from the CIC-IDS2017 dataset. The project contains a Random Forest classifier, an Isolation Forest anomaly detector, offline evaluation scripts, a FastAPI service, and a React dashboard.

## What it does

- Predicts a known traffic class with a trained Random Forest.
- Flags unusual flows with a trained Isolation Forest.
- Accepts CIC-IDS2017 compatible CSV files through the dashboard/API.
- Accepts PCAP, PCAPNG, and `.pcap10` captures after optional CICFlowMeter conversion.
- Shows flow totals, predicted class counts, anomaly counts, and downloadable predictions.
- Includes capture-level evaluation scripts to examine performance on held-out attack captures.

The Random Forest class, generalized binary classifier, source-burst heuristic, and Isolation Forest anomaly flag are separate signals. Learned model counts exclude source-burst and port-diversity heuristics; heuristics are shown separately because they can match benign high-volume traffic and do not identify an attack family. The source-rate threshold was calibrated on CIC-IDS2017 and is not validated for every dataset. Counts are flow-level, not deduplicated incidents. Calibration and development-capture results are documented in `reports/port_diversity_rule_evaluation.txt`.

## PCAP feature compatibility and the v2 Random Forest

Python CICFlowMeter (used for PCAP uploads) measures packet length as the whole Ethernet frame and header length from the IP header, while CIC-IDS2017 (Java CICFlowMeter) uses payload bytes and the TCP/UDP header. A bare SYN probe is 58 bytes in one and 0 in the other, which pushed scan traffic out of the training distribution. `src/cic_compat.py` patches CICFlowMeter in-process (called from `convert_pcap_to_csv`) so lengths and header sizes follow CIC-IDS2017. CSV uploads are unaffected.

That alone did not make the original Random Forest detect a scan from another tool, because the model also depends on CIC-specific artifacts (for example its flag-count columns). `src/train_rf_with_pcap.py` retrains the Random Forest on a CIC-IDS2017 sample plus a labelled capture and writes `models/random_forest_v2.joblib`; `hybrid_detector.py` uses it when present (delete or rename it to use `random_forest.joblib`). Results are in `reports/random_forest_v2_report.txt`.

```powershell
python src/train_rf_with_pcap.py --sample cicids2017_sample.csv.gz --flows flows.csv --scanner-ip 192.168.2.5
```

Limits: capture labels are assumed (source = scanner IP is a scan, everything else benign), the capture is one scanner on one network so held-out-host scores are optimistic, and the sample is a subset of CIC-IDS2017. `src/retrain_aux_models.py` is an experiment, not used by the app: a retrained binary classifier caught the scan but lost cross-capture DDoS recall, and the Isolation Forest did not change (`reports/auxiliary_models_v2_report.txt`).

## Check the accuracy of the project

`check_accuracy.py` runs every detector on labelled data and prints one table (attack detection rate, false alarm rate, precision, accuracy) plus a per-attack-type breakdown. It saves `reports/accuracy_summary.md` and `.csv`.

```powershell
.venv\Scripts\python.exe check_accuracy.py --csv path\to\labelled_flows.csv
.venv\Scripts\python.exe check_accuracy.py --pcap scan_A.pcap --attacker-ip 192.168.2.5
```

A CSV needs a `Label` column. A pcap has no labels, so `--attacker-ip` says which source IP is the attacker (assumption). Scores on data the models were trained on are optimistic; use a capture or CSV the models have never seen for an honest number.

## Test captures

`make_test_pcaps.py` generates labelled synthetic captures in `test_pcaps/` (benign only, a connect scan, a SYN flood, a UDP flood, and benign traffic with a hidden scan) plus `ground_truth.csv` with the attacker IP for each. Nothing is sent on a network; the files are written to disk. Upload one in the dashboard or run `check_accuracy.py --pcap test_pcaps/<file> --attacker-ip <ip>` (several IPs may be separated by commas). The traffic is simulated, so it is a sanity test, not a substitute for real captures.

## Flood rule (unanswered flows to one target)

`target_flood_analysis` in `src/traffic_features.py` flags a destination that receives more than 1,000 flows in one minute when at least 90% of them get no reply packet and have at most 2 forward packets, which is the shape of a SYN or UDP flood. The size check means a capture that only recorded one direction of traffic (replies missing) does not trip the rule: on a simulated busy one-way capture of 9,276 normal flows the rule without the size check flagged 6,035 flows and with it flagged none. It is a behavioral rule, shown with the other behavioral signals and never counted as a model prediction. The thresholds are development defaults and are NOT calibrated on benign data (the CIC sample is down-sampled, so per-target counts cannot be calibrated from it). Expect false alarms on, for example, a scan of a firewalled host, and validate on your own network before relying on it. On the synthetic `test_pcaps` it catches the SYN and UDP floods and raises no false alarms on the benign, scan and mixed captures.

## Random Forest v3 (adds UDPLag)

`src/train_rf_v3.py` extends v2 with a sample of CIC-DDoS2019 `UDPLag.csv`. `hybrid_detector.py` loads `models/random_forest_v3.joblib` first, then v2, then the original `random_forest.joblib`; rename a file to fall back.

```powershell
python src/train_rf_v3.py --sample cicids2017_sample_ids2017.csv.gz --flows flows.csv --scanner-ip 192.168.2.5 --udplag udplag_sample.csv.gz
```

UDPLag rows are split by time, so the test uses minutes the model never saw. Results are in `reports/random_forest_v3_report.txt`: on the unseen UDPLag minute v3 detects 99.8% of attack flows with 0.13% benign false alarms (original and v2: 0%). Nothing is lost elsewhere: with v2 and v3 both trained without the test rows, the CIC-IDS2017 holdout scores 99.46% accuracy / 0.20% benign false alarms for v2 and 99.49% / 0.17% for v3, and the scan capture's held-out hosts score 100% for both. Most remaining CIC errors are Web Attack Brute Force vs XSS mix-ups, a near-identical pair in that dataset. The saved `random_forest_v2.joblib` and `random_forest_v3.joblib` are refit on all the data, so do not score them on the same rows to compare them.

Two caveats worth stating in any write-up. First, in `UDPLag.csv` only the first burst (13:04) is real UDP; about 96% of the labelled attack flows (13:29-13:30) are TCP with no payload and no reply, so the label covers two different kinds of traffic. A strict split that trains only on the UDP burst detects almost none of the TCP bursts (`reports/random_forest_v3_report_strict_time_split.txt`). Second, the data is one capture on one network and is down-sampled, so the numbers do not show that other UDP attacks are detected.

## Repository layout

- `src/hybrid_detector.py`: model loading, feature alignment, and inference.
- `src/train_*.py`: training scripts for the saved models.
- `src/evaluate_*.py`: held-out capture evaluation scripts.
- `api.py`: FastAPI upload and analysis service.
- `src/pcap_converter.py`: PCAP-to-flow conversion and feature-column alignment.
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

Uploads may be up to 20 GiB; converted flow CSV output may be up to 40 GiB. There is no fixed flow-count limit. Large captures need substantial free disk space in the system temporary directory and enough memory to hold the flow data and analysis results.

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


### Important: external dataset evaluation

The Random Forest predicts only the attack families present in its saved `classes_`
from CIC-IDS2017. A dataset with a new label (for example, `UDP-lag` or `WebDDoS`)
will not become a supported class merely because its columns have similar names.
Review the API's `unsupported_actual_label_counts` before interpreting multiclass
metrics on another capture. The generalized binary model and Isolation Forest are
separate signals: anomaly flags are not confirmed attacks, and their counts must not
be added together as if they were one classifier's predictions.

The serialized models were created with scikit-learn 1.9.1. Install the pinned
version in `requirements.txt` before loading them; scikit-learn does not guarantee
compatibility when loading joblib/pickle artifacts created by a different version.
