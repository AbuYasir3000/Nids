
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"

TEST_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

FILES = [
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

DROP_COLUMNS = [
    "Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"
]

def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Could not find: {path}")

    df = pd.read_csv(path, encoding="cp1252", low_memory=False)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError(f"No Label column in {filename}")

    df["Label"] = df["Label"].astype(str).str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)

    # Binary classification: BENIGN versus any attack.
    y = (df["Label"].str.upper() != "BENIGN").astype("int8")

    X = df.drop(
        columns=[c for c in DROP_COLUMNS if c in df.columns],
        errors="ignore",
    )

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    return X, y, df["Label"].copy()

print("Loading training captures...")
train_parts = []
labels_parts = []

for filename in FILES:
    print(f"  {filename}")
    X_part, y_part, _ = load_csv(filename)
    train_parts.append(X_part)
    labels_parts.append(y_part)

X_train_all = pd.concat(train_parts, ignore_index=True, sort=False)
y_train_all = pd.concat(labels_parts, ignore_index=True)

# Align columns and impute using training data only.
X_train_all = X_train_all.reindex(
    columns=sorted(X_train_all.columns)
)
imputer = SimpleImputer(strategy="median")
X_train_all = pd.DataFrame(
    imputer.fit_transform(X_train_all),
    columns=X_train_all.columns,
).astype("float32")

print(f"Training rows: {len(X_train_all):,}")
print(f"Features: {X_train_all.shape[1]}")

print("\nTraining binary Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
    min_samples_leaf=2,
)
model.fit(X_train_all, y_train_all)

print("\nLoading held-out DDoS capture...")
X_test, y_test, original_labels = load_csv(TEST_FILE)

# Match the exact training feature columns.
X_test = X_test.reindex(columns=X_train_all.columns)
X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_train_all.columns,
).astype("float32")

print("Predicting...")
predictions = model.predict(X_test)

actual_attack = y_test.to_numpy() == 1
detected_mask = actual_attack & (predictions == 1)
missed_mask = actual_attack & (predictions == 0)

print(f"\nActual attack flows: {actual_attack.sum():,}")
print(f"Detected attack flows: {detected_mask.sum():,}")
print(f"Missed attack flows: {missed_mask.sum():,}")
print("\nBinary confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# Keep original feature values, including zeros, for inspection.
raw_test, _, raw_labels = load_csv(TEST_FILE)
raw_test = raw_test.reindex(columns=X_train_all.columns)

detected_indices = np.flatnonzero(detected_mask)
missed_indices = np.flatnonzero(missed_mask)

rng = np.random.default_rng(42)
sample_size = min(10, len(detected_indices), len(missed_indices))

detected_sample = rng.choice(
    detected_indices, size=sample_size, replace=False
)
missed_sample = rng.choice(
    missed_indices, size=sample_size, replace=False
)

# Select informative features from the previous distribution analysis.
preferred_features = [
    "Packet Length Variance",
    "Packet Length Std",
    "Bwd Packet Length Std",
    "Bwd Packet Length Mean",
    "Total Length of Bwd Packets",
    "Total Backward Packets",
    "Packet Length Mean",
    "Average Packet Size",
    "Flow Duration",
    "Total Fwd Packets",
    "Flow Packets/s",
]

selected_features = [
    feature for feature in preferred_features
    if feature in raw_test.columns
]

# Also retain original labels and model predictions.
def make_sample(indices, group):
    result = raw_test.iloc[indices][selected_features].copy()
    result.insert(0, "original_label", raw_labels.iloc[indices].to_numpy())
    result.insert(1, "predicted_binary", predictions[indices])
    result.insert(2, "group", group)
    result.insert(3, "row_index", indices)
    return result.reset_index(drop=True)

comparison = pd.concat(
    [
        make_sample(detected_sample, "DETECTED_ATTACK"),
        make_sample(missed_sample, "MISSED_ATTACK"),
    ],
    ignore_index=True,
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
output = REPORT_DIR / "ddos_detected_vs_missed_samples.csv"
comparison.to_csv(output, index=False)

print("\nSAMPLED FLOWS (10 detected + 10 missed, where available)")
print(comparison.to_string(index=False))
print(f"\nSaved sample comparison to:\n{output}")
