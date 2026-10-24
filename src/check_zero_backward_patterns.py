
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

FILES = [
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

TEST_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

FEATURES = [
    "Total Backward Packets",
    "Total Length of Bwd Packets",
    "Packet Length Variance",
    "Packet Length Mean",
]

def read_file(filename):
    path = DATA_DIR / filename
    df = pd.read_csv(path, encoding="cp1252", low_memory=False)
    df.columns = df.columns.str.strip()

    for feature in FEATURES:
        if feature not in df.columns:
            raise ValueError(f"Missing feature: {feature} in {filename}")
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df["Label"] = df["Label"].astype(str).str.strip()
    return df

print("Loading the seven training captures...")
training = []

for filename in FILES:
    print(" ", filename)
    df = read_file(filename)
    training.append(df[["Label"] + FEATURES])

train = pd.concat(training, ignore_index=True)

print("\nLoading held-out DDoS capture...")
test = read_file(TEST_FILE)

# Focus on whether backward traffic is absent.
backward_feature = "Total Backward Packets"
variance_feature = "Packet Length Variance"

train["zero_backward"] = train[backward_feature].eq(0)
train["zero_variance"] = train[variance_feature].eq(0)

test["zero_backward"] = test[backward_feature].eq(0)
test["zero_variance"] = test[variance_feature].eq(0)

print("\nTRAINING CAPTURES: ZERO-BACKWARD RATE BY LABEL")
summary = (
    train.groupby("Label")
    .agg(
        flows=("Label", "size"),
        zero_backward=("zero_backward", "sum"),
        zero_variance=("zero_variance", "sum"),
    )
)

summary["zero_backward_pct"] = (
    100 * summary["zero_backward"] / summary["flows"]
)
summary["zero_variance_pct"] = (
    100 * summary["zero_variance"] / summary["flows"]
)

print(
    summary.sort_values("zero_backward_pct", ascending=False)
    .round(2)
    .to_string()
)

print("\nHELD-OUT DDOS CAPTURE")
for name, mask in [
    ("All DDoS flows", pd.Series(True, index=test.index)),
    ("DDoS with zero backward packets", test["zero_backward"]),
    ("DDoS with nonzero backward packets", ~test["zero_backward"]),
]:
    subset = test.loc[mask]
    if len(subset) == 0:
        continue

    print(f"\n{name}: {len(subset):,} flows")
    print(
        "  Zero packet-length variance: "
        f"{subset['zero_variance'].mean() * 100:.2f}%"
    )
    print(
        "  Median total backward packets: "
        f"{subset[backward_feature].median():.2f}"
    )
    print(
        "  Median packet-length variance: "
        f"{subset[variance_feature].median():.2f}"
    )

print("\nDone. No models were changed.")
