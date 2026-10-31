"""Make a small stratified sample of the CIC-IDS2017 CSVs.

Usage (from the folder that holds the *.pcap_ISCX.csv files):
    python make_sample.py <csv_folder> [rows_per_label_per_file]

Writes cicids2017_sample.csv.gz next to this script. Keeps up to N rows per
(file, Label) -- default 8000 -- so rare attacks stay whole and big classes
(BENIGN, DoS Hulk, PortScan, DDoS) are shrunk. A 'SourceFile' column is added
so the capture each row came from is still known.
"""
import sys
from pathlib import Path
import pandas as pd

folder = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
files = sorted(folder.rglob("*.csv"))
if not files:
    sys.exit(f"No CSV files found under {folder}")

parts = []
for f in files:
    # latin-1: some CIC files contain a non-UTF8 dash in 'Web Attack - ...'
    df = pd.read_csv(f, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()
    if "Label" not in df.columns:
        print("skip (no Label):", f.name)
        continue
    df["Label"] = df["Label"].astype(str).str.strip()
    sampled = pd.concat(
        [g.sample(min(len(g), n), random_state=42) for _, g in df.groupby("Label")]
    )
    sampled["SourceFile"] = f.name
    parts.append(sampled)
    print(f"{f.name}: {len(df):,} -> {len(sampled):,}")

out = pd.concat(parts, ignore_index=True)
dest = Path(__file__).with_name("cicids2017_sample.csv.gz")
out.to_csv(dest, index=False, compression="gzip")
print(f"\nWrote {dest} ({dest.stat().st_size/1e6:.1f} MB, {len(out):,} rows)")
print(out["Label"].value_counts().to_string())
