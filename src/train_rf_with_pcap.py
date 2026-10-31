"""Retrain the Random Forest on a CIC-IDS2017 sample plus a labelled capture.

Example:
    python src/train_rf_with_pcap.py --sample cicids2017_sample.csv.gz \
        --flows flows.csv --scanner-ip 192.168.2.5

--flows is the CICFlowMeter CSV made by src/pcap_converter.py (with the
cic_compat patch). Flows whose source is --scanner-ip are labelled PortScan,
all other flows in that capture BENIGN. That labelling is an assumption: check
it matches what you know about the capture.

Writes models/random_forest_v2.joblib (same artifact layout as
random_forest.joblib) and reports/random_forest_v2_report.txt.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import train_test_split

try:
    from pcap_converter import normalize_flow_columns
except ImportError:
    from .pcap_converter import normalize_flow_columns

ROOT = Path(__file__).resolve().parents[1]
LABEL_FIX = {
    "Web Attack  Brute Force": "Web Attack – Brute Force",
    "Web Attack  Sql Injection": "Web Attack – Sql Injection",
    "Web Attack  XSS": "Web Attack – XSS",
}


def numeric(df, cols, medians=None):
    x = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x.fillna(medians if medians is not None else x.median()).astype("float32")


def load_capture(flows_csv, scanner_ip, cols):
    """Label a converted capture and split it by host.

    Flows from ``scanner_ip`` are PortScan, everything else BENIGN (an
    assumption). Whole hosts are held out (scanner flows by target host,
    other flows by source host) so the test is not near-duplicate flows.
    Returns (all, train, test, held_out_groups).
    """
    cap = normalize_flow_columns(pd.read_csv(flows_csv), cols, convert_time_units=True)
    cap["Label"] = np.where(cap["Source IP"] == scanner_ip, "PortScan", "BENIGN")
    key = np.where(cap["Label"] == "PortScan", cap["Destination IP"], cap["Source IP"])
    cap["_group"] = pd.Series(key, index=cap.index).astype(str)
    groups = sorted(cap["_group"].unique())
    held = set(groups[2::3])
    held_mask = cap["_group"].isin(held)
    return cap, cap[~held_mask], cap[held_mask], held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--flows", required=True)
    ap.add_argument("--scanner-ip", required=True)
    ap.add_argument("--trees", type=int, default=200)
    a = ap.parse_args()

    old = joblib.load(ROOT / "models" / "random_forest.joblib")
    cols = list(old["feature_columns"])

    cic = pd.read_csv(a.sample, low_memory=False)
    cic.columns = cic.columns.str.strip()
    cic["Label"] = cic["Label"].astype(str).str.strip().replace(LABEL_FIX)

    cap, cap_train, cap_test, held = load_capture(a.flows, a.scanner_ip, cols)

    cic_train, cic_test = train_test_split(
        cic, test_size=0.2, stratify=cic["Label"], random_state=42
    )
    train = pd.concat([cic_train[cols + ["Label"]], cap_train[cols + ["Label"]]])
    medians = numeric(train, cols).median()
    Xtr = numeric(train, cols, medians)

    model = RandomForestClassifier(
        n_estimators=a.trees, n_jobs=-1, random_state=42,
        class_weight="balanced_subsample", min_samples_leaf=1,
    ).fit(Xtr, train["Label"])

    lines = [
        "Random Forest v2 (CIC-IDS2017 sample + labelled capture)",
        f"Train rows: {len(train):,} (CIC {len(cic_train):,}, capture {len(cap_train):,})",
        f"Held-out capture hosts: {sorted(held)}",
        "Capture labels are ASSUMED: source == %s -> PortScan, else BENIGN." % a.scanner_ip,
        "",
    ]
    for name, test in (("CIC sample (20% stratified holdout)", cic_test),
                       ("Capture, held-out hosts", cap_test)):
        X = numeric(test, cols, medians)
        for mname, m, Xm in (
            ("v2", model, X),
            ("current (old)", old["model"], numeric(test, cols, pd.Series(old["medians"]))),
        ):
            pred = m.predict(Xm)
            y = test["Label"].astype(str)
            attack = y != "BENIGN"
            lines.append(f"[{name}] {mname}: accuracy {np.mean(pred == y.values):.4f}; "
                         f"attack recall (any non-benign) {np.mean(pred[attack.values] != 'BENIGN'):.4f}; "
                         f"benign false-positive {np.mean(pred[~attack.values] != 'BENIGN'):.4f}")
        lines.append("")
    pred = model.predict(numeric(cap_test, cols, medians))
    lines.append("v2 on held-out capture hosts:\n" + classification_report(
        cap_test["Label"], pred, zero_division=0))
    lines.append("NOTE: capture flows come from one scanner/tool and one network, so the "
                 "host split is only a partial test of generalization.")
    report = "\n".join(lines)
    print(report)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "random_forest_v2_report.txt").write_text(report, encoding="utf-8")

    # Final model: refit on everything so no data is wasted.
    full = pd.concat([cic[cols + ["Label"]], cap[cols + ["Label"]]])
    med_full = numeric(full, cols).median()
    final = RandomForestClassifier(
        n_estimators=a.trees, n_jobs=-1, random_state=42,
        class_weight="balanced_subsample",
    ).fit(numeric(full, cols, med_full), full["Label"])
    joblib.dump({"model": final, "feature_columns": cols, "medians": med_full.to_dict()},
                ROOT / "models" / "random_forest_v2.joblib")
    print("saved models/random_forest_v2.joblib")


if __name__ == "__main__":
    main()
