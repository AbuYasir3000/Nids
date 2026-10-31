"""Train Random Forest v3: CIC-IDS2017 sample + labelled scan capture + UDPLag.

    python src/train_rf_v3.py --sample cicids2017_sample_ids2017.csv.gz ^
        --flows flows.csv --scanner-ip 192.168.2.5 --udplag udplag_sample.csv.gz

UDPLag rows are split by time: everything before --cutoff trains the model and
the later rows (the model never sees them) test it. The same three held-out
sets are scored for the original, v2 and v3 models so regressions show up.
Writes models/random_forest_v3.joblib and reports/random_forest_v3_report.txt.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    from train_rf_with_pcap import LABEL_FIX, load_capture, numeric
except ImportError:
    from .train_rf_with_pcap import LABEL_FIX, load_capture, numeric

ROOT = Path(__file__).resolve().parents[1]


def summarize(name, model_name, model, medians, df, cols):
    pred = model.predict(numeric(df, cols, medians))
    y = df["Label"].astype(str).to_numpy()
    attack = y != "BENIGN"
    recall = (pred[attack] != "BENIGN").mean() if attack.any() else float("nan")
    fpr = (pred[~attack] != "BENIGN").mean() if (~attack).any() else float("nan")
    exact = (pred == y).mean()
    return (f"  {model_name:12s} attack recall {recall:7.2%}   benign false alarms {fpr:6.2%}"
            f"   exact-label accuracy {exact:7.2%}   ({int(attack.sum()):,} attacks, {int((~attack).sum()):,} benign)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--flows", required=True)
    ap.add_argument("--scanner-ip", required=True)
    ap.add_argument("--udplag", required=True)
    ap.add_argument("--cutoff", default="2018-12-01 13:30:00",
                    help="attack rows at or after this time are test rows")
    ap.add_argument("--benign-cutoff", default="2018-12-01 13:25:00",
                    help="benign rows at or after this time are test rows (too few benign flows late)")
    ap.add_argument("--tag", default="", help="suffix for the report file name")
    ap.add_argument("--trees", type=int, default=200)
    a = ap.parse_args()

    old = joblib.load(ROOT / "models" / "random_forest.joblib")
    cols = list(old["feature_columns"])

    cic = pd.read_csv(a.sample, low_memory=False)
    cic.columns = cic.columns.str.strip()
    cic["Label"] = cic["Label"].astype(str).str.strip().replace(LABEL_FIX)
    cap, cap_train, cap_test, held = load_capture(a.flows, a.scanner_ip, cols)
    udp = pd.read_csv(a.udplag, low_memory=False)
    udp.columns = udp.columns.str.strip()
    udp["Label"] = udp["Label"].astype(str).str.strip()
    missing = [c for c in cols if c not in udp.columns]
    if missing:
        raise SystemExit(f"UDPLag file is missing model features: {missing[:5]}")
    when = pd.to_datetime(udp["Timestamp"], errors="coerce")
    is_test = np.where(udp["Label"] == "BENIGN", when >= a.benign_cutoff, when >= a.cutoff)
    udp_train, udp_test = udp[~is_test], udp[is_test]

    cic_train, cic_test = train_test_split(cic, test_size=0.2, stratify=cic["Label"], random_state=42)
    train = pd.concat([cic_train[cols + ["Label"]], cap_train[cols + ["Label"]], udp_train[cols + ["Label"]]])
    medians = numeric(train, cols).median()
    model = RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, random_state=42,
                                   class_weight="balanced_subsample")
    model.fit(numeric(train, cols, medians), train["Label"])

    # Fair comparison: "v2" is retrained here on exactly the same training rows
    # as v3 minus UDPLag, so both models have never seen the test rows. (The saved
    # random_forest_v2.joblib is refit on everything, so scoring it on these
    # test rows would be leaky.)
    train_v2 = pd.concat([cic_train[cols + ["Label"]], cap_train[cols + ["Label"]]])
    med_v2 = numeric(train_v2, cols).median()
    v2_holdout = RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, random_state=42,
                                        class_weight="balanced_subsample")
    v2_holdout.fit(numeric(train_v2, cols, med_v2), train_v2["Label"])

    lines = ["Random Forest v3 (CIC-IDS2017 sample + scan capture + UDPLag)",
             f"Train rows: {len(train):,} (CIC {len(cic_train):,}, scan capture {len(cap_train):,}, UDPLag {len(udp_train):,})",
             f"UDPLag split by time: attacks before {a.cutoff} and benign before {a.benign_cutoff} train ({len(udp_train):,} rows); the later rows test ({len(udp_test):,} rows, never seen)",
             f"Scan-capture held-out hosts: {sorted(held)}",
             f"Capture labels ASSUMED: source == {a.scanner_ip} -> PortScan, else BENIGN.", ""]
    sets = (("UDPLag, later time window (never trained on)", udp_test),
            ("Scan capture, held-out hosts", cap_test),
            ("CIC-IDS2017 sample, 20% holdout", cic_test))
    for title, df in sets:
        lines.append(title)
        lines.append(summarize(title, "original*", old["model"], pd.Series(old["medians"]), df, cols))
        lines.append(summarize(title, "v2 (no UDP)", v2_holdout, med_v2, df, cols))
        lines.append(summarize(title, "v3", model, medians, df, cols))
        lines.append("")
    lines.append("* original: trained on the full CIC-IDS2017 data, so it may have seen the CIC rows; its numbers are not a clean holdout.")
    lines.append("v2 (no UDP) and v3 are trained here on the same CIC and scan-capture rows (v3 adds UDPLag) and have never seen any test row.")
    lines.append("")
    lines.append("UDPLag test rows, per label (v3 exact class):")
    pred = model.predict(numeric(udp_test, cols, medians))
    for label, grp in udp_test.groupby("Label"):
        lines.append(f"  {label:10s} {len(grp):6,} flows -> {(pred[grp.index.map(udp_test.index.get_loc)] == label).mean():.2%} predicted as {label}")
    lines.append("")
    lines.append("Dataset note: in UDPLag.csv the 13:04 'UDP-lag' flows are real UDP, but the 13:29-13:30 flows")
    lines.append("(about 96% of the attack) are TCP with no payload and no reply, i.e. SYN-flood-like.")
    lines.append("CAUTION: UDPLag is one attack on one network; the time split tests later bursts of the")
    lines.append("same attack, not other UDP attacks. The sample is down-sampled (8,000 UDP-lag rows of 366,461).")
    report = "\n".join(lines)
    print(report)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"random_forest_v3_report{a.tag}.txt").write_text(report + "\n", encoding="utf-8")

    full = pd.concat([cic[cols + ["Label"]], cap[cols + ["Label"]], udp[cols + ["Label"]]])
    med_full = numeric(full, cols).median()
    final = RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, random_state=42,
                                   class_weight="balanced_subsample")
    final.fit(numeric(full, cols, med_full), full["Label"])
    joblib.dump({"model": final, "feature_columns": cols, "medians": med_full.to_dict()},
                ROOT / "models" / "random_forest_v3.joblib", compress=("zlib", 6))
    print("saved models/random_forest_v3.joblib")


if __name__ == "__main__":
    main()
