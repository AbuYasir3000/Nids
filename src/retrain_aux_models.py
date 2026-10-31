"""Retrain the Isolation Forest and the binary attack classifier.

Uses the same inputs as train_rf_with_pcap.py (a CIC-IDS2017 sample plus a
labelled, converted capture). EXPERIMENT, not wired into the app: writes isolation_forest_v2.joblib and
novel_attack_generalizer_v2.joblib next to the originals, and
hybrid_detector.py does NOT load them. In our run the new binary classifier
caught the scanner (99.6% on held-out hosts) but lost cross-capture DDoS recall
(21% -> 0%) and the Isolation Forest did not change; see
reports/auxiliary_models_v2_report.txt.

    python src/retrain_aux_models.py --sample cicids2017_sample.csv.gz \
        --flows flows.csv --scanner-ip 192.168.2.5

Capture labels are an assumption: source == --scanner-ip -> attack.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    from train_rf_with_pcap import LABEL_FIX, load_capture, numeric
except ImportError:
    from .train_rf_with_pcap import LABEL_FIX, load_capture, numeric

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PORTSCAN_CAPTURE = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
CALIBRATION_CAPTURE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
CALIBRATION_FPR = 0.001
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--flows", required=True)
    ap.add_argument("--scanner-ip", required=True)
    a = ap.parse_args()

    rf = joblib.load(MODELS / "random_forest.joblib")
    cols = list(rf["feature_columns"])          # IF must share the RF order
    old_gen = joblib.load(MODELS / "novel_attack_generalizer.joblib")
    gen_cols = list(old_gen["feature_columns"])

    cic = pd.read_csv(a.sample, low_memory=False)
    cic.columns = cic.columns.str.strip()
    cic["Label"] = cic["Label"].astype(str).str.strip().replace(LABEL_FIX)
    cic["is_attack"] = (cic["Label"] != "BENIGN").astype(int)
    cap, cap_train, cap_test, held = load_capture(a.flows, a.scanner_ip, sorted(set(cols) | set(gen_cols)))
    cap["is_attack"] = (cap["Label"] != "BENIGN").astype(int)
    cap_train = cap.loc[cap_train.index]
    cap_test = cap.loc[cap_test.index]
    report = ["Retrained auxiliary models (CIC-IDS2017 sample + labelled capture)",
              f"Held-out capture hosts: {sorted(held)}",
              f"Capture labels ASSUMED: source == {a.scanner_ip} -> attack.", ""]

    # ---------------- Isolation Forest (benign traffic only) ----------------
    benign_train = pd.concat([
        cic[cic.is_attack == 0].sample(frac=0.8, random_state=SEED)[cols],
        cap_train[cap_train.is_attack == 0][cols],
    ])
    medians = numeric(benign_train, cols).median()
    scaler = StandardScaler().fit(numeric(benign_train, cols, medians))
    iso = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=SEED, n_jobs=-1)
    iso.fit(scaler.transform(numeric(benign_train, cols, medians)))

    def flagged(df):
        x = scaler.transform(numeric(df, cols, medians))
        return iso.predict(x) == -1

    cic_test = cic.drop(cic[cic.is_attack == 0].sample(frac=0.8, random_state=SEED).index)
    old_if = joblib.load(MODELS / "isolation_forest.joblib")

    def old_flagged(df):
        x = numeric(df, cols, pd.Series(old_if["medians"]))
        return old_if["model"].predict(old_if["scaler"].transform(x)) == -1

    report.append("Isolation Forest: share of flows flagged as anomalous")
    for name, df in (("CIC benign (held-out 20%)", cic_test[cic_test.is_attack == 0]),
                     ("CIC attacks", cic_test[cic_test.is_attack == 1]),
                     ("Capture benign, held-out hosts", cap_test[cap_test.is_attack == 0]),
                     ("Capture scanner flows, held-out hosts", cap_test[cap_test.is_attack == 1])):
        report.append(f"  {name:42s} new {flagged(df).mean():7.2%}   old {old_flagged(df).mean():7.2%}")
    joblib.dump({"model": iso, "scaler": scaler, "feature_columns": cols,
                 "medians": medians.to_dict(), "threshold": float(iso.offset_)},
                MODELS / "isolation_forest_v2.joblib", compress=3)

    # ---------------- Binary attack classifier ----------------
    # As in the original design, the PortScan and DDoS captures stay out of
    # training (the DDoS capture calibrates the threshold, PortScan is a test).
    held_out_files = {PORTSCAN_CAPTURE, CALIBRATION_CAPTURE}
    train_cic = cic[~cic.SourceFile.isin(held_out_files)]
    train = pd.concat([train_cic[gen_cols + ["is_attack"]],
                       cap_train[gen_cols + ["is_attack"]]])
    imputer = SimpleImputer(strategy="median").fit(numeric(train, gen_cols))
    Xtr = imputer.transform(numeric(train, gen_cols)).astype("float32")
    clf = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=31,
                                         learning_rate=0.1, l2_regularization=1.0,
                                         class_weight="balanced", random_state=SEED)
    clf.fit(Xtr, train["is_attack"])

    def score(df):
        return clf.predict_proba(imputer.transform(numeric(df, gen_cols)).astype("float32"))[:, 1]

    calib = pd.concat([cic[(cic.SourceFile == CALIBRATION_CAPTURE) & (cic.is_attack == 0)],
                       cap_test[cap_test.is_attack == 0]])
    threshold = float(np.quantile(score(calib), 1 - CALIBRATION_FPR, method="higher"))
    old_thr = float(old_gen["decision_threshold"])

    def old_score(df):
        x = numeric(df, gen_cols, pd.Series(old_gen["imputer"].statistics_, index=gen_cols))
        x = old_gen["imputer"].transform(x).astype("float32")
        return old_gen["model"].predict_proba(x)[:, 1]

    report += ["", f"Binary classifier: share of flows flagged (threshold new {threshold:.4f}, old {old_thr:.4f})"]
    # Note: capture test hosts' benign flows also helped calibrate the threshold.
    for name, df in (("CIC PortScan capture (never trained on)", cic[(cic.SourceFile == PORTSCAN_CAPTURE) & (cic.is_attack == 1)]),
                     ("CIC DDoS capture attacks (calibration capture)", cic[(cic.SourceFile == CALIBRATION_CAPTURE) & (cic.is_attack == 1)]),
                     ("Capture scanner flows, held-out hosts", cap_test[cap_test.is_attack == 1]),
                     ("Capture benign, held-out hosts", cap_test[cap_test.is_attack == 0]),
                     ("CIC PortScan capture benign", cic[(cic.SourceFile == PORTSCAN_CAPTURE) & (cic.is_attack == 0)])):
        report.append(f"  {name:48s} new {(score(df) > threshold).mean():7.2%}   old {(old_score(df) > old_thr).mean():7.2%}")
    report.append("Held-out-host benign rows were used to calibrate the new threshold, so their")
    report.append("false-positive rate is optimistic. One scanner and one network: not independent proof.")

    artifact = dict(old_gen)
    artifact.update({
        "model": clf, "imputer": imputer, "feature_columns": gen_cols,
        "decision_threshold": threshold,
        "calibration_observed_fpr": float((score(calib) > threshold).mean()),
        "calibration_capture": CALIBRATION_CAPTURE + " benign + held-out capture benign",
        "training_rows": int(len(train)),
    })
    joblib.dump(artifact, MODELS / "novel_attack_generalizer_v2.joblib", compress=3)

    text = "\n".join(report)
    print(text)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "auxiliary_models_v2_report.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
