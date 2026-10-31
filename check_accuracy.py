"""One-command accuracy check for SentinelAI.

Runs every detector (Random Forest, binary classifier, Isolation Forest and
the behavioral rule) on labelled data and prints one table.

Examples (run from the project folder):
    python check_accuracy.py --csv path\\to\\labelled_flows.csv
    python check_accuracy.py --pcap scan_A.pcap --attacker-ip 192.168.2.5
    python check_accuracy.py --csv cicids2017_sample.csv.gz

--csv   a CIC-IDS2017-style flow CSV that has a Label column (.gz is fine).
--pcap  a capture; flows are made with CICFlowMeter. A pcap has no labels, so
        you must give --attacker-ip (several IPs separated by commas): flows from those IPs count as attacks, all
        other flows as benign. That is YOUR assumption; the report repeats it.

Writes reports/accuracy_summary.md and reports/accuracy_summary.csv.
"""
import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.hybrid_detector import HybridNIDS  # noqa: E402
from src.pcap_converter import convert_pcap_to_csv, normalize_flow_columns  # noqa: E402

def clean_label(value):
    """Normalise labels; CIC's 'Web Attack <odd dash> X' often arrives garbled."""
    text = str(value).strip()
    if text.lower().startswith("web attack"):
        for key, name in (("brute", "Brute Force"), ("sql", "Sql Injection"), ("xss", "XSS")):
            if key in text.lower():
                return f"Web Attack \u2013 {name}"
    return text


def read_csv_any_encoding(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def load_labelled(args, detector):
    """Return (flows with a clean 'Label' column, notes about the data)."""
    notes = []
    if args.csv:
        df = read_csv_any_encoding(args.csv)
        df.columns = df.columns.str.strip()
        lab = next((c for c in df.columns if c.lower() == "label"), None)
        if lab is None:
            sys.exit("The CSV has no Label column, so accuracy cannot be computed.")
        df = normalize_flow_columns(df, detector.feature_columns, convert_time_units=False)
        df["Label"] = df[lab].map(clean_label)
        df = df[~df["Label"].str.lower().isin(["", "nan"])].reset_index(drop=True)
        notes.append(f"Input: {Path(args.csv).name} (labels from the file's Label column)")
        if "sample" in Path(args.csv).name.lower():
            notes.append("This looks like a down-sampled file: the models were trained on data "
                         "like this, so scores are optimistic, and the behavioral rule needs "
                         "complete captures (rows were removed), so it is skipped.")
        return df, notes
    if not args.attacker_ip:
        sys.exit("--pcap needs --attacker-ip (a pcap has no labels).")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "flows.csv"
        print("Converting capture to flows (about a minute for 15 MB)...")
        convert_pcap_to_csv(args.pcap, out)
        df = normalize_flow_columns(pd.read_csv(out), detector.feature_columns,
                                    convert_time_units=True)
    attackers = [ip.strip() for ip in args.attacker_ip.split(",") if ip.strip()]
    df["Label"] = np.where(df["Source IP"].isin(attackers), "ATTACK", "BENIGN")
    notes.append(f"Input: {Path(args.pcap).name}; flows from {args.attacker_ip} labelled ATTACK, "
                 "all other flows BENIGN (assumption).")
    notes.append("If the Random Forest was retrained on this capture, scores are optimistic "
                 "(not independent).")
    return df, notes


def md_table(df, index=True):
    """Markdown table without needing the optional 'tabulate' package."""
    frame = df.reset_index() if index else df
    if index:
        frame = frame.rename(columns={frame.columns[0]: "Detector"})
    cols = [str(c) for c in frame.columns]
    rows = [[str(v) for v in row] for row in frame.itertuples(index=False)]
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(r) + " |" for r in rows])


def rates(truth_attack, flagged):
    t, f = np.asarray(truth_attack, bool), np.asarray(flagged, bool)
    tp, fp = int((t & f).sum()), int((~t & f).sum())
    fn, tn = int((t & ~f).sum()), int((~t & ~f).sum())
    n = tp + fp + fn + tn
    return {
        "accuracy": (tp + tn) / n if n else np.nan,
        "attack detection rate (recall)": tp / (tp + fn) if tp + fn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "false alarm rate on benign": fp / (fp + tn) if fp + tn else np.nan,
        "attacks caught": f"{tp:,} / {tp + fn:,}",
        "benign flagged": f"{fp:,} / {fp + tn:,}",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv")
    g.add_argument("--pcap")
    ap.add_argument("--attacker-ip")
    args = ap.parse_args()

    print("Loading models...")
    det = HybridNIDS()
    df, notes = load_labelled(args, det)
    truth = df["Label"].astype(str).str.upper().ne("BENIGN").to_numpy()
    print(f"Running detectors on {len(df):,} flows "
          f"({int(truth.sum()):,} attacks, {int((~truth).sum()):,} benign)...")
    res = det.predict(df)

    rf_attack = res["predicted_class"].astype(str).str.upper().ne("BENIGN").to_numpy()
    rule = (res["is_source_burst_alert"] | res["is_high_rate_multiport_activity"]
            | res["is_target_flood_activity"]).to_numpy()
    skip_rule = any("down-sampled" in n for n in notes)
    detectors = {
        "Random Forest (any attack class)": rf_attack,
        "Binary classifier": res["is_binary_classifier_attack"].to_numpy(),
        "Isolation Forest (anomaly)": res["is_anomaly"].to_numpy(),
    }
    if not skip_rule:
        detectors["Behavioral rule (not a model)"] = rule
        detectors["ANY of the above"] = rf_attack | res["is_binary_classifier_attack"].to_numpy() | rule
    else:
        detectors["Any model (RF / binary / IF)"] = (rf_attack | res["is_binary_classifier_attack"].to_numpy()
                                                      | res["is_anomaly"].to_numpy())

    table = pd.DataFrame({k: rates(truth, v) for k, v in detectors.items()}).T
    pct = ["accuracy", "attack detection rate (recall)", "precision", "false alarm rate on benign"]
    show = table.copy()
    for c in pct:
        show[c] = table[c].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1%}")

    # Random Forest multi-class accuracy on classes it was trained on.
    classes = set(map(str, det.rf_model.classes_))
    supported = df["Label"].isin(classes).to_numpy()
    rf_multi = float((res["predicted_class"].astype(str).to_numpy()[supported]
                      == df["Label"].to_numpy()[supported]).mean()) if supported.any() else float("nan")
    unsupported = df.loc[~supported, "Label"].value_counts()

    per_class = []
    for label, grp in df.groupby("Label"):
        idx = grp.index.to_numpy()
        per_class.append({
            "Label": label, "flows": len(idx),
            "RF predicts exact class": f"{(res['predicted_class'].astype(str).to_numpy()[idx] == label).mean():.1%}"
                                       if label in classes else "not a trained class",
            "RF flags as any attack": f"{rf_attack[idx].mean():.1%}",
            "binary classifier": f"{res['is_binary_classifier_attack'].to_numpy()[idx].mean():.1%}",
            "isolation forest": f"{res['is_anomaly'].to_numpy()[idx].mean():.1%}",
            **({} if skip_rule else {"behavioral rule": f"{rule[idx].mean():.1%}"}),
        })
    per_class = pd.DataFrame(per_class).sort_values("flows", ascending=False)

    lines = ["# SentinelAI accuracy check", "", *[f"- {n}" for n in notes], "",
             "## Overall (BENIGN vs attack, per flow)", "", md_table(show), "",
             f"Random Forest exact-class accuracy on classes it was trained on: **{rf_multi:.1%}** "
             f"({int(supported.sum()):,} flows)" if supported.any() else
             "Random Forest exact-class accuracy: no trained classes present in this data.",
             ""]
    if len(unsupported):
        lines += ["Labels the Random Forest was never trained on (it cannot name these): "
                  + ", ".join(f"{k} ({v:,})" for k, v in unsupported.items()), ""]
    lines += ["## By attack type (share of flows each detector flags)", "", md_table(per_class, index=False), "",
              "How to read it: *attack detection rate* = attacks caught / all attacks; "
              "*false alarm rate* = benign flows wrongly flagged / all benign. Accuracy alone is "
              "misleading when one class dominates, so quote detection rate and false alarm rate."]
    report = "\n".join(lines)
    print("\n" + report)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "accuracy_summary.md").write_text(report + "\n", encoding="utf-8")
    table.to_csv(out / "accuracy_summary.csv")
    print(f"\nSaved: {out / 'accuracy_summary.md'}")


if __name__ == "__main__":
    main()
