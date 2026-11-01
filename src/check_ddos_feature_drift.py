
from pathlib import Path
import gc

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

HELD_OUT_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
MAX_SAMPLE_PER_GROUP = 50_000
RANDOM_STATE = 42


def load_csv(path):
    df = pd.read_csv(path, encoding="cp1252", low_memory=False)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError(f"Missing Label column: {path.name}")

    df["Label"] = df["Label"].astype("string").str.strip()
    return df


def numeric_features(df):
    excluded = {
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "Label",
    }

    X = df.drop(
        columns=[c for c in excluded if c in df.columns]
    )

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    return X


def sample_rows(df, max_rows):
    if len(df) > max_rows:
        return df.sample(
            n=max_rows,
            random_state=RANDOM_STATE,
        )
    return df


def main():
    held_out_path = RAW_DIR / HELD_OUT_FILE

    if not held_out_path.exists():
        raise FileNotFoundError(
            f"Held-out file not found: {held_out_path}"
        )

    print("SENTINELAI FEATURE-DRIFT ANALYSIS")
    print("=" * 45)

    print("\nLoading held-out DDoS capture...")
    test_df = load_csv(held_out_path)

    test_labels = test_df["Label"].copy()
    X_test = numeric_features(test_df)

    print(f"Test rows: {len(X_test):,}")
    print(f"Test numeric features: {len(X_test.columns)}")

    del test_df
    gc.collect()

    # Load all other captures as the reference training distribution.
    train_parts = []

    for path in sorted(RAW_DIR.rglob("*.csv")):
        if path.name == HELD_OUT_FILE:
            continue

        print(f"Reading reference data: {path.name}")
        df = load_csv(path)

        # Match the same binary task: BENIGN versus ATTACK.
        benign = df["Label"].eq("BENIGN")
        attack = df["Label"].notna() & df["Label"].ne("") & ~benign

        # Retain representative samples from both groups.
        for mask, label in [
            (benign, "BENIGN"),
            (attack, "ATTACK"),
        ]:
            part = df.loc[mask]

            if len(part) > MAX_SAMPLE_PER_GROUP:
                part = part.sample(
                    n=MAX_SAMPLE_PER_GROUP,
                    random_state=RANDOM_STATE,
                )

            if not part.empty:
                part = part.copy()
                part["_binary_label"] = label
                train_parts.append(part)

        del df
        gc.collect()

    if not train_parts:
        raise RuntimeError("No reference data was loaded.")

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
        sort=False,
    )

    del train_parts
    gc.collect()

    y_reference = train_df["_binary_label"].copy()
    X_reference = numeric_features(train_df)

    del train_df
    gc.collect()

    # Compare only features present in both datasets.
    common_features = sorted(
        set(X_reference.columns) & set(X_test.columns)
    )
    reference_only = sorted(
        set(X_reference.columns) - set(X_test.columns)
    )
    test_only = sorted(
        set(X_test.columns) - set(X_reference.columns)
    )

    print(f"\nReference sampled rows: {len(X_reference):,}")
    print(f"Common numeric features: {len(common_features)}")

    if reference_only:
        print("Reference-only features:", reference_only)

    if test_only:
        print("Test-only features:", test_only)

    # Separate reference BENIGN and ATTACK data.
    reference_benign = X_reference.loc[
        y_reference.eq("BENIGN"), common_features
    ]
    reference_attack = X_reference.loc[
        y_reference.eq("ATTACK"), common_features
    ]

    test_benign = X_test.loc[
        test_labels.eq("BENIGN"), common_features
    ]
    test_attack = X_test.loc[
        test_labels.notna() & test_labels.ne("BENIGN"),
        common_features,
    ]

    results = []

    print("\nComparing feature distributions...")

    for index, feature in enumerate(common_features, start=1):
        # Compare benign-to-benign to detect capture/domain shift.
        ref_b = reference_benign[feature].dropna()
        tst_b = test_benign[feature].dropna()

        # Compare attack-to-attack to detect attack distribution shift.
        ref_a = reference_attack[feature].dropna()
        tst_a = test_attack[feature].dropna()

        def compare(a, b):
            if len(a) < 2 or len(b) < 2:
                return np.nan, np.nan

            ks = ks_2samp(a, b)
            return float(ks.statistic), float(ks.pvalue)

        benign_ks, benign_p = compare(ref_b, tst_b)
        attack_ks, attack_p = compare(ref_a, tst_a)

        results.append({
            "feature": feature,
            "reference_benign_nonmissing": len(ref_b),
            "test_benign_nonmissing": len(tst_b),
            "benign_ks_statistic": benign_ks,
            "benign_ks_pvalue": benign_p,
            "reference_attack_nonmissing": len(ref_a),
            "test_attack_nonmissing": len(tst_a),
            "attack_ks_statistic": attack_ks,
            "attack_ks_pvalue": attack_p,
            "reference_missing_pct": (
                X_reference[feature].isna().mean() * 100
            ),
            "test_missing_pct": (
                X_test[feature].isna().mean() * 100
            ),
        })

        if index % 20 == 0:
            print(f"Compared {index}/{len(common_features)} features")

    report = pd.DataFrame(results)

    # Large KS statistic means distributions differ more.
    report = report.sort_values(
        "attack_ks_statistic",
        ascending=False,
        na_position="last",
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = REPORT_DIR / "ddos_feature_drift.csv"
    report.to_csv(output_csv, index=False)

    output_txt = REPORT_DIR / "ddos_feature_drift_summary.txt"

    with output_txt.open("w", encoding="utf-8") as file:
        file.write("Argus Feature-Drift Analysis\n")
        file.write("=" * 40 + "\n\n")
        file.write(f"Held-out file: {HELD_OUT_FILE}\n")
        file.write(f"Reference sampled rows: {len(X_reference):,}\n")
        file.write(f"Held-out rows: {len(X_test):,}\n")
        file.write(f"Common features: {len(common_features)}\n\n")

        file.write(
            "Note: KS statistics measure distribution differences. "
            "They do not prove that a feature caused misclassification.\n\n"
        )

        file.write("Top 20 features by attack distribution difference:\n")
        file.write(
            report[
                [
                    "feature",
                    "attack_ks_statistic",
                    "attack_ks_pvalue",
                    "benign_ks_statistic",
                    "reference_missing_pct",
                    "test_missing_pct",
                ]
            ].head(20).to_string(index=False)
        )

    print("\nTOP 20 FEATURES BY ATTACK DISTRIBUTION DIFFERENCE")
    print("-" * 85)
    print(
        report[
            [
                "feature",
                "attack_ks_statistic",
                "benign_ks_statistic",
                "reference_missing_pct",
                "test_missing_pct",
            ]
        ].head(20).to_string(index=False)
    )

    print("\nSaved reports:")
    print(output_csv)
    print(output_txt)
    print("\nNo models or dashboard files were changed.")


if __name__ == "__main__":
    main()