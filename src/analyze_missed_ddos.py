
from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

HELD_OUT_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
EXCLUDED_COLUMNS = {
    "Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"
}


def load_csv(path):
    df = pd.read_csv(path, encoding="cp1252", low_memory=False)
    df.columns = df.columns.str.strip()
    if "Label" not in df.columns:
        raise ValueError(f"Missing Label column: {path.name}")
    df["Label"] = df["Label"].astype("string").str.strip()
    return df


def make_features(df, columns=None, medians=None):
    X = df.drop(
        columns=[c for c in EXCLUDED_COLUMNS if c in df.columns]
    )
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if columns is None:
        X = X.dropna(axis=1, how="all")
        columns = X.columns.tolist()
        medians = X.median().fillna(0)

    X = X.reindex(columns=columns)
    X = X.fillna(medians).fillna(0).astype("float32")
    return X, columns, medians


def main():
    test_path = RAW_DIR / HELD_OUT_FILE
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    print("SENTINELAI MISSED-DDOS ANALYSIS")
    print("=" * 45)

    # Load training captures, excluding the entire held-out DDoS file.
    frames = []
    for path in sorted(RAW_DIR.rglob("*.csv")):
        if path.name == HELD_OUT_FILE:
            print("Excluded:", path.name)
            continue
        print("Loading training data:", path.name)
        df = load_csv(path)
        df = df.dropna(subset=["Label"])
        df = df[df["Label"] != ""]
        frames.append(df)

    train_df = pd.concat(frames, ignore_index=True, sort=False)
    del frames
    gc.collect()

    y_train = np.where(train_df["Label"].eq("BENIGN"), "BENIGN", "ATTACK")
    X_train, columns, medians = make_features(train_df)
    del train_df
    gc.collect()

    print(f"Training rows: {len(X_train):,}")
    print(f"Features: {len(columns)}")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    start = time.time()
    model.fit(X_train, y_train)
    print(f"Training time: {time.time() - start:.1f} seconds")

    del X_train, y_train
    gc.collect()

    # Prepare the held-out capture.
    print("\nLoading held-out DDoS capture...")
    test_df = load_csv(test_path)
    test_df = test_df.dropna(subset=["Label"])
    test_df = test_df[test_df["Label"] != ""]

    actual_attack = test_df["Label"].ne("BENIGN").to_numpy()
    X_test, _, _ = make_features(test_df, columns, medians)

    print("Predicting held-out traffic...")
    predicted = model.predict(X_test)

    # Focus only on actual DDoS flows.
    missed_mask = actual_attack & (predicted == "BENIGN")
    detected_mask = actual_attack & (predicted == "ATTACK")

    print(f"\nActual attack flows: {actual_attack.sum():,}")
    print(f"Detected attack flows: {detected_mask.sum():,}")
    print(f"Missed attack flows: {missed_mask.sum():,}")

    if missed_mask.sum() < 2 or detected_mask.sum() < 2:
        raise RuntimeError("Not enough detected or missed attacks to compare.")

    # Sample equally sized groups so very large groups remain manageable.
    rng = np.random.default_rng(42)
    detected_indices = np.flatnonzero(detected_mask)
    missed_indices = np.flatnonzero(missed_mask)
    sample_size = min(50_000, len(detected_indices), len(missed_indices))

    detected_indices = rng.choice(
        detected_indices, sample_size, replace=False
    )
    missed_indices = rng.choice(
        missed_indices, sample_size, replace=False
    )

    detected = X_test.iloc[detected_indices]
    missed = X_test.iloc[missed_indices]

    results = []
    print("\nComparing detected versus missed DDoS flows...")

    for feature in columns:
        a = detected[feature].replace([np.inf, -np.inf], np.nan).dropna()
        b = missed[feature].replace([np.inf, -np.inf], np.nan).dropna()

        if len(a) < 2 or len(b) < 2:
            statistic, pvalue = np.nan, np.nan
        else:
            result = ks_2samp(a, b)
            statistic, pvalue = float(result.statistic), float(result.pvalue)

        results.append({
            "feature": feature,
            "ks_statistic": statistic,
            "p_value": pvalue,
            "detected_median": a.median() if len(a) else np.nan,
            "missed_median": b.median() if len(b) else np.nan,
            "detected_missing_pct": (
                X_test.iloc[detected_indices][feature].isna().mean() * 100
            ),
            "missed_missing_pct": (
                X_test.iloc[missed_indices][feature].isna().mean() * 100
            ),
        })

    report = pd.DataFrame(results).sort_values(
        "ks_statistic", ascending=False, na_position="last"
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "missed_ddos_feature_analysis.csv"
    txt_path = REPORT_DIR / "missed_ddos_feature_analysis.txt"
    report.to_csv(csv_path, index=False)

    top = report.head(20)
    with txt_path.open("w", encoding="utf-8") as file:
        file.write("Detected vs Missed DDoS Feature Analysis\n\n")
        file.write(f"Detected attacks: {detected_mask.sum():,}\n")
        file.write(f"Missed attacks: {missed_mask.sum():,}\n")
        file.write(f"Sample size per group: {sample_size:,}\n\n")
        file.write(
            "KS statistics measure differences, not causation.\n\n"
        )
        file.write(top.to_string(index=False))

    print("\nTOP 20 FEATURES: DETECTED VS MISSED DDOS")
    print(top.to_string(index=False))
    print("\nSaved:")
    print(csv_path)
    print(txt_path)


if __name__ == "__main__":
    main()
