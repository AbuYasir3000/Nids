
from pathlib import Path
import sys
import tempfile

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_detector import HybridNIDS
from pcap_converter import convert_pcap_to_csv, normalize_flow_columns
from traffic_features import source_minute_event_table


st.set_page_config(
    page_title="Hybrid NIDS",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Hybrid Network Intrusion Detection System")
st.caption(
    "Random Forest classification + Isolation Forest anomaly detection"
)


@st.cache_resource
def load_detector():
    return HybridNIDS()


try:
    detector = load_detector()
    st.success("Both detection models loaded successfully.")
    if detector.generalized_model_ready:
        st.success("Cross-capture attack model loaded.")
    else:
        st.warning(
            "Cross-capture attack model is missing. Run "
            "python src/train_generalized_attack_model.py to enable it."
        )
except Exception as error:
    st.error(f"Could not load the trained models: {error}")
    st.info(
        "Check that models/random_forest.joblib and "
        "models/isolation_forest.joblib exist."
    )
    st.stop()


st.subheader("Analyze network traffic")
st.write(
    "Upload a compatible flow CSV or a PCAP/PCAPNG capture. Packet captures "
    "are converted to flow features with CICFlowMeter."
)

uploaded_file = st.file_uploader(
    "Choose a network-flow CSV or packet capture",
    type=["csv", "pcap", "pcapng", "pcap10"],
)
st.caption("Maximum file size: 20 GiB. No fixed flow-count limit per analysis.")

if uploaded_file is not None:
    try:
        suffix = Path(uploaded_file.name).suffix.lower()
        is_pcap = suffix in {".pcap", ".pcapng", ".pcap10"}
        analyze_now = False
        if is_pcap:
            analyze_now = st.button("Convert and analyze capture", type="primary")
            if not analyze_now:
                st.info("Click to extract flow features and run detection.")
                st.stop()
            with st.spinner("Extracting network flows from the packet capture..."):
                with tempfile.TemporaryDirectory(prefix="sentinelai_pcap_") as work_dir:
                    capture_path = Path(work_dir) / f"capture{suffix}"
                    flows_path = Path(work_dir) / "flows.csv"
                    capture_path.write_bytes(uploaded_file.getvalue())
                    convert_pcap_to_csv(capture_path, flows_path)
                    data = pd.read_csv(
                        flows_path,
                        low_memory=False,
                    )
            data = normalize_flow_columns(
                data,
                detector.feature_columns,
                convert_time_units=True,
            )
        else:
            data = pd.read_csv(
                uploaded_file,
                low_memory=False,
            )
            data = normalize_flow_columns(
                data,
                detector.feature_columns,
                convert_time_units=False,
            )

        missing_features = [
            feature for feature in detector.feature_columns if feature not in data.columns
        ]
        if missing_features:
            st.error(
                f"Input is missing {len(missing_features)} required model features: "
                f"{', '.join(missing_features[:8])}. Provide a CIC-IDS-compatible "
                "flow CSV or PCAP."
            )
            st.stop()

        if data.empty:
            st.warning("The uploaded CSV contains no records.")
            st.stop()

        st.write(f"Records uploaded: {len(data):,}")

        st.subheader("Input preview")
        st.dataframe(data.head(10), use_container_width=True)

        if analyze_now or (not is_pcap and st.button("Analyze traffic", type="primary")):
            with st.spinner(
                f"Analyzing {len(data):,} network flows..."
            ):
                results = detector.predict(data)

            total = len(results)
            predicted_attacks = int(
                (results["predicted_class"] != "BENIGN").sum()
            )
            anomalies = int(results["is_anomaly"].sum())

            st.subheader("Detection summary")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total flows", f"{total:,}")
            col2.metric(
                "Predicted known attacks",
                f"{predicted_attacks:,}",
            )
            col3.metric("Anomalous flows", f"{anomalies:,}")
            generalized_alerts = int(
                results["is_generalized_attack"].fillna(False).astype(bool).sum()
            ) if "is_generalized_attack" in results.columns else 0
            col4.metric("Learned binary model flags", f"{generalized_alerts:,}")
            model_alerts = int(
                results["is_model_generalized_attack"]
                .fillna(False)
                .astype(bool)
                .sum()
            ) if "is_model_generalized_attack" in results.columns else 0
            source_burst_flow_matches = int(
                results["is_source_burst_alert"]
                .fillna(False)
                .astype(bool)
                .sum()
            ) if "is_source_burst_alert" in results.columns else 0
            high_rate_multiport_flows = int(
                results["is_high_rate_multiport_activity"]
                .fillna(False)
                .astype(bool)
                .sum()
            ) if "is_high_rate_multiport_activity" in results.columns else 0
            event_table = source_minute_event_table(
                data,
                results["unique_destination_ports_per_source_minute"].to_numpy(),
                results["unique_destination_ips_per_source_minute"].to_numpy(),
                detector.source_burst_threshold,
            )
            burst_events = int(event_table["source_burst_rule"].sum()) if not event_table.empty else 0
            multiport_events = int(event_table["high_rate_multiport_rule"].sum()) if not event_table.empty else 0
            st.caption(
                f"Learned classifier flags: {model_alerts:,}. Separate heuristic: "
                f"{burst_events:,} source-minute events cover {source_burst_flow_matches:,} flow rows; "
                f"{multiport_events:,} also exceed the port-diversity threshold. "
                "These are not model attack flags."
            )
            if high_rate_multiport_flows:
                st.warning(
                    "Behavioral heuristic matches: "
                    f"{multiport_events:,} source-minute events ({high_rate_multiport_flows:,} associated flows) matched high source-rate "
                    "and destination-port-diversity thresholds. This is not a "
                    "validated attack prediction or attack-family classification."
                )
            if not event_table.empty:
                st.dataframe(event_table, use_container_width=True)
            st.caption("Counts are flow-level; the same flow may match multiple signals, and incidents are not deduplicated.")

            st.caption(
                "These are model predictions. An anomaly is not proof "
                "of an attack, and a predicted class is not a verified incident."
            )

            st.subheader("Predicted traffic classes")
            class_counts = results["predicted_class"].value_counts()
            st.bar_chart(class_counts)

            st.subheader("Anomaly detection")
            anomaly_counts = (
                results["is_anomaly"]
                .map({True: "Anomaly", False: "No anomaly"})
                .value_counts()
            )
            st.bar_chart(anomaly_counts)

            st.subheader("Flow-level results")
            st.dataframe(results, use_container_width=True)

            csv_data = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download detection results",
                data=csv_data,
                file_name="nids_detection_results.csv",
                mime="text/csv",
            )

    except Exception as error:
        st.error(f"Could not analyze the uploaded file: {error}")

else:
    st.info("Upload a compatible network-flow CSV to begin.")
