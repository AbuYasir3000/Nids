
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_detector import HybridNIDS


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
except Exception as error:
    st.error(f"Could not load the trained models: {error}")
    st.info(
        "Check that models/random_forest.joblib and "
        "models/isolation_forest.joblib exist."
    )
    st.stop()


st.subheader("Analyze network traffic")
st.write(
    "Upload a CSV containing network-flow features compatible with "
    "the CIC-IDS2017 dataset."
)

uploaded_file = st.file_uploader(
    "Choose a network-flow CSV file",
    type=["csv"],
)

MAX_ROWS = 300_000

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, low_memory=False)

        if data.empty:
            st.warning("The uploaded CSV contains no records.")
            st.stop()

        if len(data) > MAX_ROWS:
            st.warning(
                f"This dashboard currently supports up to "
                f"{MAX_ROWS:,} rows per upload. "
                f"Your file contains {len(data):,} rows."
            )
            st.stop()

        st.write(f"Records uploaded: {len(data):,}")

        st.subheader("Input preview")
        st.dataframe(data.head(10), use_container_width=True)

        if st.button("Analyze traffic", type="primary"):
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

            col1, col2, col3 = st.columns(3)
            col1.metric("Total flows", f"{total:,}")
            col2.metric(
                "Predicted known attacks",
                f"{predicted_attacks:,}",
            )
            col3.metric("Anomalous flows", f"{anomalies:,}")

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
