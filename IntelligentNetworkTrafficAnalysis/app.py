from pathlib import Path
import json
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_traffic_model.joblib"
FEATURE_PATH = BASE_DIR / "models" / "feature_columns.json"
LIVE_TRAFFIC_FILE = BASE_DIR / "reports" / "live_traffic.csv"
LIVE_ALERT_FILE = BASE_DIR / "reports" / "live_alerts.csv"
LIVE_PREDICTIONS_FILE = (
    BASE_DIR / "reports" / "live_flow_predictions.csv"
)

def read_csv_tail(path, max_rows=100):
    """
    Read the CSV header and only the most recent rows.
    This avoids repeatedly loading the entire growing log.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    with path.open("rb") as file:
        header = file.readline()
        header_end = file.tell()

        file.seek(0, 2)
        position = file.tell()

        if position <= header_end:
            return pd.DataFrame()

        chunks = []
        newline_count = 0
        block_size = 8192

        while position > header_end and newline_count <= max_rows:
            size = min(block_size, position - header_end)
            start = position - size

            file.seek(start)
            block = file.read(size)

            chunks.append(block)
            newline_count += block.count(b"\n")
            position = start

        tail = b"".join(reversed(chunks))
        lines = tail.splitlines()

        # The first line may be incomplete if reading started
        # in the middle of a CSV row.
        if position > header_end and lines:
            lines = lines[1:]

        lines = lines[-max_rows:]

        content = header + b"\n".join(lines)

        if lines:
            content += b"\n"

    if not lines:
        return pd.DataFrame()

    return pd.read_csv(io.BytesIO(content))
st.set_page_config(
    page_title="Intelligent Network Traffic Analyzer",
    page_icon="🛡️",
    layout="wide",
)

# --------------------------------------------------
# 2. Load saved model and feature list
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)

    with open(FEATURE_PATH, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    return model, feature_columns


st.title("🛡️ Intelligent Network Traffic Analyzer")

st.write(
    "Analyze network-flow records using a trained "
    "machine learning model to classify traffic as "
    "Normal or potentially malicious."
)

if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
    st.error(
        "Model artifacts are missing. Run "
        "notebook/06_train_models.py first."
    )
    st.stop()

try:
    model, feature_columns = load_artifacts()
except Exception as error:
    st.error(f"Could not load model artifacts: {error}")
    st.stop()

with st.sidebar:
    st.header("Model information")
    st.write("Algorithm: Random Forest")
    st.write(f"Expected features: {len(feature_columns)}")
    st.caption(
        "Predictions are model estimates, not proof "
        "that a connection is malicious."
    )

# --------------------------------------------------
# 3. Upload CSV
# --------------------------------------------------

st.subheader("Upload network traffic data")
# --------------------------------------------------
# LIVE NETWORK MONITORING PANEL
# --------------------------------------------------

st.divider()
st.subheader("Live Network Monitoring")

st.caption(
    "Reads packet and alert logs written by "
    "notebook/07_live_monitor.py. "
    "Updates every 3 seconds while this page is open."
)


@st.fragment(run_every="3s")
def display_live_monitor():
    recent_traffic = read_csv_tail(
        LIVE_TRAFFIC_FILE,
        max_rows=300,
    )

    recent_alerts = read_csv_tail(
        LIVE_ALERT_FILE,
        max_rows=100,
    )

    # Traffic summary
    st.markdown("#### Recent packet activity")

    if recent_traffic.empty:
        st.info(
            "No traffic records found. Start the live "
            "monitor in a separate terminal."
        )
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Recent packet records",
            f"{len(recent_traffic):,}",
        )

        col2.metric(
            "Recent TCP records",
            int(
                recent_traffic["protocol"].eq("TCP").sum()
            ),
        )

        col3.metric(
            "Recent UDP records",
            int(
                recent_traffic["protocol"].eq("UDP").sum()
            ),
        )

        st.caption(
            "Statistics describe the most recent 300 "
            "records, not total traffic since startup."
        )

        if "protocol" in recent_traffic.columns:
            protocol_counts = (
                recent_traffic["protocol"]
                .value_counts()
                .rename_axis("Protocol")
                .to_frame("Packets")
            )

            st.bar_chart(protocol_counts)

        st.dataframe(
            recent_traffic.tail(20).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

    # Alert summary
    st.markdown("#### Recent security alerts")

    if recent_alerts.empty:
        st.success("No alert records found.")
    else:
        st.metric(
            "Recent logged alerts",
            len(recent_alerts),
        )

        st.dataframe(
            recent_alerts.tail(20).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download recent alert records",
            data=recent_alerts.to_csv(index=False).encode(
                "utf-8"
            ),
            file_name="recent_network_alerts.csv",
            mime="text/csv",
            key="download_live_alerts",
        )

    st.caption(
        "Alerts are heuristic indicators and may include "
        "false positives. They do not confirm an attack."
    )


display_live_monitor()
# --------------------------------------------------
# EXPERIMENTAL LIVE ML PREDICTIONS
# --------------------------------------------------

@st.fragment(run_every="3s")
def display_live_ml_predictions():
    st.subheader("Live ML Flow Predictions")

    st.caption(
        "Shows predictions from the latest completed capture batch. "
        "These predictions remain experimental."
    )

    if (
        not LIVE_PREDICTIONS_FILE.exists()
        or LIVE_PREDICTIONS_FILE.stat().st_size == 0
    ):
        st.info(
            "No predictions yet. Start the continuous live "
            "ML monitor in a separate terminal."
        )
        return

    try:
        predictions = pd.read_csv(
            LIVE_PREDICTIONS_FILE,
            low_memory=False,
        )
    except Exception as error:
        st.error(f"Could not read prediction file: {error}")
        return

    if predictions.empty:
        st.info("The latest capture batch contains no flow records.")
        return

    if "predicted_traffic" not in predictions.columns:
        st.error("Prediction output is missing its class column.")
        return

    predicted = predictions["predicted_traffic"].astype(str)

    normal_count = int(predicted.eq("Normal").sum())
    attack_count = int(predicted.eq("Attack").sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Flows in latest batch", f"{len(predictions):,}")
    col2.metric("Normal predictions", f"{normal_count:,}")
    col3.metric("Attack predictions", f"{attack_count:,}")

    if attack_count:
        st.warning(
            f"The model flagged {attack_count} flow(s) as Attack. "
            "Investigate these experimental predictions; they do "
            "not confirm malicious activity."
        )
    else:
        st.success(
            "No flows were classified as Attack in the latest batch."
        )

    st.dataframe(
        predictions.tail(25).iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )


display_live_ml_predictions()
st.divider()
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help=(
        "Use a CIC-IDS2017 CSV file or the project's "
        "prepared test.csv file."
    ),
)

st.info(
    "For initial testing, upload "
    "data/processed/split/test.csv. "
    "It contains labels that allow us to compare "
    "predictions against known results."
)

if uploaded_file is None:
    st.stop()

# --------------------------------------------------
# 4. Read and validate uploaded data
# --------------------------------------------------

try:
    data = pd.read_csv(uploaded_file, low_memory=False)
except Exception as error:
    st.error(f"Could not read the uploaded CSV: {error}")
    st.stop()

# Original CIC files may have leading spaces in column names.
data.columns = data.columns.astype(str).str.strip()

if data.empty:
    st.error("The uploaded CSV contains no records.")
    st.stop()

if data.columns.duplicated().any():
    st.error("Duplicate column names were found after cleaning.")
    st.stop()

missing_features = [
    col for col in feature_columns
    if col not in data.columns
]

if missing_features:
    st.error(
        f"The file is missing {len(missing_features)} "
        "required model features."
    )
    st.write("Missing feature names:")
    st.code("\n".join(missing_features))
    st.stop()

# --------------------------------------------------
# 5. Extract actual labels if available
# --------------------------------------------------

actual_labels = None

if "Target" in data.columns:
    values = data["Target"].astype("string").str.strip()
    non_null = values.dropna()

    if (
        not non_null.empty
        and non_null.isin(["Normal", "Attack"]).all()
    ):
        actual_labels = values

elif "Original_Label" in data.columns:
    raw_labels = (
        data["Original_Label"].astype("string").str.strip()
    )

    actual_labels = pd.Series(
        index=data.index,
        dtype="object",
    )

    valid = raw_labels.notna() & raw_labels.ne("")

    actual_labels.loc[valid] = np.where(
        raw_labels.loc[valid].str.upper().eq("BENIGN"),
        "Normal",
        "Attack",
    )

elif "Label" in data.columns:
    raw_labels = data["Label"].astype("string").str.strip()

    actual_labels = pd.Series(
        index=data.index,
        dtype="object",
    )

    valid = raw_labels.notna() & raw_labels.ne("")

    actual_labels.loc[valid] = np.where(
        raw_labels.loc[valid].str.upper().eq("BENIGN"),
        "Normal",
        "Attack",
    )

# --------------------------------------------------
# 6. Prepare features and predict
# --------------------------------------------------

X = data[feature_columns].copy()

for column in feature_columns:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce",
    )

X = X.replace([np.inf, -np.inf], np.nan)

with st.spinner("Analyzing network traffic..."):
    try:
        predictions = model.predict(X)
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

# --------------------------------------------------
# 7. Summary metrics
# --------------------------------------------------

prediction_series = pd.Series(
    predictions,
    name="Predicted_Traffic",
)

normal_count = int((prediction_series == "Normal").sum())
attack_count = int((prediction_series == "Attack").sum())
total_count = len(prediction_series)

st.subheader("Traffic analysis summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Records analyzed", f"{total_count:,}")
col2.metric("Normal predictions", f"{normal_count:,}")
col3.metric("Attack predictions", f"{attack_count:,}")
col4.metric(
    "Potential attack rate",
    f"{attack_count / total_count * 100:.2f}%",
)

# --------------------------------------------------
# 8. Charts
# --------------------------------------------------

st.subheader("Predicted traffic distribution")

counts = pd.Series(
    predictions
).value_counts().reindex(
    ["Normal", "Attack"],
    fill_value=0,
)

left, right = st.columns(2)

with left:
    st.bar_chart(counts)

with right:
    st.write("Prediction percentages")

    percentages = (
        counts / total_count * 100
    ).round(2)

    st.dataframe(
        pd.DataFrame({
            "Traffic class": percentages.index,
            "Percentage": percentages.values,
            "Records": counts.values,
        }),
        hide_index=True,
        use_container_width=True,
    )

# --------------------------------------------------
# 9. Result table and download
# --------------------------------------------------

st.subheader("Prediction results")

useful_columns = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
]

display_columns = [
    col for col in useful_columns
    if col in data.columns
]

results = data[display_columns].copy()
results["Predicted_Traffic"] = predictions

if actual_labels is not None:
    results["Actual_Traffic"] = actual_labels.to_numpy()

st.dataframe(
    results.head(200),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Showing the first 200 records. The download contains "
    "all processed records."
)

csv_data = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download prediction results (CSV)",
    data=csv_data,
    file_name="network_traffic_predictions.csv",
    mime="text/csv",
)

# --------------------------------------------------
# 10. Evaluation when ground-truth labels exist
# --------------------------------------------------

if actual_labels is not None:
    valid = actual_labels.notna()

    y_true = actual_labels.loc[valid].astype(str)
    y_pred = prediction_series.loc[valid].astype(str)

    if len(y_true) > 0:
        st.subheader("Comparison with known labels")

        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric(
            "Accuracy",
            f"{accuracy_score(y_true, y_pred):.4f}",
        )

        metric2.metric(
            "Attack precision",
            f"{precision_score(
                y_true, y_pred,
                pos_label='Attack',
                zero_division=0
            ):.4f}",
        )

        metric3.metric(
            "Attack recall",
            f"{recall_score(
                y_true, y_pred,
                pos_label='Attack',
                zero_division=0
            ):.4f}",
        )

        metric4.metric(
            "Attack F1-score",
            f"{f1_score(
                y_true, y_pred,
                pos_label='Attack',
                zero_division=0
            ):.4f}",
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=["Normal", "Attack"],
        )

        st.write("Confusion matrix")

        st.dataframe(
            pd.DataFrame(
                matrix,
                index=["Actual Normal", "Actual Attack"],
                columns=["Predicted Normal", "Predicted Attack"],
            ),
            use_container_width=True,
        )

        st.caption(
            "These metrics are meaningful as a generalization "
            "test only when the uploaded labeled records were "
            "not used to fit or select the model."
        )

st.success("Traffic analysis completed.")