from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(
    r"C:\CN_PROJECT\IntelligentNetworkTrafficAnalysis"
)

MODEL_FILE = BASE_DIR / "models" / "best_traffic_model.joblib"
FEATURE_FILE = BASE_DIR / "models" / "feature_columns.json"
TRAIN_FILE = BASE_DIR / "data" / "processed" / "split" / "train.csv"
FLOW_FILE = BASE_DIR / "reports" / "live_flows.csv"
OUTPUT_FILE = BASE_DIR / "reports" / "live_flow_predictions.csv"

# Model feature name -> PyFlowMeter feature name
FEATURE_MAP = {
    "Destination Port": "dst_port",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "tot_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",
    "Total Length of Fwd Packets": "totlen_fwd_pkts",
    "Total Length of Bwd Packets": "totlen_bwd_pkts",
    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "flow_byts_s",
    "Flow Packets/s": "flow_pkts_s",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_tot",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_tot",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Fwd URG Flags": "fwd_urg_flags",
    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",
    "Fwd Packets/s": "fwd_pkts_s",
    "Bwd Packets/s": "bwd_pkts_s",
    "Min Packet Length": "pkt_len_min",
    "Max Packet Length": "pkt_len_max",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",
    "FIN Flag Count": "fin_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
    "RST Flag Count": "rst_flag_cnt",
    "PSH Flag Count": "psh_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "URG Flag Count": "urg_flag_cnt",
    "CWE Flag Count": "cwe_flag_count",
    "ECE Flag Count": "ece_flag_cnt",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "pkt_size_avg",
    "Avg Fwd Segment Size": "fwd_seg_size_avg",
    "Avg Bwd Segment Size": "bwd_seg_size_avg",
    "Fwd Header Length.1": "fwd_header_len",
    "Subflow Fwd Packets": "subflow_fwd_pkts",
    "Subflow Fwd Bytes": "subflow_fwd_byts",
    "Subflow Bwd Packets": "subflow_bwd_pkts",
    "Subflow Bwd Bytes": "subflow_bwd_byts",
    "Init_Win_bytes_forward": "init_fwd_win_byts",
    "Init_Win_bytes_backward": "init_bwd_win_byts",
    "act_data_pkt_fwd": "fwd_act_data_pkts",
    "min_seg_size_forward": "fwd_seg_size_min",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
}


def main():
    for path in [MODEL_FILE, FEATURE_FILE, TRAIN_FILE, FLOW_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    with open(FEATURE_FILE, encoding="utf-8") as file:
        expected_features = json.load(file)

    missing_mappings = [
        feature for feature in expected_features
        if feature not in FEATURE_MAP
    ]

    if missing_mappings:
        raise ValueError(
            "No mapping exists for these model features:\n"
            + "\n".join(missing_mappings)
        )

    # Verify that the two source training columns are equivalent
    # before duplicating one PyFlowMeter feature into both columns.
    source_check = pd.read_csv(
        TRAIN_FILE,
        usecols=["Fwd Header Length", "Fwd Header Length.1"],
        low_memory=False,
    )

    first = pd.to_numeric(
        source_check["Fwd Header Length"], errors="coerce"
    )
    second = pd.to_numeric(
        source_check["Fwd Header Length.1"], errors="coerce"
    )

    if not np.allclose(
        first.to_numpy(dtype=float),
        second.to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise ValueError(
            "The training header-length columns differ. "
            "We cannot safely map both to fwd_header_len."
        )

    live = pd.read_csv(FLOW_FILE, low_memory=False)
    live.columns = live.columns.str.strip()

    if live.empty:
        raise ValueError("The live-flow CSV contains no records.")

    missing_live_columns = [
        FEATURE_MAP[feature]
        for feature in expected_features
        if FEATURE_MAP[feature] not in live.columns
    ]

    if missing_live_columns:
        raise ValueError(
            "PyFlowMeter output is missing required features:\n"
            + "\n".join(sorted(set(missing_live_columns)))
        )

    # Build the model input in exactly the saved feature order.
    source_columns = [
        FEATURE_MAP[feature] for feature in expected_features
    ]

    X = live.loc[:, source_columns].copy()
    X.columns = expected_features

    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    # Save the exact 70 model input features for validation.
    ALIGNED_FILE = (
        BASE_DIR / "reports" / "live_flow_features_aligned.csv"
    )
    X.to_csv(ALIGNED_FILE, index=False)
    print("Aligned feature matrix saved:", ALIGNED_FILE)
    
    # Load the saved preprocessing/model pipeline.
    model = joblib.load(MODEL_FILE)

    print("Live flows:", len(live))
    print("Model features:", len(expected_features))
    print("Running experimental predictions...")

    predictions = model.predict(X)

    result_columns = [
        col for col in [
            "timestamp", "src_ip", "dst_ip",
            "src_port", "dst_port", "protocol"
        ]
        if col in live.columns
    ]

    results = live[result_columns].copy()
    results["predicted_traffic"] = predictions

    # Random Forest scores are not automatically calibrated
    # probabilities or guarantees that a flow is malicious.
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if "Attack" in classes:
            attack_index = classes.index("Attack")
            results["model_attack_score"] = (
                model.predict_proba(X)[:, attack_index]
            ).round(4)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nPrediction counts:")
    print(results["predicted_traffic"].value_counts())

    print("\nSaved experimental predictions to:")
    print(OUTPUT_FILE)

    print("\nWARNING:")
    print(
        "PyFlowMeter feature calculations may differ from "
        "the calculations used in CIC-IDS2017. Validate "
        "feature parity before treating these predictions "
        "as reliable security detections."
    )


if __name__ == "__main__":
    main()