from pathlib import Path
from datetime import datetime
import csv
import subprocess
import sys
import time

import pandas as pd
from scapy.all import sniff, wrpcap, IP, TCP, UDP

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(
    r"C:\CN_PROJECT\IntelligentNetworkTrafficAnalysis"
)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

INTERFACE = "Wi-Fi"
CAPTURE_SECONDS = 30
PROCESS_TIMEOUT = 180
ALERT_COOLDOWN_SECONDS = 300

PCAP_FILE = REPORT_DIR / "live_capture_test.pcap"
FLOW_FILE = REPORT_DIR / "live_flows.csv"
PREDICTIONS_FILE = REPORT_DIR / "live_flow_predictions.csv"

TRAFFIC_FILE = REPORT_DIR / "live_traffic.csv"
ALERT_FILE = REPORT_DIR / "live_alerts.csv"

FLOW_SCRIPT = BASE_DIR / "notebook" / "09_process_pcap.py"
PREDICTION_SCRIPT = (
    BASE_DIR / "notebook" / "10_predict_live_flows.py"
)

TRAFFIC_FIELDS = [
    "timestamp",
    "source_ip",
    "destination_ip",
    "protocol",
    "source_port",
    "destination_port",
    "packet_length",
    "tcp_flags",
]

ALERT_FIELDS = [
    "timestamp",
    "severity",
    "rule",
    "source_ip",
    "destination_ip",
    "details",
]

last_ml_alert = {}


# --------------------------------------------------
# CSV helpers
# --------------------------------------------------

def timestamp_now():
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def append_rows(path, fieldnames, rows):
    if not rows:
        return

    needs_header = (
        not path.exists() or path.stat().st_size == 0
    )

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        if needs_header:
            writer.writeheader()

        writer.writerows(rows)


# --------------------------------------------------
# Record packet metadata for the existing dashboard
# --------------------------------------------------

def log_packet_metadata(packets):
    rows = []

    for packet in packets:
        if IP not in packet:
            continue

        ip = packet[IP]

        protocol = {
            1: "ICMP",
            6: "TCP",
            17: "UDP",
        }.get(int(ip.proto), f"IP-{int(ip.proto)}")

        src_port = ""
        dst_port = ""
        tcp_flags = ""

        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            tcp_flags = str(packet[TCP].flags)

        elif UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport

        rows.append({
            "timestamp": timestamp_now(),
            "source_ip": ip.src,
            "destination_ip": ip.dst,
            "protocol": protocol,
            "source_port": src_port,
            "destination_port": dst_port,
            "packet_length": len(packet),
            "tcp_flags": tcp_flags,
        })

    append_rows(TRAFFIC_FILE, TRAFFIC_FIELDS, rows)

    return len(rows)


# --------------------------------------------------
# Run an existing pipeline script and detect failures
# --------------------------------------------------

def run_script(script_path):
    print(f"\nRunning: {script_path.name}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=PROCESS_TIMEOUT,
    )

    if result.stdout:
        print(result.stdout, end="")

    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")

    # PyFlowMeter can raise an exception inside a background
    # thread even when its parent Python process exits normally.
    thread_failed = (
        "Exception in thread AsyncSniffer" in result.stderr
        or "Traceback (most recent call last)" in result.stderr
    )

    if result.returncode != 0 or thread_failed:
        raise RuntimeError(
            f"{script_path.name} failed. "
            "Review the output above before restarting."
        )


# --------------------------------------------------
# Log experimental Random Forest attack predictions
# --------------------------------------------------

def log_ml_alerts(predictions):
    attack_rows = predictions[
        predictions["predicted_traffic"].astype(str).eq("Attack")
    ].copy()

    if attack_rows.empty:
        print("ML alert status: no flows predicted as Attack.")
        return

    required = {"src_ip", "dst_ip"}

    if not required.issubset(attack_rows.columns):
        raise ValueError(
            "Prediction CSV lacks source/destination IP fields."
        )

    grouped = attack_rows.groupby(
        ["src_ip", "dst_ip"],
        dropna=False,
    ).size()

    alerts = []
    now_epoch = time.time()
    now_text = timestamp_now()

    for (src_ip, dst_ip), count in grouped.items():
        key = (str(src_ip), str(dst_ip))

        previous = last_ml_alert.get(key, 0)

        if now_epoch - previous < ALERT_COOLDOWN_SECONDS:
            continue

        last_ml_alert[key] = now_epoch

        alerts.append({
            "timestamp": now_text,
            "severity": "EXPERIMENTAL",
            "rule": "Random Forest live-flow prediction",
            "source_ip": str(src_ip),
            "destination_ip": str(dst_ip),
            "details": (
                f"Model predicted {int(count)} flow(s) as Attack "
                f"in the latest {CAPTURE_SECONDS}-second capture "
                "window. Experimental result: feature calculations "
                "have not been fully validated against the training "
                "extractor. Investigate; this is not confirmation "
                "of malicious activity."
            ),
        })

    append_rows(ALERT_FILE, ALERT_FIELDS, alerts)

    print("Flows predicted as Attack:", len(attack_rows))
    print("New ML alert records written:", len(alerts))


# --------------------------------------------------
# One capture -> extraction -> prediction cycle
# --------------------------------------------------

def run_cycle():
    print("\n" + "=" * 68)
    print("INTELLIGENT NETWORK TRAFFIC ANALYSIS")
    print("Experimental live ML monitoring")
    print("=" * 68)

    print(
        f"Capturing on {INTERFACE} for "
        f"{CAPTURE_SECONDS} seconds..."
    )

    packets = sniff(
        iface=INTERFACE,
        timeout=CAPTURE_SECONDS,
        store=True,
        promisc=False,
    )

    print("Packets captured:", f"{len(packets):,}")

    if not packets:
        print("No packets captured in this window. Skipping.")
        return

    # Write a new PCAP for this capture window.
    wrpcap(str(PCAP_FILE), packets)

    logged_packets = log_packet_metadata(packets)
    print("IP packet records appended:", f"{logged_packets:,}")

    del packets

    # Remove old outputs so a failed run cannot reuse stale results.
    for file in [FLOW_FILE, PREDICTIONS_FILE]:
        if file.exists():
            file.unlink()

    # 1. Extract flow features from the new PCAP.
    run_script(FLOW_SCRIPT)

    if not FLOW_FILE.exists() or FLOW_FILE.stat().st_size == 0:
        raise RuntimeError("Flow extraction produced no output.")

    # 2. Apply the saved model to the new flows.
    run_script(PREDICTION_SCRIPT)

    if (
        not PREDICTIONS_FILE.exists()
        or PREDICTIONS_FILE.stat().st_size == 0
    ):
        raise RuntimeError("Prediction output is missing or empty.")

    predictions = pd.read_csv(
        PREDICTIONS_FILE,
        low_memory=False,
    )

    if "predicted_traffic" not in predictions.columns:
        raise ValueError(
            "Prediction file has no predicted_traffic column."
        )

    print("\nPrediction summary:")
    print(
        predictions["predicted_traffic"]
        .value_counts(dropna=False)
        .to_string()
    )

    # 3. Save an alert only when the model predicts Attack.
    log_ml_alerts(predictions)

    print("\nCycle completed:", timestamp_now())


# --------------------------------------------------
# Continuous loop
# --------------------------------------------------

def main():
    print("Interface:", INTERFACE)
    print("Capture duration:", CAPTURE_SECONDS, "seconds")
    print("Packet log:", TRAFFIC_FILE)
    print("Alert log:", ALERT_FILE)
    print(
        "\nThis is experimental batch-based monitoring. "
        "Press Ctrl+C to stop between cycles."
    )

    try:
        while True:
            run_cycle()
            print("\nStarting next capture window in 2 seconds...")
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nStop requested. Monitoring stopped.")

    except Exception as error:
        print("\nMonitoring stopped due to an error:")
        print(error)
        print("Fix the error before restarting.")


if __name__ == "__main__":
    main()