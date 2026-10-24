from pathlib import Path
from pyflowmeter.sniffer import create_sniffer

BASE_DIR = Path(
    r"C:\CN_PROJECT\IntelligentNetworkTrafficAnalysis"
)

PCAP_FILE = BASE_DIR / "reports" / "live_capture_test.pcap"
FLOW_FILE = BASE_DIR / "reports" / "live_flows.csv"

if not PCAP_FILE.exists() or PCAP_FILE.stat().st_size == 0:
    raise FileNotFoundError(f"PCAP file missing or empty: {PCAP_FILE}")

# Remove the previous empty output so the result is unambiguous.
if FLOW_FILE.exists():
    FLOW_FILE.unlink()

sniffer = create_sniffer(
    input_file=str(PCAP_FILE),
    to_csv=True,
    output_file=str(FLOW_FILE),
    verbose=False,
)

print("Processing existing PCAP...")
print("Input:", PCAP_FILE)
print("Output:", FLOW_FILE)

sniffer.start()
sniffer.join()

if FLOW_FILE.exists() and FLOW_FILE.stat().st_size > 0:
    print("Flow extraction produced a non-empty CSV.")
    print("File size:", FLOW_FILE.stat().st_size, "bytes")
else:
    print("ERROR: Flow extraction produced no CSV data.")
    raise RuntimeError("PyFlowMeter did not produce flow records.")