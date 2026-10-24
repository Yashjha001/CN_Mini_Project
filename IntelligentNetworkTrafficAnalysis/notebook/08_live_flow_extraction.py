from pathlib import Path
import time

from pyflowmeter.sniffer import create_sniffer

BASE_DIR = Path(
    r"C:\CN_PROJECT\IntelligentNetworkTrafficAnalysis"
)

OUTPUT_FILE = BASE_DIR / "reports" / "live_flows.csv"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

sniffer = create_sniffer(
    input_interface="Wi-Fi",
    to_csv=True,
    output_file=str(OUTPUT_FILE),
    verbose=False,
)

try:
    sniffer.start()

    print("Live flow extraction started.")
    print("Interface: Wi-Fi")
    print("Output:", OUTPUT_FILE)
    print("Press Ctrl+C to stop.")

    # Poll rather than blocking indefinitely in join().
    while sniffer.running:
        time.sleep(0.25)

except KeyboardInterrupt:
    print("\nStop requested. Stopping packet capture...")

except Exception as error:
    print("Flow extraction error:", error)

finally:
    try:
        if sniffer.running:
            # Request a stop without blocking here.
            sniffer.stop(join=False)

        # Allow time for the capture thread to finish.
        sniffer.join(timeout=10)

        if sniffer.running:
            print(
                "WARNING: Capture thread has not stopped. "
                "Use the terminal's Kill button if necessary."
            )
        else:
            print("Flow extraction stopped.")

    except Exception as error:
        print("Error during shutdown:", error)

    print("Output file:", OUTPUT_FILE)