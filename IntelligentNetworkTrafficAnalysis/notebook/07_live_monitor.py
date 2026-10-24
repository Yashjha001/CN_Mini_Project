from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import argparse
import csv
import time

from scapy.all import (
    conf,
    sniff,
    IP,
    TCP,
    UDP,
    ICMP,
    get_if_addr,
)

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

BASE_DIR = Path(
    r"C:\CN_PROJECT\IntelligentNetworkTrafficAnalysis"
)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAFFIC_FILE = REPORT_DIR / "live_traffic.csv"
ALERT_FILE = REPORT_DIR / "live_alerts.csv"

WINDOW_SECONDS = 10
LOCAL_IP = None

# Initial thresholds for a student demonstration.
# They are configurable heuristics, not universal
# definitions of a network attack.
PORT_SCAN_UNIQUE_PORTS = 15
PORT_SCAN_MIN_PACKETS = 20
SYN_BURST_THRESHOLD = 100
# Initial demonstration threshold: packets per source
# within the rolling 10-second window.
HIGH_PACKET_RATE_THRESHOLD = 3000
ALERT_COOLDOWN_SECONDS = 30

# State maintained over the recent time window
source_windows = defaultdict(deque)
pair_windows = defaultdict(deque)
inbound_scan_windows = defaultdict(deque)
destination_syn_windows = defaultdict(deque)
last_alert_times = {}

traffic_writer = None
alert_writer = None
traffic_handle = None
alert_handle = None
packet_count = 0


def timestamp_text():
    """Return the current local timestamp."""
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def prune_window(queue, now):
    """Remove events older than the monitoring window."""
    while queue and now - queue[0][0] > WINDOW_SECONDS:
        queue.popleft()


# --------------------------------------------------
# 2. Save alert events
# --------------------------------------------------

def create_alert(rule, severity, src_ip, dst_ip, details):
    """Print and persist an alert, with a cooldown."""
    now = time.time()
    if rule == "Unusually high packet rate":
        # One cooldown per source, regardless of destination.
        key = (rule, src_ip, "*")
    elif rule == "Possible SYN flood or connection burst":
        # One cooldown per destination for this alert type.
        key = (rule, "*", dst_ip)
    else:
        key = (rule, src_ip, dst_ip)

    previous = last_alert_times.get(key, 0)

    if now - previous < ALERT_COOLDOWN_SECONDS:
        return

    last_alert_times[key] = now
    timestamp = timestamp_text()

    alert = {
        "timestamp": timestamp,
        "severity": severity,
        "rule": rule,
        "source_ip": src_ip,
        "destination_ip": dst_ip,
        "details": details,
    }

    alert_writer.writerow(alert)
    alert_handle.flush()

    print("\n" + "!" * 68)
    print(f"[{timestamp}] {severity} ALERT: {rule}")
    print(f"Source      : {src_ip}")
    print(f"Destination : {dst_ip}")
    print(f"Details     : {details}")
    print("!" * 68)


# --------------------------------------------------
# 3. Process each captured packet
# --------------------------------------------------

def handle_packet(packet):
    global packet_count

    if IP not in packet:
        return

    now = time.time()

    ip = packet[IP]
    src_ip = ip.src
    dst_ip = ip.dst

    src_port = ""
    dst_port = ""
    tcp_flags = ""

    protocol_number = int(ip.proto)

    protocol = {
        1: "ICMP",
        6: "TCP",
        17: "UDP",
    }.get(protocol_number, f"IP-{protocol_number}")

    is_syn_without_ack = False

    if TCP in packet:
        tcp = packet[TCP]
        protocol = "TCP"
        src_port = tcp.sport
        dst_port = tcp.dport
        tcp_flags = str(tcp.flags)

        flags_value = int(tcp.flags)

        is_syn_without_ack = (
            bool(flags_value & 0x02)
            and not bool(flags_value & 0x10)
        )

    elif UDP in packet:
        udp = packet[UDP]
        protocol = "UDP"
        src_port = udp.sport
        dst_port = udp.dport

    elif ICMP in packet:
        protocol = "ICMP"

    packet_length = len(packet)
    packet_count += 1

    # Save packet metadata. Payloads are not logged.
    traffic_writer.writerow({
        "timestamp": timestamp_text(),
        "source_ip": src_ip,
        "destination_ip": dst_ip,
        "protocol": protocol,
        "source_port": src_port,
        "destination_port": dst_port,
        "packet_length": packet_length,
        "tcp_flags": tcp_flags,
    })

    # Flush periodically instead of on every packet.
    if packet_count % 25 == 0:
        traffic_handle.flush()

    # ------------------------------------------------
    # Rule A: High packet volume from one source
    # ------------------------------------------------

    source_queue = source_windows[src_ip]
    source_queue.append((now,))

    prune_window(source_queue, now)

    if len(source_queue) >= HIGH_PACKET_RATE_THRESHOLD:
        create_alert(
            rule="Unusually high packet rate",
            severity="MEDIUM",
            src_ip=src_ip,
            dst_ip="MULTIPLE",
            details=(
                f"{len(source_queue)} packets from this source "
                f"within {WINDOW_SECONDS} seconds, across "
                "all destinations."
            ),
        )

    # ------------------------------------------------
    # Rule B: Possible port scanning
    # Many destination ports for one source/destination
    # pair within a short time window.
    # ------------------------------------------------

    # Only examine outbound initial TCP SYN packets
    # originating from our own interface's IPv4 address.
    if (
        TCP in packet
        and is_syn_without_ack
        and LOCAL_IP is not None
        and src_ip == LOCAL_IP
        and dst_port != ""
    ):
        pair_key = (src_ip, dst_ip)
        pair_queue = pair_windows[pair_key]

        pair_queue.append((now, int(dst_port)))
        prune_window(pair_queue, now)

        distinct_ports = {
            event[1] for event in pair_queue
        }

        if (
            len(pair_queue) >= PORT_SCAN_MIN_PACKETS
            and len(distinct_ports) >= PORT_SCAN_UNIQUE_PORTS
        ):
            create_alert(
                rule="Possible outbound port scan",
                severity="HIGH",
                src_ip=src_ip,
                dst_ip=dst_ip,
                details=(
                    f"{len(distinct_ports)} distinct TCP "
                    f"destination ports in "
                    f"{WINDOW_SECONDS} seconds."
                ),
            )

        # ------------------------------------------------
    # Rule B2: Possible inbound port scan
    # ------------------------------------------------

    if (
        TCP in packet
        and is_syn_without_ack
        and LOCAL_IP is not None
        and dst_ip == LOCAL_IP
        and src_ip != LOCAL_IP
        and dst_port != ""
    ):
        pair_key = (src_ip, dst_ip)
        scan_queue = inbound_scan_windows[pair_key]

        scan_queue.append((now, int(dst_port)))
        prune_window(scan_queue, now)

        distinct_ports = {
            event[1] for event in scan_queue
        }

        if (
            len(scan_queue) >= PORT_SCAN_MIN_PACKETS
            and len(distinct_ports) >= PORT_SCAN_UNIQUE_PORTS
        ):
            create_alert(
                rule="Possible inbound port scan",
                severity="HIGH",
                src_ip=src_ip,
                dst_ip=dst_ip,
                details=(
                    f"{len(distinct_ports)} distinct TCP "
                    f"destination ports targeted within "
                    f"{WINDOW_SECONDS} seconds."
                ),
            )

    # ------------------------------------------------
    # Rule C: Possible high-volume SYN activity
    # ------------------------------------------------

    if (
        is_syn_without_ack
        and LOCAL_IP is not None
        and dst_ip == LOCAL_IP
    ):
        syn_queue = destination_syn_windows[dst_ip]
        syn_queue.append((now, src_ip))
        prune_window(syn_queue, now)

        if len(syn_queue) >= SYN_BURST_THRESHOLD:
            unique_sources = len({
                event[1] for event in syn_queue
            })

            create_alert(
                rule="Possible SYN flood or connection burst",
                severity="HIGH",
                src_ip=src_ip,
                dst_ip=dst_ip,
                details=(
                    f"{len(syn_queue)} SYN packets without ACK "
                    f"observed within {WINDOW_SECONDS} seconds; "
                    f"{unique_sources} source address(es)."
                ),
            )


# --------------------------------------------------
# 4. List available network interfaces
# --------------------------------------------------

def list_interfaces():
    interfaces = list(conf.ifaces.values())

    print("\nAvailable network interfaces\n")

    for index, interface in enumerate(interfaces):
        print(f"[{index}] {interface.name}")
        print(
            "    "
            + str(getattr(interface, "description", ""))
        )

    print(
        "\nUse the interface index shown above when "
        "starting the monitor."
    )


def resolve_interface(value):
    interfaces = list(conf.ifaces.values())

    if value.isdigit():
        index = int(value)

        if index < 0 or index >= len(interfaces):
            raise ValueError("Invalid interface index.")

        return interfaces[index]

    matches = [
        interface
        for interface in interfaces
        if value.casefold() == str(interface.name).casefold()
    ]

    if not matches:
        raise ValueError(
            "Interface not found. Run the script with "
            "--list-interfaces and choose an available index."
        )

    return matches[0]


# --------------------------------------------------
# 5. Main
# --------------------------------------------------

def main():
    global LOCAL_IP
    global traffic_writer
    global alert_writer
    global traffic_handle
    global alert_handle

    parser = argparse.ArgumentParser(
        description="Live network traffic monitor"
    )

    parser.add_argument(
        "--list-interfaces",
        action="store_true",
        help="List available network interfaces",
    )

    parser.add_argument(
        "--interface",
        help="Interface index or exact interface name",
    )

    args = parser.parse_args()

    if args.list_interfaces:
        list_interfaces()
        return

    if not args.interface:
        parser.error(
            "Choose an interface using --interface INDEX. "
            "First run --list-interfaces."
        )

    try:
        interface = resolve_interface(args.interface)
    except ValueError as error:
        parser.error(str(error))

    try:
        LOCAL_IP = get_if_addr(interface.name)
        if not LOCAL_IP or LOCAL_IP == "0.0.0.0":
            LOCAL_IP = None
    except Exception:
        LOCAL_IP = None

    print("Local IPv4 address:", LOCAL_IP or "Not detected")

    # Start CSV logs
    traffic_exists = (
        TRAFFIC_FILE.exists() and TRAFFIC_FILE.stat().st_size > 0
    )
    alerts_exist = (
        ALERT_FILE.exists() and ALERT_FILE.stat().st_size > 0
    )

    traffic_handle = open(
        TRAFFIC_FILE, "a", newline="", encoding="utf-8"
    )
    alert_handle = open(
        ALERT_FILE, "a", newline="", encoding="utf-8"
    )

    traffic_writer = csv.DictWriter(
        traffic_handle,
        fieldnames=[
            "timestamp",
            "source_ip",
            "destination_ip",
            "protocol",
            "source_port",
            "destination_port",
            "packet_length",
            "tcp_flags",
        ],
    )

    alert_writer = csv.DictWriter(
        alert_handle,
        fieldnames=[
            "timestamp",
            "severity",
            "rule",
            "source_ip",
            "destination_ip",
            "details",
        ],
    )

    if not traffic_exists:
        traffic_writer.writeheader()

    if not alerts_exist:
        alert_writer.writeheader()

    traffic_handle.flush()
    alert_handle.flush()

    print("\n" + "=" * 68)
    print("INTELLIGENT NETWORK TRAFFIC MONITOR")
    print("=" * 68)

    print("Interface:", interface.name)
    print("Traffic log:", TRAFFIC_FILE)
    print("Alert log:", ALERT_FILE)

    print("\nMonitoring is active.")
    print("Press CTRL+C to stop.")
    print(
        "Alerts are rule-based indicators, not confirmation "
        "of an attack."
    )
    print(
        "This captures traffic visible on the selected adapter; "
        "it does not automatically monitor every router client."
    )

    try:
        # Promiscuous mode is disabled.
        # The callback ignores non-IP packets.
        sniff(
            iface=interface.name,
            prn=handle_packet,
            store=False,
            promisc=False,
        )

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

    except Exception as error:
        print("\nPacket capture failed:", error)
        print(
            "Check Npcap installation, interface selection, "
            "permissions, and whether another capture driver "
            "is conflicting."
        )

    finally:
        if traffic_handle:
            traffic_handle.flush()
            traffic_handle.close()

        if alert_handle:
            alert_handle.flush()
            alert_handle.close()

        print(f"Packets observed: {packet_count:,}")
        print("Traffic log saved:", TRAFFIC_FILE)
        print("Alert log saved:", ALERT_FILE)


if __name__ == "__main__":
    main()