"""PCAP conversion using the optional CICFlowMeter command-line tool."""

from pathlib import Path
import struct
import tempfile

import pandas as pd


CONVERSION_TIMEOUT_SECONDS = 4 * 60 * 60
_PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1": "<",
    b"\xa1\xb2\xc3\xd4": ">",
    b"\x4d\x3c\xb2\xa1": "<",
    b"\xa1\xb2\x3c\x4d": ">",
}

_FEATURE_ALIASES = {
    "flow_duration": "Flow Duration",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "src_ip": "Source IP",
    "dst_ip": "Destination IP",
    "src_port": "Source Port",
    "dst_port": "Destination Port",
    "tot_fwd_pkts": "Total Fwd Packets",
    "tot_bwd_pkts": "Total Backward Packets",
    "totlen_fwd_pkts": "Total Length of Fwd Packets",
    "totlen_bwd_pkts": "Total Length of Bwd Packets",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "bwd_pkt_len_std": "Bwd Packet Length Std",
    "flow_byts_s": "Flow Bytes/s",
    "flow_pkts_s": "Flow Packets/s",
    "fwd_iat_tot": "Fwd IAT Total",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_tot": "Bwd IAT Total",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    "fwd_psh_flags": "Fwd PSH Flags",
    "bwd_psh_flags": "Bwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "bwd_urg_flags": "Bwd URG Flags",
    "fwd_header_len": "Fwd Header Length",
    "fwd_header_len.1": "Fwd Header Length",
    "bwd_header_len": "Bwd Header Length",
    "fwd_pkts_s": "Fwd Packets/s",
    "bwd_pkts_s": "Bwd Packets/s",
    "pkt_len_min": "Min Packet Length",
    "pkt_len_max": "Max Packet Length",
    "pkt_len_mean": "Packet Length Mean",
    "pkt_len_std": "Packet Length Std",
    "pkt_len_var": "Packet Length Variance",
    "fin_flag_cnt": "FIN Flag Count",
    "syn_flag_cnt": "SYN Flag Count",
    "rst_flag_cnt": "RST Flag Count",
    "psh_flag_cnt": "PSH Flag Count",
    "ack_flag_cnt": "ACK Flag Count",
    "urg_flag_cnt": "URG Flag Count",
    "cwe_flag_count": "CWE Flag Count",
    "ece_flag_cnt": "ECE Flag Count",
    "down_up_ratio": "Down/Up Ratio",
    "pkt_size_avg": "Average Packet Size",
    "fwd_seg_size_avg": "Avg Fwd Segment Size",
    "bwd_seg_size_avg": "Avg Bwd Segment Size",
    "fwd_byts_b_avg": "Fwd Avg Bytes/Bulk",
    "fwd_pkts_b_avg": "Fwd Avg Packets/Bulk",
    "fwd_blk_rate_avg": "Fwd Avg Bulk Rate",
    "bwd_byts_b_avg": "Bwd Avg Bytes/Bulk",
    "bwd_pkts_b_avg": "Bwd Avg Packets/Bulk",
    "bwd_blk_rate_avg": "Bwd Avg Bulk Rate",
    "subflow_fwd_pkts": "Subflow Fwd Packets",
    "subflow_fwd_byts": "Subflow Fwd Bytes",
    "subflow_bwd_pkts": "Subflow Bwd Packets",
    "subflow_bwd_byts": "Subflow Bwd Bytes",
    "init_fwd_win_byts": "Init_Win_bytes_forward",
    "init_bwd_win_byts": "Init_Win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
    "cwr_flag_count": "CWE Flag Count",
}

_SECOND_TO_MICROSECOND_FEATURES = (
    "Flow Duration",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
)


def normalize_flow_columns(data, expected_columns=(), convert_time_units=True):
    """Map CICFlowMeter fields to model headers and normalize its time units.

    Python CICFlowMeter reports duration/IAT/active/idle values in seconds;
    CIC-IDS2017 model features use microseconds. Set ``convert_time_units`` to
    false for CIC CSV input, which already uses the training units.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Flow data must be a pandas DataFrame.")
    rename = {}
    expected_by_key = {
        str(name).strip().casefold(): str(name) for name in expected_columns
    }
    for column in data.columns:
        key = str(column).strip().casefold()
        target = _FEATURE_ALIASES.get(key)
        if target is None:
            target = expected_by_key.get(key)
        target_present = target and any(
            str(candidate).strip() == target
            for candidate in data.columns
            if candidate != column
        )
        if target and not target_present:
            rename[column] = target
    normalized = data.rename(columns=rename).copy()

    # The CIC-IDS CSV header contains a duplicate of Fwd Header Length.
    # The local training data confirms the two columns are identical.
    if (
        convert_time_units
        and "Fwd Header Length.1" in normalized.columns
        and "Fwd Header Length" not in normalized.columns
    ):
        normalized = normalized.rename(columns={"Fwd Header Length.1": "Fwd Header Length"})
    expected_names = set(map(str, expected_columns))
    if (
        convert_time_units
        and "Fwd Header Length.1" in expected_names
        and "Fwd Header Length" in normalized.columns
        and "Fwd Header Length.1" not in normalized.columns
    ):
        # CIC-IDS2017 stores this exact duplicate header feature twice; Python
        # CICFlowMeter emits it once. The local training CSV was verified equal.
        normalized["Fwd Header Length.1"] = normalized["Fwd Header Length"]

    if convert_time_units:
        for column in _SECOND_TO_MICROSECOND_FEATURES:
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce") * 1_000_000
    return normalized


def _pcap_header(path):
    with Path(path).open("rb") as capture:
        header = capture.read(24)
    if len(header) != 24 or header[:4] not in _PCAP_MAGICS:
        return None
    endian = _PCAP_MAGICS[header[:4]]
    return header, endian, struct.unpack(f"{endian}I", header[20:24])[0]


def _rewrite_sll2_as_ethernet(input_path, output_path):
    """Wrap Linux SLL2 network packets in Ethernet headers for Scapy tools."""
    parsed = _pcap_header(input_path)
    if parsed is None:
        raise ValueError("Capture is not a supported classic PCAP file.")
    header, endian, _ = parsed
    output_header = header[:20] + struct.pack(f"{endian}I", 1)

    with Path(input_path).open("rb") as source, Path(output_path).open("wb") as target:
        source.seek(24)
        target.write(output_header)
        while True:
            record_header = source.read(16)
            if not record_header:
                break
            if len(record_header) != 16:
                raise ValueError("Capture ends with an incomplete packet header.")
            seconds, fraction, captured_length, original_length = struct.unpack(
                f"{endian}IIII", record_header
            )
            packet = source.read(captured_length)
            if len(packet) != captured_length:
                raise ValueError("Capture ends with an incomplete packet record.")
            if len(packet) < 20:
                continue

            # SLL2 contains protocol type in bytes 0-1 and sender address in
            # bytes 12-19. CICFlowMeter needs the IP packet, not this pseudoheader.
            ethernet_frame = b"\x00" * 6 + packet[12:18] + packet[:2] + packet[20:]
            adjusted_original_length = max(14, original_length - 6)
            target.write(
                struct.pack(
                    f"{endian}IIII",
                    seconds,
                    fraction,
                    len(ethernet_frame),
                    adjusted_original_length,
                )
            )
            target.write(ethernet_frame)


def convert_pcap_to_csv(input_path, output_path):
    """Convert one capture into CICFlowMeter flow CSV, returning its path."""
    try:
        from cicflowmeter.flow_session import FlowSession
        from scapy.sendrecv import AsyncSniffer
    except ImportError as exc:
        raise FileNotFoundError(
            "PCAP conversion requires CICFlowMeter and Scapy. Install them with "
            "'.\\.venv\\Scripts\\python.exe -m pip install -r requirements-pcap.txt' "
            "and restart the API."
        ) from exc

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    capture_path = input_path
    normalized_path = None
    parsed_header = _pcap_header(input_path)
    if parsed_header is not None and parsed_header[2] == 276:
        with tempfile.NamedTemporaryFile(
            suffix=".pcap",
            prefix="sentinelai_ethernet_",
            dir=input_path.parent,
            delete=False,
        ) as normalized:
            normalized_path = Path(normalized.name)
        try:
            _rewrite_sll2_as_ethernet(input_path, normalized_path)
        except Exception:
            normalized_path.unlink(missing_ok=True)
            raise
        capture_path = normalized_path
    try:
        session = FlowSession(output_mode="csv", output=str(output_path))
        sniffer = AsyncSniffer(
            offline=str(capture_path),
            prn=session.process,
            store=False,
        )
        sniffer.start()
        sniffer.join(timeout=CONVERSION_TIMEOUT_SECONDS)
        if sniffer.running:
            sniffer.stop(join=True)
            raise TimeoutError(
                "PCAP conversion exceeded 4 hours. Try a smaller capture."
            )
        session.flush_flows()
    except TimeoutError:
        raise
    except Exception as exc:
        raise RuntimeError(f"CICFlowMeter could not convert this capture: {exc}") from exc
    finally:
        if normalized_path is not None:
            normalized_path.unlink(missing_ok=True)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("CICFlowMeter produced no flow records for this capture.")
    return output_path
