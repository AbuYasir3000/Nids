import numpy as np
import pandas as pd


PORT_DIVERSITY_BENIGN_QUANTILE = 0.999
PORT_DIVERSITY_THRESHOLD = 54  # q99.9 of 176,985 benign CIC source-minute groups

# Unanswered-flow flood rule (SYN/UDP floods aimed at one target). These are
# development defaults, NOT calibrated on benign CIC data: the available CIC
# sample is down-sampled, so per-target counts cannot be calibrated from it.
TARGET_FLOOD_FLOW_THRESHOLD = 1000   # flows to one destination in one minute
TARGET_FLOOD_UNANSWERED_SHARE = 0.9  # share of those flows with no reply packet
# An "unanswered" flood flow is also tiny. Real conversations whose replies simply
# were not captured (one-direction captures) have many forward packets and are not counted.
TARGET_FLOOD_MAX_FORWARD_PACKETS = 2


def high_rate_multiport_flags(
    source_flows, unique_destination_ports, source_burst_threshold
):
    """Flag source-minute groups matching calibrated rate and port thresholds."""
    return (
        (np.asarray(source_flows) > source_burst_threshold)
        & (np.asarray(unique_destination_ports) > PORT_DIVERSITY_THRESHOLD)
    )


# Compatibility alias for older scripts. This rule does not establish a scan.
possible_port_scan_flags = high_rate_multiport_flags


def source_flows_per_minute(data):
    """Count rows per source IP and minute for batch scan-burst detection.

    Identifiers are used only to group flows. They are not passed to a
    classifier. Rows without a usable source IP or timestamp get count one.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    normalized = {str(column).strip().lower(): column for column in data.columns}
    source_column = normalized.get("source ip")
    timestamp_column = normalized.get("timestamp")
    counts = pd.Series(1, index=data.index, dtype="int64")
    if source_column is None or timestamp_column is None or data.empty:
        return counts.to_numpy()

    source = data[source_column].astype("string").str.strip()
    timestamp = pd.to_datetime(data[timestamp_column], errors="coerce")
    windows = timestamp.dt.floor("min")
    usable = source.notna() & source.ne("") & windows.notna()
    if usable.any():
        groups = pd.DataFrame(
            {
                "source": source.loc[usable],
                "window": windows.loc[usable],
            },
            index=data.index[usable],
        )
        counts.loc[usable] = groups.groupby(
            ["source", "window"], sort=False
        )["source"].transform("size").astype("int64")
    return counts.to_numpy()


def source_minute_event_table(data, unique_destination_ports,
                              unique_destination_ips, source_burst_threshold):
    """Deduplicate heuristic matches into source-IP/minute events."""
    from collections import Counter

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    normalized = {str(column).strip().casefold(): column for column in data.columns}
    source_column = normalized.get("source ip") or normalized.get("src_ip")
    timestamp_column = normalized.get("timestamp")
    destination_column = normalized.get("destination ip") or normalized.get("dst_ip")
    port_column = normalized.get("destination port") or normalized.get("dst_port")
    if (
        source_column is None
        or timestamp_column is None
        or source_burst_threshold is None
        or data.empty
    ):
        return pd.DataFrame()

    source = data[source_column].astype("string").str.strip()
    timestamp = pd.to_datetime(data[timestamp_column], errors="coerce")
    valid = source.notna() & source.ne("") & timestamp.notna()
    if not valid.any():
        return pd.DataFrame()
    rows = pd.DataFrame({
        "source_ip": source.loc[valid].to_numpy(),
        "timestamp": timestamp.loc[valid].to_numpy(),
        "unique_destination_ports": np.asarray(unique_destination_ports)[valid.to_numpy()],
        "unique_destination_ips": np.asarray(unique_destination_ips)[valid.to_numpy()],
    })
    if destination_column is not None:
        rows["destination_ip"] = data.loc[valid, destination_column].astype("string").to_numpy()
    if port_column is not None:
        rows["destination_port"] = data.loc[valid, port_column].to_numpy()

    rows["window_start"] = rows["timestamp"].dt.floor("min")
    records = []
    for (source_ip, window_start), group in rows.groupby(
        ["source_ip", "window_start"], sort=True
    ):
        flow_count = int(len(group))
        unique_ports = int(group["unique_destination_ports"].max())
        unique_ips = int(group["unique_destination_ips"].max())
        source_burst = flow_count > int(source_burst_threshold)
        multiport = source_burst and unique_ports > PORT_DIVERSITY_THRESHOLD
        if not source_burst:
            continue
        destinations = []
        if "destination_ip" in group:
            destinations = [
                {"destination_ip": str(value), "flows": int(count)}
                for value, count in Counter(group["destination_ip"].dropna().astype(str)).most_common(5)
            ]
        ports = []
        if "destination_port" in group:
            ports = [
                {"destination_port": str(value), "flows": int(count)}
                for value, count in Counter(group["destination_port"].dropna().astype(str)).most_common(5)
            ]
        reasons = [
            f"{flow_count:,} source flows in this minute exceeded threshold "
            f"{int(source_burst_threshold):,}"
        ]
        if multiport:
            reasons.append(
                f"{unique_ports:,} unique destination ports exceeded threshold "
                f"{PORT_DIVERSITY_THRESHOLD}"
            )
        records.append({
            "event_id": f"{source_ip}|{pd.Timestamp(window_start).isoformat()}",
            "source_ip": str(source_ip),
            "window_start": pd.Timestamp(window_start).isoformat(),
            "window_end": (pd.Timestamp(window_start) + pd.Timedelta(minutes=1)).isoformat(),
            "last_observed_at": pd.Timestamp(group["timestamp"].max()).isoformat(),
            "flow_count": flow_count,
            "unique_destination_ports": unique_ports,
            "unique_destination_ips": unique_ips,
            "source_flow_threshold": int(source_burst_threshold),
            "destination_port_diversity_threshold": PORT_DIVERSITY_THRESHOLD,
            "top_destination_ips": destinations,
            "top_destination_ports": ports,
            "source_burst_rule": True,
            "high_rate_multiport_rule": bool(multiport),
            "classification": "High-rate multiport activity" if multiport else "High source-rate activity",
            "reasons": reasons,
        })
    return pd.DataFrame.from_records(records)


def target_flood_analysis(data, flow_threshold=TARGET_FLOOD_FLOW_THRESHOLD,
                          unanswered_share=TARGET_FLOOD_UNANSWERED_SHARE,
                          max_forward_packets=TARGET_FLOOD_MAX_FORWARD_PACKETS):
    """Flag destination-minute groups that receive a flood of unanswered flows.

    A group is flagged when it holds more than ``flow_threshold`` flows and at
    least ``unanswered_share`` of them have no backward packet (no reply) and no
    more than ``max_forward_packets`` forward packets, which is the shape of a SYN
    or UDP flood. Flows with many forward packets are real conversations even if
    the capture did not record the replies, so they are not counted. Pass
    ``max_forward_packets=None`` to disable that size check. A single scanner probing a filtered host
    can look the same, so this is a behavioral signal, not an attack family.

    Returns (flags, events): a boolean array aligned with ``data`` and a table
    with one row per flagged destination-minute.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    flags = np.zeros(len(data), dtype=bool)
    cols = {str(c).strip().casefold(): c for c in data.columns}
    dst = cols.get("destination ip") or cols.get("dst_ip")
    ts = cols.get("timestamp")
    bwd = cols.get("total backward packets") or cols.get("tot_bwd_pkts")
    fwd = cols.get("total fwd packets") or cols.get("tot_fwd_pkts")
    src = cols.get("source ip") or cols.get("src_ip")
    if dst is None or ts is None or bwd is None or data.empty:
        return flags, pd.DataFrame()
    no_reply = pd.to_numeric(data[bwd], errors="coerce").fillna(0) <= 0
    if max_forward_packets is not None and fwd is not None:
        small = pd.to_numeric(data[fwd], errors="coerce").fillna(0) <= max_forward_packets
        no_reply = no_reply & small

    frame = pd.DataFrame({
        "destination_ip": data[dst].astype("string").str.strip().to_numpy(),
        "window": pd.to_datetime(data[ts], errors="coerce").dt.floor("min").to_numpy(),
        "unanswered": no_reply.to_numpy(),
        "source_ip": (data[src].astype("string").to_numpy() if src is not None else None),
    })
    valid = frame["destination_ip"].notna() & frame["destination_ip"].ne("") & frame["window"].notna()
    if not valid.any():
        return flags, pd.DataFrame()
    grouped = frame[valid].groupby(["destination_ip", "window"], sort=False)
    stats = grouped.agg(flows=("unanswered", "size"), unanswered=("unanswered", "sum"))
    if src is not None:
        stats["unique_sources"] = grouped["source_ip"].nunique()
    stats["unanswered_share"] = stats["unanswered"] / stats["flows"]
    hit = stats[(stats["flows"] > flow_threshold) & (stats["unanswered_share"] >= unanswered_share)]
    if hit.empty:
        return flags, pd.DataFrame()
    key = pd.MultiIndex.from_frame(frame.loc[valid, ["destination_ip", "window"]])
    flags[np.flatnonzero(valid.to_numpy())] = key.isin(hit.index)
    events = hit.reset_index().rename(columns={"window": "window_start"})
    events["window_start"] = events["window_start"].map(lambda t: pd.Timestamp(t).isoformat())
    events["flows"] = events["flows"].astype(int)
    events["unanswered"] = events["unanswered"].astype(int)
    events["unanswered_share"] = events["unanswered_share"].round(4)
    events["flow_threshold"] = int(flow_threshold)
    events["classification"] = "High-rate unanswered flows to one target (possible SYN/UDP flood)"
    return flags, events.sort_values("flows", ascending=False).reset_index(drop=True)
