import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_features import target_flood_analysis  # noqa: E402


def _flows(n, dst="10.0.0.1", answered=0.0, sources=20, t0="2025-01-01 10:00:05"):
    answered_n = int(n * answered)
    return pd.DataFrame({
        "Source IP": [f"9.9.9.{i % sources}" for i in range(n)],
        "Destination IP": dst,
        "Timestamp": pd.Timestamp(t0),
        "Total Backward Packets": [1] * answered_n + [0] * (n - answered_n),
    })


def test_flood_of_unanswered_flows_is_flagged():
    flags, events = target_flood_analysis(_flows(1500))
    assert flags.all()
    row = events.iloc[0]
    assert row["destination_ip"] == "10.0.0.1"
    assert row["flows"] == 1500 and row["unique_sources"] == 20
    assert row["unanswered_share"] == 1.0


def test_busy_server_with_replies_is_not_flagged():
    flags, events = target_flood_analysis(_flows(5000, answered=0.8))
    assert not flags.any() and events.empty


def test_small_burst_is_not_flagged():
    flags, _ = target_flood_analysis(_flows(500))
    assert not flags.any()


def test_only_the_flooded_target_is_flagged():
    data = pd.concat([_flows(1500, dst="10.0.0.1"), _flows(300, dst="10.0.0.2")], ignore_index=True)
    flags, events = target_flood_analysis(data)
    assert flags[:1500].all() and not flags[1500:].any()
    assert list(events["destination_ip"]) == ["10.0.0.1"]


def test_missing_columns_return_no_flags():
    flags, events = target_flood_analysis(pd.DataFrame({"a": [1, 2, 3]}))
    assert not flags.any() and events.empty


def _with_fwd(frame, fwd):
    frame = frame.copy()
    frame["Total Fwd Packets"] = fwd
    return frame


def test_one_direction_capture_of_real_conversations_is_not_flagged():
    # Replies were not captured (no backward packets), but each flow has many packets.
    flags, events = target_flood_analysis(_with_fwd(_flows(2000), 9))
    assert not flags.any() and events.empty


def test_tiny_unanswered_flows_are_still_flagged():
    flags, events = target_flood_analysis(_with_fwd(_flows(2000), 1))
    assert flags.all() and len(events) == 1


def test_size_check_can_be_disabled():
    flags, _ = target_flood_analysis(_with_fwd(_flows(2000), 9), max_forward_packets=None)
    assert flags.all()
