import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

scapy = pytest.importorskip("scapy.all")
pytest.importorskip("cicflowmeter")

import cic_compat  # noqa: E402


def _syn():
    # 14 Ethernet + 20 IP + 24 TCP (MSS option) = 58-byte frame, no payload.
    packet = (scapy.Ether() / scapy.IP(src="1.1.1.1", dst="2.2.2.2")
              / scapy.TCP(sport=1, dport=80, flags="S", options=[("MSS", 1460)]))
    # Round-trip so header fields are parsed like packets read from a pcap.
    return scapy.Ether(bytes(packet))


def test_payload_length_ignores_headers():
    packet = _syn()
    assert len(packet) == 58
    assert len(cic_compat._PayloadLenPacket(packet)) == 0


def test_payload_length_counts_tcp_data():
    packet = _syn() / scapy.Raw(b"x" * 37)
    assert len(cic_compat._PayloadLenPacket(packet)) == 37


def test_udp_payload_and_header():
    packet = scapy.Ether() / scapy.IP() / scapy.UDP() / scapy.Raw(b"abc")
    assert len(cic_compat._PayloadLenPacket(packet)) == 3
    assert cic_compat._transport_header_len(packet) == 8


def test_tcp_header_length_uses_data_offset():
    assert cic_compat._transport_header_len(_syn()) == 24


def test_wrapper_delegates_to_scapy_packet():
    wrapped = cic_compat._PayloadLenPacket(_syn())
    assert "TCP" in wrapped
    assert wrapped["TCP"].dport == 80
    assert wrapped.proto == 6


def test_apply_is_idempotent():
    cic_compat.apply()
    cic_compat.apply()
