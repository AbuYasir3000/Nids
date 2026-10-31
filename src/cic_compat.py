"""Make Python CICFlowMeter's packet-size features match CIC-IDS2017.

The models were trained on CIC-IDS2017 (Java CICFlowMeter), which measures
*payload* bytes (TCP/UDP data) and takes "header length" from the TCP/UDP
header. Python CICFlowMeter 0.5.0 instead uses ``len(packet)`` (the whole
Ethernet frame) and the IP header length. For a bare SYN probe that gives
58 bytes instead of 0, which pushes every packet-length feature far outside
the training distribution and hides port scans from the models.

``apply()`` patches CICFlowMeter in-process (idempotent). It does not touch
the installed package on disk.
"""

_APPLIED = False


class _PayloadLenPacket:
    """Delegates to a scapy packet but reports ``len()`` as payload bytes."""

    __slots__ = ("_p", "_n")

    def __init__(self, packet):
        object.__setattr__(self, "_p", packet)
        object.__setattr__(self, "_n", _payload_len(packet))

    def __len__(self):
        return self._n

    def __getitem__(self, item):
        return self._p[item]

    def __contains__(self, item):
        return item in self._p

    def __getattr__(self, name):
        return getattr(self._p, name)


def _payload_len(packet):
    for layer in ("TCP", "UDP"):
        if layer in packet:
            return len(bytes(packet[layer].payload))
    return len(bytes(packet["IP"].payload)) if "IP" in packet else 0


def _transport_header_len(packet):
    if "TCP" in packet:
        return int(packet["TCP"].dataofs or 5) * 4
    if "UDP" in packet:
        return 8
    return int(packet["IP"].ihl or 5) * 4 if "IP" in packet else 0


def apply():
    global _APPLIED
    if _APPLIED:
        return
    from cicflowmeter.flow import Flow
    from cicflowmeter.features.packet_length import PacketLength
    from cicflowmeter.features.flow_bytes import FlowBytes
    original_init = Flow.__init__
    original_add = Flow.add_packet

    def init(self, packet, direction, *args, **kwargs):
        original_init(self, _PayloadLenPacket(packet), direction, *args, **kwargs)

    def add_packet(self, packet, direction):
        original_add(self, _PayloadLenPacket(packet), direction)

    def header_length(self, packet_direction=None):
        items = [
            _transport_header_len(p)
            for p, d in self.flow.packets
            if packet_direction is None or d == packet_direction
        ]
        return items

    def flow_header_size(self, packet):
        return _transport_header_len(packet)

    Flow.__init__ = init
    Flow.add_packet = add_packet
    PacketLength.get_header_length = header_length
    FlowBytes._header_size = flow_header_size
    _APPLIED = True
