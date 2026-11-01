"""Create labelled synthetic test captures for Argus (files only; nothing is sent
on any network). Output goes to ./test_pcaps together with ground_truth.csv.

    python make_test_pcaps.py

Then test one, for example:
    python check_accuracy.py --pcap test_pcaps/mixed_scan_and_benign.pcap --attacker-ip 10.9.0.66
or upload the file in the dashboard.
"""
import csv
import random
from pathlib import Path

from scapy.all import Ether, IP, TCP, UDP, Raw, PcapWriter

OUT = Path(__file__).resolve().parent / "test_pcaps"
OUT.mkdir(exist_ok=True)
rng = random.Random(7)
MAC_A, MAC_B = "02:00:00:00:00:01", "02:00:00:00:00:02"


def pkt(t, src, dst, l4, payload=b""):
    p = Ether(src=MAC_A, dst=MAC_B) / IP(src=src, dst=dst, ttl=rng.choice([64, 128])) / l4
    if payload:
        p = p / Raw(payload)
    p.time = t
    return p


def tcp_session(t, c, s, sport, dport, n_exchanges):
    """Full TCP session: handshake, request/response pairs, FIN."""
    out, seq_c, seq_s = [], rng.randrange(1 << 30), rng.randrange(1 << 30)
    opts = [("MSS", 1460), ("SAckOK", b""), ("WScale", 7)]
    out.append(pkt(t, c, s, TCP(sport=sport, dport=dport, flags="S", seq=seq_c, window=64240, options=opts)))
    t += 0.0005
    out.append(pkt(t, s, c, TCP(sport=dport, dport=sport, flags="SA", seq=seq_s, ack=seq_c + 1, window=65160, options=opts)))
    t += 0.0005
    seq_c += 1; seq_s += 1
    out.append(pkt(t, c, s, TCP(sport=sport, dport=dport, flags="A", seq=seq_c, ack=seq_s, window=502)))
    for _ in range(n_exchanges):
        t += rng.uniform(0.001, 0.05)
        req = bytes(rng.randrange(256) for _ in range(rng.randint(60, 400)))
        out.append(pkt(t, c, s, TCP(sport=sport, dport=dport, flags="PA", seq=seq_c, ack=seq_s, window=502), req))
        seq_c += len(req); t += rng.uniform(0.001, 0.03)
        resp = bytes(rng.randrange(256) for _ in range(rng.randint(200, 1400)))
        out.append(pkt(t, s, c, TCP(sport=dport, dport=sport, flags="PA", seq=seq_s, ack=seq_c, window=510), resp))
        seq_s += len(resp)
    t += 0.01
    out.append(pkt(t, c, s, TCP(sport=sport, dport=dport, flags="FA", seq=seq_c, ack=seq_s, window=502)))
    out.append(pkt(t + 0.0005, s, c, TCP(sport=dport, dport=sport, flags="FA", seq=seq_s, ack=seq_c + 1, window=510)))
    out.append(pkt(t + 0.001, c, s, TCP(sport=sport, dport=dport, flags="A", seq=seq_c + 1, ack=seq_s + 1, window=502)))
    return out


def benign(t0=0.0, sessions=350):
    out, clients = [], [f"10.1.0.{i}" for i in range(10, 30)]
    servers = [("10.1.1.10", 80), ("10.1.1.10", 443), ("10.1.1.20", 8080), ("10.1.1.30", 22)]
    for _ in range(sessions):
        t = t0 + rng.uniform(0, 120)
        c, (s, port) = rng.choice(clients), rng.choice(servers)
        out += tcp_session(t, c, s, rng.randint(32768, 60999), port, rng.randint(1, 8))
    for _ in range(300):                                # DNS lookups
        t = t0 + rng.uniform(0, 120); c = rng.choice(clients); sp = rng.randint(32768, 60999)
        out.append(pkt(t, c, "10.1.1.53", UDP(sport=sp, dport=53), bytes(rng.randrange(256) for _ in range(40))))
        out.append(pkt(t + 0.01, "10.1.1.53", c, UDP(sport=53, dport=sp), bytes(rng.randrange(256) for _ in range(90))))
    return out


def connect_scan(attacker, victim, t0=0.0, ports=range(1, 1001)):
    """nmap -sT style: full handshake to open ports, RST from closed ones."""
    out, t, open_ports = [], t0, {22, 80, 443, 8080}
    for port in ports:
        sport, seq = rng.randint(32768, 60999), rng.randrange(1 << 30)
        opts = [("MSS", 1460), ("SAckOK", b""), ("Timestamp", (1, 0)), ("WScale", 7)]
        out.append(pkt(t, attacker, victim, TCP(sport=sport, dport=port, flags="S", seq=seq, window=64240, options=opts)))
        t += 0.0004
        if port in open_ports:
            out.append(pkt(t, victim, attacker, TCP(sport=port, dport=sport, flags="SA", seq=5000, ack=seq + 1, window=65160, options=opts)))
            out.append(pkt(t + 0.0003, attacker, victim, TCP(sport=sport, dport=port, flags="A", seq=seq + 1, ack=5001, window=502)))
            out.append(pkt(t + 0.0006, attacker, victim, TCP(sport=sport, dport=port, flags="RA", seq=seq + 1, ack=5001, window=0)))
        else:
            out.append(pkt(t, victim, attacker, TCP(sport=port, dport=sport, flags="RA", seq=0, ack=seq + 1, window=0)))
        t += rng.uniform(0.0005, 0.003)
    return out


def syn_flood(attackers, victim, port=80, n=20000, t0=0.0):
    out, t = [], t0
    for i in range(n):
        a = attackers[i % len(attackers)]
        t += rng.uniform(0.00001, 0.0003)
        out.append(pkt(t, a, victim, TCP(sport=rng.randint(1024, 65535), dport=port, flags="S",
                                          seq=rng.randrange(1 << 30), window=rng.choice([512, 1024, 8192]))))
    return out


def udp_flood(attackers, victim, n=20000, t0=0.0):
    out, t = [], t0
    for i in range(n):
        a = attackers[i % len(attackers)]
        t += rng.uniform(0.00001, 0.0003)
        out.append(pkt(t, a, victim, UDP(sport=rng.randint(1024, 65535), dport=rng.choice([53, 123, 1900, 11211])),
                       bytes(rng.randrange(256) for _ in range(rng.randint(40, 160)))))
    return out


def write(name, packets):
    packets.sort(key=lambda p: float(p.time))
    base = 1_700_000_000
    w = PcapWriter(str(OUT / name), linktype=1, sync=False)
    for p in packets:
        p.time = base + float(p.time)
        w.write(p)
    w.close()
    print(f"{name}: {len(packets):,} packets")


if __name__ == "__main__":
    truth = []
    write("benign_only.pcap", benign())
    truth.append(("benign_only.pcap", "", "No attack. Detector should flag nothing."))
    write("connect_scan.pcap", connect_scan("10.9.0.66", "10.1.1.10"))
    truth.append(("connect_scan.pcap", "10.9.0.66", "TCP connect port scan of 1000 ports"))
    bots = [f"10.9.1.{i}" for i in range(1, 21)]
    write("syn_flood.pcap", syn_flood(bots, "10.1.1.10"))
    truth.append(("syn_flood.pcap", ",".join(bots), "SYN flood from 20 sources"))
    write("udp_flood.pcap", udp_flood(bots[:5], "10.1.1.53"))
    truth.append(("udp_flood.pcap", ",".join(bots[:5]), "UDP flood from 5 sources"))
    write("mixed_scan_and_benign.pcap", benign(sessions=500) + connect_scan("10.9.0.66", "10.1.1.10", t0=20, ports=range(1, 1501)))
    truth.append(("mixed_scan_and_benign.pcap", "10.9.0.66", "Normal traffic with a port scan hidden in it"))
    with open(OUT / "ground_truth.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "attacker_ip", "description"]); w.writerows(truth)
    print("Wrote", OUT / "ground_truth.csv")
