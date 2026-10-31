import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from check_accuracy import clean_label, rates  # noqa: E402


def test_rates_counts_hits_and_false_alarms():
    truth = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    flagged = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0], bool)
    r = rates(truth, flagged)
    assert r["attack detection rate (recall)"] == 0.75
    assert r["false alarm rate on benign"] == 1 / 6
    assert r["precision"] == 0.75
    assert r["accuracy"] == 0.8
    assert r["attacks caught"] == "3 / 4"


def test_rates_handles_no_attacks():
    r = rates(np.zeros(5, bool), np.zeros(5, bool))
    assert np.isnan(r["attack detection rate (recall)"])
    assert r["false alarm rate on benign"] == 0


def test_clean_label_repairs_garbled_web_attack_names():
    for raw in ("Web Attack  Brute Force", "Web Attack Â Brute Force", "Web Attack – Brute Force"):
        assert clean_label(raw) == "Web Attack – Brute Force"
    assert clean_label(" Web Attack - XSS ") == "Web Attack – XSS"
    assert clean_label("DoS Hulk ") == "DoS Hulk"
