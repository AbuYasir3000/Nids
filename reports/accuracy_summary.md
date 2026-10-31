# SentinelAI accuracy check

- Input: benign_only.pcap; flows from 0.0.0.0 labelled ATTACK, all other flows BENIGN (assumption).
- If the Random Forest was retrained on this capture, scores are optimistic (not independent).

## Overall (BENIGN vs attack, per flow)

| Detector | accuracy | attack detection rate (recall) | precision | false alarm rate on benign | attacks caught | benign flagged |
|---|---|---|---|---|---|---|
| Random Forest (any attack class) | 100.0% | n/a | n/a | 0.0% | 0 / 0 | 0 / 650 |
| Binary classifier | 100.0% | n/a | n/a | 0.0% | 0 / 0 | 0 / 650 |
| Isolation Forest (anomaly) | 100.0% | n/a | n/a | 0.0% | 0 / 0 | 0 / 650 |
| Behavioral rule (not a model) | 100.0% | n/a | n/a | 0.0% | 0 / 0 | 0 / 650 |
| ANY of the above | 100.0% | n/a | n/a | 0.0% | 0 / 0 | 0 / 650 |

Random Forest exact-class accuracy on classes it was trained on: **100.0%** (650 flows)

## By attack type (share of flows each detector flags)

| Label | flows | RF predicts exact class | RF flags as any attack | binary classifier | isolation forest | behavioral rule |
|---|---|---|---|---|---|---|
| BENIGN | 650 | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |

How to read it: *attack detection rate* = attacks caught / all attacks; *false alarm rate* = benign flows wrongly flagged / all benign. Accuracy alone is misleading when one class dominates, so quote detection rate and false alarm rate.
