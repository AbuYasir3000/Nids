import unittest

import numpy as np
import pandas as pd

from src.diagnose_generalized_model import binary_metrics, group_port_diversity


class GeneralizedModelDiagnosticsTests(unittest.TestCase):
    def test_binary_metrics_uses_attack_positive_and_explicit_counts(self):
        result = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.2], 0.5)
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (1, 1, 1, 1))
        self.assertEqual(result["benign_fpr"], 0.5)

    def test_group_port_diversity_counts_unique_ports_per_source_minute(self):
        frame = pd.DataFrame({
            "Source IP": ["10.0.0.1", "10.0.0.1", "10.0.0.1", "10.0.0.2"],
            "Timestamp": ["2020-01-01 00:00:01", "2020-01-01 00:00:59", "2020-01-01 00:01:00", "2020-01-01 00:00:01"],
            "Destination Port": [80, 443, 443, 53],
        })
        np.testing.assert_array_equal(group_port_diversity(frame), [2, 2, 1, 1])


if __name__ == "__main__":
    unittest.main()
