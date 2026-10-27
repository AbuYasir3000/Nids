import unittest

import pandas as pd

from src.pcap_converter import normalize_flow_columns


class PcapFeatureNormalizationTests(unittest.TestCase):
    def test_maps_case_and_duplicate_header_and_converts_time_units(self):
        flows = pd.DataFrame(
            {
                "protocol": [17],
                "flow_duration": [0.25],
                "flow_iat_mean": [0.125],
                "fwd_header_len.1": [40],
            }
        )
        normalized = normalize_flow_columns(
            flows,
            ["Protocol", "Flow Duration", "Flow IAT Mean", "Fwd Header Length"],
        )

        self.assertEqual(normalized.loc[0, "Protocol"], 17)
        self.assertEqual(normalized.loc[0, "Flow Duration"], 250_000)
        self.assertEqual(normalized.loc[0, "Flow IAT Mean"], 125_000)
        self.assertEqual(normalized.loc[0, "Fwd Header Length"], 40)

    def test_rate_features_are_not_scaled_with_time_features(self):
        flows = pd.DataFrame({"Flow Duration": [0.25], "Flow Bytes/s": [100]})
        normalized = normalize_flow_columns(flows)
        self.assertEqual(normalized.loc[0, "Flow Duration"], 250_000)
        self.assertEqual(normalized.loc[0, "Flow Bytes/s"], 100)

    def test_cic_csv_mode_preserves_microsecond_training_units(self):
        flows = pd.DataFrame({"Flow Duration": [10_000]})
        normalized = normalize_flow_columns(flows, convert_time_units=False)
        self.assertEqual(normalized.loc[0, "Flow Duration"], 10_000)

    def test_does_not_fuzzy_map_semantically_unverified_headers(self):
        flows = pd.DataFrame({"FlowDuration": [0.25]})
        normalized = normalize_flow_columns(flows, ["Flow Duration"])
        self.assertNotIn("Flow Duration", normalized.columns)
        self.assertIn("FlowDuration", normalized.columns)


if __name__ == "__main__":
    unittest.main()
