import unittest

import numpy as np
import pandas as pd

from src.hybrid_detector import HybridNIDS
from src.traffic_features import (
    high_rate_multiport_flags,
    source_flows_per_minute,
    source_minute_event_table,
)


class FakeClassifier:
    def predict(self, features):
        return np.array(["BENIGN", "DDoS"][: len(features)])


class FakeScaler:
    def transform(self, features):
        return features.to_numpy()


class FakeIsolationForest:
    def predict(self, features):
        return np.array([1, -1][: len(features)])

    def decision_function(self, features):
        return np.array([0.2, -0.3][: len(features)])


class FakeImputer:
    statistics_ = np.array([3.0, 4.0])

    def transform(self, features):
        return features.to_numpy(dtype=np.float32)


class FakeGeneralizedClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        return np.array([[0.1, 0.9], [0.9, 0.1]])[: len(features)]


def make_detector():
    detector = HybridNIDS.__new__(HybridNIDS)
    detector.rf_model = FakeClassifier()
    detector.feature_columns = ["feature_a", "feature_b"]
    detector.medians = pd.Series({"feature_a": 3.0, "feature_b": 4.0})
    detector.if_feature_columns = list(detector.feature_columns)
    detector.if_medians = detector.medians.copy()
    detector.if_scaler = FakeScaler()
    detector.if_model = FakeIsolationForest()
    detector.generalized_model = None
    detector.generalized_imputer = None
    detector.generalized_feature_columns = []
    detector.generalized_threshold = None
    detector.source_burst_threshold = None
    return detector


class ReversedClassOrderClassifier:
    classes_ = np.array([1, 0])

    def predict_proba(self, features):
        # Column zero is ATTACK because classes_ is [1, 0].
        return np.array([[0.9, 0.1], [0.2, 0.8]])[: len(features)]


class HybridDetectorTests(unittest.TestCase):
    def test_positive_probability_uses_class_labels_not_column_position(self):
        features = pd.DataFrame({"x": [1, 2]})
        result = HybridNIDS._positive_class_probability(
            ReversedClassOrderClassifier(), features
        )
        np.testing.assert_allclose(result, [0.9, 0.2])

    def test_feature_alignment_imputation_and_exclusions(self):
        detector = make_detector()
        incoming = pd.DataFrame(
            {
                " feature_a ": [1, None],
                "feature_b": [2, 8],
                "Label": ["BENIGN", "DDoS"],
                "Source IP": ["10.0.0.1", "10.0.0.2"],
                "unused": [99, 99],
            }
        )

        features = detector.prepare_features(incoming)

        self.assertEqual(list(features.columns), ["feature_a", "feature_b"])
        np.testing.assert_allclose(
            features.to_numpy(),
            np.array([[1, 2], [3, 8]], dtype=np.float32),
        )
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_missing_required_features_are_rejected(self):
        detector = make_detector()
        with self.assertRaisesRegex(ValueError, "missing required features"):
            detector.prepare_features({"feature_a": 7})

    def test_all_invalid_required_feature_is_rejected(self):
        detector = make_detector()
        with self.assertRaisesRegex(ValueError, "no usable numeric values"):
            detector.prepare_features(
                pd.DataFrame({"feature_a": [1, 2], "feature_b": ["bad", None]})
            )

    def test_source_flow_rate_groups_by_source_and_minute(self):
        frame = pd.DataFrame(
            {
                "Source IP": ["10.0.0.1"] * 3 + ["10.0.0.2"] * 2 + ["10.0.0.1"],
                "Timestamp": [
                    "2017-05-07 12:00:01",
                    "2017-05-07 12:00:10",
                    "2017-05-07 12:00:59",
                    "2017-05-07 12:00:05",
                    "2017-05-07 12:00:30",
                    "2017-05-07 12:01:00",
                ],
            }
        )
        self.assertEqual(
            source_flows_per_minute(frame).tolist(), [3, 3, 3, 2, 2, 1]
        )

    def test_possible_port_scan_requires_both_calibrated_signals(self):
        flags = high_rate_multiport_flags(
            source_flows=[2086, 2086, 3000, 5000],
            unique_destination_ports=[55, 54, 431, 100],
            source_burst_threshold=2085,
        )
        self.assertEqual(flags.tolist(), [True, False, True, True])

    def test_unsupported_input_type_is_rejected(self):
        detector = make_detector()
        with self.assertRaises(TypeError):
            detector.prepare_features([{"feature_a": 1}])

    def test_predict_combines_known_anomaly_and_generalized_alerts(self):
        detector = make_detector()
        detector.generalized_model = FakeGeneralizedClassifier()
        detector.generalized_imputer = FakeImputer()
        detector.generalized_feature_columns = ["feature_a", "feature_b"]
        detector.generalized_threshold = 0.8
        detector.source_burst_threshold = 10
        results = detector.predict(
            pd.DataFrame({"feature_a": [1, 2], "feature_b": [2, 3]})
        )

        self.assertEqual(results["predicted_class"].tolist(), ["BENIGN", "DDoS"])
        self.assertEqual(results["is_anomaly"].tolist(), [False, True])
        self.assertEqual(
            results["is_generalized_attack"].tolist(), [True, False]
        )
        self.assertEqual(
            results["is_source_burst_alert"].tolist(), [False, False]
        )
        self.assertEqual(
            results["hybrid_status"].tolist(),
            [
                "Suspicious - binary attack signal; family unknown",
                "Known attack: DDoS",
            ],
        )
        self.assertEqual(results["binary_classifier_result"].tolist(), ["ATTACK", "BENIGN"])
        self.assertEqual(results["decision_label"].tolist(), results["hybrid_status"].tolist())
        self.assertEqual(results["severity"].tolist(), ["high", "high"])
        self.assertEqual(results["flow_identifier"].tolist(), ["flow-000001", "flow-000002"])
        self.assertEqual(results["imputed_feature_count"].tolist(), [0, 0])
        np.testing.assert_allclose(results["anomaly_score"], [0.2, -0.3])

    def test_generalized_model_is_optional(self):
        detector = make_detector()
        results = detector.predict(
            pd.DataFrame({"feature_a": [1], "feature_b": [2]})
        )

        self.assertFalse(results.loc[0, "is_generalized_attack"])
        self.assertFalse(results.loc[0, "is_source_burst_alert"])
        self.assertTrue(pd.isna(results.loc[0, "generalized_attack_probability"]))
        self.assertEqual(results.loc[0, "hybrid_status"], "Insufficient evidence")
        self.assertFalse(results.loc[0, "is_high_rate_multiport_activity"])

    def test_source_burst_stays_separate_from_learned_model_flags(self):
        detector = make_detector()
        detector.generalized_model = FakeGeneralizedClassifier()
        detector.generalized_imputer = FakeImputer()
        detector.generalized_feature_columns = ["feature_a", "feature_b"]
        detector.generalized_threshold = 0.8
        detector.source_burst_threshold = 1
        frame = pd.DataFrame({
            "feature_a": [1, 2],
            "feature_b": [2, 3],
            "Source IP": ["10.0.0.1", "10.0.0.1"],
            "Timestamp": ["2020-01-01 00:00:01", "2020-01-01 00:00:02"],
        })

        results = detector.predict(frame)

        self.assertEqual(results["is_source_burst_alert"].tolist(), [True, True])
        self.assertEqual(results["is_generalized_attack"].tolist(), [True, False])

    def test_source_minute_rule_matches_deduplicate_to_one_event(self):
        frame = pd.DataFrame({
            "Source IP": ["10.0.0.1"] * 55,
            "Destination IP": ["192.0.2.1"] * 55,
            "Destination Port": list(range(55)),
            "Timestamp": ["2020-01-01 00:00:01"] * 55,
        })
        flow_count = np.repeat(55, len(frame))
        ports = np.repeat(55, len(frame))
        destinations = np.ones(len(frame), dtype=int)

        events = source_minute_event_table(frame, ports, destinations, 10)

        self.assertEqual(len(events), 1)
        self.assertEqual(events.loc[0, "flow_count"], 55)
        self.assertEqual(events.loc[0, "unique_destination_ips"], 1)
        self.assertTrue(events.loc[0, "high_rate_multiport_rule"])


if __name__ == "__main__":
    unittest.main()
