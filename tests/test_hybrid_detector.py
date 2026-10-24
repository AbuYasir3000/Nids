import unittest

import numpy as np
import pandas as pd

from src.hybrid_detector import HybridNIDS


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


def make_detector():
    detector = HybridNIDS.__new__(HybridNIDS)
    detector.rf_model = FakeClassifier()
    detector.feature_columns = ["feature_a", "feature_b"]
    detector.medians = pd.Series({"feature_a": 3.0, "feature_b": 4.0})
    detector.if_scaler = FakeScaler()
    detector.if_model = FakeIsolationForest()
    return detector


class HybridDetectorTests(unittest.TestCase):
    def test_feature_alignment_imputation_and_exclusions(self):
        detector = make_detector()
        incoming = pd.DataFrame(
            {
                " feature_a ": [1, None],
                "Label": ["BENIGN", "DDoS"],
                "Source IP": ["10.0.0.1", "10.0.0.2"],
                "unused": [99, 99],
            }
        )

        features = detector.prepare_features(incoming)

        self.assertEqual(list(features.columns), ["feature_a", "feature_b"])
        np.testing.assert_allclose(
            features.to_numpy(),
            np.array([[1, 4], [3, 4]], dtype=np.float32),
        )
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_dict_and_series_are_single_flow_inputs(self):
        detector = make_detector()

        from_dict = detector.prepare_features({"feature_a": 7})
        from_series = detector.prepare_features(pd.Series({"feature_b": 8}))

        self.assertEqual(from_dict.shape, (1, 2))
        self.assertEqual(from_dict.iloc[0].tolist(), [7.0, 4.0])
        self.assertEqual(from_series.iloc[0].tolist(), [3.0, 8.0])

    def test_unsupported_input_type_is_rejected(self):
        detector = make_detector()

        with self.assertRaises(TypeError):
            detector.prepare_features([{"feature_a": 1}])

    def test_predict_returns_class_and_anomaly_outputs(self):
        detector = make_detector()
        results = detector.predict(pd.DataFrame({"feature_a": [1, 2]}))

        self.assertEqual(results["predicted_class"].tolist(), ["BENIGN", "DDoS"])
        self.assertEqual(results["is_anomaly"].tolist(), [False, True])
        self.assertEqual(
            results["hybrid_status"].tolist(),
            ["No anomaly detected", "Suspicious - anomaly detected"],
        )
        np.testing.assert_allclose(results["anomaly_score"], [0.2, -0.3])


if __name__ == "__main__":
    unittest.main()
