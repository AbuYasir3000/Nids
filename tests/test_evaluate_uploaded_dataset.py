import unittest

import numpy as np

from src.evaluate_uploaded_dataset import binary_metrics


class UploadedDatasetEvaluationTests(unittest.TestCase):
    def test_binary_attack_metrics_use_benign_zero_attack_one(self):
        metrics = binary_metrics(
            y_true=np.array([0, 0, 1, 1]),
            y_pred=np.array([0, 1, 0, 1]),
        )

        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["benign_false_positive_rate"], 0.5)
        self.assertEqual(metrics["recall_attack"], 0.5)
        self.assertEqual(metrics["precision_attack"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
