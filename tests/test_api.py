import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import httpx

import api


class FakeDetector:
    def __init__(self):
        self.received = None

    def predict(self, frame):
        self.received = frame.copy()
        return pd.DataFrame(
            {
                "predicted_class": ["BENIGN", "DDoS"][: len(frame)],
                "is_anomaly": [False, True][: len(frame)],
                "anomaly_score": [0.2, -0.3][: len(frame)],
                "hybrid_status": [
                    "No anomaly detected",
                    "Suspicious - anomaly detected",
                ][: len(frame)],
            }
        )


class ApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.detector = FakeDetector()
        self.results_dir_patch = patch.object(
            api, "RESULTS_DIR", Path(self.temp_dir.name)
        )
        self.detector_patch = patch.object(api, "detector", self.detector)
        self.results_dir_patch.start()
        self.detector_patch.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.detector_patch.stop()
        self.results_dir_patch.stop()
        self.temp_dir.cleanup()

    async def test_health_reports_ready_detector(self):
        response = await self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["detector_ready"])

    async def test_analysis_excludes_label_from_features_and_downloads_all_rows(self):
        csv = (
            "feature_a,Label\n"
            "1,BENIGN\n"
            "2,DDoS\n"
        )
        response = await self.client.post(
            "/analyze",
            files={"file": ("flows.csv", csv, "text/csv")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["total_flows"], 2)
        self.assertEqual(payload["anomaly_count"], 1)
        self.assertEqual(payload["preview_rows"], 2)
        self.assertNotIn("Label", self.detector.received.columns)
        self.assertEqual(payload["evaluation"]["evaluated_rows"], 2)

        download = await self.client.get(payload["download_url"])
        self.assertEqual(download.status_code, 200)
        self.assertIn("predicted_class", download.text)
        self.assertIn("actual_class_for_evaluation", download.text)

    async def test_rejects_non_csv_upload(self):
        response = await self.client.post(
            "/analyze",
            files={"file": ("flows.txt", "data", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)

    async def test_returns_service_unavailable_when_models_are_missing(self):
        with patch.object(api, "detector", None), patch.object(
            api, "detector_error", "FileNotFoundError: models missing"
        ):
            response = await self.client.post(
                "/analyze",
                files={"file": ("flows.csv", "feature_a\n1\n", "text/csv")},
            )

        self.assertEqual(response.status_code, 503)

    async def test_rejects_malformed_download_id(self):
        response = await self.client.get("/download/not-a-uuid")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
