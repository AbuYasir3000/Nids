import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import pandas as pd
from types import SimpleNamespace

import api


class FakeDetector:
    generalized_model_ready = True
    feature_columns = ["feature_a"]
    source_burst_threshold = 1

    def __init__(self):
        self.rf_model = SimpleNamespace(classes_=["BENIGN", "DDoS"])

        self.received = None

    def predict(self, frame):
        self.received = frame.copy()
        return pd.DataFrame(
            {
                "predicted_class": ["BENIGN", "DDoS"][: len(frame)],
                "is_anomaly": [False, True][: len(frame)],
                "anomaly_score": [0.2, -0.3][: len(frame)],
                "generalized_attack_probability": [0.2, 0.95][: len(frame)],
                "is_generalized_attack": [False, True][: len(frame)],
                "is_high_rate_multiport_activity": [True, True][: len(frame)],
                "is_source_burst_alert": [True, True][: len(frame)],
                "source_ip": ["10.0.0.1", "10.0.0.1"][: len(frame)],
                "flow_identifier": ["flow-000001", "flow-000002"][: len(frame)],
                "unique_destination_ports_per_source_minute": [1, 55][: len(frame)],
                "unique_destination_ips_per_source_minute": [1, 1][: len(frame)],
                "hybrid_status": [
                    "No anomaly detected",
                    "Suspicious - generalized attack alert",
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
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        self.detector_patch.stop()
        self.results_dir_patch.stop()
        self.temp_dir.cleanup()

    async def test_health_reports_detector_readiness(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["detector_ready"])
        self.assertTrue(response.json()["generalized_model_ready"])

    async def test_analysis_excludes_labels_and_reports_generalized_metrics(self):
        csv = (
            "feature_a,Label,Source IP,Timestamp,Destination IP,Destination Port\n"
            "1,BENIGN,10.0.0.1,2020-01-01 00:00:01,192.0.2.1,80\n"
            "2,DDoS,10.0.0.1,2020-01-01 00:00:02,192.0.2.1,81\n"
        )
        response = await self.client.post(
            "/analyze",
            files={"file": ("flows.csv", csv, "text/csv")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["total_flows"], 2)
        self.assertEqual(payload["anomaly_count"], 1)
        self.assertEqual(payload["generalized_attack_count"], 1)
        self.assertEqual(payload["model_generalized_attack_count"], 1)
        self.assertEqual(payload["flagged_flow_count"], 1)
        self.assertIn("not deduplicated incidents", payload["counting_note"])
        findings = payload["behavioral_findings"]
        self.assertEqual(findings["source_burst_event_count"], 1)
        self.assertEqual(findings["source_burst_flow_count"], 2)
        self.assertEqual(findings["high_rate_multiport_event_count"], 1)
        self.assertEqual(findings["events"][0]["flow_count"], 2)
        self.assertEqual(findings["events"][0]["top_destination_ips"][0]["destination_ip"], "192.0.2.1")
        self.assertEqual(payload["preview_rows"], 2)
        self.assertNotIn("Label", self.detector.received.columns)
        self.assertEqual(payload["evaluation"]["evaluated_rows"], 2)
        self.assertEqual(payload["evaluation"]["unsupported_actual_label_counts"], {})
        self.assertEqual(payload["generalization_evaluation"]["recall_attack"], 1.0)

        download = await self.client.get(payload["download_url"])
        self.assertEqual(download.status_code, 200)
        self.assertIn("predicted_class", download.text)
        self.assertIn("is_generalized_attack", download.text)
        self.assertIn("is_high_rate_multiport_activity", download.text)
        self.assertIn("actual_class_for_evaluation", download.text)

    async def test_rejects_non_csv_upload(self):
        response = await self.client.post(
            "/analyze",
            files={"file": ("flows.txt", "data", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    async def test_rejects_csv_missing_required_model_features(self):
        self.detector.feature_columns = ["feature_a", "feature_b"]
        response = await self.client.post(
            "/analyze",
            files={"file": ("flows.csv", "feature_a\n1\n", "text/csv")},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("feature_b", response.json()["detail"])

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
