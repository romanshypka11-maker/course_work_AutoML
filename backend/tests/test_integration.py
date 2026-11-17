import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

os.environ["MODEL_API_KEY"] = "offline-integration-test-key"

from fastapi.testclient import TestClient
from app.workflow.nodes.coder import coder_agent
from app.workflow.nodes.refiner import refiner_agent
import app.main as api_module


class AutoMLApiIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path(tempfile.mkdtemp(prefix="automl-integration-"))
        self.original_upload_dir = api_module.UPLOAD_DIR
        self.original_artifact_dir = api_module.ARTIFACT_DIR
        self.original_plan = refiner_agent.refine_plan
        self.original_code = coder_agent.generate_code
        self.planner_calls = 0
        self.coder_calls = 0

        api_module.UPLOAD_DIR = self.temp_root / "uploads"
        api_module.ARTIFACT_DIR = self.temp_root / "artifacts"
        api_module.UPLOAD_DIR.mkdir()
        api_module.ARTIFACT_DIR.mkdir()
        refiner_agent.refine_plan = self._fake_plan
        coder_agent.generate_code = self._fake_code
        self.client = TestClient(api_module.app)

    def tearDown(self) -> None:
        api_module.UPLOAD_DIR = self.original_upload_dir
        api_module.ARTIFACT_DIR = self.original_artifact_dir
        refiner_agent.refine_plan = self.original_plan
        coder_agent.generate_code = self.original_code
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _fake_plan(self, **_: object) -> str:
        self.planner_calls += 1
        return f"deterministic plan #{self.planner_calls}"

    def _fake_code(self, **_: object) -> str:
        self.coder_calls += 1
        metric = 0.20 if self.coder_calls == 1 else 0.90
        return f"""
import json
from pathlib import Path
import pandas as pd

pd.read_csv('dataset.csv')
Path('out').mkdir(exist_ok=True)
Path('out/metrics.json').write_text(json.dumps({{'accuracy': {metric}}}), encoding='utf-8')
Path('out/model.joblib').write_bytes(b'deterministic-model')
print(json.dumps({{'accuracy': {metric}}}))
"""

    def test_api_returns_artifacts_after_quality_retry(self) -> None:
        response = self.client.post(
            "/run_automl",
            data={"target_column": "target", "user_prompt": "classify the rows"},
            files={
                "file": (
                    "customer_data.csv",
                    b"feature,target\n1,0\n2,1\n",
                    "text/csv",
                )
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["success"], result)
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["iteration_count"], 2, result)
        self.assertEqual(result["execution_plan"], "deterministic plan #2", result)
        self.assertEqual(result["metrics"], {"accuracy": 0.9}, result)
        self.assertNotIn("uploads/", result["generated_code"])
        self.assertEqual(
            set(result["artifact_urls"]), {"metrics.json", "model.joblib"}, result
        )
        for relative_url in result["artifact_urls"].values():
            artifact_response = self.client.get(relative_url)
            self.assertEqual(artifact_response.status_code, 200, artifact_response.text)

    def test_input_and_download_guards(self) -> None:
        invalid_upload = self.client.post(
            "/run_automl",
            data={"target_column": "target"},
            files={"file": ("wrong.txt", b"target\n1\n", "text/plain")},
        )
        self.assertEqual(invalid_upload.status_code, 422)
        invalid_job = self.client.get("/jobs/not-a-uuid/artifacts/model.joblib")
        self.assertEqual(invalid_job.status_code, 400)


if __name__ == "__main__":
    unittest.main()
