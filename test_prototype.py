"""Offline integration test for the FastAPI + LangGraph control flow.

It replaces the external LLM responses with deterministic planner/coder responses,
so no provider key is consumed. Run after installing requirements:
    python test_prototype.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["MODEL_API_KEY"] = "offline-integration-test-key"
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from fastapi.testclient import TestClient
from nodes.coder import coder_agent
from nodes.refiner import refiner_agent

planner_calls = 0
coder_calls = 0


def fake_plan(**_: object) -> str:
    global planner_calls
    planner_calls += 1
    return f"deterministic plan #{planner_calls}"


def fake_code(**_: object) -> str:
    global coder_calls
    coder_calls += 1
    metric = 0.20 if coder_calls == 1 else 0.90
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


refiner_agent.refine_plan = fake_plan
coder_agent.generate_code = fake_code

import main


def main_test() -> None:
    root_temp_dir = tempfile.mkdtemp(prefix="automl-integration-")
    temp_root = Path(root_temp_dir)
    try:
        main.UPLOAD_DIR = temp_root / "uploads"
        main.ARTIFACT_DIR = temp_root / "artifacts"
        main.UPLOAD_DIR.mkdir()
        main.ARTIFACT_DIR.mkdir()

        client = TestClient(main.app)
        response = client.post(
            "/run_automl",
            data={"target_column": "target", "user_prompt": "classify the rows"},
            files={"file": ("customer_data.csv", b"feature,target\n1,0\n2,1\n", "text/csv")},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["success"] is True, result
        assert result["status"] == "success", result
        assert result["iteration_count"] == 2, result
        assert result["execution_plan"] == "deterministic plan #2", result
        assert result["metrics"] == {"accuracy": 0.9}, result
        assert "uploads/" not in result["generated_code"], result["generated_code"]
        assert set(result["artifact_urls"]) == {"metrics.json", "model.joblib"}, result

        for relative_url in result["artifact_urls"].values():
            artifact_response = client.get(relative_url)
            assert artifact_response.status_code == 200, artifact_response.text

        print("PASS: endpoint returns a complete successful AutoML job after a quality retry")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        # main creates these relative folders at import time; they are empty in this test.
        shutil.rmtree("uploads", ignore_errors=True)
        shutil.rmtree("artifacts", ignore_errors=True)


if __name__ == "__main__":
    main_test()
