import tempfile
import unittest
from pathlib import Path

from app.sandbox.python_runner import PythonRunner


class PythonRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = PythonRunner(timeout_sec=2)

    def test_successful_run_persists_artifacts(self) -> None:
        code = """
import json
from pathlib import Path
import pandas as pd

pd.read_csv('dataset.csv')
Path('out/metrics.json').write_text(json.dumps({'accuracy': 0.95}))
Path('out/model.joblib').write_bytes(b'demo-model')
print('accuracy: 0.95')
"""
        with tempfile.TemporaryDirectory() as artifacts:
            result = self.runner.run_code(
                code=code,
                input_files={"dataset.csv": "feature,target\n1,0\n2,1\n"},
                expected_output_files=["out/metrics.json", "out/model.joblib"],
                artifact_dir=artifacts,
            )
            self.assertTrue(result.success, result.stderr)
            self.assertTrue(Path(result.artifact_files["metrics.json"]).is_file())
            self.assertTrue(Path(result.artifact_files["model.joblib"]).is_file())

    def test_runtime_error_is_captured(self) -> None:
        result = self.runner.run_code(
            code="raise RuntimeError('intentional failure')",
            expected_output_files=["out/metrics.json"],
        )
        self.assertFalse(result.success)
        self.assertIn("intentional failure", result.stderr)

    def test_timeout_is_enforced(self) -> None:
        result = self.runner.run_code("while True: pass")
        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)


if __name__ == "__main__":
    unittest.main()
