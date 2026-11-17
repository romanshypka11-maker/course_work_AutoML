"""Smoke tests for the generated-code runner.

Run from the repository root:
    python test_sandbox.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))
from tools.python_runner import PythonRunner


def main() -> None:
    runner = PythonRunner(timeout_sec=2)

    with tempfile.TemporaryDirectory() as artifacts:
        success_code = """
import json
from pathlib import Path
import pandas as pd

df = pd.read_csv('dataset.csv')
Path('out/metrics.json').write_text(json.dumps({'accuracy': 0.95}))
Path('out/model.joblib').write_bytes(b'demo-model')
print('accuracy: 0.95')
"""
        result = runner.run_code(
            code=success_code,
            input_files={"dataset.csv": "feature,target\n1,0\n2,1\n"},
            expected_output_files=["out/metrics.json"],
            artifact_dir=artifacts,
        )
        assert result.success, result.stderr
        assert Path(result.artifact_files["metrics.json"]).is_file()
        assert Path(result.artifact_files["model.joblib"]).is_file()
        print("PASS: successful run persists metrics and model artifacts")

    error_result = runner.run_code(
        code="raise RuntimeError('intentional failure')",
        expected_output_files=["out/metrics.json"],
    )
    assert not error_result.success and "intentional failure" in error_result.stderr
    print("PASS: runtime error is captured")

    timeout_result = runner.run_code("while True: pass")
    assert not timeout_result.success and timeout_result.timed_out
    print("PASS: timeout is enforced")


if __name__ == "__main__":
    main()
