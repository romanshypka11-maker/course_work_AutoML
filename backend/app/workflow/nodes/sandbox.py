import json
from pathlib import Path

from ...sandbox.python_runner import PythonRunner
from ...schemas.state import AutoMLState


runner = PythonRunner(timeout_sec=90)


def _load_metrics(artifact_files: dict[str, str]) -> dict[str, float]:
    metrics_path = artifact_files.get("metrics.json")
    if not metrics_path:
        return {}
    try:
        raw_metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    metrics: dict[str, float] = {}
    for key, value in raw_metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[str(key)] = float(value)
    return metrics


def sandbox_execution_node(state: AutoMLState) -> dict:
    """Execute code with a single canonical input name: dataset.csv."""
    print("--- [3/4] SANDBOX: executing generated pipeline ---")
    current_code = state.get("current_code") or ""
    source_path = Path(state["file_path"])

    if not current_code:
        return {
            "last_stderr": "No generated code was available for execution.",
            "last_success": False,
            "error_type": "runtime_error",
            "status": "retrying",
        }
    if not source_path.is_file():
        return {
            "last_stderr": "The uploaded dataset is no longer available.",
            "last_success": False,
            "error_type": "fatal",
            "status": "failed",
        }

    result = runner.run_code(
        code=current_code,
        input_files={state["dataset_filename"]: source_path.read_bytes()},
        expected_output_files=["out/metrics.json", "out/model.joblib"],
        artifact_dir=state["artifact_dir"],
    )

    if result.success:
        error_type = None
        status = "running"
    elif result.timed_out:
        error_type = "fatal"
        status = "failed"
    elif "SyntaxError" in result.stderr or "IndentationError" in result.stderr:
        error_type = "fixable_syntax"
        status = "retrying"
    else:
        error_type = "runtime_error"
        status = "retrying"

    artifact_files = result.artifact_files
    best_model_path = artifact_files.get("model.joblib")
    attempt_record = {
        "attempt_number": state.get("iteration_count", 0) + 1,
        "code": current_code,
        "success": result.success,
        "return_code": result.return_code,
        "stderr": result.stderr,
        "error_type": error_type or "none",
        "metrics": _load_metrics(artifact_files) if result.success else {},
    }

    return {
        "last_stdout": result.stdout,
        "last_stderr": result.stderr,
        "last_success": result.success,
        "last_metrics": attempt_record["metrics"],
        "artifact_files": artifact_files,
        "best_model_path": best_model_path,
        "error_type": error_type,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "history": [attempt_record],
        "status": status,
    }
