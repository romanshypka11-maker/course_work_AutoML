import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple


class ExecutionResult(NamedTuple):
    success: bool
    return_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    error_info: str | None
    artifact_files: dict[str, str]


class PythonRunner:
    """Execute generated code in a short-lived working directory.

    This isolates files for each run and enforces a timeout. It is not a complete
    security boundary for hostile code; deploy this component in a locked-down
    container with no host secrets or network access.
    """

    def __init__(self, timeout_sec: int = 90):
        self.timeout_sec = timeout_sec

    @staticmethod
    def _sanitize_stderr(stderr: str) -> str:
        return (stderr or "")[-3000:]

    @staticmethod
    def _safe_relative_path(path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"Unsafe relative file path: {path}")
        return candidate

    @staticmethod
    def _write_input_file(base_dir: Path, filename: str, content: str | bytes) -> None:
        relative = PythonRunner._safe_relative_path(filename)
        destination = base_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            destination.write_bytes(content)
        else:
            destination.write_text(content, encoding="utf-8")

    @staticmethod
    def _persist_artifacts(out_dir: Path, artifact_dir: str | Path | None) -> dict[str, str]:
        if artifact_dir is None or not out_dir.exists():
            return {}

        destination_root = Path(artifact_dir)
        destination_root.mkdir(parents=True, exist_ok=True)
        persisted: dict[str, str] = {}
        for source in out_dir.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(out_dir)
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            persisted[str(relative)] = str(destination)
        return persisted

    def run_code(
        self,
        code: str,
        input_files: dict[str, str | bytes] | None = None,
        expected_output_files: list[str] | None = None,
        artifact_dir: str | Path | None = None,
    ) -> ExecutionResult:
        """Run a script and persist its out/ artifacts before cleanup."""
        for output_file in expected_output_files or []:
            self._safe_relative_path(output_file)

        with tempfile.TemporaryDirectory(prefix="automl-") as tmpdir:
            tmp_path = Path(tmpdir)
            if input_files:
                for filename, content in input_files.items():
                    self._write_input_file(tmp_path, filename, content)

            out_dir = tmp_path / "out"
            out_dir.mkdir(exist_ok=True)
            script_path = tmp_path / "generated_script.py"
            script_path.write_text(code, encoding="utf-8")

            # Preserve PATH for the interpreter's native dependencies, but avoid
            # passing project-specific environment variables/secrets to LLM code.
            execution_env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
            }

            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=tmp_path,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    env=execution_env,
                )
                stdout = result.stdout or ""
                stderr = self._sanitize_stderr(result.stderr)
                return_code = result.returncode
                success = return_code == 0

                missing_files = []
                if success:
                    for expected in expected_output_files or []:
                        if not (tmp_path / expected).is_file():
                            missing_files.append(expected)
                    if missing_files:
                        success = False
                        missing_message = "Required output files missing: " + ", ".join(missing_files)
                        stderr = f"{stderr}\n{missing_message}".strip()

                artifact_files = self._persist_artifacts(out_dir, artifact_dir) if success else {}
                error_info = None
                if not success:
                    error_info = stderr.splitlines()[-1] if stderr else f"Exit code {return_code}"

                return ExecutionResult(
                    success=success,
                    return_code=return_code,
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=False,
                    error_info=error_info,
                    artifact_files=artifact_files,
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    success=False,
                    return_code=None,
                    stdout="",
                    stderr=f"Execution timed out after {self.timeout_sec} seconds.",
                    timed_out=True,
                    error_info=f"Process exceeded timeout of {self.timeout_sec} seconds.",
                    artifact_files={},
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    return_code=None,
                    stdout="",
                    stderr=str(exc),
                    timed_out=False,
                    error_info=str(exc),
                    artifact_files={},
                )
