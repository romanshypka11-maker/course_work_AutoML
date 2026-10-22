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

class PythonRunner:
    """
    Безпечний виконавець Python-коду в ізольованому тимчасовому середовищі
    з обмеженням часу та санітизацією помилок.
    """

    def __init__(self, timeout_sec: int = 30):
        self.timeout_sec = timeout_sec

    def _sanitize_stderr(self, stderr: str) -> str:
        """
        Очищує stderr від потенційних сирих даних або довгих рядків,
        щоб запобігти витоку конфіденційної інформації у промпти LLM.
        """
        if not stderr:
            return ""

        trimmed = stderr[-2000:]
        return trimmed

    def run_code(
        self,
        code: str,
        input_files: dict[str, str] | None = None,
        expected_output_files: list[str] | None = None,
    ) -> ExecutionResult:
        """
        Виконує Python-код у тимчасовій директорії з таймаутом.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # 1. Записуємо вхідні файли у пісочницу, якщо вони є
            if input_files:
                for filename, content in input_files.items():
                    file_path = tmp_path / filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(content, encoding="utf-8")

            # 2. Створюємо папку для вихідних артефактів (out/)
            out_dir = tmp_path / "out"
            out_dir.mkdir(exist_ok=True)

            # 3. Зберігаємо згенерований код у тимчасовий скрипт
            script_path = tmp_path / "generated_script.py"
            script_path.write_text(code, encoding="utf-8")

            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(tmp_path),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
                stdout = result.stdout or ""
                stderr = self._sanitize_stderr(result.stderr)
                return_code = result.returncode
                success = (return_code == 0)

                missing_files = []
                if expected_output_files and success:
                    for exp_file in expected_output_files:
                        if not (tmp_path / exp_file).exists():
                            missing_files.append(exp_file)
                    if missing_files:
                        success = False

                error_info = None
                if not success:
                    error_info = stderr.strip().split("\n")[-1] if stderr else f"Exit code {return_code}"

                return ExecutionResult(
                    success=success,
                    return_code=return_code,
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=False,
                    error_info=error_info,
                )

            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    success=False,
                    return_code=None,
                    stdout="",
                    stderr="Execution timed out.",
                    timed_out=True,
                    error_info=f"Process exceeded timeout of {self.timeout_sec} seconds.",
                )
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    return_code=None,
                    stdout="",
                    stderr=str(e),
                    timed_out=False,
                    error_info=str(e),
                )
