from pathlib import Path
from state import AutoMLState
# Імпортуємо ваш клас PythonRunner (припустимо, він лежить у tools/python_runner.py)
from tools.python_runner import PythonRunner

# Створюємо екземпляр з таймаутом, наприклад, 60 секунд
runner = PythonRunner(timeout_sec=60)

def sandbox_execution_node(state: AutoMLState) -> dict:
    """
    Вузол 3: Sandbox Execution (Пісочниця).
    Запускає згенерований кодом Python-скрипт в ізольованому середовищі.
    """
    print("--- [3] SANDBOX: Виконання згенерованого коду ---")

    current_code = state.get("current_code", "")
    file_path = state.get("file_path", "") # Шлях до файлу в папці uploads/
    
    if not current_code:
        return {
            "last_stderr": "Помилка: Код для виконання відсутній.",
            "last_success": False,
            "error_type": "fatal",
            "status": "running"
        }

    # 1. Читаємо вміст файлу датасету, щоб передати його через input_files у пісочницю
    input_files = {}
    if file_path and Path(file_path).exists():
        filename = Path(file_path).name
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            input_files[filename] = f.read()

    # 2. Запускаємо код за допомогою вашого PythonRunner
    # Вимагаємо, щоб скрипт обов'язково створив out/metrics.txt
    execution_result = runner.run_code(
        code=current_code,
        input_files=input_files,
        expected_output_files=["out/metrics.txt"]
    )

    # 3. Визначаємо тип помилки для маршрутизації
    error_type = "none"
    if not execution_result.success:
        if execution_result.timed_out:
            error_type = "fatal"
        elif "SyntaxError" in execution_result.stderr or "IndentationError" in execution_result.stderr:
            error_type = "fixable_syntax"
        else:
            error_type = "fixable_metric" # Наприклад, KeyError або помилка pandas/sklearn, яку кодер може виправити

    # 4. Формуємо історію спроби (AttemptHistory), якщо вона передбачена вашим архітектурним станом
    attempt_record = {
        "attempt_number": state.get("iteration_count", 0) + 1,
        "code": current_code,
        "success": execution_result.success,
        "return_code": execution_result.return_code,
        "stderr": execution_result.stderr,
        "error_type": error_type,
        "metrics": {}, # Оцінювач заповнить це пізніше
        "tokens_used": 0,
        "execution_time_sec": 0.0
    }

    # 5. Повертаємо оновлений стан у граф LangGraph
    return {
        "last_stdout": execution_result.stdout,
        "last_stderr": execution_result.stderr,
        "last_success": execution_result.success,
        "error_type": error_type,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "history": [attempt_record] # Додається до накопичувального списку через operator.add
    }