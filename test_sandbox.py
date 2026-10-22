from tools.python_runner import PythonRunner

# Ініціалізуємо виконавець (для тесту таймауту поставимо короткий ліміт — 3 секунди)
runner = PythonRunner(timeout_sec=3)

print("=" * 60)
print("ТЕСТ 1: Успішний код (вхідні дані + вихідний файл в out/)")
print("=" * 60)

# Симулюємо вхідний датасет
input_data = {
    "dataset.csv": "feature1,feature2,target\n1.5,2.0,0\n3.0,4.1,1\n5.2,6.3,0"
}

# Код, який зчитає датасет і збереже звіт у вихідну папку out/
success_code = """
import pandas as pd
from pathlib import Path

# Читаємо датасет із поточної ізольованої папки
df = pd.read_csv('dataset.csv')
print(f"Успішно завантажено датасет. Форма: {df.shape}")

# Симулюємо збереження моделі або звіту у вихідну папку out/
out_file = Path('out') / 'model_report.txt'
out_file.write_text("Model accuracy: 95.5%", encoding="utf-8")
print("Звіт успішно збережено у вихідну папку.")
"""

result_1 = runner.run_code(
    code=success_code,
    input_files=input_data,
    expected_output_files=["out/model_report.txt"]
)

print(f"Статус успіху (Success): {result_1.success}")
print(f"Код повернення (Return Code): {result_1.return_code}")
print(f"Вивід (Stdout):\n{result_1.stdout.strip()}")
print("\n" + "=" * 60)


print("ТЕСТ 2: Код із помилкою (Runtime Error)")
print("=" * 60)

error_code = """
import pandas as pd
# Спроба прочитати файл, якого немає в пісочниці
df = pd.read_csv('missing_file.csv')
"""

result_2 = runner.run_code(code=error_code)

print(f"Статус успіху (Success): {result_2.success}")
print(f"Інформація про помилку (Error Info): {result_2.error_info}")
print(f"Очищений stderr:\n{result_2.stderr.strip()}")
print("\n" + "=" * 60)


print("ТЕСТ 3: Захист від зависання (Timeout)")
print("=" * 60)

timeout_code = """
import time
print("Починаємо нескінченний цикл...")
while True:
    time.sleep(1)
"""

result_3 = runner.run_code(code=timeout_code)

print(f"Статус успіху (Success): {result_3.success}")
print(f"Чи спрацював таймаут (Timed Out): {result_3.timed_out}")
print(f"Інформація про помилку: {result_3.error_info}")
print("=" * 60)