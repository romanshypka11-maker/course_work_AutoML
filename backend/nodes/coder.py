import os
from openai import OpenAI
from state import AutoMLState

class CoderAgent:
    """
    Агент для генерації Python-коду на основі технічного плану від Стратега.
    """
    def __init__(self, model_name: str = os.environ.get("NAME_MODEL", "gemini-3.1-flash-lite")):
        self.client = OpenAI(
            base_url=os.environ.get("BASE_URL", "[https://generativelanguage.googleapis.com/v1beta/openai/](https://generativelanguage.googleapis.com/v1beta/openai/)"),
            api_key=os.environ.get("MODEL_API_KEY"),
        )
        self.model_name = model_name
        
    def generate_code(self, file_path: str, target_column: str, execution_plan: str, dataset_schema: str, last_stderr: str = None) -> str:
        
        # Секція виправлення помилок (якщо попередня ітерація коду впала в пісочниці)
        error_fix_section = ""
        if last_stderr:
            error_fix_section = f"""
            УВАГА! Попередня версія згенерованого тобою коду завершилася з технічною помилкою (stderr у пісочниці):
            {last_stderr}

            Твоє завдання — проаналізувати цей трейсбек і **виправити помилку** у новому коді!
            """

        prompt = f"""
            Ти професійний Python Data Scientist та AutoML інженер.
            Твоє завдання — написати чистий, самодостатній Python-скрипт згідно з технічним планом нижче.

            Вхідні дані:
            - Шлях до файлу датасету: '{file_path}'
            - Цільова колонка (Target): '{target_column}'
            - Схема датасету / метадані: {dataset_schema}

            Технічний план від Архітектора (Стратега):
            {execution_plan}

            {error_fix_section}

            Суворі правила для коду:
            1. Код має бути повністю робочим і виконуватися без додаткових ручних правок.
            2. Обов'язково опрацюй пропущені значення (NaN) та переведи категорійні змінні у числові (згідно з планом).
            3. Розділи дані на train/test, навчи відповідну модель.
            4. Усі фінальні артефакти чи звіти ЗБЕРЕЖУЙ виключно у спеціальну вихідну папку 'out/'. Наприклад: `out/metrics.txt`.
            5. Обов'язково прочитай збережений файл `out/metrics.txt` в кінці скрипту та виведи його вміст через `print()`, щоб результати відображалися у логах виконання.
            6. Поверни ВИКЛЮЧНО виконуваний код у блоці ```python ... ``` без зайвих коментарів.
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Ти експерт з програмування на Python. Пиши виключно виконуваний код за технічним планом."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # Нижча температура для коду, щоб уникнути зайвої «творчості» і синтаксичних помилок
        )
        raw_content = response.choices[0].message.content or ""

        return self._extract_code(raw_content)
    
    def _extract_code(self, text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1]

        # Витягуємо код із блоку ```python ... ```
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                code_block = parts[1].split("```")[0]
                return code_block.strip()
                
        # Альтернативний пошук звичайного блоку коду
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                return parts[1].strip()

        return text.strip()

# Створюємо глобальний екземпляр агента для вузла
coder_agent = CoderAgent()

def coder_node(state: AutoMLState) -> dict:
    """
    Вузол 2: Coder Agent (Агент-Кодер).
    Отримує технічний план від стратега та генерує Python-код.
    """
    print("--- [2] CODER AGENT: Генерація коду на основі плану ---")

    file_path = state.get("file_path", "")
    target_column = state.get("target_column", "")
    execution_plan = state.get("execution_plan", "")
    dataset_schema = state.get("dataset_schema", "")
    
    # Якщо це повторна ітерація і попередній код впав, забираємо помилку
    last_stderr = state.get("last_stderr")

    # Генеруємо код через наш клас
    current_code = coder_agent.generate_code(
        file_path=file_path,
        target_column=target_column,
        execution_plan=execution_plan,
        dataset_schema=dataset_schema,
        last_stderr=last_stderr
    )

    return {
        "current_code": current_code,
        "status": "running"
    }