import os 
from openai import OpenAI

class CoderAgent:
    """
        Агент для генерації Python-коду за допомогою локальної моделі DeepSeek-R1 через Ollama.
    """
    def __init__(self, model_name: str = "deepseek-r1"):
        self.client = OpenAI(
            base_url=os.environ.get("OLLAMA_API_URL", "http://host.docker.internal:11434/v1"),
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
        )
        self.model_name = model_name

    def generate_code(self, file_path: str, target_column: str, user_goal: str) -> str:
        prompt = f"""
            Ти професійний Python Data Scientist та AutoML інженер.
            Твоє завдання — написати чистий, самодостатній Python-скрипт для обробки датасету та тренування базової моделі за допомогою pandas та scikit-learn.

            Вхідні дані:
            - Шлях до файлу датасету: '{file_path}'
            - Цільова колонка (Target): '{target_column}'
            - Побажання користувача / Бізнес-ціль: '{user_goal}'

            Суворі правила для коду:
            1. Код має бути повністю робочим і виконуватися без додаткових ручних правок.
            2. Обов'язково опрацюй пропущені значення (NaN) та переведи категорійні змінні у числові (наприклад, через pd.get_dummies).
            3. Розділи дані на train/test, навчи відповідну модель (наприклад, RandomForest).
            4. Усі фінальні артефакти чи звіти ЗБЕРЕЖУЙ виключно у спеціальну вихідну папку 'out/'. Наприклад: `out/metrics.txt`.
            5. Поверни код у блоці ```python ... ```.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Ти експерт з програмування на Python. Пиши виключно виконуваний код."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        raw_content = response.choices[0].message.content or ""

        return self._extract_code(raw_content)
    
    def _extract_code(self, text: str) -> str:
        # DeepSeek-R1 часто пише міркування всередині тегів <think>...</think>. Видаляємо їх:
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