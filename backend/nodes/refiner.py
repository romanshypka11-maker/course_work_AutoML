import os
from openai import OpenAI
from state import AutoMLState

class RefinerAgent:
    """
    Агент-Стратег для аналізу завдання та створення технічного плану за допомогою gemini-3.1-flash-lite.
    """
    def __init__(self, model_name: str = os.environ.get("NAME_MODEL", "gemini-3.1-flash-lite")):
        self.client = OpenAI(
            base_url=os.environ.get("BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            api_key=os.environ.get("MODEL_API_KEY"),
        )
        self.model_name = model_name

    def refine_plan(self, user_prompt: str, target_column: str, file_path: str, dataset_schema: str, feedback: str = None) -> str:
        feedback_section = ""
        if feedback:
            feedback_section = f"""
            УВАГА! Попередня спроба тренування виконалася успішно (без технічних помилок), але Оцінювач (Evaluator) зафіксував незадовільні метрики:
            {feedback}

            Твоє завдання — проаналізувати ці результати і **скоригувати технічний план** (змінити алгоритм, підхід до препроцесингу, балансування класів тощо), щоб покращити якість моделі в наступній ітерації.
            """

        prompt = f"""
            Ти — досвідчений дата-саєнтист та архітектор AutoML систем.
            Твоє завдання — проаналізувати запит користувача, структуру датасету та результати попередньої оцінки, після чого скласти чіткий, покроковий технічний план для Python-розробника (агента-кодера).

            Вхідні дані:
            - Запит користувача: {user_prompt}
            - Цільова змінна (Target): {target_column}
            - Шлях до файлу датасету: {file_path}
            - Схема датасету / метадані: {dataset_schema}

            {feedback_section}

            Вимоги до технічного плану:
            1. Визначити тип задачі машинного навчання.
            2. Стратегія препроцесингу (робота з пропускеами, кодування, масштабування, відбір фіч).
            3. Рекомендований алгоритм моделі (з урахуванням попередніх невдалих метрик, якщо вони є).
            4. Цільова метрика якості.

            Формат відповіді: Чіткий, структурований технічний план без зайвого вступу.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Ти експерт з архітектури ML-систем. Удосконалюй плани на основі метрик."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        raw_content = response.choices[0].message.content or ""
        return self._clean_output(raw_content)

    def _clean_output(self, text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1]
        return text.strip()

refiner_agent = RefinerAgent()

def prompt_refiner_node(state: AutoMLState) -> dict:
    """
    Вузол 1: Prompt Refinement Agent (Агент-Стратег).
    Отримує метрики від evaluator та оновлює план.
    """
    print("--- [1] PROMPT REFINER AGENT: Аналіз метрик та коригування плану ---")

    user_prompt = state.get("user_prompt", "")
    target_column = state.get("target_column", "")
    file_path = state.get("file_path", "")
    dataset_schema = state.get("dataset_schema", "")

    # Беремо загальний фідбек з поточного стану
    feedback = state.get("feedback")

    execution_plan = refiner_agent.refine_plan(
        user_prompt=user_prompt,
        target_column=target_column,
        file_path=file_path,
        dataset_schema=dataset_schema,
        feedback=feedback
    )
    
    return {
        "execution_plan": execution_plan,
        "status": "running"
    }