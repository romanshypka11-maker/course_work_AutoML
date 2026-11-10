import re
from state import AutoMLState

def parse_metrics_from_stdout(stdout: str) -> dict:
    """
    Витягує числові метрики із виводу stdout (з файлу metrics.txt).
    """
    metrics = {}
    if not stdout:
        return metrics
    
    for line in stdout.split("\n"):
        match = re.search(r"['\"]?([a-zA-Z0-9_-]+)['\"]?\s*[:=]\s*([0-9]+\.[0-9]+|[0-9]+)", line)
        if match:
            key, val = match.groups()
            try:
                metrics[key] = float(val)
            except ValueError:
                pass
    return metrics

def evaluator_node(state: AutoMLState) -> dict:
    """
    Вузол Оцінювача (Evaluator / Critic).
    Аналізує результати виконання та формує адресні підказки:
    - Для Кодера (при технічних помилках)
    - Для Стратега / Refiner (при аналізі причин поганих метрик)
    """
    print("--- [4] EVALUATOR / CRITIC: Аналіз результатів та діагностика ---")

    last_success = state.get("last_success", False)
    last_stderr = state.get("last_stderr", "")
    last_stdout = state.get("last_stdout", "")
    iteration_count = state.get("iteration_count", 1)

    # ВИПАДОК 1: Технічна помилка виконання коду -> Підказка для КОДЕРА
    if not last_success:
        print(f"[-] Виявлено технічний збій у коді на ітерації {iteration_count}.")
        code_error_feedback = f"""
        ПОМИЛКА ВИКОНАННЯ КОДУ В ПІСОЧНИЦІ (stderr):
        {last_stderr[-1500:]}
        
        ВКАЗІВКА ДЛЯ КОДЕРА:
        Проаналізуй цей трейсбек. Виправ помилки у своєму скрипті (це можуть бути невірні імпорти, синтаксичні баги, помилки в назвах колонок DataFrame або проблеми з обробкою NaN). Надай повністю робочу виправлену версію коду.
        """
        return {
            "feedback": code_error_feedback.strip(),
            "last_metrics": {},
            "status": "running"
        }

    # ВИПАДОК 2: Код виконався успішно -> Аналізуємо якість метрик для СТРАТЕГА (Refiner)
    print("[+] Код виконався успішно! Парсимо та аналізуємо метрики...")
    metrics = parse_metrics_from_stdout(last_stdout)
    
    # Тут Критик оцінює якість і робить припущення "чому метрики можуть бути поганими"
    metric_analysis_feedback = f"""
    АНАЛІЗ ЯКОСТІ МОДЕЛІ (Ітерація {iteration_count}):
    Отримані метрики: {metrics}
    
    ЕКСПЕРТНИЙ ВИСНОВОК КРИТИКА (Чому результати можуть бути субіддеальними та що змінити в плані):
    - Якщо точність / ROC-AUC / R2 низькі, це може свідчити про недонавчання (underfitting), погане кодування категоріальних змінних, відсутність нормалізації чи дисбаланс класів.
    - ВКАЗІВКА ДЛЯ СТРАТЕГА (REFINER): Будь ласка, скоригуй технічний план. Запропонуй інший алгоритм (наприклад, перейди від базового дерева до Gradient Boosting / LightGBM), додай крос-валідацію, покращ обробку викидів або застосуй балансування вибірки.
    """

    return {
        "feedback": metric_analysis_feedback.strip(),
        "last_metrics": metrics,
        "status": "running"
    }


def decide_next_step(state: AutoMLState) -> str:
    """
    Функція умовної маршрутизації (Conditional Edge) для LangGraph.
    Визначає правильний потік залежно від природи проблеми:
    - Якщо помилка коду -> назад до кодера ("to_coder")
    - Якщо треба покращити якість/метрики -> назад до стратега ("to_refiner")
    """
    last_success = state.get("last_success", False)
    iteration_count = state.get("iteration_count", 1)
    max_iterations = state.get("max_iterations", 3)

    # Якщо вичерпано ліміт спроб
    if iteration_count >= max_iterations:
        print("[ROUTE] Вичерпано максимальний ліміт ітерацій. Зупикаємо граф.")
        return "limit_reached"

    # 1. Якщо сталася технічна помилка в коді -> відправляємо КОДЕРУ для виправлення
    if not last_success:
        print("[ROUTE] Збій у виконанні коду -> Маршрутизація до CODER.")
        return "to_coder"

    # 2. Якщо код успішний, але якість метрик аналізується -> відправляємо СТРАТЕГУ (Refiner) для зміни плану
    # (Можна обмежити кількість ітерацій оптимізації, наприклад, не більше 2 циклів планування)
    if iteration_count < 2:
        print("[ROUTE] Метрики потребують оптимізації -> Маршрутизація до PROMPT_REFINER (Стратега).")
        return "to_refiner"

    # 3. Якщо ітерацій достатньо і результати прийнятні -> успішне завершення
    print("[ROUTE] Досягнуто цільових ітерацій та успішного результату -> Завершення.")
    return "success"