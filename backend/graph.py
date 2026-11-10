from langgraph.graph import StateGraph, END
from state import AutoMLState

# Імпортуємо ваші реальні функції-агенти з папки nodes
from nodes.refiner import prompt_refiner_node
from nodes.coder import coder_node
from nodes.sandbox import sandbox_execution_node
from nodes.evaluator import evaluator_node

def decide_next_step(state: AutoMLState) -> str:
    """
    Визначає наступний крок на основі поточного стану.
    """
    iteration = state.get("iteration_count", 0)
    max_iters = state.get("max_iterations", 4)
    
    # 1. Захист від нескінченного циклу 
    if iteration >= max_iters:
        return "limit_reached"
    
    status = state.get("status", "running")
    error_type = state.get("error_type", "none")
    
    # 2. Якщо все успішно
    if status == "success" or error_type == "none":
        return "success"
    
    # 3. Технічні помилки (синтаксис, збої рантайму) -> повертаємо напряму до Кодера
    if error_type in ["fixable_syntax", "fatal", "runtime_error"]:
        return "to_coder"
        
    # 4. Проблеми з якістю (погані метрики) -> повертаємо до Рефайнера
    if error_type == "fixable_metric" or status == "bad_metrics":
        return "to_refiner"
        
    # Фолбек за замовчуванням
    return "to_refiner"

workflow = StateGraph(AutoMLState)

# Додаємо вузли (використовуємо імпортовані реальні функції)
workflow.add_node("prompt_refiner", prompt_refiner_node)
workflow.add_node("coder", coder_node)
workflow.add_node("sandbox", sandbox_execution_node)
workflow.add_node("evaluator", evaluator_node)

# Визначаємо стартову точку
workflow.set_entry_point("prompt_refiner")

# Лінійні зв'язки між основними етапами
workflow.add_edge("prompt_refiner", "coder")
workflow.add_edge("coder", "sandbox")
workflow.add_edge("sandbox", "evaluator")

# Умовні переходи після оцінки результату (мапінг узгоджено з decide_next_step)
workflow.add_conditional_edges(
    "evaluator",
    decide_next_step,
    {
        "to_refiner": "prompt_refiner",   # Повернення на рефлексію / доопрацювання промпту
        "to_coder": "coder",              # Виправлення коду напряму
        "success": END,                   # Успішне завершення
        "limit_reached": END              # Вичерпано ліміт спроб
    }
)

# Компіляція графа
app = workflow.compile()