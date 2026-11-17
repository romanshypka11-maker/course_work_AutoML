from langgraph.graph import END, StateGraph

from nodes.coder import coder_node
from nodes.evaluator import evaluator_node
from nodes.refiner import prompt_refiner_node
from nodes.sandbox import sandbox_execution_node
from state import AutoMLState


def decide_next_step(state: AutoMLState) -> str:
    """Route based on a completed evaluation, with a hard retry limit."""
    status = state.get("status", "failed")
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    if status == "success":
        return "success"
    if status == "failed":
        return "failed"
    if iteration >= max_iterations:
        return "limit_reached"
    if status == "bad_metrics" or state.get("error_type") == "fixable_metric":
        return "to_refiner"
    if status == "retrying" and state.get("error_type") in {"fixable_syntax", "runtime_error"}:
        return "to_coder"
    return "failed"


workflow = StateGraph(AutoMLState)
workflow.add_node("prompt_refiner", prompt_refiner_node)
workflow.add_node("coder", coder_node)
workflow.add_node("sandbox", sandbox_execution_node)
workflow.add_node("evaluator", evaluator_node)

workflow.set_entry_point("prompt_refiner")
workflow.add_edge("prompt_refiner", "coder")
workflow.add_edge("coder", "sandbox")
workflow.add_edge("sandbox", "evaluator")
workflow.add_conditional_edges(
    "evaluator",
    decide_next_step,
    {
        "to_refiner": "prompt_refiner",
        "to_coder": "coder",
        "success": END,
        "failed": END,
        "limit_reached": END,
    },
)

app = workflow.compile()
