import operator
from typing import Annotated, Dict, List, Optional, TypedDict


class AttemptHistory(TypedDict):
    """A record of one generated pipeline execution."""

    attempt_number: int
    code: str
    success: bool
    return_code: Optional[int]
    stderr: str
    error_type: str
    metrics: Dict[str, float]


class AutoMLState(TypedDict):
    # Immutable request/job context
    job_id: str
    file_path: str
    dataset_filename: str
    target_column: str
    user_prompt: str
    dataset_schema: str
    artifact_dir: str

    # Agent output and execution state
    execution_plan: str
    current_code: Optional[str]
    last_stdout: Optional[str]
    last_stderr: Optional[str]
    last_metrics: Dict[str, float]
    last_success: bool
    error_type: Optional[str]
    evaluation_report: Optional[str]
    artifact_files: Dict[str, str]
    best_model_path: Optional[str]

    # Flow control
    iteration_count: int
    max_iterations: int
    status: str  # planning | coding | running | retrying | bad_metrics | success | failed | limit_reached
    feedback: Optional[str]

    # History is appended by LangGraph instead of overwritten.
    history: Annotated[List[AttemptHistory], operator.add]
    baseline_metrics: Dict[str, float]
