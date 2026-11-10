import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class AttemptHistory(TypedDict):
  attempt_number: int
  code: str
  success: bool
  exit_code: int
  stderr: str
  error_type: str  # "none", "fatal", "fixable_syntax", "fixable_metric"
  metrics: Dict[str, float]
  tokens_used: int
  execution_time_sec: float


class AutoMLState(TypedDict):
  # 1. Статичні вхідні дані 
  file_path: str
  target_column: str
  user_prompt: str
  dataset_schema: str


  # 2. Поточний динамічний стан
  current_code: Optional[str]
  last_stdout: Optional[str]
  last_stderr: Optional[str]
  last_metrics: Dict[str, float]
  last_success: bool
  error_type: Optional[str]

  # 3. Керування циклом та ліміти
  iteration_count: int
  max_iterations: int
  status: str  # "running", "success", "failed", "limit_reached"
  feedback: Optional[str]

  # 4. Накопичувальна історія ітерацій (через Reducer / operator.add у LangGraph)
  history: Annotated[List[AttemptHistory], operator.add]

  # 5. Шляхи до фінальних артефактів (для серіалізації)
  best_model_path: Optional[str]
  baseline_metrics: Dict[str, float]