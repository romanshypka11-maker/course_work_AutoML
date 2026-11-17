from openai import OpenAI

from ...core.config import get_llm_settings
from ...schemas.state import AutoMLState


class CoderAgent:
    """Generates a self-contained, sandbox-compatible scikit-learn script."""

    def _create_client(self) -> tuple[OpenAI, str]:
        settings = get_llm_settings()
        return (
            OpenAI(base_url=settings.base_url, api_key=settings.api_key),
            settings.model_name,
        )

    def generate_code(
        self,
        target_column: str,
        execution_plan: str,
        dataset_schema: str,
        dataset_filename: str,
        last_stderr: str | None = None,
    ) -> str:
        client, model_name = self._create_client()
        error_fix_section = ""
        if last_stderr:
            error_fix_section = f"""
The previous script failed with this traceback:
{last_stderr[-1800:]}
Fix that failure in the new complete script.
"""

        prompt = f"""
You are a Python ML engineer. Return exactly one executable Python script, with no Markdown.

Technical plan:
{execution_plan}

Dataset profile:
{dataset_schema}
Target column: {target_column}

Sandbox contract:
- The input CSV is in the current working directory under this exact filename: {dataset_filename}
- Read only that filename. Never use an absolute path or an uploads/ path.
- Only use Python standard library, pandas, numpy, scikit-learn and joblib.
- Validate that the target column exists and remove rows with a missing target.
- Build preprocessing with ColumnTransformer / Pipeline so numeric and categorical features work.
- Use a deterministic random_state.
- Make a train/test split suitable for classification or regression.
- Save a fitted pipeline as out/model.joblib.
- Save a JSON object of numeric metrics to out/metrics.json.
- Print the JSON metrics object as the final stdout line.
- Create out/ before writing files.
- Do not download data, call APIs, read any other files, or use matplotlib.

{error_fix_section}
"""
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Write dependable, minimal scikit-learn programs and output code only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        raw_content = response.choices[0].message.content or ""
        return self._extract_code(raw_content)

    @staticmethod
    def _extract_code(text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1]
        if "```python" in text:
            return text.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in text:
            return text.split("```", 2)[1].strip()
        return text.strip()


coder_agent = CoderAgent()


def coder_node(state: AutoMLState) -> dict:
    print("--- [2/4] CODER: generating a sandbox-compatible pipeline ---")
    prior_error = state.get("last_stderr") if state.get("error_type") else None
    current_code = coder_agent.generate_code(
        target_column=state["target_column"],
        execution_plan=state["execution_plan"],
        dataset_schema=state["dataset_schema"],
        dataset_filename=state["dataset_filename"],
        last_stderr=prior_error,
    )
    if not current_code:
        return {
            "current_code": "",
            "last_success": False,
            "last_stderr": "The model returned an empty script.",
            "error_type": "runtime_error",
            "status": "retrying",
        }
    return {
        "current_code": current_code,
        "status": "coding",
    }
