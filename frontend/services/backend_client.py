import httpx

from config import BACKEND_URL, REQUEST_TIMEOUT_SECONDS


def run_automl(
    filename: str,
    content: bytes,
    target_column: str,
    user_prompt: str,
) -> httpx.Response:
    """Submit a CSV and its training request to the FastAPI service."""
    return httpx.post(
        f"{BACKEND_URL}/run_automl",
        data={"target_column": target_column, "user_prompt": user_prompt},
        files={"file": (filename, content, "text/csv")},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def fetch_artifact(relative_url: str) -> httpx.Response:
    """Download a backend artifact server-side so Streamlit can expose a safe button."""
    response = httpx.get(f"{BACKEND_URL}{relative_url}", timeout=30.0)
    response.raise_for_status()
    return response
