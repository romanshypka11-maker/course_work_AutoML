import os
from pathlib import Path
from urllib.parse import quote
from uuid import UUID, uuid4

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from .core.config import ConfigurationError, llm_is_configured
from .services.dataset import get_dataset_profile
from .workflow.graph import app as automl_graph


app = FastAPI(title="AutoML Course Prototype", version="0.2.0")
# These paths are relative to backend/ in Docker and local development.
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "storage/uploads"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "storage/artifacts"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_ITERATIONS = int(os.getenv("AUTOML_MAX_ITERATIONS", "3"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _artifact_urls(job_id: str, artifact_files: dict[str, str]) -> dict[str, str]:
    return {
        name: f"/jobs/{job_id}/artifacts/{quote(name)}"
        for name in artifact_files
        if len(Path(name).parts) == 1
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_configured": llm_is_configured(),
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }


@app.post("/run_automl")
async def run_automl(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    user_prompt: str = Form("Build a robust model and report useful validation metrics."),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Please upload a CSV file.")
    if not target_column.strip():
        raise HTTPException(status_code=422, detail="Choose a target column.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded CSV is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    job_id = str(uuid4())
    job_upload_dir = UPLOAD_DIR / job_id
    job_artifact_dir = ARTIFACT_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_artifact_dir.mkdir(parents=True, exist_ok=True)

    # Never use an untrusted browser filename inside a generated script.
    dataset_filename = "dataset.csv"
    file_path = job_upload_dir / dataset_filename
    async with aiofiles.open(file_path, "wb") as buffer:
        await buffer.write(content)

    try:
        dataset_schema = get_dataset_profile(str(file_path), target_column.strip())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    initial_state = {
        "job_id": job_id,
        "file_path": str(file_path),
        "dataset_filename": dataset_filename,
        "target_column": target_column.strip(),
        "user_prompt": user_prompt.strip() or "Build a robust model and report useful validation metrics.",
        "dataset_schema": dataset_schema,
        "artifact_dir": str(job_artifact_dir),
        "execution_plan": "",
        "current_code": None,
        "last_stdout": None,
        "last_stderr": None,
        "last_metrics": {},
        "last_success": False,
        "error_type": None,
        "evaluation_report": None,
        "artifact_files": {},
        "best_model_path": None,
        "iteration_count": 0,
        "max_iterations": max(1, MAX_ITERATIONS),
        "status": "planning",
        "feedback": None,
        "history": [],
        "baseline_metrics": {},
    }

    try:
        final_state = await run_in_threadpool(automl_graph.invoke, initial_state)
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AutoML pipeline failed: {exc}") from exc

    final_status = final_state.get("status", "failed")
    if (
        final_status not in {"success", "failed"}
        and final_state.get("iteration_count", 0) >= final_state.get("max_iterations", MAX_ITERATIONS)
    ):
        final_status = "limit_reached"

    artifact_files = final_state.get("artifact_files") or {}
    return {
        "job_id": job_id,
        "success": final_status == "success" and bool(final_state.get("last_success")),
        "status": final_status,
        "execution_plan": final_state.get("execution_plan"),
        "generated_code": final_state.get("current_code"),
        "stdout": final_state.get("last_stdout"),
        "stderr": final_state.get("last_stderr"),
        "metrics": final_state.get("last_metrics") or {},
        "evaluation_report": final_state.get("evaluation_report"),
        "iteration_count": final_state.get("iteration_count", 0),
        "artifact_urls": _artifact_urls(job_id, artifact_files),
    }


@app.get("/jobs/{job_id}/artifacts/{artifact_name}")
def download_artifact(job_id: str, artifact_name: str):
    try:
        safe_job_id = str(UUID(job_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid job ID.") from exc

    requested = Path(artifact_name)
    if requested.is_absolute() or len(requested.parts) != 1 or requested.name != artifact_name:
        raise HTTPException(status_code=400, detail="Invalid artifact name.")

    artifact_path = ARTIFACT_DIR / safe_job_id / requested.name
    if not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(artifact_path, filename=requested.name)
