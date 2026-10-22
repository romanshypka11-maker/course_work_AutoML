from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pathlib import Path
import aiofiles

from agents.coder_agent import CoderAgent
from tools.python_runner import PythonRunner

app = FastAPI(title="AutoML Course",)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/run_automl")
async def run_automl(file: UploadFile = File(...),
                     target_column: str = Form(...),
                     user_prompt: str = Form(...) ):

    file_path = UPLOAD_DIR / file.filename
    async with aiofiles.open(file_path, "wb") as buffer:
        content = await file.read()
        await buffer.write(content)

    try:
        # 2. Генерація коду за допомогою CoderAgent (наприклад, з DeepSeek-R1)
        coder = CoderAgent()
        generated_code = coder.generate_code(
            file_path=file.filename,
            target_column=target_column,
            user_goal=user_prompt  # Передаємо user_prompt як ціль для агента
        )

        # 3. Виконання в пісочниці PythonRunner
        runner = PythonRunner(timeout_sec=45)
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            file_content = f.read()

        execution_result = runner.run_code(
            code=generated_code,
            input_files={file.filename: file_content},
            expected_output_files=["out/metrics.txt"]
        )

        return {
            "success": execution_result.success,
            "generated_code": generated_code,
            "stdout": execution_result.stdout,
            "stderr": execution_result.stderr,
            "error_info": execution_result.error_info,
            "timed_out": execution_result.timed_out
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))