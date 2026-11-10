from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pathlib import Path
import aiofiles

from utils import get_dataset_profile
from graph import app as automl_graph  # Імпортуємо наш скомпільований граф під ім'ям app

app = FastAPI(title="AutoML Course")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/run_automl")
async def run_automl(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    user_prompt: str = Form(...)
):
    file_path = UPLOAD_DIR / file.filename
    
    # 1. Зберігаємо завантажений файл локально
    async with aiofiles.open(file_path, "wb") as buffer:
        content = await file.read()
        await buffer.write(content)

    try:
        # 2. Генеруємо паспорт датасету через утиліту
        dataset_schema = get_dataset_profile(str(file_path), target_column)

        # 3. Формуємо початковий стан (State) для LangGraph
        initial_state = {
            "user_prompt": user_prompt,
            "file_path": str(file_path),
            "target_column": target_column,
            "dataset_schema": dataset_schema,
            "execution_plan": "",
            "current_code": "",
            "last_stderr": None,
            "evaluation_report": None,
            "status": "started"
        }

        # 4. Запускаємо граф за допомогою .invoke(initial_state)
        final_state = automl_graph.invoke(initial_state)

        # 5. Повертаємо фінальний результат клієнту
        return {
            "success": final_state.get("success", False),
            "execution_plan": final_state.get("execution_plan"),
            "generated_code": final_state.get("current_code"),
            "stdout": final_state.get("stdout"),
            "stderr": final_state.get("last_stderr"),
            "evaluation_report": final_state.get("evaluation_report"),
            "status": final_state.get("status")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))