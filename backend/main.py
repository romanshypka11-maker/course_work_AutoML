from fastapi import FastAPI,File,UploadFile,Form
from pathlib import Path
import aiofiles

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

    return {
        "status": "received",
        "filename": file.filename,
        "target_column": target_column,
        "user_prompt": user_prompt,
        "message": "File successfully uploaded and saved for pipeline processing."
    }