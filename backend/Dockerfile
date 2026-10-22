FROM python:3.11-slim

WORKDIR /backend

# Копіюємо та встановлюємо залежності бекенда
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо решту коду бекенда
COPY . .

# Запуск Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]