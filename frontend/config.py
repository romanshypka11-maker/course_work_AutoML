import os


BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
