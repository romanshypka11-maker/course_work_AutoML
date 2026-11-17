import os

import httpx
import pandas as pd
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))

st.set_page_config(page_title="AutoML Course", page_icon="🤖", layout="wide")
st.title("🤖 AutoML Course Prototype")
st.caption("Завантажте CSV, оберіть target — система побудує та перевірить ML-пайплайн.")

uploaded_file = st.file_uploader("CSV-файл", type=["csv"])

if uploaded_file is None:
    st.info("Почніть із завантаження табличного CSV-файлу.")
    st.stop()

try:
    uploaded_file.seek(0)
    df_preview = pd.read_csv(uploaded_file, nrows=20)
    columns = list(df_preview.columns)
    if not columns:
        st.error("CSV не містить колонок.")
        st.stop()
except Exception as exc:
    st.error(f"Не вдалося прочитати CSV: {exc}")
    st.stop()

st.success(f"Файл готовий: {uploaded_file.name}")
with st.expander("Попередній перегляд даних", expanded=False):
    st.dataframe(df_preview, use_container_width=True)

with st.form("automl_form"):
    selected_target = st.selectbox("Цільова колонка (target)", columns)
    user_prompt = st.text_area(
        "Бізнес-ціль або вимоги до моделі",
        value="Побудуй надійну модель і покажи корисні метрики валідації.",
        help="Наприклад: важливо не пропускати клієнтів, які можуть піти.",
    )
    submitted = st.form_submit_button("Запустити AutoML", type="primary")

if not submitted:
    st.stop()

uploaded_file.seek(0)
payload = {
    "target_column": selected_target,
    "user_prompt": user_prompt,
}
files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}

try:
    with st.spinner("Плануємо, генеруємо та перевіряємо ML-пайплайн…"):
        response = httpx.post(
            f"{BACKEND_URL}/run_automl",
            data=payload,
            files=files,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
except httpx.HTTPError as exc:
    st.error(f"Не вдалося зв’язатися з бекендом: {exc}")
    st.stop()

if response.status_code != 200:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    st.error(f"Бекенд повернув {response.status_code}: {detail}")
    st.stop()

result = response.json()
if result.get("success"):
    st.success(f"Пайплайн успішно завершено за {result.get('iteration_count', 0)} ітерац(ій).")
elif result.get("status") == "limit_reached":
    st.warning("Досягнуто ліміт спроб. Перегляньте помилку та останній згенерований код нижче.")
else:
    st.error(f"Пайплайн не завершився успішно. Статус: {result.get('status', 'unknown')}")

metrics = result.get("metrics") or {}
if metrics:
    st.subheader("Метрики")
    metric_columns = st.columns(min(4, len(metrics)))
    for idx, (name, value) in enumerate(metrics.items()):
        metric_columns[idx % len(metric_columns)].metric(name, f"{value:.4f}")

if result.get("evaluation_report"):
    st.info(result["evaluation_report"])

with st.expander("Технічний план", expanded=False):
    st.write(result.get("execution_plan") or "План не було збережено.")
with st.expander("Згенерований Python-код", expanded=False):
    st.code(result.get("generated_code") or "Код відсутній.", language="python")
with st.expander("Логи виконання", expanded=not result.get("success")):
    if result.get("stdout"):
        st.code(result["stdout"], language="text")
    if result.get("stderr"):
        st.error(result["stderr"])
    if not result.get("stdout") and not result.get("stderr"):
        st.write("Логи відсутні.")

artifact_urls = result.get("artifact_urls") or {}
if artifact_urls:
    st.subheader("Артефакти")
    for name, relative_url in artifact_urls.items():
        try:
            artifact_response = httpx.get(
                f"{BACKEND_URL}{relative_url}", timeout=30.0
            )
            artifact_response.raise_for_status()
            st.download_button(
                label=f"Завантажити {name}",
                data=artifact_response.content,
                file_name=name,
                key=f"download-{result.get('job_id')}-{name}",
            )
        except httpx.HTTPError as exc:
            st.warning(f"Артефакт {name} створено, але не вдалося завантажити: {exc}")
