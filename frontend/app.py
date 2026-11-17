import httpx
import pandas as pd
import streamlit as st

from services.backend_client import run_automl
from views.results import render_result


st.set_page_config(page_title="AutoML Course", page_icon="🤖", layout="wide")
st.title("🤖 AutoML Course Prototype")
st.caption("Завантажте CSV, оберіть target — система побудує та перевірить ML-пайплайн.")

uploaded_file = st.file_uploader("CSV-файл", type=["csv"])
if uploaded_file is None:
    st.info("Почніть із завантаження табличного CSV-файлу.")
    st.stop()

try:
    uploaded_file.seek(0)
    preview_frame = pd.read_csv(uploaded_file, nrows=20)
    columns = list(preview_frame.columns)
    if not columns:
        st.error("CSV не містить колонок.")
        st.stop()
except Exception as exc:
    st.error(f"Не вдалося прочитати CSV: {exc}")
    st.stop()

st.success(f"Файл готовий: {uploaded_file.name}")
with st.expander("Попередній перегляд даних", expanded=False):
    st.dataframe(preview_frame, use_container_width=True)

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
try:
    with st.spinner("Плануємо, генеруємо та перевіряємо ML-пайплайн…"):
        response = run_automl(
            filename=uploaded_file.name,
            content=uploaded_file.getvalue(),
            target_column=selected_target,
            user_prompt=user_prompt,
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

render_result(response.json())
