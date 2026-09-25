import httpx
import pandas as pd
import streamlit as st

st.title("AutoML Course")
st.write(
    "Завантаж свої табличні дані (CSV), щоб запустити інтерактивний пайплайн."
)

uploaded_file = st.file_uploader("Оберіть CSV файл", type=["csv"])

if uploaded_file is not None:
    st.success(f"Файл успішно завантажено: {uploaded_file.name}")

    # 1. Зчитування колонок для UI
    try:
        uploaded_file.seek(0)
        df_preview = pd.read_csv(uploaded_file, nrows=5)
        columns = list(df_preview.columns)
    except Exception as e:
        st.error(f"Не вдалося прочитати CSV файл: {e}")
        st.stop()

    # 2. Віджети UI
    selected_target = st.selectbox("Оберіть цільову колонку (Target):", columns)
    user_prompt = st.text_area(
        "Опишіть вашу бізнес-ціль та побажання до моделі:",
        placeholder=(
            "Наприклад: Хочу передбачити відтік клієнтів. Важливо мінімізувати"
            " хибно-негативні помилки."
        ),
    )

    # 3. Відправка на бекенд при натисканні кнопки
    if st.button("Запустити AutoML"):
        with st.spinner("Передаємо файл та параметри на бекенд..."):
            uploaded_file.seek(0)

            data_payload = {
                "target_column": selected_target,
                "user_prompt": user_prompt,
            }
            files_payload = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv",
                )
            }

            try:
                response = httpx.post(
                    "http://backend:8000/run_automl",
                    data=data_payload,
                    files=files_payload,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    st.json(response.json())
                else:
                    st.error(
                        f"Помилка сервера: {response.status_code} -"
                        f" {response.text}"
                    )
            except httpx.ConnectError:
                st.error(
                    "Не вдалося підключитися до FastAPI бекенду. Перевір, чи"
                    " запущений контейнер backend на порту 8000."
                )
            except Exception as e:
                st.error(f"Вини кла помилка під час відправки запиту: {e}")