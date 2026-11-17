import httpx
import streamlit as st

from services.backend_client import fetch_artifact


def render_result(result: dict) -> None:
    """Render one completed backend response in the Streamlit UI."""
    if result.get("success"):
        st.success(
            f"Пайплайн успішно завершено за {result.get('iteration_count', 0)} ітерац(ій)."
        )
    elif result.get("status") == "limit_reached":
        st.warning(
            "Досягнуто ліміт спроб. Перегляньте помилку та останній згенерований код нижче."
        )
    else:
        st.error(
            f"Пайплайн не завершився успішно. Статус: {result.get('status', 'unknown')}"
        )

    metrics = result.get("metrics") or {}
    if metrics:
        st.subheader("Метрики")
        metric_columns = st.columns(min(4, len(metrics)))
        for index, (name, value) in enumerate(metrics.items()):
            metric_columns[index % len(metric_columns)].metric(name, f"{value:.4f}")

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
                artifact_response = fetch_artifact(relative_url)
                st.download_button(
                    label=f"Завантажити {name}",
                    data=artifact_response.content,
                    file_name=name,
                    key=f"download-{result.get('job_id')}-{name}",
                )
            except httpx.HTTPError as exc:
                st.warning(
                    f"Артефакт {name} створено, але не вдалося завантажити: {exc}"
                )
