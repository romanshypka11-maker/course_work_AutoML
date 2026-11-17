from pathlib import Path

import pandas as pd


def get_dataset_profile(file_path: str, target_column: str) -> str:
    """Read a CSV and return a compact, useful profile for the planning agent."""
    path = Path(file_path)
    if not path.is_file():
        raise ValueError("Uploaded dataset was not saved correctly.")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read the uploaded CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    if not target_column or target_column not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            f"Target column '{target_column}' is absent from the CSV. Available columns: {available}"
        )

    buffer = [
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Target column: {target_column}",
        "Column summary:",
    ]
    for col in df.columns:
        series = df[col]
        buffer.append(
            f"- {col}: dtype={series.dtype}, missing={int(series.isna().sum())}, "
            f"unique={int(series.nunique(dropna=True))}"
        )

    # Examples help the LLM recognise categorical columns, but keep the prompt bounded.
    buffer.append("Sample rows (first 3):")
    buffer.append(str(df.head(3).to_dict(orient="records")))
    return "\n".join(buffer)
