import pandas as pd

def get_dataset_profile(file_path: str, target_column: str) -> str:
    """
    Зчитує CSV-файл та формує текстовий паспорт датасету 
    для подальшої передачі Агенту-Стратегу.
    """
    df = pd.read_csv(file_path)
    
    buffer = []
    buffer.append(f"Кількість рядків: {df.shape[0]}, Кількість колонок: {df.shape[1]}")
    buffer.append(f"Цільова змінна (Target): {target_column}")
    buffer.append("\nТипи даних та пропуски:")
    
    for col in df.columns:
        dtype = df[col].dtype
        n_missing = df[col].isnull().sum()
        n_unique = df[col].nunique()
        buffer.append(f"- {col}: тип={dtype}, пропусків={n_missing}, унікальних={n_unique}")
        
    buffer.append("\nПерші 3 рядки даних:")
    buffer.append(str(df.head(3).to_dict(orient="records")))
    
    return "\n".join(buffer)