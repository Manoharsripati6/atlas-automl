import pandas as pd
import numpy as np

def run_eda(df):

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_values": {
            col: int(df[col].nunique())
            for col in df.columns
        }
    }

    if len(numeric_cols) > 1:
        report["correlations"] = (
            df[numeric_cols].corr().round(3).to_dict()
        )
    else:
        report["correlations"] = {}

    return report