import numpy as np


def run_eda(df):

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols
    ]

    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum())
    }

    report["numeric_summary"] = {}

    for col in numeric_cols:

        report["numeric_summary"][col] = {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4)
        }

    if len(numeric_cols) > 1:

        report["correlations"] = (
            df[numeric_cols]
            .corr()
            .round(3)
            .to_dict()
        )

    else:

        report["correlations"] = {}

    return report