import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


def preprocess_data(df):

    df = df.copy()

    metadata = {
        "column_map": {},
        "reverse_column_map": {},
        "encoders": {},
        "scalers": {}
    }

    actions = []

    # remove duplicates

    dup = df.duplicated().sum()

    if dup > 0:
        df = df.drop_duplicates()
        actions.append(f"Removed {dup} duplicate rows")

    # anonymize columns

    rename_map = {}

    num_idx = 1
    cat_idx = 1

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            new_name = f"NUM_{num_idx:03d}"
            num_idx += 1

        else:

            new_name = f"CAT_{cat_idx:03d}"
            cat_idx += 1

        rename_map[col] = new_name

        metadata["column_map"][new_name] = col
        metadata["reverse_column_map"][col] = new_name

    df = df.rename(columns=rename_map)

    # missing values

    for col in df.columns:

        if df[col].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):

            imp = SimpleImputer(strategy="median")
            df[col] = imp.fit_transform(df[[col]]).ravel()

        else:

            imp = SimpleImputer(strategy="most_frequent")
            df[col] = imp.fit_transform(df[[col]]).ravel()

    # encode categoricals

    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in cat_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col].astype(str))

        metadata["encoders"][col] = {
            int(i): str(v)
            for i, v in enumerate(le.classes_)
        }

    # normalize numerics

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:

        scaler = MinMaxScaler()

        df[col] = scaler.fit_transform(df[[col]])

        metadata["scalers"][col] = {
            "min": float(scaler.data_min_[0]),
            "max": float(scaler.data_max_[0])
        }

    return df, metadata, actions