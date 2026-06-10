from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):

    actions = []

    dup = df.duplicated().sum()

    if dup > 0:
        df = df.drop_duplicates()
        actions.append(f"Removed {dup} duplicate rows")

    for col in df.columns:

        missing = df[col].isnull().sum()

        if missing == 0:
            continue

        if df[col].dtype == "object":

            imputer = SimpleImputer(strategy="most_frequent")
            df[col] = imputer.fit_transform(df[[col]]).ravel()

            actions.append(f"Filled {col} using mode")

        else:

            imputer = SimpleImputer(strategy="median")
            df[col] = imputer.fit_transform(df[[col]]).ravel()

            actions.append(f"Filled {col} using median")

    for col in df.select_dtypes(include="object").columns:

        if df[col].nunique() <= 50:

            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

            actions.append(f"Encoded {col}")

    return df, actions