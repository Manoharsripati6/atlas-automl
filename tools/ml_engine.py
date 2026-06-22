import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    mean_absolute_error
)

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)


def infer_task_type(y):

    if y.dtype == "object":
        return "classification"

    if str(y.dtype) == "bool":
        return "classification"

    unique_count = y.nunique()
    unique_ratio = unique_count / len(y)

    if pd.api.types.is_integer_dtype(y):

        if unique_count <= 20:
            return "classification"

        if unique_ratio < 0.05:
            return "classification"

    if pd.api.types.is_float_dtype(y):
        return "regression"

    return "regression"


def train_models(df):

    target = df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target]

    task_type = infer_task_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    leaderboard = []

    # ==================================================
    # CLASSIFICATION
    # ==================================================

    if task_type == "classification":

        models = {
            "Logistic Regression":
                LogisticRegression(max_iter=3000),

            "Decision Tree":
                DecisionTreeClassifier(),

            "Random Forest":
                RandomForestClassifier(),

            "Gradient Boosting":
                GradientBoostingClassifier()
        }

        best_score = -1
        best_estimator = None
        best_model = None

        for model_name, model in models.items():

            try:

                model.fit(
                    X_train,
                    y_train
                )

                preds = model.predict(
                    X_test
                )

                accuracy = accuracy_score(
                    y_test,
                    preds
                )

                f1 = f1_score(
                    y_test,
                    preds,
                    average="weighted"
                )

                entry = {
                    "model": model_name,
                    "accuracy": round(
                        accuracy,
                        4
                    ),
                    "f1_score": round(
                        f1,
                        4
                    )
                }

                leaderboard.append(entry)

                if accuracy > best_score:

                    best_score = accuracy
                    best_estimator = model
                    best_model = entry

            except Exception as e:

                print(
                    f"{model_name} failed: {e}"
                )

        leaderboard = sorted(
            leaderboard,
            key=lambda x: x["accuracy"],
            reverse=True
        )

        return {
            "task_type": "classification",
            "target_column": target,
            "best_model": best_model,
            "top_models": leaderboard[:3],
            "trained_model": best_estimator,
            "X_test": X_test,
            "y_test": y_test
        }

    # ==================================================
    # REGRESSION
    # ==================================================

    models = {

        "Linear Regression":
            LinearRegression(),

        "Decision Tree":
            DecisionTreeRegressor(),

        "Random Forest":
            RandomForestRegressor(),

        "Gradient Boosting":
            GradientBoostingRegressor()
    }

    best_score = -999999
    best_estimator = None
    best_model = None

    for model_name, model in models.items():

        try:

            model.fit(
                X_train,
                y_train
            )

            preds = model.predict(
                X_test
            )

            r2 = r2_score(
                y_test,
                preds
            )

            mae = mean_absolute_error(
                y_test,
                preds
            )

            entry = {
                "model": model_name,
                "r2_score": round(
                    r2,
                    4
                ),
                "mae": round(
                    mae,
                    4
                )
            }

            leaderboard.append(entry)

            if r2 > best_score:

                best_score = r2
                best_estimator = model
                best_model = entry

        except Exception as e:

            print(
                f"{model_name} failed: {e}"
            )

    leaderboard = sorted(
        leaderboard,
        key=lambda x: x["r2_score"],
        reverse=True
    )

    return {
        "task_type": "regression",
        "target_column": target,
        "best_model": best_model,
        "top_models": leaderboard[:3],
        "trained_model": best_estimator,
        "X_test": X_test,
        "y_test": y_test
    }