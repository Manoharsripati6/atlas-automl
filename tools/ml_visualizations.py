import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix
)

os.makedirs("outputs/ml_plots", exist_ok=True)


def generate_ml_visualizations(
    ml_results,
    original_df,
    metadata
):

    outputs = []

    task = ml_results["task_type"]

    model = ml_results["trained_model"]

    X_test = ml_results["X_test"]

    y_test = ml_results["y_test"]

    predictions = model.predict(X_test)

    # ==========================================
    # CLASSIFICATION
    # ==========================================

    if task == "classification":

        cm = confusion_matrix(
            y_test,
            predictions
        )

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d"
        )

        plt.title(
            "Confusion Matrix"
        )

        path = (
            "outputs/ml_plots/"
            "confusion_matrix.png"
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        outputs.append(path)

    # ==========================================
    # REGRESSION
    # ==========================================

    else:

        # Actual vs Predicted

        plt.figure(figsize=(8, 6))

        plt.scatter(
            y_test,
            predictions
        )

        min_val = min(
            y_test.min(),
            predictions.min()
        )

        max_val = max(
            y_test.max(),
            predictions.max()
        )

        plt.plot(
            [min_val, max_val],
            [min_val, max_val]
        )

        plt.xlabel("Actual")

        plt.ylabel("Predicted")

        plt.title(
            "Actual vs Predicted"
        )

        path = (
            "outputs/ml_plots/"
            "actual_vs_predicted.png"
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        outputs.append(path)

        # Residual Plot

        residuals = (
            y_test -
            predictions
        )

        plt.figure(figsize=(8, 6))

        plt.scatter(
            predictions,
            residuals
        )

        plt.axhline(0)

        plt.xlabel(
            "Predicted"
        )

        plt.ylabel(
            "Residual"
        )

        plt.title(
            "Residual Plot"
        )

        path = (
            "outputs/ml_plots/"
            "residual_plot.png"
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        outputs.append(path)

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    if hasattr(
        model,
        "feature_importances_"
    ):

        importances = (
            model.feature_importances_
        )

        feature_names = []

        for col in X_test.columns:

            feature_names.append(
                metadata["column_map"].get(
                    col,
                    col
                )
            )

        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })

        fi = fi.sort_values(
            "importance",
            ascending=False
        ).head(15)

        plt.figure(figsize=(10, 6))

        sns.barplot(
            data=fi,
            x="importance",
            y="feature"
        )

        plt.title(
            "Feature Importance"
        )

        path = (
            "outputs/ml_plots/"
            "feature_importance.png"
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        outputs.append(path)

    return outputs