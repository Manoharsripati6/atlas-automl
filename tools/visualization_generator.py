import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs/plots", exist_ok=True)

sns.set_theme(style="whitegrid")


def generate_visualizations(df, plan, metadata):

    outputs = []

    column_map = metadata["column_map"]

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    for viz in plan:

        try:

            chart_type = viz.get("chart_type")
            anon_features = viz.get("features", [])

            features = [
                column_map.get(col, col)
                for col in anon_features
            ]

            # ----------------------------------------
            # HEATMAP
            # ----------------------------------------

            if chart_type == "heatmap":

                heatmap_cols = [
                    column_map.get(col, col)
                    for col in anon_features
                    if column_map.get(col, col) in df.columns
                ]

                if len(heatmap_cols) < 2:
                    continue

                plt.figure(figsize=(10, 8))

                corr = df[heatmap_cols].corr()

                sns.heatmap(
                    corr,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm"
                )

                plt.title("Correlation Heatmap")

                plt.tight_layout()

                path = "outputs/plots/heatmap.png"

                plt.savefig(
                    path,
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close()

                outputs.append({
                    "type": "heatmap",
                    "features": heatmap_cols,
                    "path": path
                })

            # ----------------------------------------
            # HISTOGRAM
            # ----------------------------------------

            elif chart_type == "histogram":

                if len(features) < 1:
                    continue

                col = features[0]

                if col not in df.columns:
                    continue

                plt.figure(figsize=(8, 5))

                sns.histplot(
                    df[col],
                    kde=True
                )

                plt.title(f"Distribution of {col}")
                plt.xlabel(col)

                plt.tight_layout()

                path = (
                    f"outputs/plots/"
                    f"{col}_hist.png"
                )

                plt.savefig(
                    path,
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close()

                outputs.append({
                    "type": "histogram",
                    "features": [col],
                    "path": path
                })

            # ----------------------------------------
            # BOXPLOT
            # ----------------------------------------

            elif chart_type == "boxplot":

                if len(features) < 1:
                    continue

                col = features[0]

                if col not in df.columns:
                    continue

                plt.figure(figsize=(6, 5))

                sns.boxplot(
                    y=df[col]
                )

                plt.title(f"Boxplot of {col}")
                plt.ylabel(col)

                plt.tight_layout()

                path = (
                    f"outputs/plots/"
                    f"{col}_box.png"
                )

                plt.savefig(
                    path,
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close()

                outputs.append({
                    "type": "boxplot",
                    "features": [col],
                    "path": path
                })

            # ----------------------------------------
            # COUNTPLOT
            # ----------------------------------------

            elif chart_type == "countplot":

                if len(features) < 1:
                    continue

                col = features[0]

                if col not in df.columns:
                    continue

                if df[col].nunique() > 50:
                    continue

                plt.figure(figsize=(10, 6))

                sns.countplot(
                    data=df,
                    x=col
                )

                plt.xticks(rotation=45)

                plt.title(f"Countplot of {col}")
                plt.xlabel(col)

                plt.tight_layout()

                path = (
                    f"outputs/plots/"
                    f"{col}_count.png"
                )

                plt.savefig(
                    path,
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close()

                outputs.append({
                    "type": "countplot",
                    "features": [col],
                    "path": path
                })

            # ----------------------------------------
            # SCATTER
            # ----------------------------------------

            elif chart_type == "scatter":

                if len(features) < 2:
                    continue

                x = features[0]
                y = features[1]

                if x not in df.columns:
                    continue

                if y not in df.columns:
                    continue

                plt.figure(figsize=(8, 6))

                sns.scatterplot(
                    data=df,
                    x=x,
                    y=y
                )

                plt.title(f"{x} vs {y}")
                plt.xlabel(x)
                plt.ylabel(y)

                plt.tight_layout()

                path = (
                    f"outputs/plots/"
                    f"{x}_{y}_scatter.png"
                )

                plt.savefig(
                    path,
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close()

                outputs.append({
                    "type": "scatter",
                    "features": [x, y],
                    "path": path
                })

        except Exception as e:

            print(
                f"Visualization generation failed: {e}"
            )

    return outputs