import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs/plots", exist_ok=True)


def generate_visualizations(df, plan):

    outputs = []

    for viz in plan:

        try:
            chart_type = viz.get("chart_type")
            features = viz.get("features", [])

            # ---------------- HEATMAP ----------------
            if chart_type == "heatmap":

                num = df.select_dtypes(include=np.number)

                if len(num.columns) > 1:

                    plt.figure(figsize=(10, 8))
                    sns.heatmap(num.corr(), annot=True)

                    path = "outputs/plots/heatmap.png"
                    plt.savefig(path)
                    plt.close()

                    outputs.append(path)

            # ---------------- HISTOGRAM ----------------
            elif chart_type == "histogram":

                col = features[0]

                plt.figure()
                sns.histplot(df[col])

                path = f"outputs/plots/{col}_hist.png"
                plt.savefig(path)
                plt.close()

                outputs.append(path)

            # ---------------- BOXPLOT ----------------
            elif chart_type == "boxplot":

                col = features[0]

                plt.figure()
                sns.boxplot(y=df[col])

                path = f"outputs/plots/{col}_box.png"
                plt.savefig(path)
                plt.close()

                outputs.append(path)

            # ---------------- COUNTPLOT ----------------
            elif chart_type == "countplot":

                col = features[0]

                plt.figure(figsize=(10, 6))
                sns.countplot(x=df[col])

                plt.xticks(rotation=45)

                path = f"outputs/plots/{col}_count.png"
                plt.savefig(path, bbox_inches="tight")
                plt.close()

                outputs.append(path)

            # ---------------- SCATTER ----------------
            elif chart_type == "scatter":

                x = features[0]
                y = features[1]

                plt.figure()
                sns.scatterplot(data=df, x=x, y=y)

                path = f"outputs/plots/{x}_{y}.png"
                plt.savefig(path)
                plt.close()

                outputs.append(path)

            # ---------------- BAR (optional extension) ----------------
            elif chart_type == "bar":

                col = features[0]

                plt.figure(figsize=(10, 6))
                df[col].value_counts().plot(kind="bar")

                path = f"outputs/plots/{col}_bar.png"
                plt.savefig(path, bbox_inches="tight")
                plt.close()

                outputs.append(path)

            # ---------------- VIOLIN (optional extension) ----------------
            elif chart_type == "violin":

                col = features[0]

                plt.figure()
                sns.violinplot(y=df[col])

                path = f"outputs/plots/{col}_violin.png"
                plt.savefig(path)
                plt.close()

                outputs.append(path)

        except Exception:
            continue

    return outputs