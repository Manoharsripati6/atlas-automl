import pandas as pd

from tools.eda_tool import run_eda
from tools.preprocessing_tool import preprocess_data
from tools.visualization_planner import create_visualization_plan
from tools.visualization_generator import generate_visualizations


def format_report(eda, preprocessing, viz_plan, plots):

    print("\n==============================")
    print("📊 DATASET ANALYSIS REPORT")
    print("==============================\n")

    # ---------------- EDA ----------------
    print("🔹 Dataset Overview")
    print(f"Rows: {eda['rows']}")
    print(f"Columns: {eda['columns']}")
    print(f"Duplicate rows: {eda['duplicate_rows']}\n")

    print("🔹 Features")
    print("Numeric:", eda["numeric_columns"])
    print("Categorical:", eda["categorical_columns"], "\n")

    print("🔹 Missing Values")
    for k, v in eda["missing_values"].items():
        print(f"{k} → {v}")

    print("\n🔹 Correlations")
    for k, v in eda.get("correlations", {}).items():
        print(f"{k}: {v}")

    # ---------------- PREPROCESSING ----------------
    print("\n==============================")
    print("🧹 PREPROCESSING SUMMARY")
    print("==============================\n")

    if preprocessing is None or len(preprocessing) == 0:
        print("No preprocessing required (clean dataset)")
    else:
        for p in preprocessing:
            print("-", p)

    # ---------------- VISUALIZATION ----------------
    print("\n==============================")
    print("📈 VISUALIZATION PLAN")
    print("==============================\n")

    for i, v in enumerate(viz_plan, 1):
        print(f"{i}. {v['chart_type'].upper()}")
        print(f"   Analysis Type: {v['analysis_type']}")
        print(f"   Features: {v['features']}")
        print(f"   Reason: {v['reason']}")
        print(f"   Priority: {v['priority']}\n")

    # ---------------- PLOTS ----------------
    print("\n==============================")
    print("📊 GENERATED PLOTS")
    print("==============================\n")

    for p in plots:
        print("✔", p)


if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("datasets/sample.csv")

    # 1. EDA
    eda_report = run_eda(df)

    # 2. Preprocessing
    processed_df, actions = preprocess_data(df)

    # 3. Visualization planning (LLM)
    visualization_plan = create_visualization_plan(eda_report)

    # 4. Generate plots
    plots = generate_visualizations(processed_df, visualization_plan)

    # 5. Final report
    format_report(
        eda_report,
        actions,
        visualization_plan,
        plots
    )