import os
import json
import pandas as pd

from tools.privacy_preprocessor import preprocess_data
from tools.eda_engine import run_eda
from tools.insight_engine import generate_insights
from tools.visualization_planner import create_visualization_plan
from tools.visualization_generator import generate_visualizations
from tools.ml_engine import train_models
from tools.ml_visualizations import generate_ml_visualizations
from tools.llm_report_writer import generate_report
from tools.de_anonymizer import deanonymize_text


def _safe_llm_data_context(anonymized_df):

    sample = (
        anonymized_df
        .head(25)
        .round(6)
        .to_dict(orient="records")
    )

    return {
        "shape": {
            "rows": int(anonymized_df.shape[0]),
            "columns": int(anonymized_df.shape[1])
        },
        "columns": anonymized_df.columns.tolist(),
        "dtypes": {
            col: str(dtype)
            for col, dtype in anonymized_df.dtypes.items()
        },
        "sample_records": sample
    }


def _deanonymize_structure(value, metadata):

    column_map = metadata["column_map"]

    if isinstance(value, dict):

        return {
            column_map.get(key, key): _deanonymize_structure(val, metadata)
            for key, val in value.items()
        }

    if isinstance(value, list):

        return [
            _deanonymize_structure(item, metadata)
            for item in value
        ]

    if isinstance(value, str):

        return deanonymize_text(
            value,
            metadata
        )

    return value


def run_pipeline(file_path):

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    df = pd.read_csv(file_path)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/ml_plots", exist_ok=True)

    # ==================================================
    # PRIVACY PREPROCESSING
    # ==================================================

    print("\n" + "=" * 60)
    print("PRIVACY PREPROCESSING")
    print("=" * 60)

    anonymized_df, metadata, actions = preprocess_data(df)

    print("Completed")

    if actions:

        print("\nActions Performed:")

        for action in actions:
            print(f"• {action}")

    # ==================================================
    # EDA
    # ==================================================

    print("\n" + "=" * 60)
    print("RUNNING EDA")
    print("=" * 60)

    eda_report = run_eda(anonymized_df)

    print("Completed")

    # ==================================================
    # INSIGHT EXTRACTION
    # ==================================================

    print("\n" + "=" * 60)
    print("GENERATING INSIGHTS")
    print("=" * 60)

    insights = generate_insights(
        eda_report
    )

    print(
        f"Generated {len(insights)} insights"
    )

    # ==================================================
    # ML TRAINING
    # ==================================================

    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    ml_results = train_models(
        anonymized_df
    )

    # SAFE VERSION FOR LLM + JSON
    ml_summary = {
        "task_type":
            ml_results["task_type"],

        "target_column":
            ml_results["target_column"],

        "best_model":
            ml_results["best_model"],

        "top_models":
            ml_results["top_models"]
    }

    print(
        f"Detected Task Type: "
        f"{ml_summary['task_type']}"
    )

    print(
        f"Best Model: "
        f"{ml_summary['best_model']['model']}"
    )

    # ==================================================
    # ML VISUALIZATIONS
    # ==================================================

    print("\n" + "=" * 60)
    print("GENERATING ML VISUALIZATIONS")
    print("=" * 60)

    ml_plots = generate_ml_visualizations(
        ml_results,
        df,
        metadata
    )

    print(
        f"Generated {len(ml_plots)} ML plots"
    )

    # ==================================================
    # VISUALIZATION PLANNING
    # ==================================================

    print("\n" + "=" * 60)
    print("VISUALIZATION PLANNING")
    print("=" * 60)

    viz_plan = create_visualization_plan(
        eda_report
    )

    print(
        f"Planned {len(viz_plan)} visualizations"
    )

    # ==================================================
    # VISUALIZATION GENERATION
    # ==================================================

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    plots = generate_visualizations(
        df,          # original dataframe
        viz_plan,    # anonymized plan
        metadata
    )

    print(
        f"Generated {len(plots)} plots"
    )

    # ==================================================
    # REPORT GENERATION
    # ==================================================

    print("\n" + "=" * 60)
    print("GENERATING REPORT")
    print("=" * 60)

    encoded_data_context = _safe_llm_data_context(
        anonymized_df
    )

    report = generate_report(
        insights=insights,
        ml_results=ml_summary,
        viz_plan=viz_plan,
        encoded_data_context=encoded_data_context
    )

    print("Completed")

    # ==================================================
    # DE-ANONYMIZATION
    # ==================================================

    print("\n" + "=" * 60)
    print("RESTORING COLUMN NAMES")
    print("=" * 60)

    final_report = deanonymize_text(
        report,
        metadata
    )

    display_eda_report = _deanonymize_structure(
        eda_report,
        metadata
    )

    display_insights = _deanonymize_structure(
        insights,
        metadata
    )

    display_ml_summary = _deanonymize_structure(
        ml_summary,
        metadata
    )

    display_visualization_plan = _deanonymize_structure(
        viz_plan,
        metadata
    )

    print("Completed")

    # ==================================================
    # SAVE REPORT
    # ==================================================

    report_path = (
        "outputs/final_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            final_report
        )

    print(
        f"\nReport Saved: {report_path}"
    )

    # ==================================================
    # SAVE EDA REPORT
    # ==================================================

    with open(
        "outputs/eda_report.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            eda_report,
            f,
            indent=4
        )

    # ==================================================
    # SAVE ML SUMMARY
    # ==================================================

    with open(
        "outputs/ml_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            ml_summary,
            f,
            indent=4
        )

    # ==================================================
    # SAVE INSIGHTS
    # ==================================================

    with open(
        "outputs/insights.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            insights,
            f,
            indent=4
        )

    # ==================================================
    # RETURN EVERYTHING
    # ==================================================

    return {
        "actions": actions,
        "eda_report": eda_report,
        "insights": insights,

        # internal use
        "ml_results": ml_results,

        # safe for API/reporting
        "ml_summary": ml_summary,

        "visualization_plan": viz_plan,
        "display_eda_report": display_eda_report,
        "display_insights": display_insights,
        "display_ml_summary": display_ml_summary,
        "display_visualization_plan": display_visualization_plan,
        "encoded_data_context": encoded_data_context,

        "plots": plots,
        "ml_plots": ml_plots,

        "report": final_report
    }


if __name__ == "__main__":

    DATASET_PATH = "datasets/sample.csv"

    result = run_pipeline(
        DATASET_PATH
    )

    print("\n")
    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    print(
        result["report"]
    )

    print("\n")
    print("=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(
        result["ml_summary"]["best_model"]
    )
