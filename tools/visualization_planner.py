def create_visualization_plan(eda_report):

    plan = []

    nums = eda_report["numeric_columns"]

    cats = eda_report["categorical_columns"]

    if len(nums) > 1:

        plan.append({
            "chart_type": "heatmap",
            "features": nums
        })

    for col in nums[:3]:

        plan.append({
            "chart_type": "histogram",
            "features": [col]
        })

        plan.append({
            "chart_type": "boxplot",
            "features": [col]
        })

    for col in cats[:3]:

        plan.append({
            "chart_type": "countplot",
            "features": [col]
        })

    return plan[:8]