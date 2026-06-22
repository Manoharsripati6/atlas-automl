def generate_insights(eda_report):

    insights = []

    for col, miss in eda_report["missing_values"].items():

        if miss > 0:

            insights.append(
                f"{col} contains {miss} missing values"
            )

    corrs = eda_report.get("correlations", {})

    seen = set()

    for c1 in corrs:

        for c2, corr in corrs[c1].items():

            if c1 == c2:
                continue

            pair = tuple(sorted([c1, c2]))

            if pair in seen:
                continue

            seen.add(pair)

            if abs(corr) > 0.7:

                insights.append(
                    f"{c1} and {c2} show strong correlation ({corr})"
                )

    return insights