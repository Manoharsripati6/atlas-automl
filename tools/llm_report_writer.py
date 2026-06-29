import json

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from config import GROQ_API_KEY
from config import MODEL_NAME


llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0
)


def generate_report(
    insights,
    ml_results,
    viz_plan=None,
    encoded_data_context=None
):

    if viz_plan is None:
        viz_plan = []

    if encoded_data_context is None:
        encoded_data_context = {}

    prompt = ChatPromptTemplate.from_template(
        """
You are a Senior Data Scientist.

Generate a professional business-style data analysis report.

IMPORTANT:
- Use ONLY the information provided.
- Do NOT invent statistics.
- Do NOT invent columns.
- Do NOT invent findings.
- Column names are anonymized. Use the anonymized names exactly as provided.
- The encoded normalized data may reveal patterns, but it does not contain the original column names.
- Explain findings in business language.
- Write professionally.

Encoded + Normalized Dataset Context:
{encoded_data_context}

Dataset Insights:
{insights}

Machine Learning Results:
{ml_results}

Visualizations Generated:
{viz_plan}

Generate the report with the following sections:

# Executive Summary

Provide a concise summary of the dataset.

# Data Quality Assessment

Discuss:
- Missing values
- Duplicates
- Potential quality concerns

# Exploratory Data Analysis Findings

Discuss:
- Trends
- Relationships
- Correlations
- Outliers

# Visualization Summary

Explain why the generated visualizations are useful.

# Machine Learning Results

Explain:
- Detected task type
- Best model
- Model performance
- What the results suggest

# Recommendations

Provide practical recommendations based only on the findings.

# Conclusion

Provide a final summary.

Return plain text only.
"""
    )

    chain = prompt | llm

    result = chain.invoke(
        {
            "insights": json.dumps(
                insights,
                indent=2
            ),
            "ml_results": json.dumps(
                ml_results,
                indent=2
            ),
            "viz_plan": json.dumps(
                viz_plan,
                indent=2
            ),
            "encoded_data_context": json.dumps(
                encoded_data_context,
                indent=2
            )
        }
    )

    return result.content
