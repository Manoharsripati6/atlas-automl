import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from config import GROQ_API_KEY, MODEL_NAME

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0
)

def create_visualization_plan(eda_report):

    prompt = ChatPromptTemplate.from_template(
        """
        You are a senior data scientist and expert in exploratory data analysis (EDA).

        You will be given a dataset summary (EDA report).

        Your task is to select ONLY the most useful and meaningful visualizations that help understand the dataset deeply.

        Focus on:
        - Data distributions
        - Feature relationships
        - Correlations
        - Outliers
        - Categorical patterns

        Rules:
        - Do NOT generate unnecessary or redundant plots
        - Maximum 8 visualizations only
        - Prefer insights over quantity
        - Each visualization must answer a question about the data
        - Use both single-feature and multi-feature analysis when useful

        Return ONLY valid JSON (no explanation, no markdown).

        Use this format:

        [
        {{
            "analysis_type": "distribution | relationship | categorical | correlation | outlier",
            "chart_type": "histogram | boxplot | countplot | scatter | heatmap | bar | violin",
            "features": ["feature1"] ,
            "reason": "short explanation of why this visualization is useful",
            "priority": "high | medium | low"
        }}
        ]

        Dataset Summary:
        {eda_report}
        """
    )

    chain = prompt | llm

    response = chain.invoke({
        "eda_report": json.dumps(eda_report, indent=2)
    })

    content = response.content.replace("```json", "").replace("```", "").strip()

    return json.loads(content)