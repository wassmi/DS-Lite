import json
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Dict, Optional, Union
from openai import OpenAI
from rich.console import Console
from rich.table import Table
import os
import sqlite3

# Initialize rich console
console = Console()

# Pydantic Models
class DataOverview(BaseModel):
    file_name: str
    num_rows: int
    num_columns: int
    column_types: Dict[str, str]

class AnalysisSuggestion(BaseModel):
    title: str
    description: str
    type: str
    status: str = "pending"
    parameters: Optional[Dict[str, Union[str, List[str]]]] = None

class AnalysisPlanStep(BaseModel):
    step: int
    action: str

class AnalysisPlan(BaseModel):
    steps: List[AnalysisPlanStep]

class ExecutionResults(BaseModel):
    outputs: List[str]
    status: str

class PythonCode(BaseModel):
    code: str

    @field_validator("code")
    def validate_code(cls, value):
        """Ensure the code is a valid Python code string without markdown formatting."""
        if value.strip().startswith("```python"):
            # Strip the triple backticks and markdown formatting
            value = value[9:-3].strip()
        return value

# Data Loader Factory
def load_data(file_path: str):
    """
    Load data from a file or database based on the file extension or input type.
    Supports CSV, JSON, and SQLite.
    """
    if file_path.endswith(".csv"):
        # Load CSV without modifying column names
        data = pd.read_csv(file_path)
        return data
    elif file_path.endswith(".json"):
        # Load JSON without modifying column names
        data = pd.read_json(file_path)
        return data
    elif file_path.endswith(".db") or file_path.endswith(".sqlite"):
        # Load data from SQLite database
        connection = sqlite3.connect(file_path)
        query = "SELECT * FROM data"  # Replace 'data' with your table name
        data = pd.read_sql_query(query, connection)
        return data
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

# Generate data overview
def generate_data_overview(data: pd.DataFrame, file_name: str):
    overview = DataOverview(
        file_name=file_name,
        num_rows=data.shape[0],
        num_columns=data.shape[1],
        column_types={col: str(data[col].dtype) for col in data.columns}
    )
    return overview

# Data Understanding Agent
def data_understanding_agent(overview: DataOverview):
    prompt = """
    Generate exactly 3 analysis suggestions that are highly relevant to the dataset overview. 
    The suggestions should cover a variety of data science tasks (e.g., regression, classification, clustering, time series, distribution analysis, feature engineering, etc.) and be tailored to the dataset's structure and column types.

    **Rules for Suggestions:**
    1. **Relevance**: Each suggestion must be directly related to the dataset's columns and their data types (e.g., numerical, categorical, time-based).
    2. **Diversity**: Ensure the suggestions cover different types of analysis (e.g., regression, clustering, visualization).
    3. **Actionable**: Each suggestion should include clear parameters (e.g., target column, feature columns) that can be used in the analysis.
    4. **Insightful**: The description should explain what insights the analysis will provide and why it is useful.

    Return ONLY a JSON array with objects containing these exact fields:
    - title: string (short title of the analysis)
    - description: string (2-3 sentences explaining what insights this analysis will provide)
    - type: string (one of: regression, classification, clustering, time_series, distribution, correlation, feature_engineering)
    - status: string (always set to "pending")
    - parameters: object (optional parameters specific to this analysis, e.g., target_column, feature_columns)

    The response must be a valid JSON array. It should never begin with ' ```json' or '```'. I will show you 
    an example format, use it only for the structure, but be creative as a data scientist to come up with suggestions of data science analyses. Example format:
    [
        {
            "title": "Price Prediction Model",
            "description": "Develop a regression model to predict home selling prices based on features like living area, number of rooms, and taxes.",
            "type": "regression",
            "status": "pending",
            "parameters": {
                "target_column": "Sell",
                "feature_columns": ["Living", "Rooms", "Taxes"]
            }
        }
    ]

    Here is the dataset overview: 
    """ + overview.model_dump_json()

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    suggestions = json.loads(response.choices[0].message.content)
    return [AnalysisSuggestion(**suggestion) for suggestion in suggestions]

# Planning Agent
def planning_agent(suggestion: AnalysisSuggestion):
    prompt = """
    Generate a detailed step-by-step coding plan for the following analysis suggestion: 
    """ + suggestion.model_dump_json() + """. 
    The plan should include all data science steps starting from data cleaning to results visualization. 
    Each step should be clear, actionable, and specific to the analysis type.

    **Rules for the Plan:**
    1. **Comprehensive**: Cover all necessary steps, including data loading, cleaning, exploration, modeling, evaluation, and visualization.
    2. **Actionable**: Each step should describe a specific action (e.g., "Handle missing values by imputing the mean for numerical columns").
    3. **Visual-Focused**: Ensure the final steps include saving visual outputs (e.g., plots, charts, tables) and avoid text-based summaries.
    4. **No Model Saving**: Do not include steps for saving trained models to files.
    5. **Dynamic**: The plan should not hardcode column names or assumptions about the dataset structure.

    Return ONLY a JSON array with objects containing these exact fields:
    - step: integer (step number, starting from 1)
    - action: string (detailed description of the action to be performed)

    The response must be a valid JSON array. It should never begin with ' ```json' or '```'. Example format:
    [
        {
            "step": 1,
            "action": "Load the dataset and inspect for missing values."
        },
        {
            "step": 2,
            "action": "Perform data cleaning by imputing missing values using the mean for numerical columns."
        }
    ]
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    plan_steps = json.loads(response.choices[0].message.content)
    return AnalysisPlan(steps=[AnalysisPlanStep(**step) for step in plan_steps])

# Coding Agent
def coding_agent(plan: AnalysisPlan, file_path: str):
    prompt = """
    Generate executable Python code to implement the following analysis plan: 
    """ + plan.model_dump_json() + """. 
    The code should be complete, ready to run, and include all necessary steps from data loading to results visualization. 
    Ensure the code saves any visualizations or results to files in the current directory. 
    All results must be visual (e.g., plots, graphs, tables) and never saved as .txt or .json files.
    The code should dynamically adapt to the dataset's structure and not hardcode any column names. Instead, use the first column as the target variable and the remaining columns as features.

    **Allowed Modules:**
    You are allowed to use the following Python modules for data science tasks:
    - Data Manipulation: `pandas`, `numpy`, `scipy`
    - Data Visualization: `matplotlib`, `seaborn`, `plotly`
    - Machine Learning: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
    - Model Evaluation: `sklearn.metrics`, `yellowbrick`
    - Feature Engineering: `featuretools`, `category_encoders`
    - Time Series Analysis: `statsmodels`, `prophet`
    - Deep Learning: `tensorflow`, `keras` (optional, only if necessary)
    - Utilities: `os`, `joblib`, `warnings`

    **Important Rules:**
    1. **No Text-Based Outputs**: Do not generate any text-based outputs, summaries, or reports. All results must be visual (plots, charts, tables).
    2. **No Model Saving**: Do not save trained models to files. The focus is on generating visual insights, not model deployment.
    3. **Dynamic Column Handling**: Do not hardcode column names. Use the first column as the target variable and the remaining columns as features.
    4. **Visualize Evaluation Metrics**: Metrics like ROC AUC, MAE, MSE, R-squared, etc., must be visualized (e.g., plots, bar charts) and not printed as text.
    5. **Save Visual Outputs**: Save all visual outputs (plots, charts, tables) to the current directory with appropriate file names (e.g., 'feature_distributions.png', 'roc_curve.png').
    6. ** Not every thing needs to be visualized. For example, missing values don't need to be visualized. 
    7. **Error Handling**: Include basic error handling to ensure the code runs smoothly even if the dataset structure is unexpected.
    8. **Don't change the input data file names. Use the file name provided in the file_path variable.

    **Visualization Quality Guidelines:**
    - **Titles**: Every plot must have a clear, descriptive title that explains what the visualization represents (e.g., "Distribution of House Prices").
    - **Axis Labels**: All axes must be labeled with clear, descriptive names (e.g., "Living Area (sq ft)" for the x-axis, "Price ($)" for the y-axis).
    - **Legends**: If multiple categories or groups are shown, include a legend with clear labels.
    - **Color Schemes**: Use color schemes that are easy to interpret (e.g., avoid overly bright or clashing colors). Use colorblind-friendly palettes if possible.
    - **Font Sizes**: Ensure all text (titles, labels, legends) is large enough to be easily readable.
    - **Gridlines**: Add gridlines to plots where appropriate to make it easier to interpret values.
    - **Annotations**: If necessary, add annotations (e.g., highlighting specific data points or trends) to make the visualization more informative.
    - **Consistency**: Use consistent styling (e.g., font sizes, colors) across all visualizations.

    Return ONLY the Python code as a string. Do not include any additional text, explanations, or markdown formatting like '```python'.
    The dataset is located at: """ + file_path

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    # Validate the generated code using Pydantic
    try:
        python_code = PythonCode(code=response.choices[0].message.content)
        return python_code.code
    except ValidationError as e:
        console.print(f"[bold red]Validation Error in Generated Code:[/bold red] {e}")
        raise
# Execution Engine
def execute_code(code: str):
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Replace all occurrences of plt.savefig('filename') with plt.savefig('output/filename')
    code = code.replace("plt.savefig('", f"plt.savefig('{output_dir}/")

    try:
        exec(code)
    except Exception as e:
        console.print(f"[bold red]Error during code execution:[/bold red] {e}")
        return ExecutionResults(
            outputs=[],
            status="failed"
        )

    return ExecutionResults(
        outputs=[f"{output_dir}/{f}" for f in os.listdir(output_dir)],
        status="completed"
    )

# Main workflow
def main():
    file_path = "path/to/your/file"  # Replace with your actual file path (CSV, JSON, or SQLite)
    data = load_data(file_path)
    overview = generate_data_overview(data, file_path)
    console.print("[bold green]Data Overview:[/bold green]", overview.model_dump_json())

    suggestions = data_understanding_agent(overview)
    console.print("[bold green]Analysis Suggestions:[/bold green]", [s.model_dump_json() for s in suggestions])

    plan = planning_agent(suggestions[0])
    console.print("[bold green]Analysis Plan:[/bold green]", plan.model_dump_json())

    code = coding_agent(plan, file_path)
    console.print("[bold green]Generated Code:[/bold green]", code)

    results = execute_code(code)
    console.print("[bold green]Execution Results:[/bold green]", results.model_dump_json())

if __name__ == "__main__":
    client = OpenAI(api_key="your_deepseek_api_key", base_url="https://api.deepseek.com")  # Replace with your actual API key
    main()
