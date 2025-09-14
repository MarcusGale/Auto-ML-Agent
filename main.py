import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv
import io
import os
from openai import OpenAI
from daytona import Daytona, DaytonaConfig
from io import StringIO

# Load environment variables
load_dotenv()

st.title("Autonomous ML Agent 🚀")
st.markdown("Upload a CSV to automatically clean, train models, and optimize.")


# ---- File Upload ----
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])


def summarize_dataset(dataframe: pd.DataFrame) -> str:
    """
    Generate a comprehensive summary of the dataset for LLM context.
    """
    try:
        # Create an in-memory buffer to hold a CSV-formatted sample of the dataframe
        buffer = StringIO()

        # Limit the sample size to the first 30 rows
        sample_rows = min(30, len(dataframe))

        # Convert sample rows to CSV and write into the buffer
        dataframe.head(sample_rows).to_csv(buffer, index=False)
        sample_csv = buffer.getvalue()
        # Get column data types as a dictionary: {column_name: dtype}
        dtypes = dataframe.dtypes.astype(str).to_dict()
        # Count non-null values per column
        non_null_counts = dataframe.notnull().sum().astype(int).to_dict()
        # Count null values per column
        null_counts = dataframe.isnull().sum().astype(int).to_dict()
        # Count number of unique values per column (cardinality)
        cardinality = dataframe.nunique(dropna=True).to_dict()

        # Identify numeric columns (useful for statistical summaries)
        numeric_cols = [
            c for c in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[c])
        ]
        # Generate descriptive statistics for numeric columns (if any exist)
        desc = dataframe[numeric_cols].describe().to_dict() if numeric_cols else {}

        # Store summary lines here
        lines = []
        # Schema: show each column and its dtype
        lines.append("Schema (dtype):")
        for k, v in dtypes.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

        # Null / Non-null counts
        lines.append("Null/Non-null counts:")
        for c in dataframe.columns:
            lines.append(
                f"- {c}: nulls = {int(null_counts[c])}, non_nulls = {int(non_null_counts[c])}"
            )
        lines.append("")

        # Cardinality (number of unique values)
        lines.append("Cardinality (nunique):")
        for k, v in cardinality.items():
            lines.append(f"- {k}: {int(v)}")
        lines.append("")

        # Numeric column summary stats
        if desc:
            lines.append("Numeric summary stats (describe):")
            for col, stats in desc.items():
                # Build a line of stats for each numeric column (mean, std, min, etc.)
                stat_line = ", ".join(
                    [
                        f"{s}: {round(float(val), 4) if pd.notnull(val) else 'nan'}"
                        for s, val in stats.items()
                    ]
                )
                lines.append(f"- {col}: {stat_line}")
            lines.append("")
        # Sample rows of data (in CSV format)
        lines.append("Sample data (CSV head):")
        lines.append(sample_csv)
        # Return the entire summary as one string
        return "\n".join(lines)

    except Exception as e:
        # If anything goes wrong, return the error message
        return f"Error summarizing dataframe: {e}"


def build_cleaning_prompt(df, selected_column: str):
    data_summary = summarize_dataset(df)
    prompt = f"""
    You are an expert data scientist, specifically in the field of data cleaning.
    You are given a dataframe and you need to clean it.
    Dataset summary:
    {data_summary}
    
    Please clean the data and return the cleaned data.
    Make sure to handle the following:
    - Drop the column '{selected_column}' from the dataframe
    - Handle missing values
    - Remove duplicate values
    - Detect and handle outliers
    - Standardize the data accordingly 
    - Use one-hot encoding for categorical variables
    
    GENERATE A STANDALONE python script to clean the data, based on the data summary provided,
    in a JSON property called 'script'.
    - DO NOT PRINT ANYTHING TO STDOUT OR STDERR.
    ## IMPORTANT ##
    - Load the data from a file called 'input.csv'.
    - Save the cleaned data to a new file called 'cleaned.csv'.
    """
    return prompt


def get_openai_script(prompt: str) -> str:
    """
    Call OpenAI's API to get a Python script for data cleaning based on the provided prompt.
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": (
                    "You are a senior data engineer. "
                    "Always return a strict JSON object "
                    "matching the user's requested schema."
                )},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )

        if not response or not getattr(response, 'choices', None):
            return None

        text = response.choices[0].message.content or ""
        data = json.loads(text)
        script_val = data.get("script", "")
        if isinstance(script_val, str) and script_val.strip():
            return script_val.strip()

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


def execute_in_daytona(script: str, csv_bytes: bytes):
    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        raise ValueError("DAYTONA_API_KEY environment variable not set")

    client = Daytona(DaytonaConfig(api_key=api_key))
    sandbox = client.create()
    exec_info = {}
    try:
        sandbox.fs.upload_file(csv_bytes, "input.csv")
        cmd = "python -u - << 'PY'\n" + script + "\nPY"
        result = sandbox.process.exec(cmd, timeout=300, env={"PYTHONUNBUFFERED": "1"})

        exec_info["exit_code"] = getattr(result, "exit_code", None)
        exec_info["stdout"] = getattr(result, "result", "")
        exec_info["stderr"] = getattr(result, "error", "")

        try:
            cleaned_bytes = sandbox.fs.download_file("cleaned.csv")
            return cleaned_bytes, exec_info
        except Exception as e:
            print(f"Error downloading cleaned.csv: {e}")
            return None, exec_info

    except Exception as e:
        print(f"Execution error in Daytona: {e}")
        return None, exec_info


def train_model_in_daytona(cleaned_bytes: bytes, target_column: str, chosen_metric: str, chosen_model: str):
    """
    Upload cleaned dataset to Daytona, train a model, evaluate with chosen metric, and save model.pkl
    """
    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        raise ValueError("DAYTONA_API_KEY environment variable not set")

    client = Daytona(DaytonaConfig(api_key=api_key))
    sandbox = client.create()
    exec_info = {}

    try:
        # Upload cleaned dataset
        sandbox.fs.upload_file(cleaned_bytes, "cleaned.csv")

        # Training script (with dynamic model + metric)
        train_script = f"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

# Load dataset
df = pd.read_csv("cleaned.csv")

# Separate features (X) and target (y)
X = df.drop(columns=["{target_column}"])
y = df["{target_column}"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Choose model
if "{chosen_model}" == "Logistic Regression":
    model = LogisticRegression(max_iter=1000, random_state=42)
elif "{chosen_model}" == "Gradient Boosting":
    model = GradientBoostingClassifier(random_state=42)
else:
    model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
metrics = {{
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
    "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
}}

print("All Metrics:", metrics)
print("Chosen Metric ({chosen_metric}):", metrics["{chosen_metric}"])

# Save model
joblib.dump(model, "model.pkl")
"""

        cmd = "python -u - << 'PY'\n" + train_script + "\nPY"
        result = sandbox.process.exec(cmd, timeout=300, env={"PYTHONUNBUFFERED": "1"})

        exec_info["exit_code"] = getattr(result, "exit_code", None)
        exec_info["stdout"] = getattr(result, "result", "")
        exec_info["stderr"] = getattr(result, "error", "")

        # Download model.pkl
        model_bytes = sandbox.fs.download_file("model.pkl")
        return model_bytes, exec_info

    except Exception as e:
        print(f"Error training model in Daytona: {e}")
        return None, exec_info


# --- Streamlit workflow ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    selected_column = st.selectbox(
        "Select a column to predict",
        df.columns.tolist(),
        help="The column to predict"
    )

    chosen_model = st.selectbox(
        "Choose Model",
        ["Random Forest", "Logistic Regression", "Gradient Boosting"]
    )

    chosen_metric = st.selectbox(
        "Optimize for Metric",
        ["Accuracy", "Precision", "Recall", "F1-score"]
    )

    if st.button("Run AutoML"):
        with st.spinner("Running AutoML..."):
            # Step 1: Cleaning
            cleaning_prompt = build_cleaning_prompt(df, selected_column)
            with st.expander("Cleaning Prompt"):
                st.write(cleaning_prompt)

            script = get_openai_script(cleaning_prompt)
            with st.expander("Script"):
                st.code(script)

            input_csv_bytes = df.to_csv(index=False).encode("utf-8")
            cleaned_bytes, exec_info = execute_in_daytona(script, input_csv_bytes)

            with st.expander("Execution Info"):
                st.write(exec_info)

            if cleaned_bytes:
                with st.expander("Cleaned Data"):
                    cleaned_df = pd.read_csv(io.BytesIO(cleaned_bytes))
                    st.dataframe(cleaned_df)

                # Step 2: Training
                with st.spinner("Training model in Daytona..."):
                    model_bytes, train_info = train_model_in_daytona(
                        cleaned_bytes, selected_column, chosen_metric, chosen_model
                    )

                    with st.expander("Training Info"):
                        st.write(train_info)

                    if model_bytes:
                        st.success(f"Model trained and optimized for {chosen_metric} using {chosen_model}!")

                        # Step 3: Download trained model
                        st.download_button(
                            label="Download Trained Model",
                            data=model_bytes,
                            file_name="model.pkl",
                            mime="application/octet-stream"
                        )
