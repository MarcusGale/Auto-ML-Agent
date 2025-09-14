# Autonomous ML Agent 🚀
Autonomous ML Agent is a Streamlit app that lets users upload CSVs, automatically cleans data using OpenAI, and trains ML models in a secure Daytona sandbox. It supports model selection, metric optimization, and downloads the trained model, all with a simple UI and robust error handling. Users can upload a CSV file, select a target column, choose a model and metric, and the app will:

- Summarize and analyze the dataset.
- Generate a custom Python cleaning script using OpenAI.
- Execute the script securely in a Daytona sandbox.
- Train a machine learning model (Random Forest, Logistic Regression, or Gradient Boosting) on the cleaned data.
- Optimize for a chosen metric (Accuracy, Precision, Recall, F1-score).
- Allow download of the trained model.

## Features

- **Automated Data Cleaning:** Uses OpenAI to generate a cleaning script tailored to your dataset.
- **Secure Execution:** Runs all scripts in an isolated Daytona environment.
- **Model Training:** Supports multiple model types and evaluation metrics.
- **User-Friendly UI:** Built with Streamlit for easy interaction.
- **Downloadable Model:** Export your trained model as a `.pkl` file.

## How It Works

1. **Upload CSV:** The app reads and displays your data.
2. **Select Target & Model:** Choose the column to predict, model type, and metric.
3. **AutoML Pipeline:** The app summarizes your data, generates a cleaning script, runs it in Daytona, and trains a model.
4. **Download Model:** Download the trained model for use in your own projects.

## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
