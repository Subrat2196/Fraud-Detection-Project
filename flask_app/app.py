import json
import os
import pickle
import logging
import numpy as np
import pandas as pd
import mlflow
import dagshub
from flask import Flask, render_template, request
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.logger import logging

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not mlflow_tracking_uri:
    raise EnvironmentError("MLFLOW_TRACKING_URI environment variable is not set")
mlflow.set_tracking_uri(mlflow_tracking_uri)
dagshub.init(repo_owner="bahuguna.subrat211996", repo_name="model_evaluation_repo", mlflow=True)

# Configuration
MODEL_INFO_PATH = "reports/experiment_info.json"
PREPROCESSOR_PATH = "models/power_transformer.pkl"

# Initialize Flask app
app = Flask(__name__)

# Custom Metrics for Monitoring
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency of requests", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Count of predictions", ["prediction"], registry=registry)

# Load latest model using stored run_id from local json
def load_latest_model_from_local():
    try:
        with open(MODEL_INFO_PATH, 'r') as f:
            model_info = json.load(f)
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logging.info(f"Loading model from URI: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logging.error(f"Failed to load latest model: {e}")
        return None


def load_preprocessor(preprocessor_path):
    try:
        with open(preprocessor_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading PowerTransformer: {e}")
        return None

# Load ML components
model = load_latest_model_from_local()
power_transformer = load_preprocessor(PREPROCESSOR_PATH)

# Feature names for the dataset
FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Helper Functions
def preprocess_input(data):
    try:
        input_array = np.array(data).reshape(1, -1)
        return power_transformer.transform(input_array)
    except Exception as e:
        logging.error(f"Preprocessing Error: {e}")
        return None

# Routes
@app.route("/", methods=["GET", "POST"])
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    prediction = None
    input_values = [""] * len(FEATURE_NAMES)

    if request.method == "POST":
        csv_input = request.form.get("csv_input", "").strip()
        if csv_input:
            try:
                values = list(map(float, csv_input.split(","))) # put the comma seperated values in a list
                if len(values) != len(FEATURE_NAMES):
                    raise ValueError(f"Expected {len(FEATURE_NAMES)} values, but got {len(values)}") # checking same values provided as the number of columns
                input_values = values
                transformed = preprocess_input(input_values) # Transforming/ Power Scaler applied to values
                if transformed is not None and model:
                    result = model.predict(transformed) # Predicting the value
                    prediction = "Fraud" if result[0] == 1 else "Non-Fraud"
                else:
                    prediction = "Error: Model or Transformer not loaded properly."
            except ValueError as ve:
                prediction = f"Input Error: {ve}"
            except Exception as e:
                prediction = f"Processing Error: {e}"

    return render_template("index.html", result=prediction, csv_input=",".join(map(str, input_values))) # rendering the file

@app.route("/predict", methods=["POST"]) # when we press the predict button
def predict():
    csv_input = request.form.get("csv_input", "").strip() # capturing here the values that user enters , strip to seperate the values
    if not csv_input:
        return "Error: No input provided."

    try:
        values = list(map(float, csv_input.split(",")))
        if len(values) != len(FEATURE_NAMES):
            return f"Error: Expected {len(FEATURE_NAMES)} values, but got {len(values)}"

        transformed = preprocess_input(values)
        if transformed is not None and model:
            result = model.predict(transformed)
            return "Fraud" if result[0] == 1 else "Non-Fraud"
        return "Error: Model or Transformer not loaded properly."
    except Exception as e:
        return f"Error processing input: {e}"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
