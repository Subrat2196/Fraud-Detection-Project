import os
import pickle
import logging
import numpy as np
from flask import Flask, render_template, request
import sys
import mlflow
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# ------------------- Configuration -------------------
MODEL_PATH = "models/latest_random_forest_model.pkl"
PREPROCESSOR_PATH = "models/power_transformer.pkl"
FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ------------------- Load Components -------------------
def load_model(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logging.info("Model loaded from local path.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def load_preprocessor(preprocessor_path):
    try:
        with open(preprocessor_path, "rb") as f:
            transformer = pickle.load(f)
        logging.info("Transformer loaded from local path.")
        return transformer
    except Exception as e:
        logging.error(f"Error loading transformer: {e}")
        return None

model = load_model(MODEL_PATH)
power_transformer = load_preprocessor(PREPROCESSOR_PATH)

# ------------------- Flask App -------------------
app = Flask(__name__)

def preprocess_input(data):
    try:
        input_array = np.array(data).reshape(1, -1)
        return power_transformer.transform(input_array)
    except Exception as e:
        logging.error(f"Preprocessing Error: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    input_values = [""] * len(FEATURE_NAMES)

    if request.method == "POST":
        csv_input = request.form.get("csv_input", "").strip()
        if csv_input:
            try:
                values = list(map(float, csv_input.split(",")))
                if len(values) != len(FEATURE_NAMES):
                    raise ValueError(f"Expected {len(FEATURE_NAMES)} values, but got {len(values)}")
                input_values = values
                transformed = preprocess_input(input_values)
                if transformed is not None and model:
                    result = model.predict(transformed)
                    prediction = "Fraud" if result[0] == 1 else "Non-Fraud"
                else:
                    prediction = "Error: Model or Transformer not loaded properly."
            except ValueError as ve:
                prediction = f"Input Error: {ve}"
            except Exception as e:
                prediction = f"Processing Error: {e}"

    return render_template("index.html", result=prediction, csv_input=",".join(map(str, input_values)))

@app.route("/predict", methods=["POST"])
def predict():
    csv_input = request.form.get("csv_input", "").strip()
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
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
