import os
import sys
import json
import mlflow
import dagshub
from dotenv import load_dotenv

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.logger import logging

# Load environment variables
load_dotenv(override=True)
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
if not mlflow_tracking_uri:
    logging.error("MLFLOW_TRACKING_URI environment variable is not set.")
    raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")

# Set MLflow tracking URI and initialize DagsHub
mlflow.set_tracking_uri(mlflow_tracking_uri)
dagshub.init(repo_owner='bahuguna.subrat211996', repo_name='model_evaluation_repo', mlflow=True)

def load_model_info(file_path: str) -> dict:
    """Load model info from JSON file (contains run_id and model_path)."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info(f"✅ Model info loaded from {file_path}")
        return model_info
    except FileNotFoundError:
        logging.error(f"❌ Model info file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"❌ Unexpected error while loading model info: {e}")
        raise

def register_model(model_name: str, model_info: dict):
    """
    Simulate model registration by printing model URI.
    DagsHub does not support MLflow model registry APIs.
    """
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logging.info(f"Model URI: {model_uri}")
        logging.warning("DagsHub does not support MLflow model registry APIs. Skipping formal registration.")
        print(f"🔗 Track your model at: {model_uri}")
    except Exception as e:
        logging.error(f"❌ Error during model URI logging: {e}")
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error(f"Failed to complete model URI logging: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
