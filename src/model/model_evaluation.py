import json
import os
import pickle
import dagshub
import numpy as np
import pandas as pd
import mlflow
import sys
from dotenv import load_dotenv
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import resample
from src.logger import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score
import yaml
import logging

load_dotenv(override=True)
# get mlflow tracking uri from environment variable
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
if not mlflow_tracking_uri:
    logging.error("MLFLOW_TRACKING_URI environment variable is not set.")
    raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")

mlflow.set_tracking_uri(mlflow_tracking_uri)
dagshub.init(repo_owner='bahuguna.subrat211996', repo_name='model_evaluation_repo', mlflow=True)

def load_model(model_path: str):
    """Load the model from the specified path."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logging.info(f"Model loaded successfully from {model_path}.")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model using the test data."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

        logging.info(f"Model evaluation metrics: {metrics_dict}")
        return metrics_dict
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise
# we use json.dump for saving metrics due to it being serializable, interoperable, human readable and reusable
# It need to have a python dictionary to convert it to a proper json format
def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4) # dump is used to save as a json object
        logging.info(f"Metrics saved successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the run id and path to the json file"""
    try:
        model_info = {"run_id": run_id,"model_path": model_path}
        with open(file_path,'w') as file:
            json.dump(model_info,file,indent=4)
        logging.debug("Model info saved to %s",file_path)
    except Exception as e:
        logging.error('Error occured while saving the model info: %s', e)
        raise


def main():
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            model = load_model('./models/random_forest_model.pkl')
            test_data = load_data("./data/processed/test_transformed.csv")

            X_test = test_data.drop(columns='Class').values
            y_test = test_data['Class'].values

            metrics = evaluate_model(model,X_test,y_test)
            save_metrics(metrics, "reports/metrics.json")

            # Log metrics to MlFlow

            for metric_name, metric_values in metrics.items():
                mlflow.log_metric(metric_name,metric_values)

            # log model parameters to MLFlow
            if hasattr(model, 'get_params'):
                params = model.get_params() # It will return a dictionary of hyperparameters
                for param_name,param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # # You're missing this line:
            # mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.log_artifact('./models/random_forest_model.pkl')

            save_model_info(run.info.run_id,"model",'reports/experiment_info.json')

            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
    main()


    



