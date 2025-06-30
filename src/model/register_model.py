import json
import os
import pickle
import dagshub
import mlflow.tracking
import mlflow.tracking.client
import numpy as np
import pandas as pd
import mlflow
import sys
from dotenv import load_dotenv
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.logger import logging
import yaml
import logging


mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
if not mlflow_tracking_uri:
    logging.error("MLFLOW_TRACKING_URI environment variable is not set.")
    raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")

mlflow.set_tracking_uri(mlflow_tracking_uri)
dagshub.init(repo_owner='bahuguna.subrat211996', repo_name='model_evaluation_repo', mlflow=True)

def load_model_info(file_path: str) -> dict:
    '''load model info from json file , jo humne run_id aur model path save kia tha'''
    try:
        with open(file_path,'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    '''register model to mlflow registry'''
    try:
        client = mlflow.tracking.MlflowClient()

        # Register the model
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logging.info(model_uri)
        model_version = mlflow.register_model(model_uri, model_name) 
        # Create a new model version in model registry for the model files specified by model_uri.
        # we will give the model a name for it to register

        # Transition Model to "Staging"

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version, # out of the versions in the model_version , get the latest one
            stage="Staging" # Directly putting our model to staging area
        )
        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')

    except Exception as e:
        logging.error('Error during model and transformer registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"

        register_model(model_name, model_info)
        
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
