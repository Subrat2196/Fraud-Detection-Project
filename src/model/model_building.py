import os
import pickle
import numpy as np
import pandas as pd
import mlflow
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import resample
from src.logger import logging
import yaml
import logging
from sklearn.ensemble import RandomForestClassifier

def load_params():
    """Load parameters from params.yaml."""
    try:
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file)  # params is a dictionary
        logging.info("Parameters loaded successfully.")
        return params
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        sys.exit(1)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def train_model(X_train: np.ndarray,y_train: np.ndarray) -> RandomForestClassifier:
    """Train a Random Forest Classifier model with the best parameters."""
    try:
        params = load_params()
        model_params = params['model']['params']
        n_estimators = model_params['n_estimators']
        max_depth = model_params['max_depth']
        random_state = model_params['random_state']

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        sys.exit(1)


def save_model(model: RandomForestClassifier, model_dir_path: str) -> None:
    try:
        with open(model_dir_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info(f"Model saved successfully at {model_dir_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        sys.exit(1)

def main():
    try:
        # Load Parameters
        params = load_params()
        train_data = load_data('./data/processed/train_transformed.csv')
        X_train = train_data.drop(columns='Class', axis=1).values # We are taking .values to convert the DataFrame to a numpy array
        y_train = train_data['Class'].values # Since we are using RandomForestClassifier which takes numpy array as input


        # Train the model
        model = train_model(X_train, y_train)

        # Save the model
        save_model(model, './models/random_forest_model.pkl')
        logging.info("✅ Model training pipeline completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")
        raise

if __name__ == "__main__":
    main()
    logging.info("✅ Model building pipeline completed successfully.")
