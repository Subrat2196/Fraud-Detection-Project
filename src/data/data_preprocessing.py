import os
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


# Load parameters from params.yaml

def load_params():
    """Load parameters from params.yaml."""
    try:
        with open('params.yaml', 'r') as file:
            params = yaml.safe_load(file) # params is a dictionary
        logging.info("Parameters loaded successfully.")
        return params
    except Exception as e:
        logging.error(f"Error loading parameters: {e}")
        sys.exit(1)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, scaling, and balancing."""
    try:
        # Handle missing values
        df.fillna(df.median(), inplace=True)
        logging.info("Missing values handled.")

        # Scale features using PowerTransformer
        scaler = PowerTransformer()
        X = df.drop(columns='Class', axis=1)
        y = df['Class']
        X_scaled = scaler.fit_transform(X)
        logging.info("Features scaled using PowerTransformer.")

        # Create a new DataFrame with scaled features
        df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        df_scaled['Class'] = y.values

        # Handle class imbalance
        majority_class = df_scaled[df_scaled['Class'] == 0]
        minority_class = df_scaled[df_scaled['Class'] == 1]

        # downsample majority class
        majority_downsampled = resample(majority_class,
                                      replace=True,
                                      n_samples=len(minority_class),
                                      random_state=42)

        # Combine minority class with downsampled majority class
        df_balanced = pd.concat([minority_class, majority_downsampled])
        logging.info("Class imbalance handled by downsampling majority class.")

        return df_balanced

    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        sys.exit(1)


def main():
    """Main function to run the data preprocessing pipeline."""
    try:
        # Load parameters
        params = load_params()

        # Load data
        df = pd.read_csv('data/raw/credit_card_raw.csv') # artifact of data ingestion
        logging.info(f"Data loaded successfully. Shape: {df.shape}")

        # Preprocess data
        df_processed = preprocess_data(df)
        logging.info(f"Data preprocessing completed. Shape: {df_processed.shape}")

        # Split data into features and target variable
        X = df_processed.drop(columns='Class', axis=1)
        y = df_processed['Class']

        # Split into training and test sets
        train_data,test_data = train_test_split(df_processed, test_size=params['data_preprocessing']['test_size'], random_state=42)
        logging.info(f"Data split into training and test sets. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
        # save the processed data into interim folder
        processed_data_path = os.path.join("./data","interim")
        os.makedirs(processed_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(processed_data_path, 'train_preprocessed.csv'), index=False)
        test_data.to_csv(os.path.join(processed_data_path, 'test_preprocessed.csv'), index=False)

        logging.info("Processed data saved successfully.")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(1) 


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
    logging.info("✅ Data preprocessing pipeline completed successfully.")