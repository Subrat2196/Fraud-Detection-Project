import os
import numpy as np
import pandas as pd
import sys
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from src.logger import logging
import yaml
import logging

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def apply_transformation(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    """Apply PowerTransformer to the training and test data."""
    try:
        scaler = PowerTransformer()

        X_train = train_data.drop(columns='Class', axis=1)
        y_train = train_data['Class']
        X_test = test_data.drop(columns='Class', axis=1)
        y_test = test_data['Class']

        # finding numerical columns
        # numerical_cols = X_train.columns[X_train.dtypes != 'object'].tolist() # we are doing .tolist() to convert the Index object to a list
        # using select_dtypes also we can find numerical columns like shown below
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist() 
        # Here we are doing .columns to get the column names of the DataFrame , then .tolist() to convert the Index object to a list
        # if we dont do .columns then we will get a DataFrame with the numerical columns and we cannot use it directly
        # if we dont do .tolist() then we will get an Index object which is not a list and we cannot use it directly
        logging.info(f"Numerical columns identified: {numerical_cols}")

        # Apply PowerTransformer only to numerical columns
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        logging.info("PowerTransformer applied to numerical columns.")

        # Create new DataFrames with scaled features
        train_data_scaled = pd.DataFrame(X_train, columns=X_train.columns)
        train_data_scaled['Class'] = y_train.values
        test_data_scaled = pd.DataFrame(X_test, columns=X_test.columns)
        test_data_scaled['Class'] = y_test.values

        logging.info("Data transformed successfully.")
        return train_data_scaled, test_data_scaled
    except Exception as e:
        logging.error(f"Error in applying transformation: {e}")
        sys.exit(1)

def save_data(df: pd.DataFrame, save_dir: str, file_name: str) -> None:
    """Save the DataFrame to a CSV file."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)
        df.to_csv(file_path, index=False)
        logging.info(f"Data saved successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")
        raise

def main():
    """Main function to run the feature engineering pipeline."""
    try:    
        train_data = load_data('data/interim/train_preprocessed.csv')
        test_data = load_data('data/interim/test_preprocessed.csv')
        logging.info("Data loaded successfully for feature engineering.")

        # Apply transformation
        train_data_transformed, test_data_transformed = apply_transformation(train_data, test_data)
        logging.info("Feature engineering completed successfully.")

        # Save the transformed data
        save_data(train_data_transformed, 'data/processed', 'train_transformed.csv')
        save_data(test_data_transformed, 'data/processed', 'test_transformed.csv')  

    except Exception as e:
        logging.error(f"An error occurred during feature engineering: {e}")
        raise

    
if __name__ == "__main__":
    main()
    logging.info("✅ Feature engineering pipeline completed successfully.")
# This code is part of the feature engineering pipeline for a fraud detection project.
# It loads preprocessed training and test data, applies PowerTransformer to scale numerical features,
# and saves the transformed data to a specified directory.
# The code includes error handling and logging to track the progress and any issues that arise during execution


