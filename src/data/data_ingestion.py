import os
import pandas as pd
import sys
from src.connections import s3_connections
from dotenv import load_dotenv
load_dotenv(override=True)
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.logger import logging
def load_data(file_path: str) -> pd.DataFrame:    
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist.")
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    if not file_path.endswith('.csv'):
        logging.warning("Provided file is not a CSV.")
    
    df = pd.read_csv(file_path)
    logging.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
    return df

def save_data(df: pd.DataFrame, save_dir: str) -> None:
    if df.empty:
        logging.warning("Trying to save an empty DataFrame.")
    
    raw_data_path = os.path.join(save_dir, 'raw')
    os.makedirs(raw_data_path, exist_ok=True)

    file_path = os.path.join(raw_data_path, 'credit_card_raw.csv')
    df.to_csv(file_path, index=False)
    logging.info(f"Data saved successfully at {file_path}")

def main():
    try:
        # Replace these with your actual AWS credentials and S3 details
        bucket_name = os.getenv('S3_BUCKET_NAME')
        aws_access_key = os.getenv('AWS_ACCESS_KEY')
        aws_secret_key = os.getenv('AWS_SECRET_KEY')
        FILE_KEY = "creditcard.csv"  # Path inside S3 bucket

        s3 = s3_connections.s3_operations(bucket_name, aws_access_key, aws_secret_key)
        df = s3.fetch_file_from_s3(FILE_KEY)
        # df = load_data('notebooks/creditcard.csv')
        save_data(df, 'data/')
    except Exception as e:
        logging.error(f"An error occurred during data ingestion: {e}")
        raise

if __name__ == "__main__":
    main()
    logging.info("✅ Data ingestion pipeline completed successfully.")
