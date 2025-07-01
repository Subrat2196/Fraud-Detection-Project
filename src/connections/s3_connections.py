import boto3
import pandas as pd
import os
import sys
import logging
from src.logger import logging
from io import StringIO

class s3_operations:
    def __init__(self, bucket_name, aws_access_key, aws_secret_key, region_name="us-east-1"):
        self.bucket_name=bucket_name
        self.s3_client = boto3.client(
            's3', #aws service name
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
    logging.info("Data Ingestions from S3 bucket")

    def fetch_file_from_s3(self, file_key):
        """
        Fetches a csv file from s3 bucket and returns it as a Pandas DataFrame
        """

        try:
            logging.info(f"fetching file '{file_key}' from s3 bucket '{self.bucket_name}'...")
            # self.s3_client.get_object is a boto3 call to download a file
            obj = self.s3_client.get_object(Bucket=self.bucket_name,Key=file_key)
            # obj['Body'].read() reads the raw binary content of the file from the S3 stream.
            # .decode('utf-8') converts those raw bytes into a regular Python string (assuming the file is encoded in UTF-8).
            # StringIO converts a string into a file like object
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8'))) # conversion of csv to pandas dataframe
            logging.info(f"Successfully fetched and loaded '{file_key}' from S3 that has {len(df)} records.")
            return df
        except Exception as e:
            logging.exception(f"❌ Failed to fetch '{file_key}' from S3: {e}")
            return None

    
    
# # Example usage
# if __name__ == "__main__":
#     # Replace these with your actual AWS credentials and S3 details
#     BUCKET_NAME = ''
#     AWS_ACCESS_KEY = ""
#     AWS_SECRET_KEY = ""
#     FILE_KEY = ""  # Path inside S3 bucket

#     data_ingestion = s3_operations(BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY)
#     df = data_ingestion.fetch_file_from_s3(FILE_KEY)

#     if df is not None:
#         print(f"Data fetched with {len(df)} records..")  # Display first few rows of the fetched DataFrame
        