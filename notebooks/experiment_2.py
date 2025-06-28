"""
This script trains several baseline models on a credit card fraud detection dataset while applying different feature engineering techniques.
Performance is logged in MLflow, and the best model is selected for hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler,PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
from dotenv import load_dotenv
load_dotenv()

import os


# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "notebooks/creditcard_resampled.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "your Mlflow Tracking URI",
    "dagshub_repo_owner": "your Dagshub Repo Owner",
    "dagshub_repo_name": "your Dagshub Repo Name",
    "experiment_name": "feature_engineering_experiment"
    }

# ========================== Setup MLFlow and Dagshub ==========================

mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== Feature Engineering Techniques ==========================

FE_TECH = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
    "PowerTransformer": PowerTransformer(method='yeo-johnson')
}

# =========================== MODELS ==========================
MODELS = {
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier()
}

# =========================== Load Data ==========================

def load_data(file_path) -> pd.DataFrame:
    """
    Load the dataset from the specified path.
    """
    df = pd.read_csv(file_path)
    return df

# =========================== Train and Evaluate Models ==========================

def train_and_evaluate_model(df):
    """
    Train and evaluate a model with the given scaler.
    """
    X = df.drop(columns='Class',axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42)

    with mlflow.start_run(run_name="Feature Engineering and models") as parent_run:
        for fe_name,fe_method in FE_TECH.items():
            try:
                X_train_scaled = fe_method.fit_transform(X_train)
                X_test_scaled = fe_method.transform(X_test)

                for model_name, model in MODELS.items():
                    with mlflow.start_run(run_name=f"{fe_name}_{model_name}", nested=True) as child_run:
                        try:
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)

                            # log preprocesing parameters
                            mlflow.log_params({
                                "feature_engineering": fe_name,
                                "model": model_name,
                                "test_size": CONFIG["test_size"]
                            })

                            # log model metrics
                            metrics = {
                                "accuracy": accuracy_score(y_test, y_pred),
                                "precision": precision_score(y_test, y_pred),
                                "recall": recall_score(y_test, y_pred),
                                "f1_score": f1_score(y_test, y_pred),
                            }

                            mlflow.log_metrics(metrics)

                            print(f"Model: {model_name}, Feature Engineering: {fe_name}, Metrics: {metrics}")
                        except Exception as e:
                            print(f"Error training {model_name} with {fe_name}: {e}")
                            mlflow.log_param("error", str(e))
            except Exception as e:
                print(f"Error applying feature engineering {fe_name}: {e}")
                mlflow.log_param("error", str(e))

          
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    df = load_data(CONFIG["data_path"])
    train_and_evaluate_model(df)
    
    # End the parent run
    mlflow.end_run()
    
    print("Experiment completed and logged in MLflow.")
