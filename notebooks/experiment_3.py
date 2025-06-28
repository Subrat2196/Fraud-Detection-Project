import os
import numpy as np 
import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from dotenv import load_dotenv
load_dotenv()

# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "notebooks/creditcard_resampled.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "Your Mlflow Tracking URI",
    "dagshub_repo_owner": "Your Dagshub Repo Owner",
    "dagshub_repo_name": "Your Dagshub Repo Name",
    "experiment_name": "grid_search_experiment"
    }

# ========================== Setup MLFlow and Dagshub ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])    

# =========================== Load Data ==========================

def load_and_prepare_data(file_path) -> pd.DataFrame:
    """
    Load the dataset from the specified path and prepare it for training.
    """
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Class'])
    y = df['Class']
    
  # Apply PowerTransformer to normalize the data
    scaler = PowerTransformer(method='yeo-johnson')
    X_transformed = scaler.fit_transform(X)

    return train_test_split(X_transformed, y, test_size=CONFIG["test_size"], random_state=42), scaler

# =========================== Model Training and Evaluation ==========================

def train_and_evaluate_model(X_train, X_test, y_train, y_test, scaler):
    """   
    train_and_evaluate_model trains a Logistic Regression model using GridSearchCV
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    # Here we need to find only the best hyperparameters for the RandomForestClassifier
    # We will use GridSearchCV to find the best hyperparameters for the RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    with mlflow.start_run():
        grid_search = GridSearchCV(model,param_grid,cv=5, scoring='f1', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        for params, mean_score, std_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score']):
            with mlflow.start_run(run_name=f"GridSearch_{params}", nested=True):
                model= RandomForestClassifier(**params, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred)
                }
                 
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                print(f"Params: {params}, Mean Score: {mean_score}, Std Score: {std_score}, Metrics: {metrics}")

        #log the best model and its parameters
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_score", best_score) 

        print(f"Best Params: {best_params}, Best Score: {best_score}")

if __name__ == "__main__":
    # Load and prepare the data
    (X_train, X_test, y_train, y_test), scaler = load_and_prepare_data(CONFIG["data_path"])
    
    # Train and evaluate the model
    train_and_evaluate_model(X_train, X_test, y_train, y_test, scaler)
# This script performs feature engineering using PowerTransformer and trains a RandomForestClassifier with GridSearchCV to find the best hyperparameters.
# It logs the results to MLflow and Dagshub for tracking and visualization. 
# The script is designed to be run as a standalone program.
# It loads the dataset, applies feature engineering, trains the model, and evaluates its performance.
# It also logs the model parameters and metrics to MLflow for later analysis.
# The script is modular and can be easily extended to include more feature engineering techniques or models.
# This script is designed to perform feature engineering and model training using PowerTransformer and RandomForestClassifier.
# It uses GridSearchCV to find the best hyperparameters for the model and logs the results to MLflow and Dagshub.



