import os
import sys
import time
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.metrics import (
    r2_score, 
    mean_squared_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    explained_variance_score
)
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, save_json

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    model_performance_file_path = os.path.join("artifacts", "model_performance.json")
    model_metadata_file_path = os.path.join("artifacts", "model_metadata.json")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        try:
            report = {}
            best_model_score = -float("inf")
            best_model_name = ""
            best_model_instance = None

            for model_name, model in models.items():
                logging.info(f"Starting training for {model_name}")
                print(f"Training {model_name}...")
                start_time = time.time()
                
                para = params.get(model_name, {})
                # Add estimator__ prefix to params for MultiOutputRegressor
                gs_params = {f"estimator__{k}": v for k, v in para.items()}

                gs = GridSearchCV(model, gs_params, cv=5, n_jobs=-1, verbose=1, refit=True)
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                
                # Predictions
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                # Metrics
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                mae = mean_absolute_error(y_test, y_test_pred)
                mape = mean_absolute_percentage_error(y_test, y_test_pred)
                evs = explained_variance_score(y_test, y_test_pred)
                
                end_time = time.time()
                training_duration = end_time - start_time

                logging.info(f"{model_name} Results: R2={test_model_score:.4f}, RMSE={rmse:.4f}")
                print(f"Finished {model_name} - R2: {test_model_score:.4f}")

                report[model_name] = {
                    "r2_score_test": test_model_score,
                    "r2_score_train": train_model_score,
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "explained_variance": evs,
                    "best_params": gs.best_params_,
                    "training_time_sec": training_duration
                }

                if test_model_score > best_model_score:
                    best_model_score = test_model_score
                    best_model_name = model_name
                    best_model_instance = best_model
            
            return report, best_model_name, best_model_instance, best_model_score

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            
            # Assuming last 6 columns are targets
            num_targets = 6
            X_train, y_train, X_test, y_test = (
                train_array[:, :-num_targets],
                train_array[:, -num_targets:],
                test_array[:, :-num_targets],
                test_array[:, -num_targets:],
            )

            models = {
                "Random Forest": MultiOutputRegressor(RandomForestRegressor(random_state=42)),
                "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
                "Extra Trees": MultiOutputRegressor(ExtraTreesRegressor(random_state=42)),
                "XGBRegressor": MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', random_state=42)), 
                "Ridge": MultiOutputRegressor(Ridge()),
                "Lasso": MultiOutputRegressor(Lasso()),
            }

            # Hyperparameter Grid
            params = {
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10],
                    "min_samples_split": [2, 5]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.1, 0.05],
                    "max_depth": [3, 5]
                },
                "Extra Trees": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10]
                },
                "XGBRegressor": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.1, 0.05],
                    "max_depth": [3, 5]
                },
                "Ridge": {
                    "alpha": [0.1, 1.0, 10.0]
                },
                "Lasso": {
                    "alpha": [0.1, 1.0, 10.0]
                }
            }

            logging.info("Starting Hyperparameter Tuning and Model Selection")
            
            model_report, best_model_name, best_model, best_model_score = self.evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models, 
                params=params
            )

            # Save Performance Report
            save_json(self.model_trainer_config.model_performance_file_path, model_report)
            logging.info(f"Model performance report saved to {self.model_trainer_config.model_performance_file_path}")

            # Save Best Model
            if best_model_score < 0.5:
                logging.warning("Best model R2 score is low (< 0.5). Improvement needed.")

            logging.info(f"Best Model Found: {best_model_name} with R2 Score: {best_model_score:.4f}")
            print(f"Best Model Found: {best_model_name} with R2 Score: {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Metadata
            metadata = {
                "best_model_name": best_model_name,
                "best_model_score": best_model_score,
                "dataset_size_train": len(X_train),
                "dataset_size_test": len(X_test),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "targets": ["Total_Customers", "LPG_Usage_kg", "Dosa_Sold", "Idly_Sold", "Vada_Sold", "Puri_Sold"]
            }
            save_json(self.model_trainer_config.model_metadata_file_path, metadata)
            logging.info("Model metadata saved.")

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
