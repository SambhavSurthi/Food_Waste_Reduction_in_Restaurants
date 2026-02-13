import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            
            # Identify X and y
            # Last 6 columns are targets (Total_Customers, LPG_Usage_kg, Dosa_Sold, Idly_Sold, Vada_Sold, Puri_Sold)
            # The input features are everything before that.
            
            num_targets = 6
            X_train, y_train, X_test, y_test = (
                train_array[:, :-num_targets],
                train_array[:, -num_targets:],
                test_array[:, :-num_targets],
                test_array[:, -num_targets:],
            )

            models = {
                "Random Forest": MultiOutputRegressor(RandomForestRegressor()),
                "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor()),
                "Linear Regression": MultiOutputRegressor(LinearRegression()),
                "XGBRegressor": MultiOutputRegressor(XGBRegressor(objective='reg:squarederror')), 
                "Extra Trees": MultiOutputRegressor(ExtraTreesRegressor()),
                "Ridge": MultiOutputRegressor(Ridge()),
                "Lasso": MultiOutputRegressor(Lasso()),
            }
            
            model_report: dict = {}
            
            logging.info("Training models and evaluating...")

            for i, (model_name, model) in enumerate(models.items()):
                model.fit(X_train, y_train) 
                y_test_pred = model.predict(X_test)
                
                # Evaluate
                r2 = r2_score(y_test, y_test_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                
                model_report[model_name] = r2
                
                print(f"{model_name} - R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")
                logging.info(f"{model_name} - R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")

            # Best model selection
            best_model_score = max(model_report.values())
            best_model_name = next(key for key, value in model_report.items() if value == best_model_score)
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.warning("Best model score is less than 0.6 - Model performance might be poor.")
                # raise CustomException("No best model found") # Optional: raise error if threshold not met

            logging.info(f"Best found model on both training and testing dataset: {best_model_name} with R2: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
