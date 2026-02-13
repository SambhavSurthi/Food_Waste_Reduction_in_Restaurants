import sys
import os
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")
            
            # Step 1: Data Ingestion
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed: {train_data_path}, {test_data_path}")

            # Step 2: Data Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            logging.info(f"Data Transformation completed. Preprocessor saved at: {preprocessor_path}")

            # Step 3: Model Training
            model_trainer = ModelTrainer()
            score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Model Training completed. Best Model Score: {score}")
            
            print(f"Training Pipeline Completed. Best Model Score: {score}")

        except Exception as e:
            logging.error(f"Training pipeline failed: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        print(e)
