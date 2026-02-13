import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('dataset.csv')
            logging.info('Read the dataset as dataframe')

            # Chronological sorting
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values(by='Date')
                logging.info('Sorted dataset chronologically by Date')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated (Chronological Split)")
            # Chronological split: Take first 80% as train, last 20% as test
            train_size = int(len(df) * 0.8)
            train_set = df.iloc[:train_size]
            test_set = df.iloc[train_size:]

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
            
if __name__ == "__main__":
    obj = DataIngestion()
    try:
        train_path, test_path = obj.initiate_data_ingestion()
        print(f"Data Ingestion Complete. Train Path: {train_path}, Test Path: {test_path}")
    except Exception as e:
        print(f"Error: {e}")
