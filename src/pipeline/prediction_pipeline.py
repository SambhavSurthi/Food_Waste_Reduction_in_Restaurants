import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Construct absolute paths
            # This file is in src/pipeline/
            # We need to go up two levels to get to root, then to artifacts
            src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # D:\Food_Waste_Reduction\src
            root_path = os.path.dirname(src_path) # D:\Food_Waste_Reduction
            
            model_path = os.path.join(root_path, "artifacts", "model.pkl")
            preprocessor_path = os.path.join(root_path, "artifacts", "preprocessing.pkl")
            
            # print(f"Model Path: {model_path}")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            # print("After Loading")
            
            # Feature Engineering: Extract Date info locally (Same as in DataTransformation)
            def extract_date_features(df):
                df['Date'] = pd.to_datetime(df['Date'])
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Day'] = df['Date'].dt.day
                df = df.drop(columns=['Date'])
                return df
            
            features = extract_date_features(features)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        date: str,
        day_of_week: str,
        is_weekend: int,
        season: str,
        weather_type: str,
        special_occasion: str,
        holiday: int):
        
        self.date = date
        self.day_of_week = day_of_week
        self.is_weekend = is_weekend
        self.season = season
        self.weather_type = weather_type
        self.special_occasion = special_occasion
        self.holiday = holiday

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Date": [self.date],
                "Day_of_Week": [self.day_of_week],
                "Is_Weekend": [self.is_weekend],
                "Season": [self.season],
                "Weather_Type": [self.weather_type],
                "Special_Occasion": [self.special_occasion],
                "Holiday": [self.holiday],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
