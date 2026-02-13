import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessing.pkl")

class DateFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            # Check if X is a DataFrame
            if isinstance(X, pd.DataFrame):
                df = X.copy()
            else:
                 # If it's a numpy array, convert to DF (assuming single column 'Date' if passed correctly)
                 # However, in ColumnTransformer, it passes the selected columns.
                 # If 'Date' is passed, it might be a 2D array or DF.
                 # To be safe, we expect a DataFrame or convertible.
                 # For simplicity, let's assume valid key access or index.
                 df = pd.DataFrame(X, columns=['Date'])
            
            # Ensure Date is datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            
            # Drop the original Date column
            df = df.drop(columns=['Date'])
            
            return df
        except Exception as e:
            raise CustomException(e, sys)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            cat_features = [
                "Day_of_Week",
                "Season",
                "Weather_Type",
                "Special_Occasion",
            ]
            
            num_features = [
                "Is_Weekend",
                "Holiday",
            ]
            
            # Date feature extraction pipeline
            # Since 'Date' needs to be transformed into 3 features, and then potentially scaled?
            # 'Year', 'Month', 'Day' are numerical.
            # We can create a pipeline that first extracts date features, and then scales them.
            # However, sklearn pipelines are sequential.
            # Valid approach:
            # 1. ColumnTransformer to split 'Date', 'Categorical', 'Numerical'
            # 2. 'Date' pipeline: DateFeatureGenerator -> StandardScaler (Wait, DateGenerator returns 3 cols)
            # Standard scaler works on 2D array.
            
            # Simplified approach:
            # We will handle Date extraction *before* passing to the main ColumnTransformer in `initiate_data_transformation`.
            # This makes the preprocessor object cleaner (just scaling and encoding).
            # The 'Year', 'Month', 'Day' will be treated as numerical columns passed to the preprocessor.
            
            # Revised Plan:
            # include DateFeatureGenerator in the pipeline?
            # If we do that, we need to know the output columns dynamics.
            # Let's stick to the prompt's implied workflow: "Create features... then Encode/Scale".
            # If I put Feature Engineering in `initiate_data_transformation` before `fit`, then `preprocessing.pkl` will NOT include feature extraction.
            # But the 'Step 7: Prediction Pipeline' says: "Accept raw input dictionary -> Convert to DF -> Apply preprocessing".
            # If `preprocessing.pkl` only scales/encodes, the prediction pipeline MUST manually extract date features first.
            # To make `preprocessing.pkl` self-contained (which is best practice), I should include the Date extraction.
            # But `ColumnTransformer` applies transformers to specific columns.
            # If I use a custom transformer suitable for ColumnTransformer, it should return numeric array.
            
            # Pipeline Construction:
            # We can have a custom transformer that takes the whole DF, drops/creates cols, and then passes to a ColumnTransformer?
            # Scikit-learn pipelines are flexible.
            
            # Let's go with:
            # 1. Define the preprocessing pipeline for "Date" column -> [DateFeatureExtractor, StandardScaler]
            #    Wait, DateFeatureExtractor returns 3 columns. StandardScaler scales them. This works.
            # 2. Define pipeline for "Categorical" -> [OneHotEncoder]
            # 3. Define pipeline for "Other Numerical" -> [StandardScaler]
            
            # But wait, date extraction creates 'Year', 'Month', 'Day'.
            # I will create a specific `DateTransformer` that returns the numpy array of Year, Month, Day.
            
            pass 

            # Let's use the robust approach of transforming within the object.
            
            numerical_columns = ["Is_Weekend", "Holiday", "Year", "Month", "Day"]
            categorical_columns = [
                "Day_of_Week",
                "Season",
                "Weather_Type",
                "Special_Occasion",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
            
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # Feature Engineering: Extract Date info locally
            # This means the extraction logic is NOT in the pickle, so must be replicated in Prediction Pipeline.
            # I will encapsulate this in a helper function or method to ensure consistency.
            
            def extract_date_features(df):
                df['Date'] = pd.to_datetime(df['Date'])
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Day'] = df['Date'].dt.day
                df = df.drop(columns=['Date'])
                return df

            logging.info("Extracting Date features (Year, Month, Day)")
            train_df = extract_date_features(train_df)
            test_df = extract_date_features(test_df)
            
            preprocessing_obj = self.get_data_transformer_object()

            target_columns = [
                "Total_Customers",
                "LPG_Usage_kg",
                "Dosa_Sold",
                "Idly_Sold",
                "Vada_Sold",
                "Puri_Sold"
            ]
            
            # Drop targets from inputs
            input_feature_train_df = train_df.drop(columns=target_columns, axis=1)
            target_feature_train_df = train_df[target_columns]

            input_feature_test_df = test_df.drop(columns=target_columns, axis=1)
            target_feature_test_df = test_df[target_columns]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # For testing purposes
    # We need paths from ingestion
    from src.components.data_ingestion import DataIngestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    
    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)
    print("Data Transformation Complete.")
    print(f"Train Array Shape: {train_arr.shape}")
