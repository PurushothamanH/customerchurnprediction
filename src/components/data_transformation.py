import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from scipy.sparse import csr_matrix, hstack
from src.utils import save_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_filepath = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformer_object(self):
        """
        Its Responsible for the data Transformation process for different columns..
        """
        try:
            numerical_columns =['SeniorCitizen','tenure','MonthlyCharges']
            categorical_columns =['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                     ("imputer",SimpleImputer(strategy="most_frequent")),
                     ("onehotencoder",OneHotEncoder(handle_unknown='ignore',sparse_output=True)),
                     ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("Numerical_pipeline",num_pipeline,numerical_columns),
                    ("Categorical_pipelie",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data completed..")
            logging.info("obtaining Preprocessing object..")

            preprocessor_obj = self.get_data_transformer_object()
            target_column_name ="Churn"

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Appling preprocessing object on both train and test data")
        
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

            # input_feature_train_arr = input_feature_train_arr.toarray()
            # input_feature_test_arr = input_feature_test_arr.toarray()
            input_feature_train_arr = np.array(input_feature_train_arr)
            input_feature_test_arr = np.array(input_feature_test_arr)

            train_arr = np.concatenate((input_feature_train_arr, target_feature_train_arr), axis=1)
            test_arr = np.concatenate((input_feature_test_arr, target_feature_test_arr), axis=1)

            logging.info(f"saved Preprocessing object...")

            save_object(
                
                file_path=self.data_transformation_config.preprocessor_obj_filepath,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_filepath
            )
            
        except Exception as e:
            raise CustomException(e,sys)