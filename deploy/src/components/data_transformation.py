import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.impute  import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger  import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig():
    preprocessing_obj_file_path=os.path.join('artifacts','preprocessing.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            num_columns=["reading_score", "writing_score"]

            cat_columns=['gender', 
                         'race_ethnicity',
                         'parental_level_of_education', 
                         'lunch', 
                         'test_preparation_course'
                         ]
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OneHotEncoder())
                ]
            )

            logging.info(f"Numerical Column:{num_columns}")
            logging.info(f"Categorical Column:{cat_columns}")

            preprocessor=ColumnTransformer(
               [
                ("num_pipeline",num_pipeline,num_columns),
                ("cat_pipeline",cat_pipeline,cat_columns)
               ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test data")

            preprocessor_obj=self.get_data_transformation_obj()
            logging.info("obtained preprocessor object")

            target_column_name="math_score"
            numerical_columns=["reading_score","writing_score"]

            y_train=train_df[target_column_name]
            X_train=train_df.drop(columns=[target_column_name])

            y_test=test_df[target_column_name]
            X_test=test_df.drop(columns=[target_column_name])

            X_train_arr=preprocessor_obj.fit_transform(X_train)
            X_test_arr=preprocessor_obj.transform(X_test)

            logging.info("Applied preprocessing object to input train and test data")

            train_arr=np.c_[X_train_arr,np.array(y_train)]
            test_arr=np.c_[X_test_arr,np.array(y_test)]
            
            print("Reached save_object step")

            save_object(
                file_path=self.data_transformation_config.preprocessing_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("saved preprocessing object")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_obj_file_path
            )

        
        except Exception as e:
            raise CustomException(e,sys)

        


        

        

        


            
