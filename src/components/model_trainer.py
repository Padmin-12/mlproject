import os
import sys

from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.logger import logging
from src.exception import CustomException

from src.utils import evaluate_models
from src.utils import save_object



@dataclass
class ModelTrainerConfig():
    trained_model_path=os.path.join('artifacts','trained_model.pkl')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_arr,test_arr):

        try:
            logging.info("Prepare train and test data for training")

            X_train=train_arr[:,:-1]
            y_train=train_arr[:,-1]
            X_test=test_arr[:,:-1]
            y_test=test_arr[:,-1]

            models={"Linear Regression":LinearRegression(),
                   "Decision Tree":DecisionTreeRegressor(),
                   "Gradient Boosting": GradientBoostingRegressor(),
                   "Linear Regression": LinearRegression(),
                   "XGBRegressor": XGBRegressor(),
                   "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                   "AdaBoost Regressor": AdaBoostRegressor(),}
            
            
            logging.info("find accuracy scores of models")
            model_report=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
         

            best_model_name=max(model_report,key=lambda k: model_report[k])

            best_model=models[best_model_name]
            logging.info(f"Best model: {best_model_name} with accuracy: {model_report[best_model_name]}")

            if model_report[best_model_name]<0.6:
                raise CustomException("No best model found",sys)
            

            logging.info("Save the trained model")
            save_object(
                file_path= self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            
            logging.info("Find r2 score")
            y_test_pred=best_model.predict(X_test)
            test_score=r2_score(y_test,y_test_pred)
          
            return test_score
    

        except Exception as e:
            raise CustomException(e,sys)

    

