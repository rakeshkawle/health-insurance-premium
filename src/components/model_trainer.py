# Basic Library
import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass


# Models
from sklearn.linear_model import LinearRegression,ElasticNet,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor

# Basic logging exception
from src.logger import logging
from src.exception import CustomException

# utils
from src.utils import save_obj
from src.utils import model_evaluation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train,y_train,X_test,y_test=(
            train_array[:,:-1],
            train_array[:,-1],
            test_array[:,:-1],
            test_array[:,-1],
            )

            models = {
            'linearregression': LinearRegression(),
            'lasso': Lasso(),
            'ridge': Ridge(),
            'elasticnet': ElasticNet(),
            'decisiontree': DecisionTreeRegressor(),
            'randomforest': RandomForestRegressor(),
            'gradientboosting' :GradientBoostingRegressor(),
            'adaboostregressor' :AdaBoostRegressor()
            }

            model_report:dict=model_evaluation(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('='*50)
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                )
            

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
