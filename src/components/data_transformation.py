# basic library
import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np
 
# preprocessing library
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

## pipelines
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# basic logging and exception
from src.logger import logging
from src.exception import CustomException

## picke obje making functions
from utils import save_obj


## Data Transformation config
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):

        try:
            logging.info('Data Transformation Started')

            # one hot encoding
            one_hot_tf=ColumnTransformer([
                ('ohe_sex',OneHotEncoder(sparse=False,handle_unknown='ignore',drop='first'),[1,4,5])]
                ,remainder='passthrough')
            
            # Scaling the data
            scale_tf=ColumnTransformer([('scale',StandardScaler(),slice(0,7))]) # sex =1,region =3, smo =1 =4  and other =3 total =7 
            
            # combining the pipe line
            preprocessor = Pipeline([
            ('ohe', one_hot_tf),
            ('scale_tf', scale_tf)
            ])

            return preprocessor

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):

        try:

            logging.info('Intiate Data Transformation')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        
            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            
            preprocessor_obj=self.get_data_transformation_obj()

            target_col='expenses'
            input_features_train_df = train_df.drop(target_col,axis=1)
            target_feature_train_df = train_df[target_col]
            
            input_features_test_df = test_df.drop(target_col,axis=1)
            target_feature_test_df = test_df[target_col]

            ## apply the transformation
            input_feature_train_arr =self.preprocessor_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=self.preprocessor_obj.transform(input_features_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_obj(
                file_path=self.data_transformation_config.processor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info('Processsor pickle in created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e,sys)