import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## intialize the data ingestion configuration

@dataclass()
class DataIngestionconfig:
    train_data_path=os.path.join('artifacts','train.csv')  ## Basecally join the path and saving  the path directory in vairable
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')

## create a data ingestion class

class DataIngestion:
    def __init__ (self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            ## just basically reading the data from path
            df=pd.read_csv(os.path.join('notebook/data','insurance.csv'))
            logging.info('Dataset read as pandas Dataframe')

            ## ek direcory bannow & path ka nam     given below hoga 
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            # data to ko export karo es path  jo above create kiya
            df.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info('Train data and Test data goinig to Split')
            train_data,test_data=train_test_split(df,test_size=0.3,random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion Is completed')
            
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)


        except Exception as e:
            logging.info('Error Ocure In Ingestion Config')

