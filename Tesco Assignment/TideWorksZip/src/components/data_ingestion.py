import os 
import sys

from src.utils.exception import customException
from src.utils.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split,GroupShuffleSplit


from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str =  os.path.join('artifacts','raw_data.pkl')
    train_data_path: str =  os.path.join('artifacts','train_data.pkl')
    test_data_path: str = os.path.join('artifacts','test_data.pkl')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info("Data Ingestion Config created")
    
    def runDataIngestion(self):
        logging.info('Data Ingestion Starts')
        try:
            #Logic for db connection and extraction can be written here
            df = pd.read_pickle('Notebooks/data/processed_data.pkl')

            logging.info('Creating the artifacts folder')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok = True)

            df.to_pickle(self.ingestion_config.raw_data_path)

            logging.info('Initiating the train and test split')


            #Ensures all txns related to receipt belong to either train or test
            gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
            train_idx, test_idx = next(gss.split(df, groups=df['receipt_id']))

            train_df = df.iloc[train_idx,:]
            test_df =df.iloc[test_idx,:]


            logging.info('Writing the train and test datasets')
            train_df.to_pickle(self.ingestion_config.train_data_path )
            test_df.to_pickle(self.ingestion_config.test_data_path)

            logging.info("Data Ingestion Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
            

        except Exception as e:
            raise customException(e,sys)



if __name__ == "__main__":
    logging.info('Executing the DataIngestion Test from Main')
    di =  DataIngestion()
    train_path,test_path = di.runDataIngestion()







