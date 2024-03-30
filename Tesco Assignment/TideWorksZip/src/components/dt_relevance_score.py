import os 
import sys 

from src.utils.exception import customException
from src.utils.logger import logging 
from src.utils.utils import saveObject

from src.components.data_ingestion import DataIngestion

import numpy as np 
import pandas as pd
from itertools import chain

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

from dataclasses import dataclass 
from typing import List





@dataclass
class DataTransformationConfig():
    preprocessor_pipeline_file_path = os.path.join('artifacts','preprocess_pipeline.pkl')

class DataTransformation():
    def __init__(self
                ,targetColumn:str
                ,corrCols:List[str]
                ,removeCols:List[str]
                ,catCols:List[str]
                ,numCols:List[str]
                ,train_path:str
                ,test_path:str
                ):
        self.preprocessorPipeConfig = DataTransformationConfig()
        self.targetColumn = targetColumn
        self.corrCols = corrCols
        self.removeCols = removeCols
        self.catCols = catCols
        self.numCols = numCols
        self.train_path = train_path
        self.test_path = test_path

    def getPreprocessorPipeline(self):
        """
        This function has the logic to built the required preprocessor pipeline for both numerical and categorical.
        """
        try: 


            logging.info('Numerical Columns :' + str(self.numCols))
            logging.info('Categorical Columns :' + str(self.catCols))

            numPipe = Pipeline(
                steps =[
                    
                ]
            )

            catPipe = Pipeline(
                steps =[
                    
                ]
            )

            preprocessorPipe = ColumnTransformer(
                [
                ],
                remainder='passthrough'
            )

            return preprocessorPipe
    
        except Exception as e: 
            raise customException(e,sys)


    def runPreprocessor(self):
        try:
            logging.info("Loading the train and test datasets")
            train_df = pd.read_pickle(self.train_path)
            test_df = pd.read_pickle(self.test_path)

            
            trainX = train_df.drop(columns = list(chain([self.targetColumn],self.removeCols,self.corrCols)) , axis=1)
            trainY = train_df[self.targetColumn]

            testX = test_df.drop(columns = list(chain([self.targetColumn],self.removeCols,self.corrCols)), axis=1)
            testY = test_df[self.targetColumn]

            logging.info("Preprocessing started")
            preprocessorPipe = self.getPreprocessorPipeline()

            trainX_arr = preprocessorPipe.fit_transform(trainX)
            testX_arr = preprocessorPipe.transform(testX)

            train_arr = np.c_[trainX_arr,np.array(trainY)]
            test_arr = np.c_[testX_arr,np.array(testY)]

            logging.info("Preprocessing completed")

            saveObject (
                file_path = self.preprocessorPipeConfig.preprocessor_pipeline_file_path,
                obj = preprocessorPipe
            )

            return (
                train_arr,
                test_arr,
                self.preprocessorPipeConfig.preprocessor_pipeline_file_path
            )


        except Exception as e:
            raise customException(e,sys)

if __name__ == "__main__":
    di = DataIngestion()
    train_path,test_path = di.runDataIngestion()

    SEED = 42
    TARGET_COLUMN = 'matched'
    CORRELATED_DROP_COLUMNS = ['DifferentPredictedTime','DifferentPredictedDate']
    REMOVE_COLUMNS = ['receipt_id','company_id','matched_transaction_id' ,'feature_transaction_id']
    CAT_COLUMNS = ['DifferentPredictedTime','TimeMappingMatch','ShortNameMatch','DifferentPredictedDate','PredictedTimeCloseMatch']
    NUM_COLUMNS = ['DateMappingMatch', 'AmountMappingMatch', 'DescriptionMatch',  'PredictedNameMatch', 'PredictedAmountMatch']

    dt = DataTransformation(
                             TARGET_COLUMN
                            ,CORRELATED_DROP_COLUMNS
                            ,REMOVE_COLUMNS
                            ,CAT_COLUMNS
                            ,NUM_COLUMNS
                            ,train_path
                            ,test_path
                            )
    train_arr,test_arr,prePipePath = dt.runPreprocessor()









