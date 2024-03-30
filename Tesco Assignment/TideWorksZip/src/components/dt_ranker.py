import os 
import sys 

from src.utils.exception import customException
from src.utils.logger import logging 
from src.utils.utils import saveObject

from src.components.data_ingestion import DataIngestion
from src.components.dt_relevance_score import DataTransformation
from src.components.model_trainer_relevance_score import ModelTrainer

import numpy as np 
import pandas as pd
from itertools import chain

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier

from dataclasses import dataclass 
from typing import List





@dataclass
class RankerDataTransformationConfig():
    preprocessor_pipeline_file_path = os.path.join('artifacts','ranker_preprocess_pipeline.pkl')

class RankerDataTransformation():
    def __init__(self
                ,corrCols:List[str]
                ,removeCols:List[str]
                ,catCols:List[str]
                ,numCols:List[str]
                ,ranker_train_path:str
                ,ranker_test_path:str
                ):
        self.preprocessorPipeConfig = RankerDataTransformationConfig()
        self.corrCols = corrCols
        self.removeCols = removeCols
        self.catCols = catCols
        self.numCols = numCols
        self.train_path = ranker_train_path
        self.test_path = ranker_test_path

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

            train_df.sort_values(by=['receipt_id', 'relevance_score'], ascending=[True, True], inplace=True)
            # Higher the rank Higher the relevance
            train_df['relevance_rank'] = train_df.groupby('receipt_id')['relevance_score'].rank(method='dense', ascending=True)
            train_df['relevance_rank'] = train_df['relevance_rank'].astype(int)
            
            test_df.sort_values(by=['receipt_id', 'relevance_score'], ascending=[True, True], inplace=True)

            test_df['relevance_rank'] = test_df.groupby('receipt_id')['relevance_score'].rank(method='dense', ascending=True) 
            
            test_df['relevance_rank'] = test_df['relevance_rank'].astype(int)

            trainX = train_df.drop(columns = list(chain(['relevance_score','relevance_rank'],self.removeCols,self.corrCols)) , axis=1)
            trainY = train_df['relevance_rank']

            testX = test_df.drop(columns = list(chain(['relevance_score','relevance_rank'],self.removeCols,self.corrCols)), axis=1)
            testY = test_df['relevance_rank']

            qids_train =train_df.groupby("receipt_id")["receipt_id"].count().to_numpy()
            qids_test =test_df.groupby("receipt_id")["receipt_id"].count().to_numpy()

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
                qids_train,
                qids_test,
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
    
    mt = ModelTrainer(train_arr,test_arr,train_path,test_path)
    aucScore,ranker_train_path,ranker_test_path = mt.getBestModel()
    print(aucScore)
    print(ranker_train_path)

    dtr = RankerDataTransformation(
                             CORRELATED_DROP_COLUMNS
                            ,REMOVE_COLUMNS
                            ,CAT_COLUMNS
                            ,NUM_COLUMNS
                            ,ranker_train_path
                            ,ranker_test_path)
    

    train_arr,test_arr,qids_train,qids_test,prePipePathRanker = dtr.runPreprocessor()
    print(train_arr[1:5])









