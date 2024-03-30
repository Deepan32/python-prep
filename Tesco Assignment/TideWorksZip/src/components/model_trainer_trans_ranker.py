import os 
import sys 

import numpy as np
import pandas as pd 
from src.utils.exception import customException
from src.utils.logger import logging
from src.utils.utils import saveObject,calculate_mrr
import warnings
warnings.filterwarnings("ignore")


from sklearn.metrics import ndcg_score
import lightgbm as lgb

from src.components.data_ingestion import DataIngestion
from src.components.dt_relevance_score import DataTransformation
from src.components.model_trainer_relevance_score import ModelTrainer
from src.components.dt_ranker import RankerDataTransformation


from dataclasses import dataclass
from collections import defaultdict



@dataclass
class RankerModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model_trans_ranker.pkl")
    

class RankerModelTrainer:
    def __init__(self,train_arr,test_arr,qids_train,qids_test,ranker_train_path,ranker_test_path):
        self.model_trainer_config=RankerModelTrainerConfig()
        self.train_arr = train_arr
        self.test_arr = test_arr
        self.qids_train = qids_train
        self.qids_test = qids_test
        self.train_path = ranker_train_path
        self.test_path = ranker_test_path


    def runModelTrainer(self):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                self. train_arr[:,:-1],
                self.train_arr[:,-1],
                self.test_arr[:,:-1],
                self.test_arr[:,-1]
            )

            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            ranker = lgb.LGBMRanker(
                    objective="lambdarank",
                    boosting_type = "gbdt",
                    n_estimators = 5,
                    importance_type = "gain",
                    metric= "ndcg",
                    num_leaves = 10,
                    learning_rate = 0.05,
                    max_depth = -1,
                    label_gain =[i for i in range(max(y_train.max(), y_test.max()) + 1)])

            # Training the model
            ranker.fit(
                  X=X_train,
                  y=y_train,
                  group=self.qids_train,
                  eval_set=[(X_train, y_train),(X_test, y_test)],
                  eval_group=[self.qids_train, self.qids_test],
                  eval_at=[4, 8])


            y_pred = ranker.predict(X_test)
            ndcg = ndcg_score(np.array(y_test).reshape(1, -1), y_pred.reshape(1,-1))
            print(f"NDCG Score for the ranker: {ndcg}")

            test_df = pd.read_pickle(self.test_path)

            #Calculating the rank based on the score for actual
            test_df.sort_values(by=['receipt_id', 'relevance_score'], ascending=[True, True], inplace=True)
            test_df['relevance_rank'] = test_df.groupby('receipt_id')['relevance_score'].rank(method='dense', ascending=True)
            test_df['relevance_rank'] = test_df['relevance_rank'].astype(int)

            #Calculating the rank based on the predicted score
            test_df.loc[:,'pred_rs'] = y_pred
            test_df.sort_values(by=['receipt_id', 'pred_rs'], ascending=[True,True],inplace=True)
            test_df['pred_relevance_rank'] = test_df.groupby('receipt_id')['pred_rs'].rank(method='dense', ascending=True)
            test_df['pred_relevance_rank'] = test_df['pred_relevance_rank'].astype(int)
            
            mrr = calculate_mrr(test_df)
            print(f"Mean Reciprocal Rank (MRR): {mrr}")
        
        
            saveObject(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=ranker
            )

            return test_df
        
        except Exception as e:
            raise customException(e,sys)

    

if __name__ =='__main__':
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

    ranker = RankerModelTrainer(
         train_arr
        ,test_arr
        ,qids_train 
        ,qids_test 
        ,ranker_train_path 
        ,ranker_test_path
    )
    
    test_df = ranker.runModelTrainer()

