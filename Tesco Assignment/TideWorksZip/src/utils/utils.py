import os
import sys

import numpy as np 
import pandas as pd

import dill
import pickle

from src.utils.exception import customException
from src.utils.logger import logging 

from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score


def saveObject(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customException(e, sys)
    

def evaluateModel(true, predicted):
    try:
        accuracy = accuracy_score(true, predicted)
        precision = precision_score(true, predicted)
        recall = recall_score(true, predicted)
        f1score = f1_score(true,predicted)
        auc = roc_auc_score(true, predicted)

        return accuracy,precision,recall,f1score,auc
    
    except Exception as e:
        raise customException(e, sys)
    
def loadObject(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise customException(e, sys)
    
def calculate_mrr(df):
    try:
        mrr_scores = []

        for receipt_id in df['receipt_id'].unique():
            # Filter DataFrame by receipt_id
            receipt_df = df[df['receipt_id'] == receipt_id]

            # Get the actual rank of the top-predicted transaction
            top_predicted = receipt_df.loc[receipt_df['pred_relevance_rank'].idxmax()]
            actual_rank_of_top_predicted = top_predicted['relevance_rank']

            # Calculate the reciprocal rank
            reciprocal_rank = 1.0 / actual_rank_of_top_predicted
            mrr_scores.append(reciprocal_rank)

        # Compute the MRR across all receipts
        mrr = sum(mrr_scores) / len(mrr_scores)
        return mrr
    except Exception as e:
        raise customException(e,sys)

