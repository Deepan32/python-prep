import os 
import sys
import pandas as pd

from src.utils.exception import customException
from src.utils.utils import loadObject
from src.utils.logger import logging 


class PredictPipeline:
    def __init__(self) -> None:
        pass
    
    def predict(self,features):
        try:
            pass
        except Exception as e:
            raise customException(e,sys)
        

    

class CustomData:
    '''
    Transform the user data into dataframe
    '''
    def __init__(self) -> None:
        pass
    def get_data_as_data_frame(self):
        pass
