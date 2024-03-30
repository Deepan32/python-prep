import os 
import sys
import pandas as pd

from src.utils.exception import customException
from src.utils.utils import loadObject
from src.utils.logger import logging 


class TrainPipeline:
    def __init__(self) -> None:
        pass
    
    def train(self,features):
        try:
            pass
        except Exception as e:
            raise customException(e,sys)