import sys 
import logging 
from src.utils.logger import logging

def getErrorMessage(errorMessage,errorDetails:sys):
    _,_,exc_tb = errorDetails.exc_info()
    fileName = exc_tb.tb_frame.f_code.co_filename
    lineNo = exc_tb.tb_lineno
    return (f"\n\nCustom Message: \nError occured in the {fileName} file at Line Number : {str(lineNo)} and additional details : {str(errorDetails)}\n\n")

class customException(Exception):
    def __init__(self,errorMessage,errorDetails:sys):
        super().__init__(errorMessage)
        self.errorMessage = getErrorMessage(errorMessage,errorDetails)
    
    def __str__(self):
        return self.errorMessage

if __name__ == "__main__":
    try:
        val = 1/0
    except Exception as e:
        logging.info("Divided by Zero Error")
        raise customException(e,sys)

