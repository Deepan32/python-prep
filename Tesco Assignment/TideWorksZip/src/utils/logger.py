import logging
import os 
from datetime import datetime

LOG_FILE_FOLDER = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

LOG_PATH = os.path.join(os.getcwd(),"Logs",LOG_FILE_FOLDER)
os.makedirs(LOG_PATH,exist_ok=True)

LOG_FILE_PATH=os.path.join(LOG_PATH,LOG_FILE)

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)

if __name__ == '__main__':
    logging.info('Logging from main')