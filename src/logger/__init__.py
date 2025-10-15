import logging
import os 
from logging.handlers import RotatingFileHandler 
from datetime import datetime
import sys
from from_root import from_root

log_dir='logs'
log_file=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
max_log_size=5*1024*1024
backup_count=3

log_dir_path=os.path.join(from_root(),log_dir)
os.makedirs(log_dir_path,exist_ok=True)
log_file_path=os.path.join(log_dir,log_file)

def configure_logger():
    """
    Configures logging with a rotating file handler and a console handler.
    """
    # Create a custom logger
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # define formatter
    formatter=logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s")

    # file handler with rotation
    file_handler=RotatingFileHandler(log_file_path,maxBytes=max_log_size,backupCount=backup_count,encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler=logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

configure_logger()