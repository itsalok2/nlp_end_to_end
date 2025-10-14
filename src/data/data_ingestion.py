import numpy as np
import pandas as pd 
import os 
import yaml 
import logging 
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.connections import s3_connections
pd.set_option('future.no_silent_downcasting', True)
from dotenv import load_dotenv
load_dotenv()

access_key=os.getenv('ACCESS_KEY')
secret_key=os.getenv('SECRET_ACCESS_KEY')

def load_params(params_path: str)-> dict:
    try:
        with open(params_path,'r') as f:
            params=yaml.safe_load(f)
        logging.debug('parameters retrieved form %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('file not found: %s',params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('yaml error: %s',e)
        raise
    except Exception as e:
        logging.error('unexpected error: %s',e)
        raise

def load_data(data_url: str)-> pd.DataFrame:
    try:
        df=pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info('pre-processing')
        final_df=df[df['sentiment'].isin(['positive','negative'])]
        final_df['sentiment']=final_df['sentiment'].replace({'positive': 1, 'negative': 0})
        logging.info('Data preprocessing completed')
        return final_df
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data:pd.DataFrame, test_data: pd.DataFrame, data_path:str)->None:
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        test_size=0.2
        # df=load_data('https://raw.githubusercontent.com/itsalok2/nlp_end_to_end/refs/heads/main/data/raw/IMDB.csv')
        s3=s3_connections.s3_operations(bucket_name="nlp-end-to-end-data",aws_access_key=access_key,aws_secret_key=secret_key)
        df=s3.fetch_file_from_s3(file_key='IMDB.csv')
        final_df=preprocess_data(df)
        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_data=train_data,test_data=test_data,data_path='./data')
    except Exception as e:
        logging.error('failded to complete the data ingestion process: %s',e)
        print(f"error: {e}")

if __name__=='__main__':
    main()