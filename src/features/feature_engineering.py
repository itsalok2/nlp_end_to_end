import numpy as np 
import pandas as pd 
import os 
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
from src.logger import logging 
import pickle

def load_params(params_path: str)-> dict:
    try:
        with open(params_path,'r') as f:
            params=yaml.safe_load(f)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise

def load_data(file_path:str)-> pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logging.info('data loaded and NaNs filled from %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_tfidf(train_data: pd.DataFrame,test_data: pd.DataFrame,max_features:int)->tuple:
    try:
        logging.info('applying tfidf')
        vec=TfidfVectorizer(max_features=max_features)

        x_train=train_data['review'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['review'].values
        y_test = test_data['sentiment'].values
        
        x_train_tfidf=vec.fit_transform(x_train)
        X_test_tfidf=vec.transform(X_test)

        train_df=pd.DataFrame(x_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df=pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test

        pickle.dump(vec,open('models/vectorizer.pkl','wb'))
        logging.info('Tfidf applied and data transformed')

        return train_df, test_df
    except Exception as e:
        logging.error('Error during Tfidf transformation: %s', e)
        raise

def save_data(df:pd.DataFrame,file_path):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logging.info('data saved to %s',file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params=load_params('params.yaml')
        max_features=params['feature_engineering']['max_features']

        train_data=load_data('./data/interim/train_processed.csv')
        test_data=load_data('./data/interim/test_processed.csv')

        train_df,test_df=apply_tfidf(train_data,test_data,max_features)
        
        save_data(train_df,os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df,os.path.join("./data", "processed", "test_tfidf.csv"))

    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()