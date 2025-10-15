import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from src.logger import logging

dagshub.init(repo_owner='itsalok2', repo_name='nlp_end_to_end', mlflow=True)

def load_model(file_path):
    try:
        with open(file_path,'rb') as file:
            model=pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path)-> pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf,x_test: np.ndarray, y_test: np.ndarray)->dict:
    try:
        y_pred=clf.predict(x_test)
        y_pred_proba=clf.predict_proba(x_test)[:,1]
        accuracy=accuracy_score(y_test,y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict ,file_path):
    try:
        with open(file_path,'w') as f:
            json.dump(metrics,f,indent=4)
        logging.info('metrics saved to %s',file_path)
    except Exception as e:
        logging.info('error occured while saving the metrics: %s',e)
        raise

def save_model_info(run_id:str ,model_path, file_path):
    try:
        model_info={'run_id': run_id,'model_path':model_path}
        with open(file_path,'w') as f:
            json.dump(model_info,f,indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    mlflow.set_experiment('my-dvc-pipeline')
    with mlflow.start_run() as run:
        try:
            clf=load_model('./models/model.pkl')
            test_data=load_data('./data/processed/test_tfidf.csv')
            x_test=test_data.iloc[:,:-1].values
            y_test=test_data.iloc[:,-1].values

            metrics=evaluate_model(clf,x_test,y_test)

            save_metrics(metrics=metrics,file_path='reports/metrics.json')

            for metric_name,metric_value in metrics.items():
                mlflow.log_metric(metric_name,metric_value)
            
            if hasattr(clf,'get_params'):
                params=clf.get_params()
                for param_name,param_value in params.items():
                    mlflow.log_param(param_name,param_value)
            
            mlflow.sklearn.log_model(clf,'model',input_example=x_test[:3])
            save_model_info(run.info.run_id,'model','reports/experiment_info.json')
            mlflow.log_artifact('reports/metrics.json')
        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()