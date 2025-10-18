import json 
import mlflow 
import logging 
from src.logger import logging 
import os 
import dagshub 

import warnings 
warnings.simplefilter('ignore',UserWarning)
warnings.filterwarnings('ignore')

dagshub.init(repo_owner='itsalok2', repo_name='nlp_end_to_end', mlflow=True)

def load_model_info(file_path)->dict:
    try:
        with open(file_path,'r') as f:
            model_info=json.load(f)
        logging.debug('model info loaded from: %s',file_path)
        return model_info 
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name,model_info):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri=f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version=mlflow.register_model(model_uri,model_name)
        client=mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='staging'
        )
        logging.debug(f'model {model_name} version {model_version.version} registerd and transitioned to staging.')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path='reports/experiment_info.json'
        model_info=load_model_info(model_info_path)
        model_name='my_model'
        register_model(model_name,model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()
