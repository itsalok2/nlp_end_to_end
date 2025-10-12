import os
import pandas as pd 
import numpy as np 
import mlflow 
import optuna
import mlflow.sklearn 
import dagshub 
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import warnings
warnings.simplefilter('ignore',UserWarning)
warnings.filterwarnings('ignore')

dagshub.init(repo_owner='itsalok2', repo_name='nlp_end_to_end', mlflow=True)
load_dotenv

mlflow_tracking_uri=os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("LoR Hyperparameter Tuning")

def load_and_prepare_data(filepath):
    df=pd.read_csv(filepath)
    x=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values

    return train_test_split(x,y,test_size=0.2,random_state=42)

def train_and_log_model(x_train, x_test, y_train, y_test):
    """Trains a Logistic Regression model with Optuna and logs results to MLflow."""

    def objective(trial):
        C=trial.suggest_uniform('C',0.001,100)
        penalty=trial.suggest_categorical('penalty',['l1','l2'])
        solver='liblinear'

        model=LogisticRegression(C=C,penalty=penalty,solver=solver)     
        scores=cross_val_score(model,x_train,y_train,cv=5,scoring='f1')
        return np.mean(scores)
    
    with mlflow.start_run():
        study=optuna.create_study(direction='maximize')
        study.optimize(objective,n_trials=10,n_jobs=-1)

        for trial in study.trials:
            with mlflow.start_run(run_name=f'LR with params: {trial.params}',nested=True):
                model=LogisticRegression(**trial.params,solver='liblinear')
                model.fit(x_train,y_train)
                y_pred=model.predict(x_test)

                metrics={
                    'accuracy':accuracy_score(y_test,y_pred),
                    'precision':precision_score(y_test,y_pred),
                    'recall':recall_score(y_test,y_pred),
                    'f1_score':f1_score(y_test,y_pred),
                    'mean_cv_score':trial.value
                }

                mlflow.log_params(trial.params)
                mlflow.log_metrics(metrics)

                print(f"params: {trial.params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        # log best model
        best_params=study.best_params
        best_model=LogisticRegression(**best_params,solver='liblinear',max_iter=1000)
        best_model.fit(x_train,y_train)
        best_f1=study.best_value

        mlflow.log_params(best_params)
        mlflow.log_metric('best_f1_score',best_f1)
        mlflow.sklearn.log_model(best_model,'model',input_example=x_train[:4])


if __name__ == "__main__":
    path='/home/alok_kumar/kubernetes/nlp_end_to_end/data/processed/embedded_data/embedded_dataframe.csv'
    (X_train, X_test, y_train, y_test) = load_and_prepare_data(path)
    train_and_log_model(X_train, X_test, y_train, y_test)
