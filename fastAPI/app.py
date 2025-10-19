import re
import time
import string
import mlflow
import pickle
import dagshub
import numpy as np
import pandas as pd 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse,Response
from prometheus_client import Counter,Histogram,generate_latest,CollectorRegistry,CONTENT_TYPE_LATEST

import warnings
warnings.simplefilter('ignore',UserWarning)
warnings.filterwarnings('ignore')

#----------------------------------for local use only-----------------------
dagshub.init(repo_owner='itsalok2', repo_name='nlp_end_to_end', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/itsalok2/nlp_end_to_end.mlflow")

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text

app=FastAPI()
templates=Jinja2Templates(directory='fastAPI/templates')

registry=CollectorRegistry()

REQUEST_COUNT= Counter('app_request_count','Total Number of requests to the app',
                       ['method','endpoint'],registry=registry)
REQUEST_LATENCY= Histogram('app_request_latency_seconds','Latency of requests in seconds',
                            ['endpoint'],registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Count of predictions for each class", 
                           ["prediction"], registry=registry)

model_name='my_model'

def get_latest_model_version(model_name):
    client=mlflow.MlflowClient()
    latest_version=client.get_latest_versions(model_name,stages=['staging'])
    if not latest_version:
        latest_version=client.get_latest_versions(model_name,stages=['None'])
    return latest_version[0].version if latest_version else None

model_version=get_latest_model_version(model_name)
model_uri=f'models:/{model_name}/{model_version}'
print(f"fetching model from: {model_uri}")

model=mlflow.pyfunc.load_model(model_uri=model_uri)
vectorizer=pickle.load(open('models/vectorizer.pkl','rb'))

@app.get("/",response_class=HTMLResponse)
def home(request:Request):
    REQUEST_COUNT.labels(method='GET',endpoint='/').inc()
    start_time=time.time()
    response=templates.TemplateResponse('index.html',{"request": request, "result": None})
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.post('/predict',response_class=HTMLResponse)
def predict(request:Request, text: str= Form(...)):
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    start_time=time.time()

    text = normalize_text(text)
    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray(),columns=[str(i) for i in range(features.shape[1])])

    result = model.predict(features_df)
    prediction = result[0]
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return templates.TemplateResponse('index.html',{'request':request, 'result':prediction})

@app.get('/metrics')
async def metrics():
    """Expose only custom Prometheus metrics."""
    return Response(content=generate_latest(registry),media_type = CONTENT_TYPE_LATEST)

if __name__=='__main__':
    import uvicorn
    uvicorn.run("fastAPI.app:app", host="0.0.0.0", port=5000, reload=True)
    