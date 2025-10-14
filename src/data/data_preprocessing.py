import numpy as np 
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging


def preprocess_dataframe(df,col='text'):
    """
    Preprocess a DataFrame by applying text preprocessing to a specific column.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        col (str): The name of the column containing text.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    lemm=WordNetLemmatizer()
    sw=set(stopwords.words('english'))

    def preprocess_text(text):
        """Helper function to preprocess a single text string."""
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove numbers
        text = ''.join([char for char in text if not char.isdigit()])
        # Convert to lowercase
        text = text.lower()
        # Remove punctuations
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text).strip()
        # Remove stop words
        text = " ".join([word for word in text.split() if word not in sw])
        # Lemmatization
        text = " ".join([lemm.lemmatize(word) for word in text.split()])
        return text
    
    df[col]=df[col].apply(preprocess_text)
    df=df.dropna(subset=[col])

    logging.info('data preprocessing completed')
    return df

def main():
    try:
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')

        train_processed_data=preprocess_dataframe(train_data,'review')
        test_processed_data=preprocess_dataframe(test_data,'review')

        data_path=os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'))
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'))

        logging.info('processed data saved to %s',data_path)
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()