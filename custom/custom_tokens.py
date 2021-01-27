import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import pickle
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import re
import pandas as pd
from collections import Counter



def tokenize(text):
    """
    Tokenize 
    Input: Raw text
           
    Output: Lemmatized texts
    """

    text = re.sub(r'http\S+', '', text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if tok not in stop_words]

    return clean_tokens

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'tokenize':
            from custom.custom_tokens import tokenize
            return tokenize
        return super().find_class(module, name)

def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

def top_words(df):
    # Get all the word tokens in dataframe for Disaster and Non-Disaster
    # - remove url, tokenize tweet into words, lowercase words
    stop = stopwords.words('english')
    corpus0 = [] 
    [corpus0.append(word.lower()) for tweet in df['message'] for word in word_tokenize(remove_url(tweet))]
    corpus0 = list(filter(lambda x: x not in stop, corpus0)) # use filter to unselect stopwords

    # Create df for word counts to use sns plots
    a = Counter(corpus0).most_common()
    df_words = pd.DataFrame(a, columns=['Word','Count'])

    return df_words