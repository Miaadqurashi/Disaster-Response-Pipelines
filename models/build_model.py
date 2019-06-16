import sys
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import nltk
nltk.download(['stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
import re

def tokenize(text):
    """
    This function cleans and tokenize the text.
    Inputs: text: text to be cleaned and tokenized.
    Outputs: words: list of tokenized words.
    """
    ## First: Normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    ## Second: Tokenize
    words = word_tokenize(text)
    ## Third: Remove Stop Words
    words = [w for w in words if w not in stopwords.words("english")]
    ## Fourth: Lemmatize
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    ## Fifth: Stem
    words = [PorterStemmer().stem(w) for w in words]
    return words
    
def build_model():
    """
    Builds classification model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10),n_jobs=1))])
    parameters = {
    'tfidf__norm': ('l1', 'l2',None),
    "clf__estimator__n_estimators" : [10,50,100]
    }
    cv = GridSearchCV(cv = 3, estimator=pipeline, param_grid=parameters, n_jobs=-1,verbose=3)
    return cv

