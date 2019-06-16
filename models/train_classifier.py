import sys
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from build_model import build_model
import pickle
import re

def load_data(database_filepath):
    """
    Load the database 
    I/P : database_filepath : filepath to database
    O/P : X : numpy ndarray for the messages
          Y : numpy ndarray for the category output in binary mode 36 columns
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql(f'SELECT * FROM disaster_responces',engine)
    X = df.message.values
    Y = df.iloc[:,4:].values
    return X,Y,df.iloc[:,4:].columns

## Tokenize and build model are in a separate file to avoid pickling error


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates model performance
    I/P : model: model that classifies
          X_test: test data
          Y_test: test labels
          category_names: list of category names
    '''
    y_test_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        print(f'{category_names[i]} Category:')
        print(classification_report(Y_test[:,i],y_test_pred[:,i]))


def save_model(model, model_filepath):
    '''
    Saves the model with pickle in specified model_filepath
    I/P : model : Sklearn classifier model
          model_filepath : path to save the pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print("***********BEST PARAMETERS*********")
        print(model.best_params_)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
