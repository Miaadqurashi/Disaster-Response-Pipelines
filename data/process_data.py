import sys
import re
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads messages and categories data
    Inputs: messages_filepath , categories_filepath
    Outputs: messages: Pandas Dataframe read from given filepath
    categories: Pandas Dataframe read from given filepath
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages,categories


def clean_data(messages,categories):
    """
    Cleans data by converting categories to a binary matrix
    Inputs: messages and categoris DataFrame
    Outputs: df: Cleaned Dataframe combined
    """
    s = categories.categories[0]
    columns = re.compile("-\d;").split(s)
    columns[-1] = columns[-1][:-2] # last label has a number at the end.
    categories = pd.concat([categories.id,
                 categories.categories.str.split(';{0,1}\D*-',expand=True).loc[:,1:]],
                       axis=1)
    categories.columns = ['id']+ columns
    for column in categories:
        # convert to ints
        categories[column] = categories[column].astype(int)
    df = pd.merge(messages,categories,on='id')
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Saves cleaned DataFrame in a database file.
    Inputs: database_filename : filename to save the database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("disaster_responces", engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages,categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages,categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
