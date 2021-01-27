import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the raw files and returns pandas dataframes
           
    Args:
      messages_filepath (str) : path to message .csv file
      categories_filepath (str) : path to categories .csv file
    Returns:
      messages (dataframe): messages dataframe
      categories (dataframe): categories dataframe
    """

    #loads messages and categories datasets
    messages = pd.read_csv("./" + messages_filepath)
    categories = pd.read_csv("./" + categories_filepath)
    
    return messages, categories
    


def clean_data(messages, categories):
    
    """
    Cleans and merges the message and categories dataframes and returns processed data.
           
    Args:
      messages (dataframe): messages dataframe
      categories (dataframe): categories dataframe
    Returns:
      df (dataframe) : cleaned and merged data for classification
    """
    
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[1, :]
    category_colnames = row.map(lambda row: row[0:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    #concat the original message dataframe with the transformed categories
    df = pd.concat([messages, categories], axis = 1)
    
    #filter out invalid categories
    select_data = ((df.iloc[:,4:].values > 1).sum(axis=1) == 0).astype('bool')
    df = df[select_data]
    
    #drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df


def save_data(df, database_filename):
    """
    Loads the raw files and returns pandas dataframes
           
    Args:
      messages (dataframe): messages dataframe
      categories (dataframe): categories dataframe
    Returns:
      df (dataframe) : cleaned and merged data for classification
    """
    engine = create_engine('sqlite:///' + database_filename)
    
    #Drop table if already exists and save the new data
    connection = engine.raw_connection()
    cursor = connection.cursor()
    command = "DROP TABLE IF EXISTS {};".format("DisasterMessageETL")
    cursor.execute(command)
    connection.commit()
    df.to_sql('DisasterMessageETL', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv DisasterResponse.db')
              


if __name__ == '__main__':
    main()