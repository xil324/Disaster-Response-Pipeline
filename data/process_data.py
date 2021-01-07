#!/usr/bin/env python
# coding: utf-8

# Two datasets used in this project: 
# 1.message.csv: disaster message
# 2.categories.csv: message categories

# In[4]:


# import libraries
import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine


# In[5]:


def load_data(messages_filepath, categories_filepath):
    '''
    input:
    message dataset contains the disaster message
    category dataset contain the categories of disaster message

    output: a merged dataset 
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge the two datasets on common id
    df = messages.merge(categories, how='left',on=['id'])
    df = df.dropna()
    return df
 
# In[6]:


#Split the values in the categories column on the ; character so that each value becomes a separate column. 
def clean_data(df):
    '''
    input: the merged dataset
    output: a dataset with addtiional 36 columns for 36 categoreis; duplicates removed
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand=True)
    #get the first row of this datase to create column name of the dataframe
    row = categories.iloc[1]
    category_colnames = row.str.split("-",expand=True)[0]
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert categories name to numeric value (0 or 1)
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    # convert column from string to numeric
        categories[column] =pd.to_numeric(categories[column])
   #replace old 'categories' column with the new categories dataframe
    df= df.drop(['categories'],axis=1)
    #concat two dataframes together 
    df = pd.concat([df,categories],axis=1)
    #remove duplicates
    # check number of duplicates
    print('number of duplicates:{}'.format(df.duplicated().sum()))
    # drop duplicates
    df=df.drop_duplicates()
    return df


# In[7]:


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('df_new', engine,if_exists = 'replace', index=False) 


# In[14]:



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        df = clean_data(df)
        
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



