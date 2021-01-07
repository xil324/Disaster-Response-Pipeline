
import sys
import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
import nltk
import re
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import type_of_target
import pickle 
nltk.download('punkt')
nltk.download('wordnet')



def load_data(database_filepath):
    '''
    input: datafile path
    output: x: disaster message
            y: message category

    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('df_new', con=engine)
    df = df.dropna()
    X=df['message'].astype(str)
    y=df.iloc[:,4:40]
    return X,y



def tokenize(text):
    '''
    not really used in building a model but this funciton tokenize the message and lemmatize the words
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    output: a mutiouptut model used to predict message category
    '''
    pipeline = Pipeline(
    [('vect', CountVectorizer(tokenizer = tokenize)), 
        ('tfidf', TfidfTransformer()), 
        ('clf',MultiOutputClassifier(KNeighborsClassifier())) 
    ])
    
    parameters={
        'clf__estimator__n_neighbors':[3,4,5]
    }
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


# In[22]:


def evaluate_model(model, X_test, y_test):
    '''
    output: classification report for each category
    '''
    y_pred = model.predict(X_test)
    for col in range(y_test.shape[1]): 
        cl = classification_report(y_test.iloc[:,col],y_pred[:,col])
        print(cl)


# In[ ]:


def save_model(model, model_filepath):
      pickle.dump(model, open(model_filepath, 'wb'))





def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '              'as the first argument and the filepath of the pickle file to '              'save the model to as the second argument. \n\nExample: python '              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
     main()
