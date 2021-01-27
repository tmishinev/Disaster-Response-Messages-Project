import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.svm import  LinearSVC
import pickle
from custom.custom_tokens import tokenize, CustomUnpickler


def load_data(database_filepath):
    
    """
    Loads SQLite data base into pandas dataframe
           
    Args:
      database_path (String)
    Returns:
      X (dataframe): feature data 
      y (dataframe): label data 
      category_names (list) : label category names 
    """
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('DisasterMessageETL', con = engine)
    #child_alone has only 0 class
    df.drop(['child_alone'], inplace = True, axis = 1)

    y = df.iloc[:, 4:]

    category_names = y.columns
    X = df['message']
    
    return X, y, category_names
    



def build_model():
    
    """
    Build the model and returns pipeline object
           
    Args:
      None
    Returns:
      pipeline (sklearn pipeline object)
    """



    pipeline = Pipeline([
        
        ('vect', CountVectorizer(tokenizer = tokenize, ngram_range = (1,2))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC(class_weight = 'balanced')))
    ])

    


    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluation function, reports the f1 score, precision and recall for the positive label of each output category
    Input: Y_test, y_pred
           
    Args:
      model (scikit learn pipeline): name of model
      X_test (dataframe): testdata features
      Y_test (dataframe): multiout put testdata labels
      category_names (list): label data category names
    Returns:
      None
    """
    
    y_pred = model.predict(X_test)
 
    #Evaluation 
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall', 'accuracy', 'support_positive_class'])
    num = 0
    
    #prints positive labes score for each category
    for cat in category_names:
     
        precision, recall, f_score, support = precision_recall_fscore_support(Y_test.values[:, num], y_pred[:, num])
        results.at[num+1,'Category'] = cat        
        results.at[num+1,'precision'] =round(precision[1]*100,2)
        results.at[num+1,'recall'] = round(recall[1]*100,2)
        results.at[num+1,'f_score'] = round(f_score[1]*100,2)
        results.at[num+1,'accuracy'] =round(accuracy_score(Y_test.values[:, num], y_pred[:, num])*100,2)
        results.at[num+1,'support_positive_class'] = support[1]
        num += 1
       
    #grand total score
    results.at[num+1,'Category'] = 'Total Score'
    results.at[num+1,'precision'] = results['f_score'].mean()
    results.at[num+1,'recall'] = results['precision'].mean()
    results.at[num+1,'f_score'] = results['recall'].mean()
    results.at[num+1,'accuracy'] = results['accuracy'].mean()
    results.at[num+1,'support_positive_class'] = results['support_positive_class'].sum()
      
    return results


def save_model(model, model_filepath):
    
    '''saving the model after training and evaluation
    
    Args:
      model (scikit learn pipeline): name of model
      database_filepath (str): name of database containing data
    Returns:
      None
    '''
    #joblib.dump(model,  model_filepath)
    pickle.dump(model, open(str(model_filepath), 'wb'))


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
        
        print('Evaluating model...')
        print(evaluate_model(model, X_test, Y_test, category_names))

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python train_classifier.py data/DisasterResponse.db models/classifier.pkl')


if __name__ == '__main__':
    main()