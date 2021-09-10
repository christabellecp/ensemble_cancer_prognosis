import pandas as pd 
import lazypredict
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from   lazypredict.Supervised        import LazyClassifier
from   category_encoders             import *
from   sklearn.compose               import *
from   sklearn.ensemble              import *
from   sklearn.linear_model          import *
from   sklearn.impute                import *
from   sklearn.metrics               import *
from   sklearn.pipeline              import *
from   sklearn.preprocessing         import *
from   sklearn.model_selection       import *
from   sklearn.neighbors             import *
from   sklearn.feature_selection     import *
from   sklearn.neural_network        import MLPRegressor
from   sklearn.calibration           import *
from   sklearn.discriminant_analysis import *
from   sklearn.model_selection       import cross_val_score

from   lightgbm                      import LGBMClassifier
from   sklearn.metrics               import log_loss
from   imblearn.over_sampling        import SMOTE


def lazy_classification(num_models, X_train, X_val, y_train, y_val):
    clf = LazyClassifier(predictions=True)
    models, _ = clf.fit(X_train, X_val, y_train, y_val)
    return models[:num_models]

def set_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

def plot_distribution(df):
    fig, ax = plt.subplots(3,7, figsize=(21,8))
    ax = ax.flatten()
    for pid in range(21):
        ax[pid].hist(df[df.problem_id == pid].target, color='steelblue')
        set_spines(ax[pid])
        ax[pid].set_title(f'Problem {pid}', fontname = 'Futura', fontsize=20)
        ax[pid].set_xticks([])
        ax[pid].set_yticks([])
    plt.tight_layout()

    
def get_partitions(n_class=None, val_size=.15, filter_out = None, specific_pid = None):
    """
    INPUT: n_class  - number of classes
           val_size - the test split desired
    
    OUTPUT: X - dataframe only consisting of a 'n_class' class problem 
            y - target values (should only contain 'n_class' values)
            test train splits (X_train, X_val, y_train, y_val) 
    """
    df = pd.read_csv('train_ml2_2021.csv')
    
    if specific_pid:
        df = df[df.problem_id == specific_pid]
        
        # remove target and problem id columns
        X = df.loc[:, df.columns != 'target']
        X = X.drop(columns = 'problem_id')
        
        y = df.target.values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, stratify=y, random_state=42)
        
    else:
        if filter_out:
            df = df[~df['problem_id'].isin(filter_out)]
        df = df[df.problem_id.isin(get_pids(df, n_class))]
        
        # remove target
        X = df.loc[:,df.columns != 'target']   
        
        # one hot encode the 'problem_id' column
        X = OneHotEncodeDF(X) 
        
        y = df.target.values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, stratify=y, random_state=42)
        
    
    return X, y, X_train, X_val, y_train, y_val




def tab_predict(pipe, X_train, y_train, X_val, y_val, name = 'Model'):
    """
    Prints train and val accuracy
    Returns validation predictions, feature importance, and fitted pipe
    """
    pipe.fit(X_train, y_train)
    val_preds = pipe.predict(X_val)
    train_preds = pipe.predict(X_train)

    print(f"{f'{name} Train Accuracy'}: {round(accuracy_score(y_train, train_preds),3)}")
    print(f"{f'{name} Valid Accuracy'}: {round(accuracy_score(y_val, val_preds),3)}")
    try:
        feat_importance = pipe.steps[0][1].feature_importances_
    except:
        feat_importance = None
  
    return val_preds, feat_importance, pipe



def get_test(X, y, pipe = None, pipes = None, n_class=None, filter_out = None, specific_pid = None):
    """
    INPUT: X - df to fit on
           y - targets to fit on
           pipe - classifier
           pipes - list of classifiers
           n_class - number of classes (if not predicting on a specific PID)
           filter_out - if classes, the list of PIDs to filter out
           specific_pid - PID to filter on
    
    OUTPUT: Dataframe of predictions
    """
    
    df = pd.read_csv('train_ml2_2021.csv')
    test = pd.read_csv('test0.csv', index_col='obs_id')
    
    if specific_pid:
        # filter the test df where problem id = 'specific_pid'
        df_test = test[test['problem_id']==specific_pid]
        X_test = df_test.loc[:,df_test.columns!='target']
        X_test = X_test.drop(columns='problem_id')
        predict_df = predict_helper(X, y, X_test, pipes, pipe)
    else:
        # filter the test df where number of classes is n_class
        test_df = test[test.problem_id.isin(get_pids(df, n_class))]
        if filter_out:
            test_df = test_df[~test_df['problem_id'].isin(filter_out)]
       
        # One Hot Encode the problem_id 
        test_df = OneHotEncodeDF(test_df)
        X_test = test_df.loc[:,test_df.columns != 'target']
        predict_df = predict_helper(X, y, X_test, pipes, pipe)
    return predict_df
    

def OneHotEncodeDF(df):
    df_without_pid = df.iloc[:, df.columns != 'problem_id']
    df_dummies_pid = pd.get_dummies(df.problem_id, prefix='p_id')
    df = pd.merge(df_without_pid, df_dummies_pid, left_index=True, right_index=True)
    return df

def get_pids(df, n_class):
    return [pid for pid in list(set(df.problem_id.values)) if len(df[df['problem_id']==pid].target.value_counts()) == n_class]


def predict_helper(X, y, X_test, pipes = None, pipe=None):
    if pipe:
        pipe.fit(X,y)
        X_test['target'] = pipe.predict(X_test).tolist()
    else:
        for pipe in pipes:
            pipe.fit(X,y)
        data = [pipe.predict(X_test).tolist() for pipe in pipes]
        X_test['target'] = list(map(max, *data))  
    return X_test[['target']]