import pandas as pd
import numpy as np

from collections import Counter
import holidays
import sys


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline


# Join two dataframes
def merge_df(df1_path, df2_path, col='Store'):
    """
    Joins two dfs using left join
    on the "Store" column by default.
    """
    print(type(df1_path))
    if (type(df1_path)==str) and (type(df2_path)==str):
        df1 = pd.read_csv(df1_path)
        df2 = pd.read_csv(df2_path)
    
        merged_df = pd.merge(df1, 
                             df2, 
                             on=col, 
                             how='left')
    else:
        merged_df = pd.merge(df1_path, 
                             df2_path, 
                             on=col, 
                             how='left')
    return merged_df

###################################
#       Sanity checks
###################################

def closed_sales(df):
    """
    It checks whether the df has
    non-zero Sales although it is closed.
    Returns a df by excluding rows where
    Open == 0 and Sales != 0
    """
    not_open_but_nonzerosales = (df.Open == 0) & (df.Sales != 0)
    train = df.loc[~not_open_but_nonzerosales]
    assert train.shape[0] == df.shape[0] - not_open_but_nonzerosales.sum()
    return train


def remzero_sales(df, col='Sales'):
    """
    Takes as input a df
    And returns a df by
    removing the zero
    Sales rows
    """
    zero_sales = df[col] == 0
    return df.loc[~zero_sales]

###################################
#      Datetime conversion 
###################################

def convert_date(df, col='Date'):
    """
    Converts the datetime column
    """
    df.Date = pd.to_datetime(df[col])
    return df

###################################
# train test split
###################################
def train_test(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

    
    
    
    