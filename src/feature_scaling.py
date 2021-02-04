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

#####################################
# Scale features
#####################################

def feature_selector(cols):
    """
    Selects which columns to use
    for which scaling
    """
    minmax = ['DayOfWeek', 'WeekOfYear', 'DayOfMonth', 'Month', 
               'CompetitionOpenSinceYear', 'CompetitionOpenSincePeriod']

    rob = ['Store_menc', 'Customer_per_store', 
               'CompetitionDistance']

    std = ['Sales_per_customer']
    
    minmax_cols = []
    rob_cols = []
    std_cols = []
    
    for _, col in enumerate(cols):
        if col in minmax:
            minmax_cols.append(col)
        elif col in rob:
            rob_cols.append(col)
        elif col in std:
            std_cols.append(col)
        else:
            pass
    return minmax_cols, rob_cols, std_cols


def feature_transformer(minmax_cols, rob_cols, std_cols):
    """
    Returns a transformer object
    """
    minmax_scaler = MinMaxScaler()
    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()
    
    ct = ColumnTransformer([('minmax', 
                         minmax_scaler, 
                         minmax_cols
                        ), 
                        ('std', 
                          std_scaler, 
                          std_cols
                        ), 
                        ('rob', 
                         rob_scaler, 
                         rob_cols
                        )], 
                       remainder='passthrough')
    return ct
