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

####################################
# Remove outliers based on percentile
####################################

def drop_outlier(df, col='Sales', per=99):
    pct = np.percentile(df.loc[:, col], 99)
    df = df.loc[df[col] < pct]
    return df

#####################################
# Outliers in years of competitor
#####################################
def comp_outliers(df, yr='CompetitionOpenSinceYear'):
    mask_1990 = df[yr] < 1990
    df = df[~mask_1990]
    return df
