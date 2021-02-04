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


##################################
# Extract datetime features
##################################

def date_converter(df, col='Date'):
    """
    Extract datetime features
    """
    df['Month'] = df[col].dt.month
    df['DayOfMonth'] = df[col].dt.day
    df['Year'] = df[col].dt.year
    df['DayOfWeek'] = df[col].dt.dayofweek
    df['WeekOfYear'] = df[col].dt.weekofyear
    return df

####################################
# Sales per customer
####################################

def sales_percust(df, by='Store', col1='Sales', col2='Customers'):
    """
    Returns the mean number
    of sales per customer
    per store
    """
    group = df.groupby(by=by).agg({col1: 'mean', col2: 'mean'})
    group['Sales_per_customer'] = group[col1] / group[col2]
    df['Sales_per_customer'] = df[by].map(group['Sales_per_customer'])
    return df

####################################
# Median customers per store
####################################

def cust_perstore(df, by='Store', col='Customers'):
    """
    Returns median number of
    customers per store
    """
    group2 = df.groupby(by=by).agg({col: 'median'})
    #group['Customer_per_store'] = group['Customers']
    df['Customer_per_store'] = df[by].map(group2[col])
    return df

####################################
# CompetitionOpenSinceYearMonth
####################################


    

def compet_openmonths(df, yr='CompetitionOpenSinceYear', mth='CompetitionOpenSinceMonth'):
    """
    Extract the number of months
    when competition open
    """
    
    df['CompetitionOpenSincePeriod'] = (12 * (df['Year'] - df[yr])) + (df['Month'] - df[mth])

    df['HasCompetitor'] = (df['CompetitionOpenSincePeriod'] > 0).astype('int')

    df['CompetitionOpenSincePeriod'] = np.where(df['CompetitionOpenSincePeriod'] < 0, 
                                                            0, 
                                                            df['CompetitionOpenSincePeriod'])

    df = df.drop([mth], axis=1)
    return df

####################################
# Promo Open Since Weeks
####################################

def promo_openweeks(df, yr='Promo2SinceYear', week='Promo2SinceWeek'):
    """
    Extract the number of weeks
    when promo was running
    """
    df['Promo2SincePeriod'] = (52 * (df['Year'] - 
                                            df[yr]
                                           ) + (df['WeekOfYear'] - 
                                                df[week]
                                               )
                                     )

    df = df.drop([week], axis=1)
    return df

####################################
# Drop rows where promo is nan
####################################

def drop_napromo(df, p1='Promo', p2='Promo2'):
    """
    Drop nan rows for Promo
    and Promo2
    """
    missing_promo = df[p1].isnull() | df[p2].isnull()
    df = df.loc[~missing_promo]
    return df

###################################
# Drop rows with missing Sales
# in case Store is Open
###################################

def dropnasales_ifopen(df, drop='Sales'):
    """
    """
    not_open = df['Open']==0
    no_customer = df['Customers']==0
    missing_sales = df[drop].isnull()

    df = df.loc[~((not_open | no_customer) & missing_sales), :]
    return df

####################################
# Impute missing Sales
####################################

def impute_sales(df):
    """
    Impute Sales based on average
    Sales per Customer times average
    Customers per Store
    """
    missing_sales = df.Sales.isnull()

    df.loc[missing_sales, 'Sales'] = df.loc[missing_sales, 'Customer_per_store'] * df.loc[missing_sales, 'Sales_per_customer']
    return df

####################################
# Impute State holiday #
####################################

def impute_holiday(df):
    years = df.Date.dt.year.unique()
    national_holidays = [day for day in holidays.Germany(years=years)]

    missing_holiday = df.StateHoliday.isnull()
    holiday_date = df.Date.isin(national_holidays)

    df.loc[missing_holiday & holiday_date, 'StateHoliday'] = 'a'
    df.loc[missing_holiday & ~holiday_date, 'StateHoliday'] = '0'
    return df

#####################################
# Drop customers
#####################################

def drop_cust(df, cols=['Customers']):
    df = df.drop(cols, axis=1)
    return df

####################################
# Drop na 
####################################

def drop_na(df, cols=['Open', 'SchoolHoliday', 'CompetitionDistance']):
    df = df.dropna(subset=cols)
    return df

####################################
# Group by mean encoding
###################################

def groupby_meanenc(df, by='Store', on='Sales'):
    df[by+'_menc'] = df.groupby(by).transform('mean')[on]
    df = df.drop(by, axis=1)
    return df

####################################
# One hot encoding
####################################

def onehot(df, cols=['StoreType', 'PromoInterval']):
    df = pd.get_dummies(df, 
                        columns=cols, 
                        drop_first=True)
    return df

####################################
# Ordinal encoding
####################################

def ordinal_enc(df, col='Assortment'):
    enc_assort = {'a':1, 'b':2, 'c':3}
    df[col] = df[col].map(enc_assort)
    return df

###################################
# Quantile based categorizer
###################################

def compdist_cat(df, q=3):
    df['CompDistCategories'] = pd.qcut(df.CompetitionDistance, q=q)
    df = pd.get_dummies(df, columns=['CompDistCategories'], drop_first=False)
    return df

###################################
# StateHolidays binarizer
###################################

def holi_binarizer(df):
    state_holiday_enc = {'0':0, 0.0:0, 'a':1, 'b':1, 'c':1}
    df['StateHoliday'] = df.StateHoliday.map(state_holiday_enc)
    return df
    
    
    