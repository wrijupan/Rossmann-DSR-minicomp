def missing_report(df, pd):
    """
    Description:
    Reports the amount of null values as total and percentage, and data type for each column of a dataframe

    Input:
    - df: dataframe
    - pd: alias for pandas package

    Output:
    - report: dataframe containing information on null values and data types

    Example:
    import pandas as pd
    ...
    report = missing_report(df, pd)
    missing_report(df, pd)
    """
    null_total = df.isnull().sum()
    null_percentage = round(df.isnull().sum() / df.shape[0] * 100, 2)
    types = df.dtypes
    report = pd.concat([null_total, null_percentage, types], axis=1)
    report.columns = ['Null (total)', 'Null (percent)', 'Type']
    return report


"""
IMPUTATION OF MISSING VALUES
"""

def const_imputation(df, columns, values):
    """
    Description:
    Modifies a dataframe object by replacing na values with a constant

    Input:
    - df: dataframe
    - columns: list of column names
    - values: list of values or single value for replacing

    Output:
    - df: modified dataframe

    Example:
    df = const_imputation(df, ['column_A'], 100)
    df = const_imputation(df, ['column_A', 'column_B'], 100)
    df = const_imputation(df, ['column_A', 'column_B'], [100, 50])
    """
    df[columns] = df[columns].fillna(values, axis=0)
    return df


def mean_imputation(df, columns, enforce_int=False):
    """
    Description:
    Modifies a dataframe object by replacing na values with the column mean

    Input:
    - df: dataframe
    - columns: list of column names
    - enforce_int: flag for enforcing mean to be of integer type

    Output:
    - df: modified dataframe

    Example:
    df = mean_imputation(df, ['column_A'])
    df = mean_imputation(df, ['column_A'], enforce_int=True)
    df = mean_imputation(df, ['column_A','column_B'])
    """
    if enforce_int:
        mean_imp_values = round(df[columns].mean())
    else:
        mean_imp_values = df[columns].mean()

    df = df.fillna(mean_imp_values, axis=0)
    return df


def holiday_imputation(df, holidays):
    """
    Description:
    Imputes na values in 'StateHoliday' column with 'a'/'0' if 'Date' is/is not a national German holiday

    Input:
    - df: dataframe
    - holidays: python holidays package

    Output:
    - df: modified dataframe

    Example:
    import holidays
    ...
    df = holiday_imputation(df, holidays)
    """
    years = df.Date.dt.year.unique()
    national_holidays = [day for day in holidays.Germany(years=years)]
    missing_holiday = df.StateHoliday.isnull()
    holiday_date = df.Date.isin(national_holidays)
    df.loc[missing_holiday & holiday_date, 'StateHoliday'] = 'a'
    df.loc[missing_holiday & ~holiday_date, 'StateHoliday'] = '0'
    return df


"""
CATEGORICAL VALUE ENCODING
"""

def freq_encoding(df, column):
    """
    Description:
    Adds a column to a dataframe that contains the frequency encoding of a given feature

    Input:
    - df: dataframe
    - column: column name as string

    Output:
    - df: modified dataframe
    - fenc_values: series containing frequency encoding values

    Example:
    df, fenc_values = freq_encoding(df, 'column_A')
    """
    fenc_values = df[column].value_counts()
    new_column = column + '_fenc'
    df.loc[:, new_column] = df[column].map(fenc_values)
    return df, fenc_values

def mean_encoding(df, column):
    """
    Description:
    Adds a column to a dataframe that contains the mean encoding of a given feature

    Input:
    - df: dataframe
    - column: column names as string

    Output:
    - df: modified dataframe
    - mean_values: series containing mean encoding values

    Example:
    df, menc_values = mean_encoding(df, 'column_A')
    """
    menc_values = df.groupby(by = column).mean()['Sales']
    new_column = column + '_menc'
    df.loc[:, new_column] = df[column].map(menc_values)
    return df, menc_values

def ordinal_encoding(df, column, ordinal_dict):
    """
    Description:
    Adds a column to a dataframe that contains the ordinal encoding of a given feature

    Input:
    - df: dataframe
    - column: column name as string
    - ordinal_dict: dictionary containing ordinal encoding values

    Output:
    - df: modified dataframe

    Example:
    df = ordinal_encoding(df, 'column_A', {'large': 3, 'medium': 2, 'small': 1})
    """
    new_column = column + '_orenc'
    df[new_column] = df[column].map(ordinal_dict)
    return df

def binary_encoding(df, column, positive_list):
    """
    Description:
    Adds a column to a dataframe that contains the binary encoding of a given feature

    Input:
    - df: dataframe
    - column: column name as string
    - positive_list: list containing values to be encoded as one

    Output:
    - df: modified dataframe

    Example:
    df = ordinal_encoding(df, 'column_A', ['category_A', 'category_B', 'category_C'])
    """
    new_column = column + '_benc'
    df[new_column] = 0
    df.loc[df[column].isin(positive_list), new_column] = 1
    return df