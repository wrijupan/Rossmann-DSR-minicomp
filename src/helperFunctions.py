def missing_report(df, percentage=False):
    """
    Description:
    Lists all columns of a dataframe together with information on null values

    Input:
    - df: dataframe
    - percentage: flag for showing percentage of null values (default 'False' means totals are shown)

    Output:
    - report: series containing information on null values
    """
    if percentage == False:
        report = df.isnull().sum()
    else:
        report = round(df.isnull().sum() / df.shape[0] * 100, 2)
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
    - values: list of values for replacing

    Output:
    - df: modified dataframe
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
    - mean_imp_values: series containing calculated mean
    """
    if enforce_int:
        mean_imp_values = np.floor(df[columns].mean())
    else:
        mean_imp_values = df[columns].mean()

    df = df.fillna(mean_imp_values, axis=0)
    return df, mean_imp_values


"""
CATEGORICAL VALUE ENCODING
"""

def freq_encoding(df, column):
    """
    Description:
    Adds a column to a dataframe that contains the frequency encoding of a given feature

    Input:
    - df: dataframe
    - column: column

    Output:
    - df: modified dataframe
    - fenc_values: series containing frequency encoding values
    """
    fenc_values = df[column].value_counts()
    new_column = column + '_fenc'
    df[new_column] = df[column].map(fenc_values)
    return df, fenc_values

def mean_encoding(df, column):
    """
    Description:
    Adds a column to a dataframe that contains the mean encoding of a given feature

    Input:
    - df: dataframe
    - column: column

    Output:
    - df: modified dataframe
    - mean_values: series containing mean encoding values
    """
    menc_values = df.groupby(by = column).mean()['Sales']
    df[column + '_menc'] = df[column].map(menc_values)
    return df, menc_values

def ordinal_encoding(df, column, ordinal_dict):
    """
    Description:
    Adds a column to a dataframe that contains the ordinal encoding of a given feature

    Input:
    - df: dataframe
    - column: column
    - ordinal_dict: dictionary containing ordinal encoding values

    Output:
    - df: modified dataframe
    """
    df[column + '_orenc'] = df[column].map(ordinal_dict)
    return df