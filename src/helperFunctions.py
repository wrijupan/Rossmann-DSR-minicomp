def missing_report(df, percentage=False):
    if percentage == False:
        report = df.isnull().sum()
    else:
        report = round(df.isnull().sum() / df.shape[0] * 100, 2)
    return report

############################
# Missing value imputation #
############################
def const_imputation(df, columns, values):
    df[columns] = df[columns].fillna(values, axis=0)
    return df

def mean_imputation(df, columns, enforce_int=False):
    if enforce_int:
        mean_imp_values = np.floor(df[columns].mean())
    else:
        mean_imp_values = df[columns].mean()
    return df.fillna(mean_imp_values, axis=0), mean_imp_values

##############################
# Categorical value encoding #
##############################
def freq_encoding(df, column):
    fenc_values = Counter(df[column])
    new_column = column + '_fenc'
    df[new_column] = df[column].map(fenc_values)
    return df, fenc_values

def mean_encoding(df, column):
    menc_values = df.groupby(by = column).mean()['Sales']
    df[column + '_menc'] = df[column].map(menc_values)
    return df, menc_values

def ordinal_encoding(df, column, ordinal_dict):
    df[column + '_orenc'] = df[column].map(ordinal_dict)
    return df