from utils import *

import os
import pandas as pd
import numpy as np
import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot

from sklearn                        import metrics, svm
from sklearn.linear_model           import LinearRegression
from sklearn.linear_model           import LogisticRegression
from sklearn.tree                   import DecisionTreeClassifier
from sklearn.ensemble               import RandomForestClassifier
from sklearn.neighbors              import KNeighborsClassifier
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis
from sklearn.naive_bayes            import GaussianNB
from sklearn.svm                    import SVC
from sklearn.metrics                import accuracy_score
from sklearn.metrics                import precision_score, recall_score, f1_score
from sklearn.model_selection        import train_test_split
from sklearn.preprocessing          import MinMaxScaler




def get_df(raw_data_folder_path, file_name):

    df = pd.read_csv(os.path.join(raw_data_folder_path, file_name), delimiter=r"\s+")
    
    return df



def remove_null_values(df):

    flag = False

    for i in df.describe().loc["min"]:
        if(i < -9000):
            flag = True
            break

    rows_to_remove = []

    if(flag == True):

        rows_to_remove = []

        for index in range(0, len(df)):
        
            for col in df.columns:

                if (df[f'{col}'].iloc[index] == -9999):

                    rows_to_remove.append(index)
                
        delete_rows = []
        [delete_rows.append(x) for x in rows_to_remove if x not in delete_rows]

        df.drop(delete_rows, inplace = True)

    return df



def date_columns(df):

    # Getting the day, month and year
    dates = np.char.mod('%d', df["Time"])

    years = []
    months = []
    days = []

    for date in dates:

        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])

        years.append(year)
        months.append(month)
        days.append(day)

    df["years"] = years
    df["months"] = months
    df["days"] = days

    df.drop(['Time'], axis=1, inplace = True)

    return df



def scale_date_variables(df):

    years = df["years"]
    months = df["months"]
    days = df["days"]
    
    scaler = MinMaxScaler()
    years = np.array(years).reshape(-1, 1)
    model = scaler.fit(years)
    scaled_years = model.transform(years)
    
    scaler = MinMaxScaler()
    months = np.array(months).reshape(-1, 1)
    model = scaler.fit(months)
    scaled_months = model.transform(months)
    
    scaler = MinMaxScaler()
    days = np.array(days).reshape(-1, 1)
    model = scaler.fit(days)
    scaled_days = model.transform(days)
    
    df["scaled_years"] = scaled_years
    df["scaled_months"] = scaled_months
    df["scaled_days"] = scaled_days
    
    df.drop(columns = ["years", "months", "days"], inplace = True)

    return df



def scale_features(df):

    X = df.loc[:, ['rh', 'SM', 'Tmean', 'e', 'SPI', 'scaled_years', 'scaled_months', 'scaled_days']]

    scaler = StandardScaler()
    scaler = scaler.fit(X)
    X_scaled = scaler.transform(X)

    return X_scaled


