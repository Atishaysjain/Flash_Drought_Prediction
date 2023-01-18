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

parser = argparse.ArgumentParser()

parser.add_argument("--num", type=int, default=0, help="The Number of latitudes and longitudes already processed")

args = parser.parse_args()

curr_dir_path = os.getcwd()
raw_data_folder_path = os.path.join(curr_dir_path, "Data")

if __name__ == '__main__':

    num = args.num

    os.chdir(raw_data_folder_path)
    list_files = os.listdir() # This stores the name of all files that have the data for each latitude and longitude
    os.chdir(curr_dir_path)

    files_remaining = list_files[num:]
    lstm_results = []

    for file_name in files_remaining:

        df = get_df(raw_data_folder_path, file_name) # Reading the file as DataFrame

        df = remove_null_values(df) # removing rows with missing values

        num += 1

        if(len(df) > 0): # If we have the values of all the features i.e. the value of a prticular feature is not Null for all the rows

            df = date_columns(df) # Adding year, month and day column dates

            df = scale_date_variables(df) # Scaling Date variables

            X_scaled = scale_features(df) # Gatting the scaled feature variables
            y = df['FLASH']

            # Now we will be converting the data into a format that could be used by the LSTM Model

            X_timeseries = []
            y_timeseries = []

            n_future = 1   # Number of days we want to look into the future based on the past days.
            n_past = 30  # Number of past days we want to use to predict the future.

            # Reformat input data into a shape: (n_samples x timesteps x n_features)

            for i in range(n_past, len(X_scaled) - n_future +1):
                X_timeseries.append(X_scaled[i - n_past:i])
                y_timeseries.append([y[i]])

            X_train, X_test, y_train, y_test = train_test_split(X_timeseries, y_timeseries, test_size=0.2, shuffle=True)
            X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

            # Creating the LSTM Model

            model = Sequential()
            model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
            model.add(LSTM(32, activation='relu', return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(y_train.shape[1]))

            model.compile(optimizer='adam', loss='mse')

            # Training the LSTM Model

            history = model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_test, y_test), verbose=0, shuffle=False)
            # plot history
            # pyplot.plot(history.history['loss'], label='train')
            # pyplot.plot(history.history['val_loss'], label='test')
            # pyplot.legend()
            # pyplot.show()

            # model = tf.keras.models.load_model(model_path)
            # model_path = f"/content/drive/MyDrive/Work/Machine Learning - FD/lstm_weights{file_name}.h5"
            # model.save(model_path)

            # Evaluating the LSTM Model

            y_pred = model.predict(X_test)
            y_pred = np.round(y_pred)

            if(len(np.unique(y_pred)) > 2):
                for prediction_index in range(0, len(y_pred)):
                    if(y_pred[prediction_index]>1):
                        y_pred[prediction_index] = 1
                    if(y_pred[prediction_index]<0):
                        y_pred[prediction_index] = 0

            if((len(np.unique(y_pred)) == 1) or (len(np.unique(y_test)) == 1)):
                accuracy = accuracy_score(y_test, y_pred)
                lstm_results.append([file_name, f'{np.round(accuracy*100, 2)}', f'{np.nan}', f'{np.nan}', f'{np.nan}', f'{num}', "\n"])

            else:
                print(np.unique(y_test), np.unique(y_pred))
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                lstm_results.append([file_name, f'{np.round(accuracy*100, 2)}', f'{np.round(precision*100, 2)}', f'{np.round(recall*100, 2)}', f'{np.round(f1*100, 2)}', f'{num}', "\n"])

            # Storing the LSTM results

            Results_dir_path = os.path.join(curr_dir_path, "Results")

            if os.path.exists(os.path.join(Results_dir_path, "lstm_results", "LstmResults.txt")) == 0:

                file1 = open(os.path.join(Results_dir_path, "lstm_results", "LstmResults.txt"), "w")
                file1.writelines(f"{lstm_results[-1][0]}, {lstm_results[-1][1]}, {lstm_results[-1][2]}, {lstm_results[-1][3]}, {lstm_results[-1][4]}, {num}\n")
                file1.close()
            
            else:

                file1 = open(os.path.join(Results_dir_path, "lstm_results", "LstmResults.txt"), "a")
                file1.writelines(f"{lstm_results[-1][0]}, {lstm_results[-1][1]}, {lstm_results[-1][2]}, {lstm_results[-1][3]}, {lstm_results[-1][4]}, {num}\n")
                file1.close()

    if (num%15 == 0):
        print(num)





