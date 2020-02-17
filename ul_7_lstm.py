import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import util_lstm as lstm

# set Pandas options
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 500)

# Local variables
randomState = 42
train_split = 0.8
n_features = 66
n_lag = 40
n_neurons = 10
n_epoch = 20
n_batch = 1
datapath = ['out/supervised_data/H09_supervised.csv', 'out/supervised_data/H10_supervised.csv',
            'out/supervised_data/H11_supervised.csv', 'out/supervised_data/H12_supervised.csv',
            'out/supervised_data/H13_supervised.csv', 'out/supervised_data/H14_supervised.csv',
            'out/supervised_data/H15_supervised.csv', 'out/supervised_data/H16_supervised.csv',
            'out/supervised_data/H17_supervised.csv']

df = pd.read_csv(datapath[0], index_col=0)

# split data
X_train, X_test, y_train, y_test = lstm.train_test_split_lstm(df, n_lag=n_lag, n_features=n_features,
                                                              split=train_split)
# Instantiate and compile model
model = lstm.build_lstm(X_train, y_train, n_lag=n_lag, n_seq=1, n_features=n_features,
                        n_batch=n_batch, n_neurons=n_neurons)

scl = StandardScaler()

for path in datapath:
    df = pd.read_csv(path, index_col=0)
    print('Training on: ' + path[-18:-15])

    # split data
    X_train, X_test, y_train, y_test = lstm.train_test_split_lstm(df, n_lag=n_lag, n_features=n_features,
                                                                  split=train_split)
    y_train = scl.fit_transform(y_train)
    y_test = scl.fit_transform(y_test)

    # fit model
    model = lstm.fit_lstm(model, X_train, X_test, y_train, y_test, n_epoch=n_epoch, n_batch=n_batch)

# load unseen data
testdatapath = ['out/supervised_data/H18_supervised.csv',
                'out/supervised_data/H19_supervised.csv']

for path in testdatapath:
    df = pd.read_csv(path, index_col=0)
    # split data
    X_train, X_test, y_train, y_test = lstm.train_test_split_lstm(df, n_lag=n_lag, n_features=n_features,
                                                                  split=0.002)

    # make some predictions
    y_pred = model.predict(X_test, batch_size=n_batch)
    df_test = pd.DataFrame(y_test)
    df_pred = pd.DataFrame(scl.inverse_transform(y_pred))
    pred = pd.concat([df_test, df_pred], axis=1)
    pred.to_csv('out/predictions_mse_20_' + path[-18:-15] + '.csv')

    lstm.evaluate_forecasts(y_test, y_pred, 10)


