import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# set Pandas options
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 500)


def train_test_split_lstm(data, n_lag, n_features, split=0.8):
    """
    Prepares and splits data for use with an LSTM model.
    :param data: All input (merged features and targets)
    :param n_lag: Number of days of lag on the input
    :param n_features: Number of features in the input
    :param split: Percentage of data to be used for training
    :return: X_train, X_test, y_train, y_test
    """
    values = data.values
    n_train_days = int(values.shape[0] * split)
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    print('Training on ' + str(train.shape[0]) + ' days')
    print('Testing on ' + str(test.shape[0]) + ' days')
    X_train, y_train = train[:, :n_lag * n_features], train[:, n_lag * n_features:]
    X_test, y_test = test[:, :n_lag * n_features], test[:, n_lag * n_features:]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_test, y_train, y_test


def build_lstm(X, y, n_lag, n_seq, n_features, n_batch, n_neurons):
    """
    Builds the LSTM model using the following parameters
    :param X: Training data
    :param y: Trainin labels
    :param n_lag: Testing Data number of lag days
    :param n_seq:
    :param n_features:
    :param n_batch:
    :param n_neurons:
    :return: Compiled LSTM model
    """
    # reshape data into [samples, timesteps, features]

    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]),
                   stateful=False))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='mae')

    print(model.summary())

    return model


def fit_lstm(model, X_train, X_test, y_train, y_test, n_epoch, n_batch):
    history = model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch,
                        validation_data=(X_test, y_test), verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    model.reset_states()
    return model


def evaluate_forecasts(y, pred, n_seq):
    for i in range(n_seq):
        rmse = sqrt(mean_squared_error(y[i], pred[i]))
        print('t+%d RMSE: %f' % ((i+1), rmse))
        r_sq = r2_score(y[i], pred[i])
        print('t+%d R-sq: %f' % ((i+1), r_sq))
        print('*****')
