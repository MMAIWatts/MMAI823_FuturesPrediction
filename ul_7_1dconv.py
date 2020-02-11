import pandas as pd
import numpy as np
from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, MaxPooling1D, GlobalAveragePooling1D
from keras.models import Model, Sequential
from keras.optimizers import SGD
from sklearn.metrics import balanced_accuracy_score

# set Pandas options
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 500)

# Local variables
X_train_path = 'data/X_train.csv'
X_test_path = 'data/X_test.csv'
y_train_path = 'data/y_train.csv'
y_test_path = 'data/y_test.csv'
randomState = 42

# read DataFrames
X_train = np.expand_dims(pd.read_csv(X_train_path, index_col=0), axis=2)
# X_train.index = pd.to_datetime(X_train.index)
X_test = np.expand_dims(pd.read_csv(X_test_path, index_col=0), axis=2)
# X_test.index = pd.to_datetime(X_test.index)

y_train = pd.read_csv(y_train_path, index_col=0)
# y_train.index = pd.to_datetime(y_train.index)
y_test = pd.read_csv(y_test_path, index_col=0)
# y_test.index = pd.to_datetime(y_test.index)

print(X_train.shape)

#  create input layer, width of 26 features
model = Sequential()
model.add(Conv1D(filters=12, kernel_size=20, strides=2, activation='relu', input_shape=(66, 1)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
print('Model compiled...')
print(model.summary())

for i in np.arange(len(y_train) - 1):
    print(y_train.columns[i])
    model.fit(X_train, y_train.iloc[:, i], batch_size=10, epochs=5, verbose=True)

for i in np.arange(len(y_test) - 1):
    print(y_test.columns[i])
    score = model.evaluate(X_test, y_test.iloc[:, i], batch_size=10)
    print(score)
