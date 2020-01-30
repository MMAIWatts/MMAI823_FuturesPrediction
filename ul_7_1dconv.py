import pandas as pd
import numpy as np
from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, MaxPooling1D, Flatten
from keras.models import Model, Sequential
from sklearn.metrics import balanced_accuracy_score

# Local variables
train_path = 'data/train_set.csv'
test_path = 'data/test_set.csv'
randomState = 42

# read DataFrame
df = pd.read_csv(train_path, index_col=0)
df_test = pd.read_csv(test_path, index_col=0)

# separate labels
X_train = np.expand_dims(df[df.columns[:-16]], axis=2)
X_test = np.expand_dims(df_test[df_test.columns[:-16]], axis=2)
y_trains = df[df.columns[-16:]]
y_tests = df_test[df.columns[-16:]]

#  create input layer for time window of 20 days, width of 26 features
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(1545, 26)))
model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_trains[y_trains.columns[-1]], epochs=100, batch_size=10, verbose=0)
_, acc = model.evaluate(X_test, y_tests[y_tests.columns[-1]], batch_size=10, verbose=0)

print(acc)
