import pandas as pd
import numpy as np
from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, MaxPooling1D, GlobalAveragePooling1D
from keras.models import Model, Sequential
from keras.optimizers import SGD
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
y_trains = df[df.columns[-8:]]
y_tests = df_test[df.columns[-8:]]

print(X_train.shape)

#  create input layer, width of 26 features
model = Sequential()
model.add(Conv1D(filters=12, kernel_size=20, activation='relu', input_shape=(26, 1)))
# model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
# model.add(MaxPooling1D(pool_size=3))
# model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
# model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(GlobalAveragePooling1D())
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='softmax'))
# model.add(Dense(100, activation='softmax'))
# model.add(Dense(10, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
print('Model compiled...')
print(model.summary())

scores = []
for i in np.arange(len(y_tests.columns)):
    # Fit Model
    print(y_trains.columns[i])
    y_train = y_trains[y_trains.columns[i]]
    y_test = y_tests[y_tests.columns[i]]
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
    print('Model fit...')
    # Score model
    score = model.evaluate(X_test, y_test, batch_size=10)
    scores.append(score[1])
    print(score)

df_score = pd.DataFrame(columns=['Commodity', 'score'])
df_score['Commodity'] = y_trains.columns
df_score['score'] = scores
print(df_score)
df_score.to_csv('out/conv1d_scores4.csv')

# Make some predictions
# y_pred = model.predict(X_test)
# y_pred_keras = [round(x[0]) for x in y_pred]
# bscore = balanced_accuracy_score(y_tests[y_tests.columns[-1]], y_pred)
# print(bscore)
