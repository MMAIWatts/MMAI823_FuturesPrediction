import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# Local variables
train_path = 'data/train_set.csv'
test_path = 'data/test_set.csv'
randomState = 42

# read DataFrame
df = pd.read_csv(train_path, index_col=0)
df_test = pd.read_csv(test_path, index_col=0)

# separate labels
X_train = df[df.columns[:-16]]
X_test = df_test[df_test.columns[:-16]]
y_trains = df[df.columns[-16:]]
y_tests = df_test[df.columns[-16:]]

scores = []
for i in np.arange(len(y_trains.columns)):
    print(y_trains.columns[i])
    y_train = y_trains[y_trains.columns[i]]
    y_test = y_tests[y_tests.columns[i]]
    rfc = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 25), random_state=randomState, )
    print('Training model...')
    rfc.fit(X_train, y_train)
    print('Predicting targets...')
    pred_y = rfc.predict(X_test)

    rfc_score = balanced_accuracy_score(y_test, pred_y)
    scores.append(rfc_score)
    print(rfc_score)

df_score = pd.DataFrame(columns=['Commodity', 'score'])
df_score['Commodity'] = y_trains.columns
df_score['score'] = scores
print(df_score)
df_score.to_csv('out/nn_scores4.csv')

# for i, val in enumerate(y_trains.columns):
#     y_train = y_trains[y_trains.columns[i]]
#     y_test = y_tests[y_tests.columns[i]]
#     rfc = MLPClassifier(hidden_layer_sizes=(100, 100, 50), random_state=randomState)
#     print('Training model...')
#     rfc.fit(X_train, y_train)
#     print('Predicting targets...')
#     pred_y = rfc.predict(X_test)
#
#     rfc_score = balanced_accuracy_score(y_test, pred_y)
#     print(rfc_score)

# Vizualization Curve, estimators


# Feature importance
# viz_feat = FeatureImportances(rfc, labels=X_train.columns, relative=False)
# viz_feat.fit(X_train, y_train)
# viz_feat.show()
