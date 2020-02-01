import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from yellowbrick.model_selection import ValidationCurve, FeatureImportances

# Local variables
train_path = 'data/train_set.csv'
test_path = 'data/test_set.csv'
randomState = 42

# read DataFrame
data = pd.read_csv(train_path, index_col=0)
data_test = pd.read_csv(test_path, index_col=0)

# separate labels
X_train = data[data.columns[:-16]]
X_test = data_test[data_test.columns[:-16]]
y_location_trains = data[data.columns[-16:]]
y_location_tests = data_test[data.columns[-16:]]
# Use copper as test


scores = []
for i in np.arange(len(y_location_trains.columns)):
    print(y_location_trains.columns[i])
    y_train = y_location_trains[y_location_trains.columns[i]]
    y_test = y_location_tests[y_location_tests.columns[i]]
    C =np.logspace(0, 4, 10)
    penalty = ['l2']
    logistic = LogisticRegression(solver='lbfgs')
    hyperparameters = dict(C=C, penalty=penalty)
    clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
    print('Training model...')
    best_model = clf.fit(X_train, y_train)
    print('Predicting targets...')
    y_pred_Logisticregression=best_model.predict(X_test)
  
    logistic_score = balanced_accuracy_score(y_test, y_pred_Logisticregression.round(), adjusted=False)
    #logistic_score_acc=accuracy_score(y_test, y_pred_Logisticregression)
    scores.append(logistic_score)
    print(logistic_score)
    #print(logistic_score_acc)


    
    
    #from sklearn.metrics import accuracy_score
    #logistic_score = balanced_accuracy_score(test_scores_encoded, y_pred_Logisticregression.round(), adjusted=False)
    #logistic_score_acc=accuracy_score(test_scores_encoded, y_pred_Logisticregression.round(), normalize=False)
    #print(logistic_score)
    #print(logistic_score_acc)
data_score = pd.DataFrame(columns=['Commodity', 'score'])
data_score['Commodity'] = y_location_trains.columns
data_score['score'] = scores
print(data_score)
data_score.to_csv('/Users/monalisa/Downloads/mmai823-project-master/out/linear_scores.csv')


print(X_train.columns)
# Feature importance
#viz_feat = FeatureImportances(rfc, labels=X_train.columns, relative=False)
from matplotlib import pyplot as plt
%matplotlib inline
viz_features = FeatureImportances(logistic, labels=X_train.columns)
viz_features.fit(X_train, y_train)
viz_features.show()
plt.tight_layout()
