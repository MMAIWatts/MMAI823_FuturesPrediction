import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score

# local variables
train_path = 'data/train_set.csv'
test_path = 'data/test_set.csv'
randomState = 42

# read DataFrame
df_test = pd.read_csv(test_path, index_col=0)

# separate labels
X_test = df_test[df_test.columns[:-8]]
# Use copper as test
y_test = df_test[df_test.columns[-1]]

# predict random choice between 0 and 1, for length of y_test
y_pred = np.random.choice(1, len(y_test))

score = balanced_accuracy_score(y_test, y_pred)
print(score)
