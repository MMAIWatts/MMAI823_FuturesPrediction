import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# local variables
d_path = 'data/joined_master.csv'
randomState = 42
train_test = 0.2

# read csv
df = pd.read_csv(d_path, index_col=0)

# drop un-need columns
df.drop(columns=df.columns[-16:], inplace=True)
print(df.info())

# Separate into features and targets
y = pd.DataFrame(df[df.columns[-8:]], columns=df.columns[-8:])
X = pd.DataFrame(df[df.columns[:-8]], columns=df.columns[:-8])

# Standard scaler
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(X))
scaled_df.columns = X.columns
scaled_df.index = X.index
X = scaled_df

print(X.info())
print(y.info())

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test, random_state=randomState)
df_test = pd.merge(X_test, y_test, left_index=True, right_index=True)
df_train = pd.merge(X_train, y_train, left_index=True, right_index=True)

# save to csv
df_test.to_csv('out/test_set.csv')
df_train.to_csv('out/train_set.csv')
