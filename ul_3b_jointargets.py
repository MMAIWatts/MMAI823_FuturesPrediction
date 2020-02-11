import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# set Pandas options
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 500)

# Filepaths
wdata_path = 'data/master_weather_norm.csv'
fdata_path = 'data/FCOJ/H_merged.csv'

df_w = pd.read_csv(wdata_path, index_col=0)
df_w.index = pd.to_datetime(df_w.index)
print('Imported weather data...')
df_f = pd.read_csv(fdata_path, index_col=0)
df_f.index = pd.to_datetime(df_f.index)

print(df_w.head())
print(df_w.info())
print(df_f.head())
print(df_f.info())

# join norms to master
df = df_w.merge(df_f, how='left', left_index=True, right_index=True)

# imput missing values for targets
nan_imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

# impute finance targets
df.iloc[:, -13:] = nan_imp.fit_transform(df.iloc[:, -13:])
print(df.describe())

print(df.head(20))
data = df.iloc[:, :-13]
targets = df.iloc[:, -13:]

# split data
train_test = 0.8
split = int(train_test*len(data))
X_train = data.iloc[:split]
y_train = targets.iloc[:split]
X_test = data.iloc[split:]
y_test = targets.iloc[split:]

X_train.to_csv('out/X_train.csv')
X_test.to_csv('out/X_test.csv')
y_train.to_csv('out/y_train.csv')
y_test.to_csv('out/y_test.csv')
