import pandas as pd
import numpy as np
import datetime
from sklearn.impute import SimpleImputer

# set Pandas options
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 500)

# Filepaths
wdata_path = 'data/master_weather_norm.csv'
fdata_path = 'data/FCOJ/H_merged.csv'
dates_path = 'data/contract_dates.csv'

df_w = pd.read_csv(wdata_path, index_col=0)
df_w.index = pd.to_datetime(df_w.index)
print('Imported weather data...')
df_f = pd.read_csv(fdata_path, index_col=0)
df_f.index = pd.to_datetime(df_f.index)
print('Imported contract data...')
df_dates = pd.read_csv(dates_path)

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

start = datetime.datetime.fromisoformat('2007-01-03')
end = datetime.datetime.fromisoformat('2007-03-09')
X, y = list(), list()
# create batches for train-test
for _ in range(len(data)):
    in_e = start + datetime.timedelta(days=40)
    out_s = start + datetime.timedelta(days=40)
    out_e = out_s + datetime.timedelta(days=10)
    if out_e <= end:
        X.append(data.loc[start:in_e, :])
        y.append(targets.loc[start:out_e, 'H07'])
    start = start + datetime.timedelta(days=2)

