import pandas as pd
import numpy as np
import datetime
from sklearn.impute import SimpleImputer
from util import series_to_supervised

# set Pandas options
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 500)

# Filepaths
wdata_path = 'data/master_weather_norm.csv'
fdata_path = 'data/FCOJ/fcoj_merged.csv'
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
df = df.fillna(0, axis=0)

print(df.head(20))
data = df.iloc[:, :-65]
data.index = pd.to_datetime(data.index)
targets = df.iloc[:, -65:]
targets.index = pd.to_datetime(targets.index, )

# Configure number of lag and target days
n_lag = 40
n_seq = 10

for i, row in df_dates.iterrows():
    start = datetime.datetime.fromisoformat(row['Start'])
    end = datetime.datetime.fromisoformat(row['End'])

    # Slice data and targets to contract limits
    local_data = data.loc[start:end, :]
    local_targets = targets.loc[start:end, row['Contract']]

    # Convert to a supervised series
    supervised = series_to_supervised(local_data, local_targets, n_in=n_lag, n_out=n_seq, dropnan=True)

    supervised.to_csv('out/supervised_data/' + row['Contract'] + '_supervised.csv')

