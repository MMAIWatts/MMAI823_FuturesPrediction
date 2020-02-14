import pandas as pd
import numpy as np
import datetime
from sklearn.impute import SimpleImputer

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
nan_imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

# impute finance targets
df.iloc[:, -65:] = nan_imp.fit_transform(df.iloc[:, -65:])
print(df.describe())

print(df.head(20))
data = df.iloc[:, :-65]
data.index = pd.to_datetime(data.index)
targets = df.iloc[:, -65:]
targets.index = pd.to_datetime(targets.index, )

X, y = list(), list()

for i, row in df_dates.iterrows():
    cstart = datetime.datetime.fromisoformat(row['Start'])
    end = datetime.datetime.fromisoformat(row['End'])
    # create batches for train-test
    if row['Contract'][0] != 'H':
        np.savetxt('out/X_Htrain.csv' , X, delimiter=',')
        break
    for x in range(len(data)):
        start = cstart + datetime.timedelta(days=2*x)
        in_e = start + datetime.timedelta(days=40)
        out_s = start + datetime.timedelta(days=40)
        out_e = out_s + datetime.timedelta(days=10)
        if out_e <= end:
            X.append(data.loc[start:in_e, :].to_numpy())
            y.append(targets.loc[out_s:out_e, row['Contract']].to_numpy())
        else:
            break
