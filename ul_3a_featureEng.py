import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 500)
# Filepaths
wdata_path = 'data/cleaned_weather_data.csv'

df_w = pd.read_csv(wdata_path, index_col=0)
df_w.index = pd.to_datetime(df_w.index)
print('Imported weather data...')
df_w.drop(['prcp_la', 'prcp_okee', 'tavg_la', 'tavg_okee'], axis=1, inplace=True)

print(df_w.head())
print(df_w.info())

# create daily norm values
daily_norm = df_w.groupby(df_w.index.dayofyear).mean()
daily_norm = daily_norm.add_prefix('norm')

# join norms to master
df = df_w.merge(daily_norm, how='left', left_on=df_w.index.dayofyear, right_on=daily_norm.index)
df.index = df_w.index
df.drop('key_0', axis=1, inplace=True)

labels = df.iloc[:, :22].add_prefix('diff').columns.values
for i in np.arange(0, 22):
    df[labels[i]] = df.iloc[:, i] - df.iloc[:, i + 22]

print(df.info())
df[['difftavg_tor', 'difftavg_lake', 'difftavg_moore', 'difftavg_orlan', 'difftavg_nyc']].plot(linewidth=0.5, alpha=0.8,
                                                                                               grid=True)
plt.show()

# imput both NaNs and '-' symbols
nan_imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
nan_mean = SimpleImputer(missing_values=np.nan, strategy='mean', )
imp = SimpleImputer(missing_values='-', strategy='constant', fill_value=0)

# impute
df.iloc[:, :11] = nan_imp.fit_transform(df.iloc[:, :11])
df.iloc[:, 11:] = nan_mean.fit_transform(df.iloc[:, 11:])
print(df.describe())

# normalize
scl = StandardScaler()
df_s = pd.DataFrame(scl.fit_transform(df))
df_s.columns = df.columns
df_s.index = df.index

print(df_s.describe())
# write to file
df.to_csv('out/master_weather_norm.csv')
