import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt

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

labels = df.iloc[:, :24].add_prefix('diff').columns.values
for i in np.arange(0, 24):
    df[labels[i]] = df.iloc[:, i] - df.iloc[:, i + 24]

print(df.info())
df[['tavg_moore', 'normtavg_moore', 'difftavg_moore']].plot(linewidth=0.5)
plt.show()

# imput both NaNs and '-' symbols
nan_imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
nan_mean = SimpleImputer(missing_values=np.nan, strategy='mean', )
imp = SimpleImputer(missing_values='-', strategy='constant', fill_value=0)

# impute
df.iloc[:, :11] = nan_imp.fit_transform(df.iloc[:, :11])
df.iloc[:, 11:] = nan_mean.fit_transform(df.iloc[:, 11:])
print(df.info())

# write to file
df.to_csv('out/master_weather.csv')
