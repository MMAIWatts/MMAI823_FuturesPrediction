import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Filepaths
fdata_path = 'data/commodities_agg.csv'
wdata_path = 'data/cleaned_weather_data.csv'

df_f = pd.read_csv(fdata_path, index_col=0)
print('Imported financial data...')
df_w = pd.read_csv(wdata_path, index_col=0)
print('Imported weather data...')

# reindex finance data
df_f['Date'] = pd.to_datetime(df_f['Date'])
df_f.set_index('Date', inplace=True)

print(df_f.head())
print(df_f.info())
print(df_w.head())
print(df_w.info())

# Merge both target and feature tables
df = pd.merge(df_w, df_f, how='left', left_index=True, right_index=True)
print(df.info())

# imput both NaNs and '-' symbols
nan_imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
nan_mean = SimpleImputer(missing_values=np.nan, strategy='mean', )
imp = SimpleImputer(missing_values='-', strategy='constant', fill_value=0)
imp_df = nan_mean.fit_transform(df[df.columns[13:25]])
imp_df = nan_imp.fit_transform(df)
imp_df = pd.DataFrame(imp.fit_transform(imp_df))
imp_df.columns = df.columns
imp_df.index = df.index
print(imp_df.head())

imp_df.to_csv('out/joined_master.csv')
