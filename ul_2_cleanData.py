import datetime
import pandas as pd


# Define local variables
wdata1_path = 'out/2019_raw_weather_data.csv'
wdata2_path = 'out/2018_raw_weather_data.csv'
wdata3_path = 'out/2017_raw_weather_data.csv'
wdata3_path = 'out/2016_raw_weather_data.csv'
wdata4_path = 'out/2015_raw_weather_data.csv'
wdata5_path = 'out/2014_raw_weather_data.csv'
wdata6_path = 'out/2013_raw_weather_data.csv'
wdata7_path = 'out/2012_raw_weather_data.csv'
wdata8_path = 'out/2011_raw_weather_data.csv'
wdata9_path = 'out/2010_raw_weather_data.csv'
wdata10_path = 'out/2009_raw_weather_data.csv'
wdata11_path = 'out/2008_raw_weather_data.csv'
wdata12_path = 'out/2007_raw_weather_data.csv'


# Read CSV and concat
df_weather = pd.concat([pd.read_csv(wdata1_path, index_col=0), pd.read_csv(wdata2_path, index_col=0),
                        pd.read_csv(wdata3_path, index_col=0), pd.read_csv(wdata4_path, index_col=0),
                        pd.read_csv(wdata5_path, index_col=0), pd.read_csv(wdata6_path, index_col=0),
                        pd.read_csv(wdata7_path, index_col=0), pd.read_csv(wdata8_path, index_col=0),
                        pd.read_csv(wdata9_path, index_col=0), pd.read_csv(wdata10_path, index_col=0),
                        pd.read_csv(wdata11_path, index_col=0), pd.read_csv(wdata12_path, index_col=0),
                        ], ignore_index=True)

# Separate PRCP (precipitation) data, drop unneeded columns
prcp = df_weather[df_weather['datatype'] == 'PRCP']
prcp.drop(['datatype', 'attributes'], axis=1, inplace=True)
# Separate TAVG and TOBS (Average Temperature) data, drop unneeded columns
tavg = df_weather[df_weather['datatype'] == ('TAVG' | 'TOBS')]
tavg.drop(['datatype', 'attributes'], axis=1, inplace=True)

# merge together, creating features out of the PRCP and TAVG values
df = prcp.merge(tavg, on=['date', 'station'])

# reset index to fix inconsistency in merge
df.reset_index(drop=True, inplace=True)

# relabel columns
df.columns = ['date', 'station', 'precip', 'avg_temp']

# reformat date since no time needed, only daily summaries
df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")

print(df.head())

df.drop_duplicates(inplace=True)
df = df.pivot(index='date', columns='station', values=['precip', 'avg_temp'])
print(df.head())
print(df.info())

df.to_csv('out/cleaned_weather_data.csv')
