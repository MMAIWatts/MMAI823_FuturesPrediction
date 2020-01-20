import datetime
import numpy as np
import pandas as pd
from util import get_weather

# NOAA API Token
myToken = 'oGjWvAZgOEWogyaRmrdJxJgyCnWwTMFq'
# Path to location list
station_path = 'data/city_codes.csv'

# get datetime of last year
lastyear = datetime.datetime.now() - datetime.timedelta(weeks=52)
lastyear_plus = lastyear + datetime.timedelta(weeks=2)

# set start and end dates, currently set to 1 day's data for testing
begin_date = lastyear.strftime("%Y-%m-%d")
end_date = lastyear_plus.strftime("%Y-%m-%d")

# import list of locations
df_stations = pd.read_csv(station_path)
stations = df_stations.StationID
print('Fetching the following stations...')
print(df_stations)

# location id for the locations of interest
locationid = 'FIPS:38'
stationid = 'GHCND:MXM00076680'
datasetid = 'GHCND'

# urls
base_url_data = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'
df_weather = pd.DataFrame()

for s in stations:
    d = get_weather(stationid=s, datasetid=datasetid, datatype='PRCP', start_date=begin_date, end_date=end_date,
                    token=myToken, base_url=base_url_data)
    df_weather = df_weather.append(d, ignore_index=True)

# separate for PRCP data
prcp = df_weather[df_weather['datatype'] == 'PRCP']
tavg = df_weather[df_weather['datatype'] == 'PRCP']

prcp = prcp.join(tavg, on=['date', 'station'], rsuffix='_1')

print(prcp)
# df_weather.to_csv('out/mx_test.csv')
