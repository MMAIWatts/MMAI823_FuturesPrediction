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
begin_date = datetime.date(year=2019, month=1, day=1)
end_date = begin_date.strftime("%Y-%m-%d")

# import list of locations
df_stations = pd.read_csv(station_path)
stations = df_stations.StationID
print('Fetching the following stations...')
print(df_stations)

# Station ID for the locations of interest
stationid = 'GHCND:MXM00076680'
datasetid = 'GHCND'

# urls
base_url_data = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'
df_weather = pd.DataFrame()

# fetch data from NOAA and store in csv
for s in stations:
    for i in np.arange(0, 12):
        start = begin_date + datetime.timedelta(weeks=int(i * 4))
        end = start + datetime.timedelta(weeks=4)
        d = get_weather(stationid=s, datasetid=datasetid, datatype='PRCP', start_date=start.strftime('%Y-%m-%d'),
                        end_date=end.strftime('%Y-%m-%d'), token=myToken, base_url=base_url_data)
        df_weather = df_weather.append(d, ignore_index=True)

# Write to CSV
df_weather.to_csv('out/raw_weather_data.csv')
